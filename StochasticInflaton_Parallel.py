import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

# Configuración de constantes (ajustar según necesidades)
H_inf = torch.tensor(1e-5)        # Escala de inflación ~ 10¹³ GeV
mass = torch.tensor(1e-6)         # Consistente con n_s ≈ 0.965
phi_0 = torch.tensor(15.0)        # Desde el plateau del potencial
sigma = torch.tensor(0.1)         # Más pequeño para mejor separación de modos
tau_init = torch.tensor(-20.0)
tau_fin = torch.tensor(-0.1)

# =============================================================================
# FUNCIONES COSMOLÓGICAS VECTORIZADAS
# =============================================================================

def cosmological_a(tau):
    return -1/(H_inf * tau)

def k_sigma_func(tau):
    return sigma * cosmological_a(tau) * H_inf

def window_derivative_de_sitter_vectorized(K, tau, sigma, delta=0.6):
    """Versión completamente vectorizada para tensores 3D"""
    k_sigma_val = k_sigma_func(tau)
    dk_sigma_dtau = sigma / (tau**2)
    
    x = K / k_sigma_val
    z = (1 - x) / delta
    
    # Cálculo numéricamente estable de sech²(z)
    mask = torch.abs(z) <= 20
    result = torch.zeros_like(K)
    
    if mask.any():
        z_valid = z[mask]
        exp_z = torch.exp(z_valid)
        exp_neg_z = torch.exp(-z_valid)
        cosh_z = (exp_z + exp_neg_z) / 2
        sech2 = 1.0 / (cosh_z * cosh_z)
        
        result[mask] = (K[mask] / (2 * delta * k_sigma_val**2)) * dk_sigma_dtau * sech2
    
    return result

def phi_k_dS_vectorized(k, tau):
    """Versión vectorizada de phi_k_dS"""
    a = cosmological_a(tau)
    # Evitar división por cero para k=0
    k_safe = k.clone()
    k_safe[k == 0] = 1e-10
    return (1/a * torch.sqrt(2*k_safe)) * (1 + 1j/k_safe*tau) * torch.exp(1j*k_safe*tau)

def pi_k_dS_vectorized(k, tau):
    """Versión vectorizada de pi_k_dS"""
    return torch.sqrt(k/2) * torch.exp(1j*k*tau)

def creat_gaussian_oper_vectorized(shape, device):
    """Generación vectorizada de operadores gaussianos"""
    X = torch.randn(shape, dtype=torch.complex64, device=device)
    Y = torch.randn(shape, dtype=torch.complex64, device=device)
    return (X + 1j * Y) / torch.sqrt(torch.tensor(2.0, dtype=torch.complex64, device=device))

# =============================================================================
# CÁLCULO DE RUIDO OPTIMIZADO
# =============================================================================

def noise_integrand_vectorized(mesh, tau, sigma, field_momentum, device):
    """Versión vectorizada - elimina el triple bucle"""
    K = mesh['K'].to(device)
    
    # Vectorizar window_derivative
    w_dot = window_derivative_de_sitter_vectorized(K, tau, sigma)
    
    # Vectorizar modos de Bunch-Davies
    if field_momentum == 'phi':
        mode_func = phi_k_dS_vectorized(K, tau)
    elif field_momentum == "pi":
        mode_func = pi_k_dS_vectorized(K, tau)
    
    # Generar operadores gaussianos vectorizados
    A_k = creat_gaussian_oper_vectorized(K.shape, device)
    
    f_k = w_dot * mode_func * A_k
    
    # Manejar k = 0
    f_k[K == 0] = 0.0
    
    return f_k

def calc_noise_vectorized(mesh, tau, sigma, field_momentum, device):
    """Versión optimizada del cálculo de ruido"""
    # Calcular integrando vectorizado
    f_k = noise_integrand_vectorized(mesh, tau, sigma, field_momentum, device)
    
    # FFT inversa
    F_x = torch.fft.ifftn(f_k, norm='ortho')
    
    # Factor de conversión
    dk = 2 * torch.pi / mesh['L']
    integral_factor = (dk**3) / (2*torch.pi)**3
    
    noise_field = torch.real(F_x) * integral_factor
    
    return noise_field

# =============================================================================
# CONFIGURACIÓN DEL ENTORNO PARALELO
# =============================================================================

def setup_parallel_environment(rank, world_size):
    """Configuración del entorno distribuido"""
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        device = torch.device('cpu')
    
    return device

# =============================================================================
# EVOLUCIÓN DE LANGEVIN PARALELIZADA
# =============================================================================

class ParallelLangevinEvolver:
    def __init__(self, mesh, tau_array, sigma, mass, device):
        self.mesh = {k: v.to(device) for k, v in mesh.items()}
        self.tau_array = tau_array.to(device)
        self.sigma = sigma.to(device)
        self.mass = mass.to(device)
        self.device = device
        self.delta_tau = tau_array[1] - tau_array[0]
        
        # Precalcular cantidades que se usan frecuentemente
        self.precomputed = self._precompute_frequent_quantities()
    
    def _precompute_frequent_quantities(self):
        """Precalcular cantidades que se usan repetidamente"""
        precomputed = {}
        for tau in self.tau_array:
            a = cosmological_a(tau)
            precomputed[tau.item()] = {
                'a': a,
                'a3': a**3
            }
        return precomputed
    
    def langevin_step_vectorized(self, phi_actual, pi_actual, tau_actual, N=1):
        """Paso de Langevin vectorizado"""
        a_data = self.precomputed[tau_actual.item()]
        a3 = a_data['a3']
        
        # Términos deterministas
        drift_phi = (N / a3) * pi_actual * self.delta_tau
        drift_pi = (-N * a3 * self.mass**2 * phi_actual) * self.delta_tau
        
        # Ruido estocástico
        xi_phi = calc_noise_vectorized(self.mesh, tau_actual, self.sigma, "phi", self.device)
        xi_pi = calc_noise_vectorized(self.mesh, tau_actual, self.sigma, "pi", self.device)
        
        # Euler-Mayurama
        phi_next = phi_actual + drift_phi + xi_phi
        pi_next = pi_actual + drift_pi + xi_pi
        
        return phi_next, pi_next
    
    def evolve_single_universe(self, phi_0, pi_0):
        """Evolución para un único universo"""
        total_steps = len(self.tau_array)
        n_x, n_y, n_z = phi_0.shape
        
        phi_evolution = torch.zeros((total_steps, n_x, n_y, n_z), device=self.device)
        pi_evolution = torch.zeros((total_steps, n_x, n_y, n_z), device=self.device)
        
        phi_actual = phi_0.clone()
        pi_actual = pi_0.clone()
        
        phi_evolution[0] = phi_actual
        pi_evolution[0] = pi_actual
        
        for i_step in range(1, total_steps):
            tau_actual = self.tau_array[i_step - 1]
            
            phi_next, pi_next = self.langevin_step_vectorized(
                phi_actual, pi_actual, tau_actual
            )
            
            phi_actual, pi_actual = phi_next, pi_next
            phi_evolution[i_step] = phi_actual
            pi_evolution[i_step] = pi_actual
        
        return phi_evolution, pi_evolution

# =============================================================================
# GENERACIÓN DE MALLA
# =============================================================================

def universe(points, lenght, device):
    """Crear malla computacional en el dispositivo especificado"""
    dx = lenght / points

    # Crear la malla espacial en el dispositivo
    x = torch.linspace(-lenght/2, lenght/2, points, endpoint=False, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

    # Espacio de frecuencias
    kx = 2 * torch.pi * torch.fft.fftfreq(points, d=dx, device=device)
    ky = 2 * torch.pi * torch.fft.fftfreq(points, d=dx, device=device)
    kz = 2 * torch.pi * torch.fft.fftfreq(points, d=dx, device=device)
    
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)

    return {
        'N': points, 'L': lenght, 'dx': dx,
        'X': X, 'Y': Y, 'Z': Z,
        'KX': KX, 'KY': KY, 'KZ': KZ, 'K': K
    }

# =============================================================================
# REALIZACIONES DISTRIBUIDAS DE UNIVERSOS
# =============================================================================

def distributed_universe_realizations(rank, world_size, n_universes, phi_0_base, pi_0_base, 
                                    mesh, tau_array, sigma, mass, save_last_n=10):
    """Distribuye el cálculo de múltiples universos across GPUs"""
    
    device = setup_parallel_environment(rank, world_size)
    
    # Dividir universos entre procesos
    universes_per_process = n_universes // world_size
    start_idx = rank * universes_per_process
    end_idx = start_idx + universes_per_process
    
    if rank == world_size - 1:  # Último proceso toma los sobrantes
        end_idx = n_universes
    
    # Inicializar el evolucionador
    evolver = ParallelLangevinEvolver(mesh, tau_array, sigma, mass, device)
    
    # Resultados locales
    local_phi_finals = []
    local_pi_finals = []
    local_full_evolutions = {
        'phi_point': [], 'pi_point': [], 
        'spatial_slices': []
    }
    
    # Punto de muestreo
    selected_point = (mesh['N']//2, mesh['N']//2, mesh['N']//2)
    xc, yc, zc = selected_point
    
    # Evolucionar universos locales con barra de progreso
    if rank == 0:
        pbar = tqdm(total=end_idx-start_idx, desc=f"Proceso {rank}")
    
    for i_local in range(start_idx, end_idx):
        # Perturbación inicial única para cada universo
        torch.manual_seed(i_local * 1000)
        initial_perturbation = torch.normal(0, 0.01, mesh['X'].shape, device=device)
        
        phi_0_current = phi_0_base.to(device) + initial_perturbation
        pi_0_current = pi_0_base.to(device) * torch.ones_like(mesh['X'])
        
        # Evolucionar
        phi_evolution, pi_evolution = evolver.evolve_single_universe(
            phi_0_current, pi_0_current
        )
        
        # Recolectar resultados
        phi_final = torch.mean(phi_evolution[-1])
        pi_final = torch.mean(pi_evolution[-1])
        
        local_phi_finals.append(phi_final.cpu())
        local_pi_finals.append(pi_final.cpu())
        
        # Guardar evoluciones detalladas para los últimos universos
        if i_local >= (end_idx - save_last_n):
            local_full_evolutions['phi_point'].append(
                phi_evolution[:, xc, yc, zc].cpu()
            )
            local_full_evolutions['pi_point'].append(
                pi_evolution[:, xc, yc, zc].cpu()
            )
            
            # Guardar slices espaciales
            spatial_slices = []
            frame_indices = torch.linspace(0, len(tau_array)-1, 
                                         min(50, len(tau_array)), dtype=torch.long)
            for frame_idx in frame_indices:
                spatial_slices.append(phi_evolution[frame_idx, :, :, 0].cpu())
            local_full_evolutions['spatial_slices'].append(spatial_slices)
        
        if rank == 0:
            pbar.update(1)
    
    if rank == 0:
        pbar.close()
    
    # Sincronizar entre procesos
    dist.barrier()
    
    # Recopilar resultados de todos los procesos
    all_phi_finals = [None for _ in range(world_size)]
    all_pi_finals = [None for _ in range(world_size)]
    all_full_evolutions = [None for _ in range(world_size)]
    
    dist.all_gather_object(all_phi_finals, local_phi_finals)
    dist.all_gather_object(all_pi_finals, local_pi_finals)
    dist.all_gather_object(all_full_evolutions, local_full_evolutions)
    
    # Combinar resultados
    phi_finals = torch.cat([torch.tensor(f) for f in all_phi_finals if f is not None])
    pi_finals = torch.cat([torch.tensor(f) for f in all_pi_finals if f is not None])
    
    # Combinar evoluciones completas
    combined_full_evolutions = {
        'phi_point': [],
        'pi_point': [],
        'spatial_slices': []
    }
    
    for evo in all_full_evolutions:
        if evo is not None:
            combined_full_evolutions['phi_point'].extend(evo['phi_point'])
            combined_full_evolutions['pi_point'].extend(evo['pi_point'])
            combined_full_evolutions['spatial_slices'].extend(evo['spatial_slices'])
    
    return phi_finals, pi_finals, combined_full_evolutions

# =============================================================================
# MATRIZ DE DIFUSIÓN OPTIMIZADA
# =============================================================================

def diffusion_matrix_vectorized(mesh, tau, sigma, delta_tau, realizations, device):
    """Cálculo vectorizado de la matriz de difusión"""
    n_x, n_y, n_z = mesh['X'].shape
    
    # Acumuladores en GPU
    xi_phi_phi = torch.zeros((n_x, n_y, n_z), device=device)
    xi_phi_pi = torch.zeros((n_x, n_y, n_z), device=device)
    xi_pi_phi = torch.zeros((n_x, n_y, n_z), device=device)
    xi_pi_pi = torch.zeros((n_x, n_y, n_z), device=device)
    
    # Paralelizar realizaciones
    for realization in range(realizations):
        torch.manual_seed(realization * 1000)
        
        xi_phi = calc_noise_vectorized(mesh, tau, sigma, "phi", device)
        xi_pi = calc_noise_vectorized(mesh, tau, sigma, "pi", device)
        
        xi_phi_phi += xi_phi * xi_phi
        xi_phi_pi += xi_phi * xi_pi
        xi_pi_phi += xi_pi * xi_phi
        xi_pi_pi += xi_pi * xi_pi
    
    # Promediar
    xi_phi_phi /= realizations
    xi_phi_pi /= realizations
    xi_pi_phi /= realizations
    xi_pi_pi /= realizations
    
    # Matrices de Pauli
    I = torch.eye(2, device=device)
    J_x = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.float32)
    J_z = torch.tensor([[1, 0], [0, -1]], device=device, dtype=torch.float32)
    
    # Calcular D_matrix vectorizada
    D_matrix = torch.zeros((n_x, n_y, n_z, 2, 2), device=device)
    
    # Aplicar fórmula de Venin vectorizada
    trace_part = 0.5 * (xi_phi_phi + xi_pi_pi)
    off_diag_part = 0.5 * (xi_phi_pi + xi_pi_phi)
    diag_diff_part = 0.5 * (xi_phi_phi - xi_pi_pi)
    
    D_matrix[..., 0, 0] = trace_part + diag_diff_part
    D_matrix[..., 0, 1] = off_diag_part
    D_matrix[..., 1, 0] = off_diag_part
    D_matrix[..., 1, 1] = trace_part - diag_diff_part
    
    # Normalizar
    D_matrix = D_matrix / delta_tau
    
    return D_matrix

def distributed_diffusion_cache(rank, world_size, mesh, tau_array, sigma, delta_tau, realizations):
    """Cache distribuido de matrices de difusión"""
    device = setup_parallel_environment(rank, world_size)
    
    # Dividir tau_array entre procesos
    tau_per_process = len(tau_array) // world_size
    start_idx = rank * tau_per_process
    end_idx = start_idx + tau_per_process
    
    if rank == world_size - 1:
        end_idx = len(tau_array)
    
    local_D_cache = {}
    
    if rank == 0:
        pbar = tqdm(total=end_idx-start_idx, desc="Calculando matriz de difusión")
    
    for i in range(start_idx, end_idx):
        tau = tau_array[i]
        D_matrix = diffusion_matrix_vectorized(
            mesh, tau, sigma, delta_tau, realizations, device
        )
        local_D_cache[tau.item()] = D_matrix.cpu()  # Mover a CPU para reducir memoria GPU
        
        if rank == 0:
            pbar.update(1)
    
    if rank == 0:
        pbar.close()
    
    # Sincronizar y combinar
    dist.barrier()
    all_caches = [None for _ in range(world_size)]
    dist.all_gather_object(all_caches, local_D_cache)
    
    # Combinar caches
    combined_cache = {}
    for cache in all_caches:
        if cache is not None:
            combined_cache.update(cache)
    
    return combined_cache

# =============================================================================
# FUNCIÓN PRINCIPAL DISTRIBUIDA
# =============================================================================

def run_distributed_simulation(rank, world_size, config):
    """Función principal para ejecución distribuida"""
    
    # Extraer configuración
    points = config['points']
    lenght = config['lenght']
    n_universes = config['n_universes']
    realizations_diffusion = config.get('realizations_diffusion', 100)
    save_last_n = config.get('save_last_n', 10)
    
    device = setup_parallel_environment(rank, world_size)
    
    # Crear tau_array en el dispositivo correcto
    tau_array = torch.arange(config['tau_init'], config['tau_fin'], 0.1, device=device)
    delta_tau = tau_array[1] - tau_array[0]
    
    if rank == 0:
        print(f"Iniciando simulación distribuida con {world_size} procesos")
        print(f"Puntos por dimensión: {points}")
        print(f"Número de universos: {n_universes}")
        print(f"Tau range: {config['tau_init']} to {config['tau_fin']}")
    
    # Crear malla
    mesh = universe(points, lenght, device)
    
    # Condiciones iniciales
    phi_0_base = torch.tensor(config['phi_0'], device=device) * torch.ones((points, points, points), device=device)
    pi_0_base = torch.zeros((points, points, points), device=device)
    
    # Ejecutar realizaciones distribuidas
    if rank == 0:
        print("\n=== EJECUTANDO REALIZACIONES DE UNIVERSOS ===")
    
    phi_finals, pi_finals, full_evolutions = distributed_universe_realizations(
        rank, world_size, n_universes, phi_0_base, pi_0_base, 
        mesh, tau_array, config['sigma'], config['mass'], save_last_n
    )
    
    # Calcular matriz de difusión distribuida
    if rank == 0:
        print("\n=== CALCULANDO MATRIZ DE DIFUSIÓN ===")
    
    D_cache = distributed_diffusion_cache(
        rank, world_size, mesh, tau_array, config['sigma'], delta_tau, realizations_diffusion
    )
    
    # Guardar resultados (solo el proceso 0)
    if rank == 0:
        results = {
            'phi_finals': phi_finals,
            'pi_finals': pi_finals,
            'full_evolutions': full_evolutions,
            'D_cache': D_cache,
            'tau_array': tau_array.cpu(),
            'config': config
        }
        
        torch.save(results, config.get('output_path', 'simulation_results.pth'))
        print(f"\nResultados guardados en: {config.get('output_path', 'simulation_results.pth')}")
    
    dist.destroy_process_group()
    
    return phi_finals, pi_finals, D_cache

# =============================================================================
# CONFIGURACIÓN Y LANZAMIENTO
# =============================================================================

def main():
    """Función principal para lanzar la simulación"""
    
    # Configuración de la simulación (ajustar según necesidades)
    config = {
        'points': 128,           # Puntos por dimensión espacial
        'lenght': 60,           # Tamaño del dominio
        'n_universes': 2000,     # Número total de realizaciones
        'realizations_diffusion': 100,  # Realizaciones para matriz de difusión
        'tau_init': tau_init,
        'tau_fin': tau_fin,
        'phi_0': phi_0,
        'sigma': sigma,
        'mass': mass,
        'save_last_n': 8,
        'output_path': 'cosmic_inflation_results.pth'
    }
    
    # Determinar número de procesos
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Detectadas {world_size} GPUs disponibles")
    else:
        world_size = 1
        print("Ejecutando en CPU")
    
    # Lanzar simulación distribuida
    if world_size > 1:
        mp.spawn(
            run_distributed_simulation,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        run_distributed_simulation(0, 1, config)

if __name__ == "__main__":
    main()