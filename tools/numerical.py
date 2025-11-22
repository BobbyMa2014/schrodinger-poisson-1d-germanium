import numpy as np
from scipy.constants import hbar, epsilon_0, e, Boltzmann, m_e, pi
from scipy.integrate import solve_bvp
from scipy.linalg import eigh_tridiagonal
from numba import njit, prange
# import torch
# There is unknown problem while importing torch. Temporarily disabled.

eV_to_J = 1.602e-19

# Mesh related
def _reset_mesh(func):
    def wrapper(*args, **kwargs):
        layers = kwargs["layers"]
        for layer in layers:
            layer["mesh"] = np.zeros(len(layer["mesh"]))
        # print("[Log][Mesh reset]")
        return func(*args, **kwargs)
    return wrapper

def generate_mesh(layers, mesh_step):
    for i, layer in enumerate(layers):
        layer["mesh"] = np.zeros(round(layer["thickness"] / mesh_step))
    mesh_total = np.concatenate([layer["mesh"] for layer in layers])
    return layers, mesh_total

def get_sub_layers(layers, slice):
    qw_region = layers[slice]
    non_qw_region = layers[:slice.start] + layers[slice.stop:]
    return qw_region, non_qw_region

# Methods for Schrodinger equation
def _assemble_schrodinger_Hamiltonian(potential, mstar_mid, grid_step, grid_number, method='matrix'):
    H_scale = - (hbar**2 / (2 * grid_step**2))
    if method=='matrix': 
        H = np.zeros((grid_number, grid_number))
        H[0,0] = -1*(1/mstar_mid[0])
        H[0,1] = 1/mstar_mid[0]
        H[grid_number-1,grid_number-1] = -1*(1/mstar_mid[grid_number-2])
        H[grid_number-1,grid_number-2] = 1/mstar_mid[grid_number-2]
        for i in prange(1, grid_number-1): # Assemble Hamiltonian from effective mass
            H[i,i] = -1 * (1 / mstar_mid[i-1] + 1 / mstar_mid [i])
            H[i,i+1] = 1 / mstar_mid[i]
            H[i,i-1] = 1 / mstar_mid[i-1]
        H = H*H_scale
        for i in prange(grid_number): # Add potential part
            H[i, i] = H[i, i] + potential[i]
        return H
    elif method=='tridiagonal': 
        d = np.zeros(grid_number)
        e = np.zeros(grid_number - 1)

        # Diagonal terms
        d[0] = -1 / mstar_mid[0]
        for i in prange(1, grid_number - 1):
            d[i] = -1 / mstar_mid[i-1] - 1 / mstar_mid[i]
        d[-1] = -1 / mstar_mid[grid_number - 2]
        # Off-diagonal terms
        for i in prange(grid_number - 1):
            e[i] = 1 / mstar_mid[i]
        # Scale and add potential
        d *= H_scale
        e *= H_scale
        if len(d)==len(potential):
            d = d + potential
        else:
            raise ValueError("Potential term doesn't fit into Hamiltonian")    
        return d, e    
    else:
        raise KeyError("Specified method does not exist")    

@njit
def _sort_eigenvalues(eigenvalues, eigenvectors): # Split sort and normalization to improve efficiency
    indices = np.argsort(eigenvalues)  # Sort the eigenvalues
    eigenvalues_sorted = eigenvalues[indices]  # Sorted eigenvalues
    eigenvectors_sorted = eigenvectors[:, indices]  # Sorted eigenvectors
    return eigenvalues_sorted, eigenvectors_sorted

@njit(parallel=True)
def _normalize_to_prob_density(eigenvectors, grid_step):
    prob_density = eigenvectors**2
    for i in prange(prob_density.shape[1]):
        prob_density[:, i] /= np.trapezoid(prob_density[:, i], dx=grid_step)
    return prob_density

def solve_1d_schrodinger(band_structure, mstar, mesh, grid_step, layers, fermi_level, Filter_method='Combined', carrier='hole'):
    mstar_mid = (mstar[:-1] + mstar[1:]) / 2
    mstar_mid *= m_e
    grid_number = len(mesh)
    if carrier == 'hole': #Flip valence band to calculate hole wavefunction
        potential = np.max(band_structure) + np.min(band_structure) - band_structure
    elif carrier == 'electron': #Don't flip conduction band
        potential = band_structure

    # Using full matrix
    # H = _assemble_schrodinger_Hamiltonian(potential, mstar_mid, grid_step, grid_number, method='matrix')
    # Eigenvalue problem: CPU
    # eigenvalues, eigenvectors = eigsh(H, k=30, which='SA')
    # Eigenvalue problem: GPU
    # H_matrix = torch.tensor(H, dtype=torch.float64, device='cuda')
    # eigenvalues_tensor, eigenvectors_tensor = torch.linalg.eigh(H_matrix)
    # eigenvalues = eigenvalues_tensor.cpu().numpy()
    # eigenvectors = eigenvectors_tensor.cpu().numpy()
    # eigenvalues_sorted, eigenvectors_sorted = _sort_eigenvalues(eigenvalues, eigenvectors)

    # Using tridiagonal method
    main_diag, off_diag = _assemble_schrodinger_Hamiltonian(potential, mstar_mid, grid_step, grid_number, method='tridiagonal')
    eigenvalues_sorted, eigenvectors_sorted = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0,19)) # Compute only 20 lowest eigenstates with sorted order

    if Filter_method is not None:
        start_index, end_index = get_qw_center_region(layers)
        prob_density = _normalize_to_prob_density(eigenvectors_sorted, grid_step)
        eigenvalues_sorted, prob_density = _filter_bound_states(eigenvalues_sorted, prob_density, band_structure, fermi_level, start_index, end_index, grid_step, Filter_method)
    
    return eigenvalues_sorted, prob_density

def get_qw_center_region(layers):
    start_index = 0
    end_index = 0
    flag = True
    for layer in layers:
        if layer["material"]["name"] == "SiGe20" and flag == True:
            start_index += len(layer["mesh"])
            flag = False
        elif layer["material"]["name"] == "Ge":
            end_index = start_index + len(layer["mesh"])
    return start_index, end_index

def _filter_bound_states(eigenenergy, prob_density, band_profile, fermi_level, start_index, end_index, mesh_step, method):
    if method == "MaxValue":
        index_range = np.where((eigenenergy < np.max(band_profile[start_index-20:end_index+20])) & (eigenenergy > np.min(band_profile)))[0]
        # index_range = np.where((eigenenergy < np.max(band_profile)) & (eigenenergy > np.min(band_profile)))[0]
        eigenenergy_bound_states = eigenenergy[index_range]
        prob_density_bound_states = prob_density[:,index_range]
    elif method == "Localisation":
        localisation_threshold = 0.65
        for index in range(np.shape(prob_density)[1]):  
            localised_prob = np.trapezoid(prob_density[start_index:end_index,index], dx=mesh_step)
            total_prob = np.trapezoid(prob_density[:,index], dx=mesh_step)
            localisation_ratio = localised_prob/total_prob
            print(f"Number {index}: energy = {eigenenergy[index]/eV_to_J}, ratio = {localisation_ratio}")
            if localisation_ratio < localisation_threshold: 
                filter_upper_bound = index 
                break
        eigenenergy_bound_states = eigenenergy[0:filter_upper_bound]
        eigenvectors_bound_states = prob_density[:,0:filter_upper_bound]
    elif method == "Combined":
        index_range = np.where((eigenenergy < np.max(band_profile[start_index-20:end_index+20])) & (eigenenergy > np.min(band_profile)))[0]
        eigenenergy_filtered = eigenenergy[index_range]
        prob_density_filtered = prob_density[:,index_range]
        threshold = 0.7
        prob_density_partial = prob_density_filtered[start_index:end_index, :]
        localization_value = np.trapezoid(prob_density_partial, dx=mesh_step, axis=0)
        # print(localization_value)
        filter_mask = localization_value >= threshold
        selected_eigenenergy = eigenenergy_filtered[filter_mask]
        selected_prob_density = prob_density_filtered[:, filter_mask]
        rest_eigenenergy = eigenenergy_filtered[~filter_mask]
        rest_prob_density = prob_density_filtered[:, ~filter_mask]

        if len(selected_eigenenergy) > 0 and len(rest_eigenenergy) > 0:
            localized_state_eigenenergy = selected_eigenenergy[0]
            localized_state_prob_density = selected_prob_density[:,0]
            tunneled_state_eigenenergy = rest_eigenenergy[0]
            tunneled_state_prob_density = rest_prob_density[:,0]
            eigenenergy_bound_states = np.array([localized_state_eigenenergy, tunneled_state_eigenenergy])
            prob_density_bound_states = np.column_stack((localized_state_prob_density, tunneled_state_prob_density))
        elif len(selected_eigenenergy) > 0:
            eigenenergy_bound_states = np.array([selected_eigenenergy[0]])
            prob_density_bound_states = selected_prob_density[:,[0]]
        elif len(rest_eigenenergy) > 0:
            eigenenergy_bound_states = np.array([rest_eigenenergy[0]])
            prob_density_bound_states = rest_prob_density[:,[0]]    
        else:
            raise ValueError("No bound states found after filtering.")
    else:
        raise ValueError("Specified method does not exist.")

    eigenenergy_bound_states = np.max(band_profile) + np.min(band_profile) - eigenenergy_bound_states

    return eigenenergy_bound_states, prob_density_bound_states

@_reset_mesh
def calculate_band_offset(layers): # Calculate band offset using Anderson's rule
    for layer in layers:
        layer["mesh"] = (layer["material"]["bandgap"] + layer["material"]["electron_affinity"]) * np.ones(len(layer["mesh"]))
    band_offset = np.concatenate([layer["mesh"] for layer in layers])
    band_offset = -1 * band_offset + np.max(band_offset)
    #band_offset = band_offset - np.min(band_offset)
    band_offset = band_offset * eV_to_J
    return band_offset

@_reset_mesh
def get_fixed_band_offset(layers): # Get band offset from literature data
    for layer in layers:
        layer["mesh"] = layer["material"]["fixed_offset"] * np.ones(len(layer["mesh"]))
    band_offset = np.concatenate([layer["mesh"] for layer in layers])
    band_offset = band_offset * eV_to_J
    return band_offset

@_reset_mesh
def calculate_effective_mass(layers):
    for layer in layers:
        layer["mesh"] = layer["material"]["mstar_hole"] * np.ones(len(layer["mesh"]))
    effective_mass = np.concatenate([layer["mesh"] for layer in layers])
    return effective_mass

# Methods for Poisson equation
@_reset_mesh
def calculate_space_charge_density(layers, distribution: str):
    if distribution == "uniform":
        for layer in layers:
            if layer["total_density"] != 0: # Transform density (m^-2) to space charge density (C/m^3)
                layer["mesh"] = np.ones(len(layer["mesh"])) * (layer["total_density"] / layer["thickness"]) * e
    elif distribution == "triangular":
        for layer in layers:
            if layer["total_density"] != 0:
                x = np.linspace(0, 1, len(layer["mesh"]))
                profile = 2*(1-x)
                profile *= layer["total_density"] / layer["thickness"] * -1 * e
                layer["mesh"] = profile

    density_profile = np.concatenate([layer["mesh"] for layer in layers])
    return density_profile

@_reset_mesh
def get_permittivity(layers): # Concatenate relative permittivity of each layer into one array
    permittivity = np.concatenate([layer["material"]["epsilon_r"] * np.ones(len(layer["mesh"])) for layer in layers])
    return permittivity

def solve_1d_poisson(charge_density, permittivity, gate_voltage, boundary_right, mesh, grid_step):
    grid_number = len(mesh)
    eps = np.flip(permittivity * epsilon_0)
    # Flipping the permittivity will yield correct result. Not figured out yet.
    x = np.arange(grid_number) * grid_step

    @njit
    def poisson_eq(x_val, y):
        eps_val = np.interp(x_val, x, eps)
        rho_val = np.interp(x_val, x, charge_density)
        d2phi_dx2 = -rho_val / eps_val
        return np.vstack((y[1], d2phi_dx2))  # [dphi/dx, d²phi/dx²]
    
    @njit
    def boundary_conditions(ya, yb):
        if boundary_right == "ground":
            return np.array([ya[0] - gate_voltage, yb[0]])
        elif boundary_right == "continuous":
            return np.array([ya[0] - gate_voltage, yb[1]])
        else: 
            raise ValueError("Invalid boundary condition")
    
    y_init = np.zeros((2, grid_number))
    solution = solve_bvp(poisson_eq, boundary_conditions, x, y_init, max_nodes=30000, tol=1e-10)
    return solution.sol(x)[0]

def build_poisson_jacobian(permittivity, mesh_step):
    N = len(permittivity)
    J = np.zeros((N, N))
    eps_mid = np.zeros(N-1)
    eps_mid[:] = 0.5 * (permittivity[:-1] + permittivity[1:])

    # Interior points
    for i in range(1, N-1):
        J[i, i-1] = -eps_mid[i-1]
        J[i, i]   = (eps_mid[i-1] + eps_mid[i])
        J[i, i+1] = -eps_mid[i]

    # Left boundary (Dirichlet): enforce V[0] = const
    J[0, 0] = 1.0
    # Right boundary (Neumann): enforce dV/dz = 0 → V[N-1] - V[N-2] = 0
    J[N-1, N-1] = 1.0
    J[N-1, N-2] = -1.0

    J = J * epsilon_0 / mesh_step**2  # Scale Jacobian at the end, testing

    return J

@njit
def poisson_laplacian_for_potential(poisson_voltage, permittivity, mesh_step):
    eps_mid = (permittivity[:-1] + permittivity[1:]) / 2
    eps_mid *= epsilon_0

    lap_scale = -1/(mesh_step**2)
    laplacian = np.zeros_like(poisson_voltage)
    grid_number = len(poisson_voltage)
    v_temp = poisson_voltage.copy()

    laplacian[0] = eps_mid[0] * (v_temp[1] - v_temp[0])
    laplacian[grid_number-1] = 0 - eps_mid[grid_number-2] * (v_temp[grid_number-1] - v_temp[grid_number-2])

    for i in prange(1, grid_number-1): 
        laplacian[i] = eps_mid[i] * (v_temp[i+1] - v_temp[i]) - eps_mid[i-1] * (v_temp[i] - v_temp[i-1])

    laplacian = laplacian * lap_scale
    laplacian[0] = 0.0
    return laplacian

def get_fermi_level(Ids): 
    Vtotal = 1.0
    # R1 = 1e+7 + 4.74e+7
    R1 = 1e+7
    Vds = Vtotal - Ids*R1
    return Vds*e

# @njit
def calculate_rho_from_subbands(eigenenergy, prob_density, fermi_level: float, effective_mass: np.ndarray):
    Temp = 4.5
    a_factor = (m_e*e)/(pi*hbar*hbar)
    b_factor = Temp * Boltzmann
    
    arg = (eigenenergy - fermi_level) / b_factor
    arg_clipped = np.clip(arg, -100, 100)
    occupation = np.log1p(np.exp(arg_clipped))
    assert len(effective_mass) == prob_density.shape[0], "Effective mass shape mismatch."
    assert len(occupation) == prob_density.shape[1], "Occupation shape mismatch."
    rho_total = (prob_density 
             * effective_mass[:, np.newaxis] 
             * occupation[np.newaxis, :]).sum(axis=1) * a_factor * b_factor
    # reduce computation cost by moving some constants outside
    if np.any(np.isnan(rho_total)) or np.any(np.isinf(rho_total)):
        raise ValueError("Charge density contains NaN or Inf, check calculation.")
    return rho_total

def mixing(iteration):# Constant-value mixing method
    if iteration < 2:
        alpha = 0.8
    elif (iteration+1) < 8:
        alpha = 0.2        
    elif (iteration+1) < 25:
        alpha = 0.15
    elif (iteration+1) < 40:
        alpha = 0.06
    else:
        alpha = 0.02
    return alpha

def adaptive_mixing(iteration, residual_history, alpha): # Linear adaptive
    # Force alpha at first few iterations to guarantee stability
    if iteration < 2:
        alpha_new = 0.8
    elif iteration < 7:
        alpha_new = 0.3
    else: # Use adaptive alpha later
        new_residual = float(residual_history[-1])
        previous_residual = float(residual_history[-2])
        if previous_residual < new_residual:
            alpha_new = alpha*1.05
        else:
            alpha_new = alpha*0.88
        alpha_new = np.clip(alpha_new, 0.04, 0.3)
    return alpha_new

def adaptive_mixing_newton(iteration, residual_history, alpha): # Return mixing factor for Newton's method
    # Force alpha at first few iterations to guarantee stability
    if iteration < 2:
        alpha_new = 0.8
    elif iteration < 7:
        alpha_new = 0.3
    else: # Use adaptive alpha later
        new_residual = float(residual_history[-1])
        previous_residual = float(residual_history[-2])
        if previous_residual < new_residual:
            alpha_new = alpha*1.05
        else:
            alpha_new = alpha*0.88
        alpha_new = np.clip(alpha_new, 1.5e-5, 0.3)
    return alpha_new

def grid_adaptive_mixing(iteration, new_residual_vector, previous_residual_vector, previous_alpha_vector): # Linear adaptive
    # Force alpha at first few iterations to guarantee stability
    alpha_vector = np.ones_like(new_residual_vector)
    if iteration < 2:
        alpha_vector *= 0.8
    elif iteration < 7:
        alpha_vector *= 0.3
    else: # Use adaptive alpha later
        mask_smaller = new_residual_vector < previous_residual_vector
        alpha_vector[mask_smaller] = previous_alpha_vector[mask_smaller] * 0.9
        alpha_vector[~mask_smaller] = previous_alpha_vector[~mask_smaller] * 1.05
        alpha_vector = np.clip(alpha_vector, 0.04, 0.3)
    return alpha_vector

def stern_damping_update(iteration, residual_history, alpha_history): # Stern, J. Computational Physics 6, 56 (1970)
    if iteration < 2: # Force fixed alpha at initial iterations
        alpha_new = 0.8
    elif iteration < 7:
        alpha_new = 0.15
    else:
        new_residual = float(residual_history[-1])
        previous_residual = float(residual_history[-2])
        previous_alpha = float(alpha_history[-1])
        if previous_residual == 0:
            residual_ratio = 0
        else:
            residual_ratio = new_residual / previous_residual

        if residual_ratio != 1:
            alpha_new = previous_alpha / (1 - residual_ratio)
        else:
            alpha_new = previous_alpha  # fallback to previous alpha if r is problematic

    return np.clip(alpha_new, 0.005, 0.8)

def is_band_bending_strong(band_profile, qw_start, qw_end):
    band_qw_region = band_profile[qw_start-10:qw_end+10]
    band_near_surface = band_profile[ :qw_start // 2]
    if np.max(band_near_surface) > np.max(band_qw_region):
        return True
    else:
        return False

