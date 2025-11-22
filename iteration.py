import numpy as np
import h5py
import tools.numerical as numerical
import tools.datamanager as datamanager
from warnings import warn
from time import time
from tqdm import tqdm
from tools.materials import get_material

def solve_self_consistent_sp_with_config(simulation_config: str):
    cfg = datamanager.load_simulation_config(simulation_config)
    Vg = float(cfg.parameter.gate_voltage)
    p_it = float(cfg.parameter.interface_density)
    fermi_level_fixed = (cfg.parameter.fixed_fermi/1000 * numerical.eV_to_J) if cfg.parameter.method_fermi == "Fermi" else numerical.get_fermi_level(cfg.parameter.source_drain)
    mesh_step = float(cfg.simulation.mesh_step)
    max_iterations = int(cfg.simulation.max_iterations)
    save_log = bool(cfg.save.save_log)
    plot_combined = bool(cfg.save.plot_combined)
    check_existing = bool(cfg.save.check_existing)
    log_folder = cfg.save.simulation_log
    simulation_data = cfg.save.results_sim
    schrodinger_results = cfg.save.results_schrodinger
    target_format = cfg.save.store_format
    datamanager.clear_all(log_folder)
    datamanager.clear_savedata(log_folder)
    p_2DHG = None
    temp_value = 0.0
    layers_interface = [
        {"material": get_material("Al2O3"), "thickness": 30e-9, "total_density": 0, "mesh": []},
        {"material": get_material("SiO2"), "thickness": 1.5e-9, "total_density": (p_it*1e4), "mesh": []},
    ]
    layers_qw = [
        {"material": get_material("SiGe20"), "thickness": 32e-9, "total_density": 0, "mesh": []},
        {"material": get_material("Ge"), "thickness": 16e-9, "total_density": 0, "mesh": []},
        {"material": get_material("SiGe20"), "thickness": 32e-9, "total_density": 0, "mesh": []},
    ]
    layers_qw, mesh_qw = numerical.generate_mesh(layers_qw, mesh_step)
    ge_start, ge_end = numerical.get_qw_center_region(layers_qw)
    layers_interface, mesh_interface = numerical.generate_mesh(layers_interface, mesh_step)
    if check_existing:
        p_2DHG, band_profile = datamanager.query_existing_p_2DHG(names=["Vg", "p_it", "E_F", "p_2DHG"], values=[Vg, p_it, fermi_level_fixed], format=target_format, path_to_file=simulation_data)
        if p_2DHG is not None:
            print("Found existing result from given input, returning previous result directly")
            return float(p_2DHG)
    if p_2DHG is None:     
        start_time = time()
        # Calculate interface trap-induced voltage shift
        density_interface = numerical.calculate_space_charge_density(layers=layers_interface, distribution="uniform")
        permittivity_interface = numerical.get_permittivity(layers=layers_interface)
        potential_interface = numerical.solve_1d_poisson(charge_density=density_interface, permittivity=permittivity_interface, gate_voltage=Vg, boundary_right="continuous", mesh=mesh_interface, grid_step=mesh_step)
        Vg_equiv = potential_interface[-1]

        band_offset = numerical.get_fixed_band_offset(layers=layers_qw)
        mstar_hole = numerical.calculate_effective_mass(layers=layers_qw)
        permittivity_qw = numerical.get_permittivity(layers=layers_qw)
        fermi_level_offset = lambda band: fermi_level_fixed - np.max(band[ge_start-10:ge_end+10]) + 0.02*(np.max(band[ge_start-10:ge_end+10]) - np.min(band[ge_start-10:ge_end+10]))
        band_offset_initial_guess = band_offset + fermi_level_offset(band_offset) # Initial kick of band profile

        # Initial run
        eigenvalues, prob_density = numerical.solve_1d_schrodinger(band_offset_initial_guess, mstar_hole, mesh_qw, mesh_step, layers_qw, fermi_level_fixed, Filter_method='Combined')
        rho0 = numerical.calculate_rho_from_subbands(eigenvalues, prob_density, fermi_level_fixed, mstar_hole)
        V = numerical.solve_1d_poisson(charge_density=rho0, permittivity=permittivity_qw, gate_voltage=Vg_equiv, boundary_right="continuous", mesh=mesh_qw, grid_step=mesh_step)

        previous_band_profile = band_offset - V * numerical.e
        convergence_history = []
        convergence_threshold = 1e-5
        consecutive_required = 10
        consecutive_counter = 0

        # Newton's method
        Jacobian = numerical.build_poisson_jacobian(permittivity_qw, mesh_step)
        alpha = 0.1

        with tqdm(total=max_iterations, desc="S-P Iteration", unit="it") as pbar: 
            if not bool(cfg.simulation.progress_bar_enable):
                pbar.disable = True
                pbar.close()
            for iteration in range(max_iterations):
                updated_band = band_offset - V * numerical.e
                eigenvalues, prob_density = numerical.solve_1d_schrodinger(updated_band, mstar_hole, mesh_qw, mesh_step, layers_qw, fermi_level_fixed, Filter_method='Combined')
                rho = numerical.calculate_rho_from_subbands(eigenvalues, prob_density, fermi_level_fixed, mstar_hole)

                # 2) Residual from current V (must match Jacobian discretization & scaling)
                laplacian = numerical.poisson_laplacian_for_potential(V, permittivity_qw, mesh_step)  # returns -d(eps dV/dz)
                residual = laplacian - rho

                # Enforce BC residual entries consistent with J's BC rows and scaling:
                # residual[0]  = (numerical.epsilon_0/mesh_step**2) * (V[0] - V_left)        # Dirichlet
                residual[-1] = (numerical.epsilon_0/mesh_step**2) * (V[-1] - V[-2])        # Neumann

                # 3) Newton correction
                deltaV = np.linalg.solve(Jacobian, -residual)

                # Optional: zero out correction at Dirichlet node so it never drifts
                deltaV[0] = 0.0

                # 4) Damped update
                alpha = numerical.adaptive_mixing_newton(iteration, convergence_history, alpha)
                V = V + alpha * deltaV

                # Update band profile for Schrodinger
                updated_band = band_offset - V * numerical.e
                current_convergence = np.max(abs(updated_band - previous_band_profile)/numerical.eV_to_J)
                convergence_history.append(current_convergence)
                pbar.set_postfix({'convergence': f'{current_convergence:.2e}', 'alpha': f'{alpha:.1e}'})

                previous_band_profile = updated_band.copy()
                # Handle save data
                if save_log:
                    datamanager.save_single_axis_data(previous_band_profile, "ValenceBandMax", (iteration+1), log_folder)
                    datamanager.save_single_axis_data(V, "PoissonPotential", (iteration+1), log_folder)
                    datamanager.save_single_axis_data(rho, "HoleConcentration", (iteration+1), log_folder)
                # Handle convergence
                pbar.update(1)
                if current_convergence < convergence_threshold:
                    consecutive_counter += 1
                else:
                    consecutive_counter = 0
                if consecutive_counter >= consecutive_required: # Do something for break
                    pbar.close()
                    print(f"Converged: Finished {iteration+1} iterations in {(time()-start_time):.1f} seconds")
                    break
            else: # If didn't converge, use previous result with minimum residual
                pbar.close()
                print(f"Converge failed: Finished {iteration+1} iterations in {(time()-start_time):.1f} seconds, using fallback results")
                # start_index = 90
                # relative_index = np.argmin(np.array(convergence_history)[start_index:])
                # smallest_residual_position = start_index + relative_index
                # print(smallest_residual_position)

        eigenvalues, prob_density = numerical.solve_1d_schrodinger(updated_band, mstar_hole, mesh_qw, mesh_step, layers_qw, fermi_level_fixed)
        rho = numerical.calculate_rho_from_subbands(eigenvalues, prob_density, fermi_level_fixed, mstar_hole)
        # valence_band_max = np.max(updated_band)
        # delta_E = (valence_band_max - fermi_level_fixed)/numerical.eV_to_J*1000 # meV
        rho_final = rho.copy()  
        p_2DHG = np.trapezoid(rho_final[ge_start-10:ge_end+10], dx=mesh_step)/numerical.e/1e+4 # cm^-2
        band_bending_strong = numerical.is_band_bending_strong(previous_band_profile, ge_start, ge_end)

        if save_log:
            # eigenvalues, prob_density = datamanager.filter_states_to_plot(eigenvalues, prob_density, ge_start, ge_end, mesh_step)
            datamanager.plot_1d_schrodinger_with_offset(previous_band_profile, prob_density, eigenvalues, fermi_level_fixed, mesh_step, Vg, p_it, p_2DHG, schrodinger_results, save=True)
            datamanager.plot_1d_schrodinger_with_offset(previous_band_profile, prob_density, eigenvalues, fermi_level_fixed, mesh_step, Vg, p_it, p_2DHG, log_folder, save=True)
            datamanager.plot_convergence(iteration, (np.array(convergence_history)), log_folder, alpha_array=None, save=True)
        if plot_combined:
            datamanager.plot_single_axis_combined(key="ValenceBandMax", folder_path=log_folder, mesh_step=mesh_step, scale_factor=datamanager.eV_to_J, save=True)
            datamanager.plot_single_axis_combined(key="PoissonPotential", folder_path=log_folder, mesh_step=mesh_step, save=True)
            datamanager.plot_single_axis_combined(key="HoleConcentration", folder_path=log_folder, mesh_step=mesh_step, save=True)  

        # Save simulation results to hdf5, supporting swmr
        if simulation_data.endswith(".h5"):
            with h5py.File(simulation_data, 'a', libver='latest') as f:
                f.swmr_mode = True
                existing_runs = [k for k in f.keys() if k.startswith("run_")]
                next_index = len(existing_runs) + 1
                run_name = f"run_{next_index:05d}"
                grp = f.create_group(run_name)
                formatted = [fmt % value for fmt, value in zip(target_format, [Vg, p_it, fermi_level_fixed, p_2DHG, current_convergence])]
                for name, value in zip(["Vg", "p_it", "E_F", "p_2DHG", "convergence"], formatted):
                    grp.attrs[name] = value
                grp.attrs["qw_start"] = int(ge_start)
                grp.attrs["qw_end"] = int(ge_end)
                grp.create_dataset("hole_density", data=rho_final)
                grp.create_dataset("valence_band_profile", data=previous_band_profile)
                grp.create_dataset("eigenenergy", data=eigenvalues)
                grp.create_dataset("wavefunction", data=prob_density)
                f.flush()
                # print(f"Saved simulation result to {simulation_data} under group {run_name}")
        elif simulation_data.endswith(".dat"):
            print("Saved data to .dat file.")
        else: 
            warn("Unsupported file format. Skipping saving result.")
    return float(p_2DHG)