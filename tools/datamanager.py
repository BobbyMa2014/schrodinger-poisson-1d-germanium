import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import yaml
import h5py
from pathlib import Path
from types import SimpleNamespace

eV_to_J = 1.602e-19

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,                # base font size (axis/ticks/legend)
    'axes.labelsize': 10,          # x/y labels
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'stix',
    'figure.figsize': (3.4, 2.5),  # single-column default
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'savefig.dpi': 600,            # high-resolution output
    'pdf.fonttype': 42,            # TrueType fonts (editable text in vector editors)
    'ps.fonttype': 42
})

# Save data
def clear_savedata(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dat'):
                os.remove(os.path.join(root, file))
    # print("Previous result cleared.")
    return None

def clear_all(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dat') or file.endswith('.png'):
                os.remove(os.path.join(root, file))
    # print("Previous result cleared.")
    return None

def get_incremented_filename(base_path, base_name, extension=".png"):
    i = 1
    while True:
        filename = f"{base_name}_{i}{extension}"
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

def save_single_axis_data(data:np.ndarray, data_name:str, iteration:int, folder_path:str):
    save_data = data
    save_name = f"{data_name}_iteration_{iteration}.dat"
    save_folder = os.path.join(folder_path, data_name)
    os.makedirs(save_folder, exist_ok=True)
    np.savetxt(os.path.join(save_folder, save_name), save_data, fmt="%.5e", delimiter="\t")
    # print(f"File saved to {save_name}")
    return None

def save_2d_array(data:np.ndarray, data_name:str, folder_path:str):
    save_name = f"{data_name}.dat"
    np.savetxt(os.path.join(folder_path, save_name), data, fmt="%.5e", delimiter="\t")
    return None

# Plot data
def get_shade_color(layer):
    if layer["material"]["name"] == "SiO2":
        return "gray"
    elif layer["material"]["name"] == "SiGe20":
        return "lightblue"
    elif layer["material"]["name"] == "Ge":
        return "darkblue"
    elif layer["material"]["name"] == "Al2O3":
        return "yellow"
    
def plot_1d_general(ydata, ydata_name, layers, mesh_step, scale_factor=1.0):
    x = np.arange(len(ydata)) * mesh_step
    plt.figure()
    ydata_plot = ydata/scale_factor
    plt.plot(x * 1e9, ydata_plot)  # x in nm
    plt.xlabel("Depth (nm)")
    plt.ylabel(ydata_name)
    boundary = np.zeros(len(layers))
    for i, layer in enumerate(layers):
        boundary[i] = boundary[i-1] + layer["thickness"]
    boundary = np.concatenate([[0], boundary])
    for i in range(len(boundary)):
        plt.axvline(x=boundary[i] * 1e9, color="red", linestyle="--", linewidth=1)
        if i > 0:
            #Shade whole area in the background between boundaries
            plt.fill([boundary[i-1] * 1e9, boundary[i] * 1e9, boundary[i] * 1e9, boundary[i-1] * 1e9], [np.min(ydata_plot), np.min(ydata_plot), np.max(ydata_plot), np.max(ydata_plot)], 
                     color=get_shade_color(layers[i-1]), alpha=0.3)
    plt.show()
    return None

def plot_1d_schrodinger_with_offset(potential_well, prob_density, eigenenergy, fermi_level, mesh_step, gate_voltage, interface, hole_density, folder_path, save=False):
    x_axis = np.arange(len(prob_density)) * mesh_step * 1e9
    potential_meV = potential_well * 1000 / eV_to_J
    fermi_level_meV = fermi_level * 1000 / eV_to_J

    # Create figure and axis
    plt.figure()
    plt.plot(x_axis, potential_meV, 'k--', label='Valence Band Max')
    plt.plot(x_axis, np.full_like(x_axis, fermi_level_meV), 'r--', label=f"Fermi level")
    # Plot each probability density shifted to its eigenenergy
    for i in range(len(eigenenergy)):
        energy_offset = eigenenergy[i] * 1000 / eV_to_J  # meV
        scaled_density = prob_density[:, i] / np.max(prob_density[:, i]) * 6  # Adjust height scaling as needed
        plt.plot(x_axis, scaled_density + energy_offset, label=f"Eigenstate {i+1}")
        plt.fill_between(x_axis, energy_offset, scaled_density + energy_offset, alpha=0.3)
    # Labels and aesthetics
    plt.xlabel("Depth (nm)")
    # plt.ylabel("Energy (meV)")
    # plt.title('1D Schrödinger Solver Results')
    # plt.legend(loc='upper right')
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Vg={gate_voltage:.3f}, pit={interface:.4e}, phole={hole_density:.4e}")
    plt.yticks([])

    if save:
        save_name = "Schrodinger"
        fig_path = get_incremented_filename(folder_path, save_name)
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None

def plot_1d_schrodinger_without_offset(potential_well, prob_density, eigenenergy, fermi_level, mesh_step, folder_path, save=False):
    x_axis = np.arange(len(prob_density)) * mesh_step * 1e9  # convert to nm
    potential_meV = potential_well * 1000 / eV_to_J
    fermi_level_meV = fermi_level * 1000 / eV_to_J
    state_energy_meV = eigenenergy * 1000 / eV_to_J

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Left y-axis: energy
    ax1.plot(x_axis, potential_meV, 'k--', label='Valence Band Max')
    ax1.plot(x_axis, np.full_like(x_axis, fermi_level_meV), 'r--', label=f"Fermi level: {fermi_level_meV:.2f} meV")

    ax2 = ax1.twinx()  # Right y-axis for probability density
    for i in range(len(state_energy_meV)):
        prob_scaled = prob_density[:, i] / np.max(prob_density[:, i]) / 5
        # ax1.plot(x_axis, np.full_like(x_axis, eigenenergy[i]), 'b-', lw=1)
        ax2.plot(x_axis, prob_scaled, label=f"State {i+1}, E = {state_energy_meV[i]:.2f} meV")
        ax2.fill_between(x_axis, 0, prob_scaled, alpha=0.3)

    # Labels and aesthetics
    ax1.set_xlabel("Depth (nm)")
    ax1.set_ylabel("Energy (meV)")
    ax2.set_ylabel("Probability density (a.u.)")
    ax2.set_ylim(ymax=1.0)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax1.set_title('1D Schrödinger Solver Results')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if save:
        save_name = "Schrodinger"
        fig_path = get_incremented_filename(folder_path, save_name)
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None

def plot_single_axis_combined(key:str, folder_path:str, mesh_step, scale_factor = 1.0, save=False):
    save_path = os.path.join(folder_path, key)
    os.makedirs(save_path, exist_ok=True)
    iterations = sum(1 for entry in os.scandir(save_path) if entry.is_file() and entry.name.startswith(key) and entry.name.endswith('.dat'))
    iterations = np.linspace(1, iterations, num=iterations, dtype=int)
    colormap = plt.cm.get_cmap('jet', 256)
    norm = plt.Normalize(vmin=min(iterations), vmax=max(iterations))
    fig, ax = plt.subplots()

    for filename in os.listdir(save_path):
        if filename.startswith(key) and filename.endswith('.dat'):
            filename_clean, ext = os.path.splitext(filename)
            iteration_number = int(filename_clean.split('_')[2])
            if iteration_number in iterations:
                color = colormap(norm(iteration_number))
                ydata = np.loadtxt(os.path.join(save_path, filename))/scale_factor
                xdata = np.arange(len(ydata)) * mesh_step
                ax.plot(xdata, ydata, label=f"Iteration {iteration_number}", color=color)
    mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
    mappable.set_array([])
    ax.set_xlabel("Depth (nm)")
    ax.set_ylabel(key)
    # ax.set_title("Combined Vxy vs Time")
    colorbar = plt.colorbar(mappable, ax=ax)
    colorbar.set_label("Iterations")
    if save:
        save_name = f"{key}"
        fig_path = get_incremented_filename(save_path, save_name)
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.close()
    return None

def plot_convergence(iterations, residual_array, save_folder, alpha_array=None, save=False):
    iteration_array = np.linspace(1, iterations+1, num=iterations+1, dtype=int)
    if alpha_array is None:
        save_2d_array(np.column_stack((iteration_array, residual_array)), "Residual_vs_iteration", save_folder)
    else:
        save_2d_array(np.column_stack((iteration_array, residual_array, alpha_array)), "Residual_alpha_vs_iteration", save_folder)    
    plt.figure()
    plt.plot(iteration_array, residual_array)
    plt.xlabel("Iterations")
    plt.ylabel("Convergence")
    plt.yscale('log')
    if save:
        save_name = "Convergence"
        fig_path = get_incremented_filename(save_folder, save_name)
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.close()
    return None

def delete_data_after_fallback(iteration, folder_path):
    return None

def save_computation_result(Vg, Ids, p_it, p_2DHG, last_residual, path_to_file):
    with open(path_to_file, 'a') as f:
        f.write(f"{Vg:.3f} {Ids:.3e} {p_it:.5e} {p_2DHG:.5e} {last_residual:.2e}\n")

def query_existing_p_2DHG(names, values, format, path_to_file):
    if not os.path.exists(path_to_file):
        return None, None
    elif path_to_file.endswith(".h5"):
        formatted_query = {}
        for i in range(len(values)):
            formatted_query[names[i]] = format[i] % values[i]
        with h5py.File(path_to_file, 'r', libver='latest', swmr=True) as f:
            for run_name in f.keys():
                grp = f[run_name]
                match = True
                for key, query_value in formatted_query.items():
                    if grp.attrs[key] != query_value:
                        match = False
                        break
                if match:
                    p_2DHG = grp.attrs[names[3]]  # Assuming p_2DHG is the 4th name in the list
                    band_profile = grp['valence_band_profile'][:]
                    return float(p_2DHG), band_profile
    elif path_to_file.endswith(".dat"): # TODO: implement for .dat files
        return None, None
    
    return None, None

def query_existing_simulation_result(names, values, format, path_to_file):
    
    def group_to_dict(grp):
        result = {}
        # 1. Datasets
        for key, item in grp.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]  # load dataset as np.array (or scalar)
            elif isinstance(item, h5py.Group):
                result[key] = group_to_dict(item)  # recursively include subgroups if you want
        # 2. Attributes
        for key, val in grp.attrs.items():
            result[key] = val
        return result
    
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"File {path_to_file} does not exist.")
    elif path_to_file.endswith(".h5"):
        formatted_query = {}
        for i in range(len(values)):
            formatted_query[names[i]] = format[i] % values[i]
        with h5py.File(path_to_file, 'r', libver='latest', swmr=True) as f:
            for run_name in f.keys():
                grp = f[run_name]
                match = True
                for key, query_value in formatted_query.items():
                    if grp.attrs[key] != query_value:
                        match = False
                        break
                if match:
                    result_dict = group_to_dict(grp)
                    return result_dict
    return None

def load_simulation_config(path_to_yaml):
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d
    with open(path_to_yaml, 'r') as file:
        data = yaml.safe_load(file)
    return dict_to_namespace(data)

def filter_states_to_plot(eigenvalues, prob_density, ge_start, ge_end, mesh_step):
    threshold = 0.8
    prob_density_partial = prob_density[ge_start:ge_end, :]
    localization_value = np.trapezoid(prob_density_partial, dx=mesh_step, axis=0)
    filter_mask = localization_value >= threshold

    return eigenvalues[filter_mask], prob_density[:, filter_mask]
