import yaml, time, os
import numpy as np
import matplotlib.pyplot as plt
from iteration import solve_self_consistent_sp_with_config

config_path = r"simulation_config.yml"

def generate_sweep_array(start, stop, step, scale=1.0):
    if stop > start:
        sweep_array = np.arange(start, stop + step, step)
        return sweep_array * scale
    else:
        sweep_array = np.flip(np.arange(stop, start + step, step))
        return sweep_array * scale

def parameter_sweep_1d(xdata_name:str, value_start, value_stop, value_step, path_to_config, **kwargs):
    with open(path_to_config, 'r') as f:
        config = yaml.safe_load(f)
    if 'scale' in kwargs:
        scale = kwargs['scale']
    else: 
        scale = 1.0
    if 'save_format' in kwargs:
        target_format = kwargs['save_format']
    else:
        target_format = ['%.5f', '%.5e']        
    sweep_values = generate_sweep_array(value_start, value_stop, value_step, scale=scale)
    folder_path = os.path.split(config['save']['results_sim'])[0]
    if 'secondary_value' in kwargs and kwargs['secondary_value'] != 'p_2DHG':
        ydata_name = kwargs['secondary_value']
    else:
        ydata_name = 'p_2DHG'    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_savename = f"{ydata_name}_vs_{xdata_name}.dat"
    file_savepath = os.path.join(folder_path, file_savename)
    if os.path.exists(file_savepath):
        os.remove(file_savepath)
    fig_savepath = os.path.join(folder_path, f"{ydata_name}_vs_{xdata_name}.png")   
    if os.path.exists(fig_savepath):
        os.remove(fig_savepath)    

    for i, value in enumerate(sweep_values):
        write_value = value.copy()
        modified_config = config.copy()
        modified_config['parameter'][xdata_name] = float(write_value)      
        modified_config_path = path_to_config
        with open(modified_config_path, 'w') as f:
            yaml.dump(modified_config, f, default_flow_style=False, sort_keys=False)
        p_2DHG, secondary_value = solve_self_consistent_sp_with_config(modified_config_path)
        if 'secondary_value' in kwargs and kwargs['secondary_value'] != 'p_2DHG':
            ydata_value = secondary_value
        else:
            ydata_value = p_2DHG

        with open(file_savepath, 'a') as f:
            f.write(f"{target_format[0] % float(write_value)}\t{target_format[1] % ydata_value}\n")
        time.sleep(1)    

    sweep_data = np.loadtxt(file_savepath)
    x = sweep_data[:,0]
    y = sweep_data[:,1]
    y = [max(val, 0.1) for val in y]  # Filter extremely small/non-physical values for log scale plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y)
    plt.xlabel(f"{xdata_name}")
    plt.ylabel(f"{ydata_name}")
    # plt.yscale("log")
    plt.title(f"{ydata_name} vs {xdata_name}")
    plt.grid(True)
    if np.max(sweep_values) < 0:
        plt.gca().invert_xaxis()
    plt.savefig(fig_savepath)
    plt.close()
        
start = -0.45
stop = -0.75
step = 0.001
parameter_sweep_1d('gate_voltage', start, stop, step, config_path, scale=1.0, save_format=['%.3f', '%.3e'], secondary_value='p_2DHG')