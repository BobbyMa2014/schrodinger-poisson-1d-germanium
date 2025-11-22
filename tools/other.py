import h5py
import re

def rename_groups(path_to_h5: str):
    with h5py.File(path_to_h5, 'r+') as f:
        old_names = [name for name in f.keys() if re.match(r"run_\d{4}$", name)]

        for old_name in old_names:
            number = int(old_name.split('_')[1])
            new_name = f"run_{number:05d}"
            f.move(old_name, new_name)
            print(f"Renamed group {old_name} to {new_name}")