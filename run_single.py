from iteration import solve_self_consistent_sp_with_config

config = r"Config\simulation_config.yml"
p_hole = solve_self_consistent_sp_with_config(config)

print(f"{p_hole:.3e} cm-2")
