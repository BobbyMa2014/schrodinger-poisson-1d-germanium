_material_database = {
    "Si": {
        "name": "Si",
        "epsilon_r": 11.7,
        "mstar_hole": 0.59,
        "m_hh": 0.49,
        "m_lh": 0.16,
        "m_so": 0.24,
        "bandgap": 1.12,
        "electron_affinity": 4.05,
    },

    "Ge": {
        "name": "Ge",
        "epsilon_r": 16.2,
        "mstar_hole": 0.0728, # From literature: ACS Appl. Mater. Interfaces 2023, 15, 28799−28805
        "m_hh": 0.33,
        "m_lh": 0.043,
        "m_so": 0.084,
        "bandgap": 0.66,
        "electron_affinity": 4.0,
        "fixed_offset": 0.114, # From literature: Semicond. Sci. Technol. 12 (1997) 1515–1549
    },

    "SiGe20": {
        "name": "SiGe20",
        "epsilon_r": 15.34,
        "mstar_hole": 0.362,
        "bandgap": 0.8587,
        "electron_affinity": 4.01,
        "fixed_offset": 0,
    },

    "Al2O3": {
        "name": "Al2O3",
        "epsilon_r": 9.9,
        "bandgap": 6.2,
    },

    "SiO2": {
        "name": "SiO2",
        "epsilon_r": 3.9,
        "bandgap": 8.0,
        "mstar_hole": 0.58,
    }

}

def get_material(name: str) -> dict:
    if name not in _material_database:
        raise ValueError(f"Material {name} not found in database")
    return _material_database[name]
