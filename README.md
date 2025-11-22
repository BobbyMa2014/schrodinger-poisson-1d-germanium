# schrodinger-poisson-1d
Numerical calculation of one-dimensional self-consistent Schrödinger-Poisson for germanium heterostructures

## Features
 - Convergence below 1e-5 achieved
 - Accelerate calculation using tridiagonal matrix, Newton's method and numba support
 - Stores simulation log in each run and archives all past simulation results

## Material Support
 - Strained germanium heterostructure (SiGe/sGe/SiGe)
 - Includes the impact of interface states at SiOx layer

## How to Use
The `iteration.py` file is the main program that runs the self-consistent S-P solver. By calling the function `solve_self_consistent_sp_with_config`, one would be able to do many things based on the S-P calculation. Some examples include: 
 - `run_single.py` file runs a single calculation with specified config.
 - `sweep_1d_simple.py` file runs a parametric sweep and stores all results.

## To-do
 - [x] Investigate current numerical error
 - [x] Fix instability at near turn-on region 
 - [x] Run simulation with parameter sweep
 - [ ] Code cleanup

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg