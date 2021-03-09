import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])  # Make fdtd package discoverable
import numpy as np
from fdtd import Simulator, DATA_PATH

# Use standard parameters for now.
SIMULATION_PARAMS = dict(
    x_cells=100,
    y_cells=100,
    pml_width=20,
    spatial_resolution=0.01,  # 1 centimeter
    courant_number=1/np.sqrt(2),
    relative_permittivity=1,
    relative_permeability=1,
    pml_conductivity_scale=1.0,
    pml_scaling_index=3,
)

sim = Simulator(**SIMULATION_PARAMS)

source_params = dict(
    courant_number=sim.courant_number,
    points_per_wavelength=10,
    offset=80,
)

x_len, y_len = sim.mesh[0].shape
source_loc = (x_len // 3, y_len // 3)
dielectric_boundaries = (
    3 * x_len // 5, 4 * x_len // 5, 2 * y_len // 5, 4 * y_len // 5
)
permittivity_shape = sim.relative_permittivity.shape
permittivity_grid = np.array(
    np.meshgrid(
        np.arange(permittivity_shape[0]).astype(float),
        np.arange(permittivity_shape[1]).astype(float),
        indexing="ij",
    )
) / 2
in_dielectric = (
    (permittivity_grid[0] > dielectric_boundaries[0]) &
    (permittivity_grid[0] < dielectric_boundaries[1]) &
    (permittivity_grid[1] > dielectric_boundaries[2]) &
    (permittivity_grid[1] < dielectric_boundaries[3])
)

dielectric_amp = 5
perturbation_scale = 0.1
size = int(in_dielectric.sum())
new_permittivity = np.ones(permittivity_shape)
new_permittivity[in_dielectric] = dielectric_amp * (
    1 + np.random.normal(scale=perturbation_scale, size=size)
)
sim.set_relative_permittivity(new_permittivity)

sim.simulate(
    Ntimes=400,
    source_func="ricker_TEz",
    source_loc=source_loc,
    source_params=source_params,
)

save_path = os.path.join(DATA_PATH, "inhomogenous_dielectric_slab.npz")
sim.write(save_path)
