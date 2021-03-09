import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])  # Make fdtd package discoverable
import numpy as np
from fdtd import Simulator, DATA_PATH

# Use standard parameters for now.
SIMULATION_PARAMS = dict(
    x_cells=200,
    y_cells=200,
    pml_width=20,
    spatial_resolution=0.01,  # 1 centimeter
    courant_number=1/np.sqrt(2),
    relative_permittivity=1,
    relative_permeability=1,
    pml_conductivity_scale=1.0,
    pml_scaling_index=2,
)

sim = Simulator(**SIMULATION_PARAMS)

source_params = dict(
    angle=0,
    amplitude=1,
    courant_number=sim.courant_number,
    points_per_wavelength=100,
    offset=0,
)

x_len, y_len = sim.mesh[0].shape
pml_width = sim.pml_width
source_loc = (pml_width, slice(pml_width, -pml_width))

sim.simulate(
    Ntimes=300,
    source_func="plane_wave",
    source_loc=source_loc,
    source_params=source_params,
)

save_path = os.path.join(DATA_PATH, "plane_wave_test.npz")
sim.write(save_path)
