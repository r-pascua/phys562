import os
import numpy as np
from fdtd import Simulator, DATA_PATH

# Use standard parameters for now.
SIMULATION_PARAMS = dict(
    x_cells=101,
    y_cells=101,
    pml_width=5,
    spatial_resolution=0.01,  # 1 centimeter
    courant_number=1/np.sqrt(2),
    relative_permittivity=1,
    relative_permeability=1,
    pml_conductivity_scale=4.0,
    pml_scaling_index=2,
)

sim = Simulator(**SIMULATION_PARAMS)

source_params = dict(
    courant_number=sim.courant_number,
    points_per_wavelength=10,
    offset=5,
)

x_len, y_len = sim.mesh[0].shape
source_loc = (x_len // 2, y_len // 2)
sim.simulate(
    source_func="ricker",
    source_loc=source_loc,
    source_params=source_params,
)

save_path = os.path.join(DATA_PATH, "wavelet_test.npz")
sim.write(save_path)
