import numpy as np

SOURCES = {}

def ricker_wavelet(
    step,
    courant_number=1,
    points_per_wavelength=10,
    offset=0,
):
    """Discrete Ricker wavelet for a FDTD simulation.
    
    Parameters
    ----------
    step: int
        Current time step.
    courant_number: float, optional
        Courant number describing the relation between the simulation's
        temporal resolution, spatial resolution, and wave speed.
    points_per_wavelength: float, optional
        Number of points per wavelength for the maximum power mode of
        the wavelet's spectrum. Should be greater than 2 in order to
        satisfy the Nyquist sampling theorem at the peak frequency.
    offset: float, optional
        Number of (possibly fractional) time steps to delay the source
        turn-on.

    Returns
    -------
    response: float
        Response of the Ricker wavelet at the provided time step according
        to the provided parameters.
    """
    prefactor = np.sqrt(2 * np.pi) * courant_number / points_per_wavelength
    x = prefactor * (step - offset)
    return (1 - x ** 2) * np.exp(-0.5 * x ** 2)

SOURCES["ricker"] = ricker_wavelet
