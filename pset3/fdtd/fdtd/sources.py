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


def ricker_TEz(
    step,
    courant_number=1,
    points_per_wavelength=10,
    offset=0,
):
    """Return Ex, Ey, Hz for a magnetic field Ricker wavelet source."""
    return 0, 0, ricker_wavelet(step, courant_number, points_per_wavelength, offset)


SOURCES["ricker_TEz"] = ricker_TEz


def sine_wave(
    step,
    amplitude=1,
    courant_number=1,
    points_per_wavelength=10,
    offset=0,
):
    return amplitude * np.sin(
        2 * np.pi * courant_number * (step - offset) / points_per_wavelength
    )


def plane_wave(
    step,
    angle=0,
    amplitude=1,
    courant_number=1,
    points_per_wavelength=10,
    offset=0,
):
    """Plane wave source in free space."""
    from astropy import constants
    impedance = np.sqrt(constants.mu0.value / constants.eps0.value)
    Hz = sine_wave(step, amplitude, courant_number, points_per_wavelength, offset)
    E0 = impedance * Hz
    return -E0 * np.sin(angle), E0 * np.cos(angle), Hz


SOURCES["plane_wave"] = plane_wave
