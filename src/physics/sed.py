import numpy as np
import astropy.units as u
from astropy.constants import h, c, k_B


def mbb(wavelength_AA, temperature, beta=2.0, norm=1.0):
    """
    Modified blackbody SED per unit wavelength.

    Parameters
    ----------
    wavelength_AA : array – wavelength in Angstrom
    temperature   : float – dust temperature in K
    beta          : float – emissivity index
    norm          : float – multiplicative normalisation
    """
    lam_m = (wavelength_AA * u.AA).to(u.m).value
    lam_m = np.where(lam_m <= 0, 1e-10, lam_m)
    temperature = np.where(temperature <= 0, 1e-10, temperature)

    x = h.value * c.value / (lam_m * k_B.value * temperature)
    x = np.clip(x, 0, 700)

    B_lam = (2 * h.value * c.value ** 2) / (lam_m ** 5) / np.expm1(x)
    emissivity = (lam_m / 100e-6) ** beta

    return norm * emissivity * B_lam


def normalised_mbb(wavelength_AA, L_FIR, temperature, beta=2.0):
    """
    MBB normalised so its integral equals L_FIR.
    Returns None if normalisation fails.
    """
    raw = mbb(wavelength_AA, temperature, beta, norm=1.0)
    if not np.all(np.isfinite(raw)):
        return None

    dlam = np.gradient(wavelength_AA)
    integral = np.sum(raw * dlam)
    if integral <= 0 or not np.isfinite(integral):
        return None

    return raw * (L_FIR / integral)