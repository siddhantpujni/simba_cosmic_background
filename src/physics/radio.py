import numpy as np
from scipy import integrate

def _chabrier_imf(m):
    """Chabrier (2003) system IMF  ξ(m) = dn/dm."""
    if m < 1.0:
        return (0.158 / m) * np.exp(
            -(np.log10(m) - np.log10(0.079)) ** 2 / (2 * 0.69 ** 2)
        )
    # Power-law for m >= 1, normalised for continuity at m = 1
    A = 0.158 * np.exp(
        -(np.log10(1.0) - np.log10(0.079)) ** 2 / (2 * 0.69 ** 2)
    )
    return A * m ** (-2.3)


def chabrier_mass_fraction(m_low=5.0, m_min=0.1, m_max=100.0):
    """
    Fraction of total SFR going into stars with M >= m_low
    for a Chabrier (2003) IMF integrated from m_min to m_max.

    Returns
    -------
    f : float   (≈ 0.17 for m_low = 5 M_sun)
    """
    mass_weighted = lambda m: m * _chabrier_imf(m)
    total, _ = integrate.quad(mass_weighted, m_min, m_max)
    above, _ = integrate.quad(mass_weighted, m_low, m_max)
    return above / total


# Pre-compute once at import time
CHABRIER_FRAC_M5 = chabrier_mass_fraction(m_low=5.0)


# ── Condon1992 radio luminosity ───────────────────────────────

def radio_luminosity_sf(sfr_total, nu_ghz=1.4, f_imf=None):
    """
    Star-formation radio luminosity (Condon1992, eqs 10+11 from Thomas2021).

    Parameters
    ----------
    sfr_total : float or array
        Total star formation rate  [M_sun / yr].
    nu_ghz : float or array
        Frequency in GHz (default 1.4 GHz).
    f_imf : float or None
        Mass fraction of SFR in stars M >= 5 M_sun.
        If None, uses the Chabrier (2003) value (≈ 0.17).

    Returns
    -------
    P_nu : float or array
        Radio luminosity in W Hz^-1.
    """
    if f_imf is None:
        f_imf = CHABRIER_FRAC_M5

    sfr_m5 = np.asarray(sfr_total) * f_imf
    nu = np.asarray(nu_ghz, dtype=float)

    P_nonthermal = 5.3e21 * nu ** (-0.8) * sfr_m5     # W Hz^-1
    P_thermal    = 5.5e20 * nu ** (-0.1) * sfr_m5      # W Hz^-1

    return P_nonthermal + P_thermal


def radio_sed_sf(sfr_total, nu_ghz_array, f_imf=None):
    """
    Full radio SED from star formation over a frequency grid.

    Parameters
    ----------
    sfr_total : float
        Total SFR  [M_sun / yr].
    nu_ghz_array : array
        Frequencies in GHz.

    Returns
    -------
    P_nu : array   -  W Hz^-1 at each frequency.
    """
    return radio_luminosity_sf(sfr_total, nu_ghz_array, f_imf)


def agn_radio_luminosity(mdot_bh, nu_ghz=1.4):
    """
    AGN radio luminosity from black hole accretion with spectral index.

    P_Rad / 1e30 erg s^-1 = Mdot_BH / 4e17 g s^-1

    scaled by a power-law SED  (nu / 1.4 GHz)^{-0.7}.

    Parameters
    ----------
    mdot_bh : float or array
        Black hole accretion rate  [M_sun / yr].
    nu_ghz : float or array
        Frequency in GHz (default 1.4 GHz).

    Returns
    -------
    P_rad : float or array
        Radio luminosity in erg s^-1.
    """
    MSUN_PER_YR_TO_G_PER_S = 6.304e25          # 1 M_sun/yr → g/s
    mdot_cgs = np.asarray(mdot_bh) * MSUN_PER_YR_TO_G_PER_S
    nu = np.asarray(nu_ghz, dtype=float)

    P_ref = (mdot_cgs / 4e17) * 1e30           # erg s^-1 at 1.4 GHz

    return P_ref * (nu / 1.4) ** (-0.7)         # erg s^-1