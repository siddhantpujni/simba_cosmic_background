import numpy as np
import h5py


def equivalent_dust_temperature(hdf5_path, redshift):
    """
    Compute equivalent dust temperature T_eqv for all galaxies.

    Returns
    -------
    T_eqv : np.ndarray  – temperature in K (NaN where invalid)
    mask  : np.ndarray   – boolean mask of valid galaxies
    """
    with h5py.File(hdf5_path, "r") as f:
        dust_mass = f["galaxy_data/dicts/masses.dust"][:]
        gas_mass = f["galaxy_data/dicts/masses.gas"][:]
        metallicity = f["galaxy_data/dicts/metallicities.mass_weighted"][:]

    delta_dzr = dust_mass / (metallicity * gas_mass)
    mask = (delta_dzr > 0) & np.isfinite(delta_dzr)

    T_eqv = np.full(len(dust_mass), np.nan)

    a, b, c = -0.05, -0.15, 0.36
    log_T = (a + b * np.log10(delta_dzr[mask] / 0.4) + c * np.log10(1 + redshift) + np.log10(25))

    T_eqv[mask] = 10**log_T
    
    #log_T = (a + b * np.log(delta_dzr[mask] / 0.4)
    #         + c * np.log(1 + redshift) + np.log(25))
    #T_eqv[mask] = np.exp(log_T)

    return T_eqv, mask