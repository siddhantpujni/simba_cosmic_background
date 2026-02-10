from pathlib import Path
import numpy as np
import caesar
import h5py
import astropy.units as u
from astropy.constants import h, c, k_B
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt

from week_2_funcs import get_redshift

CAT_DIR = Path("/home/spujni/sim/m25n256/s50/Groups/")

def mbb(wavelength, temperature, beta=2.0, norm=1.0):
    """
    Modified blackbody (MBB) luminosity per wavelength.
    wavelength: array (Angstrom)
    temperature: dust temperature (K)
    beta: emissivity index
    norm: normalisation (set so integral matches L_FIR)
    """

    wavelength = wavelength * u.AA
    lam_m = wavelength.to(u.m).value

    # Avoid zero or negative wavelengths and temperatures
    lam_m = np.where(lam_m <= 0, 1e-10, lam_m)
    temperature = np.where(temperature <= 0, 1e-10, temperature)

    x = h.value * c.value / (lam_m * k_B.value * temperature)
    x = np.clip(x, 0, 700)  # Prevent overflow in exp

    B_lam = (2 * h.value * c.value**2) / (lam_m**5) / np.expm1(x)
    emissivity = (lam_m / (100e-6))**beta
    mbb = emissivity * B_lam
    return norm * mbb

def equivalent_dust_temperature(hdf5_path):
    """
    Compute equivalent dust temperature T_eqv for all galaxies in the HDF5 file.
    Returns array of T_eqv in K.
    """ 

    path = "/home/spujni/sim/m25n256/s50/Groups/m25n256_110.hdf5"

    with h5py.File(hdf5_path, "r") as f:
        dust_mass = f["galaxy_data/dicts/masses.dust"][:] 
        gas_mass = f["galaxy_data/dicts/masses.gas"][:]
        metallicity = f["galaxy_data/dicts/metallicities.mass_weighted"][:]

    z = get_redshift(caesar.load(str(path)))
    
    #print(f"the dust mas is:{dust_mass}")
    #print(f"the metallicity is: {metallicity}")
    #print(f"the gas mass is {gas_mass}")

    delta_dzr = dust_mass / (metallicity * gas_mass)
    mask = delta_dzr > 0
    delta_dzr = delta_dzr[mask]

    a = -0.05
    b = -0.15
    c = 0.36

    log_Teqv = a + b * np.log(delta_dzr/0.4) + c * np.log(1+z) - np.log(25) #from paper romeel sent
    Teqv = np.exp(log_Teqv)

    # print(f"the equivalent temps are: {Teqv}")

    return Teqv

def summed_mbb(hdf5_path, beta=2.0, n_points=200):
    """
    Compute summed MBB SED from all galaxies' L_FIR and individual T_eqv.
    Returns wavelength (Angstrom), summed SED (erg/s/AA)
    """
    with h5py.File(hdf5_path, "r") as f:
        lfir = f["galaxy_data/L_FIR"][:]
    Teqv = equivalent_dust_temperature(hdf5_path)
    lam = np.logspace(3.5, 5, n_points)  # Angstrom

    total_sed = np.zeros_like(lam)
    for L, T in zip(lfir, Teqv):
        if np.isnan(T) or np.isnan(L) or T <= 0 or L <= 0:
            continue  # Skip invalid galaxy
    
        mbb_unnorm = mbb(lam, T, beta, norm=1.0)
        if np.any(np.isnan(mbb_unnorm)) or np.any(np.isinf(mbb_unnorm)):
            continue  # Skip if MBB is invalid
    
        dlam = np.gradient(lam)
        integral = np.sum(mbb_unnorm * dlam)
        if integral <= 0 or np.isnan(integral) or np.isinf(integral):
            continue  # Skip if normalisation fails
    
        norm = L / integral
        mbb_norm = mbb(lam, T, beta, norm=norm)
        if np.any(np.isnan(mbb_norm)) or np.any(np.isinf(mbb_norm)):
            continue  # Skip if normalised MBB is invalid
    
        total_sed += mbb_norm

    # print(f"the total sed is: {total_sed}")
    # print(f"the wavelength array is: {lam}")

    return lam, total_sed
