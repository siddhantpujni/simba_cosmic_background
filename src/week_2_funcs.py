import re
import numpy as np
import h5py
import fsps
from pathlib import Path
import astropy.units as u
from astropy.constants import c

CAT_DIR = Path("/home/spujni/sim/m50n512/s50/Groups/")

def get_redshift(obj):
    for attr in ["redshift", "z"]:
        if hasattr(obj.simulation, attr):
            return float(getattr(obj.simulation, attr))

def list_snapshots(cat_dir=CAT_DIR, prefix="m50n512_"):
    files = sorted(cat_dir.glob(f"{prefix}*.hdf5"))
    snaps = []
    for f in files:
        m = re.search(r"_(\d+)\.hdf5$", f.name)
        if m:
            snaps.append((int(m.group(1)), f))
    snaps.sort(key=lambda t: t[0], reverse=True)
    return snaps

def compute_summed_sed_from_appmags(hdf5_path, mask=None):
    """SED from apparent magnitudes, summed over all galaxies."""
    with h5py.File(hdf5_path, "r") as f:
        appmag_keys = [k for k in f["galaxy_data/dicts"].keys() if k.startswith("appmag.")]
        freqs, fluxes, labels = [], [], []

        for k in appmag_keys:
            mags = f[f"galaxy_data/dicts/{k}"][:]
            if mask is not None:
                mags = mags[mask]
            fnu_jy = 3631.0 * 10 ** (-mags / 2.5)
            filt = k.split("appmag.")[-1]
            
            try:
                fsps_filt = fsps.get_filter(filt)
                lam_eff = fsps_filt.lambda_eff * u.AA
                nu = (c / lam_eff).to_value(u.Hz)
                freqs.append(nu)
                fluxes.append(np.sum(fnu_jy))
                labels.append(filt)
            except Exception as e:
                pass

    freqs = np.asarray(freqs)
    fluxes = np.asarray(fluxes)
    labels = np.asarray(labels)
    order = np.argsort(freqs)
    
    lam_AA = (c / (freqs[order] * u.Hz)).to_value(u.AA)
    nuFnu = freqs[order] * fluxes[order]
    return lam_AA, nuFnu, labels[order]

def compute_summed_sed_from_absmags(hdf5_path, mask=None):
    """SED from absolute magnitudes, summed over selected galaxies."""
    with h5py.File(hdf5_path, "r") as f:
        absmag_keys = [k for k in f["galaxy_data/dicts"].keys() if k.startswith("absmag.")]
        freqs, fluxes, labels = [], [], []

        for k in absmag_keys:
            mags = f[f"galaxy_data/dicts/{k}"][:]
            if mask is not None:
                mags = mags[mask]
            fnu_jy = 3631.0 * 10 ** (-mags / 2.5)
            filt = k.split("absmag.")[-1]
            
            try:
                fsps_filt = fsps.get_filter(filt)
                lam_eff = fsps_filt.lambda_eff * u.AA
                nu = (c / lam_eff).to_value(u.Hz)
                freqs.append(nu)
                fluxes.append(np.sum(fnu_jy))
                labels.append(filt)
            except Exception as e:
                pass

    freqs = np.asarray(freqs)
    fluxes = np.asarray(fluxes)
    labels = np.asarray(labels)
    order = np.argsort(freqs)
    
    lam_AA = (c / (freqs[order] * u.Hz)).to_value(u.AA)
    nuFnu = freqs[order] * fluxes[order]
    return lam_AA, nuFnu, labels[order]

def classify_galaxies(hdf5_path):
    """
    Classify galaxies as star-forming or quenched.
    
    Parameters:
    - hdf5_path: Path to CAESAR HDF5 file
    - sfr_threshold: SFR threshold (Msun/yr) for classification
    - ssfr_threshold: sSFR threshold (1/yr) for classification
    
    Returns:
    - Dictionary with classification arrays
    """

    with h5py.File(hdf5_path, "r") as f:
        sfr = f["galaxy_data/sfr"][:]
        sfr_100 = f["galaxy_data/sfr_100"][:]
        stellar_mass = f["galaxy_data/dicts/masses.stellar"][:]

    ssfr = sfr / stellar_mass

    # Quenched: low SFR_100 AND low sSFR
    quenched = ssfr < 10**(-11)  # sSFR threshold defined by paper 
    star_forming = ~quenched
    
    return {
        'sfr': sfr,
        'sfr_100': sfr_100,
        'ssfr': ssfr,
        'stellar_mass': stellar_mass,
        'quenched': quenched,
        'star_forming': star_forming
    }

def get_stellar_mass_bins(stellar_mass, n_bins=5):
    """Create equal-log-spaced stellar mass bins."""
    log_mass = np.log10(stellar_mass[stellar_mass > 0])
    bins = np.logspace(log_mass.min(), log_mass.max(), n_bins + 1)
    return np.digitize(stellar_mass, bins) - 1, bins