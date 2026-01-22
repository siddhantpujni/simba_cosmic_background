import re
import numpy as np
import h5py
import fsps
from pathlib import Path
import astropy.units as u
from astropy.constants import c

CAT_DIR = Path("/home/sid/Documents/edinburgh/year_4/simba_cosmic_background/data/caesar_catalogues/m25n256")

def get_redshift(obj):
    for attr in ["redshift", "z"]:
        if hasattr(obj.simulation, attr):
            return float(getattr(obj.simulation, attr))

def list_snapshots(cat_dir=CAT_DIR, prefix="m25n256_"):
    files = sorted(cat_dir.glob(f"{prefix}*.hdf5"))
    snaps = []
    for f in files:
        m = re.search(r"_(\d+)\.hdf5$", f.name)
        if m:
            snaps.append((int(m.group(1)), f))
    snaps.sort(key=lambda t: t[0], reverse=True)
    return snaps

def compute_summed_sed_from_appmags(hdf5_path):
    """SED from apparent magnitudes, summed over all galaxies."""
    with h5py.File(hdf5_path, "r") as f:
        appmag_keys = [k for k in f["galaxy_data/dicts"].keys() if k.startswith("appmag.")]
        freqs, fluxes, labels = [], [], []

        for k in appmag_keys:
            mags = f[f"galaxy_data/dicts/{k}"][:]
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
                print(f"Skipping {filt}: {e}")

    freqs = np.asarray(freqs)
    fluxes = np.asarray(fluxes)
    labels = np.asarray(labels)
    order = np.argsort(freqs)
    
    lam_AA = (c / (freqs[order] * u.Hz)).to_value(u.AA)
    nuFnu = freqs[order] * fluxes[order]
    return lam_AA, nuFnu, labels[order]

def compute_summed_sed_from_absmags(hdf5_path):
    """SED from absolute magnitudes, summed over all galaxies."""
    with h5py.File(hdf5_path, "r") as f:
        absmag_keys = [k for k in f["galaxy_data/dicts"].keys() if k.startswith("absmag.")]
        freqs, fluxes, labels = [], [], []

        for k in absmag_keys:
            mags = f[f"galaxy_data/dicts/{k}"][:]
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
                print(f"Skipping {filt}: {e}")

    freqs = np.asarray(freqs)
    fluxes = np.asarray(fluxes)
    labels = np.asarray(labels)
    order = np.argsort(freqs)
    
    lam_AA = (c / (freqs[order] * u.Hz)).to_value(u.AA)
    nuFnu = freqs[order] * fluxes[order]
    return lam_AA, nuFnu, labels[order]