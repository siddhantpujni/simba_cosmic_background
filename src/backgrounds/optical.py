import numpy as np
import h5py
from pathlib import Path
import astropy.units as u
from astropy.constants import c
from astropy.cosmology import Planck15 as cosmo
import caesar
import os
os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')
import fsps
from src.config import SimConfig
from src.lightcone.generate import generate_lightcone

LIGHTCONE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lightcones"
SKIP_SNAPS = {150, 151}

LSUN_ERG_S = 3.828e33  # erg/s

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

def classify_galaxies(hdf5_path, redshift):
    """
    Classify galaxies as star-forming or quenched using evolving sSFR threshold.

    Parameters:
    - hdf5_path: Path to CAESAR HDF5 file
    - redshift: Redshift of the snapshot

    Returns:
    - Dictionary with classification arrays
    """

    with h5py.File(hdf5_path, "r") as f:
        sfr = f["galaxy_data/sfr"][:]
        sfr_100 = f["galaxy_data/sfr_100"][:]
        stellar_mass = f["galaxy_data/dicts/masses.stellar"][:]

    ssfr = sfr / stellar_mass

    # Evolving sSFR threshold: 0.2 / t_H(z)
    t_H = cosmo.age(redshift).to('yr').value  # Age of universe at z in years
    ssfr_thresh = 0.2 / t_H  # 1/yr

    quenched = ssfr < ssfr_thresh
    star_forming = ~quenched

    return {
        'sfr': sfr,
        'sfr_100': sfr_100,
        'ssfr': ssfr,
        'stellar_mass': stellar_mass,
        'quenched': quenched,
        'star_forming': star_forming,
        'ssfr_thresh': ssfr_thresh
    }

def get_stellar_mass_bins(stellar_mass, n_bins=5):
    """Create equal-log-spaced stellar mass bins."""
    log_mass = np.log10(stellar_mass[stellar_mass > 0])
    bins = np.logspace(log_mass.min(), log_mass.max(), n_bins + 1)
    return np.digitize(stellar_mass, bins) - 1, bins

def build_lightcone(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0):
    """Generate or load a cached lightcone."""
    LIGHTCONE_DIR.mkdir(parents=True, exist_ok=True)
    lc_path = LIGHTCONE_DIR / f"lc_{cfg.name}_a{area_deg2}_z{z_min}-{z_max}.h5"
    if lc_path.exists():
        print(f"Lightcone cached: {lc_path}")
        return lc_path
    generate_lightcone(cfg, area_deg2, z_min, z_max, lc_path, verbose=True)
    return lc_path


def lightcone_optical_background(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0):
    """
    Compute the optical/near-IR cosmic background intensity using
    Caesar's pre-computed apparent magnitudes (with and without dust).

    Returns
    -------
    lam_obs          : array (Angstrom) — filter effective wavelengths
    intensity        : array (erg/s/cm^2/Hz/sr)  — with dust
    intensity_nodust : array (erg/s/cm^2/Hz/sr)  — no dust
    """
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        gal_z = lc["z"][:]
        snap_arr = lc["snap"][:]
        gal_idx = lc["galaxy_index"][:]

    omega_sr = area_deg2 * (np.pi / 180.0) ** 2

    # We'll accumulate flux at each filter's effective wavelength
    # First pass: figure out which filters are available
    unique_snaps = np.unique(snap_arr)
    
    # Get filter list from first valid snapshot
    filter_info = {}  # filter_name -> (nu_Hz, lam_AA)
    for snap in unique_snaps:
        snap = int(snap)
        if snap in SKIP_SNAPS:
            continue
        hdf5 = cfg.hdf5_path(snap)
        if not hdf5.exists():
            continue
        with h5py.File(hdf5, "r") as f:
            if "galaxy_data/dicts" not in f:
                continue
            for k in f["galaxy_data/dicts"].keys():
                if k.startswith("appmag."):
                    filt = k.split("appmag.")[-1]
                    if filt not in filter_info:
                        try:
                            fsps_filt = fsps.get_filter(filt)
                            lam_eff = fsps_filt.lambda_eff  # Angstrom
                            nu = (c / (lam_eff * u.AA)).to_value(u.Hz)
                            filter_info[filt] = (nu, lam_eff)
                        except:
                            pass
        break  # only need first snapshot to get filter list

    if not filter_info:
        raise ValueError("No valid filters found in Caesar catalogues")

    # Sort by frequency
    filters_sorted = sorted(filter_info.keys(), key=lambda f: filter_info[f][0])
    nu_arr = np.array([filter_info[f][0] for f in filters_sorted])
    lam_arr = np.array([filter_info[f][1] for f in filters_sorted])

    total_fnu = np.zeros(len(filters_sorted))
    total_fnu_nodust = np.zeros(len(filters_sorted))

    # Cache HDF5 data per snapshot
    cache = {}

    print(f"Processing {len(gal_z)} galaxies across "
          f"{len(unique_snaps)} snapshots …")

    for snap in unique_snaps:
        snap = int(snap)
        if snap in SKIP_SNAPS:
            continue

        smask = snap_arr == snap
        hdf5 = cfg.hdf5_path(snap)
        if not hdf5.exists():
            print(f"  WARN: missing {hdf5}, skipping snap {snap}")
            continue

        if snap not in cache:
            with h5py.File(hdf5, "r") as f:
                mags = {}
                mags_nodust = {}
                for filt in filters_sorted:
                    key = f"galaxy_data/dicts/appmag.{filt}"
                    key_nd = f"galaxy_data/dicts/appmag_nodust.{filt}"
                    if key in f:
                        mags[filt] = f[key][:]
                    if key_nd in f:
                        mags_nodust[filt] = f[key_nd][:]
            cache[snap] = (mags, mags_nodust)

        mags, mags_nodust = cache[snap]
        n_gals = len(mags[filters_sorted[0]]) if filters_sorted[0] in mags else 0

        print(f"  snap {snap}: {smask.sum()} lightcone galaxies")

        for gi in gal_idx[smask]:
            gi = int(gi)
            if gi >= n_gals:
                continue

            for i, filt in enumerate(filters_sorted):
                # With dust
                if filt in mags:
                    mag = mags[filt][gi]
                    if np.isfinite(mag):
                        fnu_jy = 3631.0 * 10 ** (-mag / 2.5)
                        total_fnu[i] += fnu_jy

                # Without dust
                if filt in mags_nodust:
                    mag_nd = mags_nodust[filt][gi]
                    if np.isfinite(mag_nd):
                        fnu_jy_nd = 3631.0 * 10 ** (-mag_nd / 2.5)
                        total_fnu_nodust[i] += fnu_jy_nd

    # Convert Jy to cgs: 1 Jy = 1e-23 erg/s/cm²/Hz
    total_fnu_cgs = total_fnu * 1e-23
    total_fnu_nodust_cgs = total_fnu_nodust * 1e-23

    # Divide by solid angle to get intensity
    intensity = total_fnu_cgs / omega_sr
    intensity_nodust = total_fnu_nodust_cgs / omega_sr

    print("Done.")
    return lam_arr, intensity, intensity_nodust