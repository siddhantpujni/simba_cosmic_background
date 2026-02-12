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

def _redshift_for_snap(cfg, snap):
    """Get redshift for a snapshot from HDF5 attrs or Caesar."""
    hdf5 = cfg.hdf5_path(snap)
    if hdf5.exists():
        with h5py.File(hdf5, "r") as f:
            if "redshift" in f.attrs:
                return float(f.attrs["redshift"])
    caesar_f = cfg.caesar_path(snap)
    if caesar_f.exists():
        obj = caesar.load(str(caesar_f))
        return get_redshift(obj)
    raise FileNotFoundError(f"Cannot get redshift for snap {snap}")

def build_lightcone(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0):
    """Generate or load a cached lightcone."""
    LIGHTCONE_DIR.mkdir(parents=True, exist_ok=True)
    lc_path = LIGHTCONE_DIR / f"lc_{cfg.name}_a{area_deg2}_z{z_min}-{z_max}.h5"
    if lc_path.exists():
        print(f"Lightcone cached: {lc_path}")
        return lc_path
    generate_lightcone(cfg, area_deg2, z_min, z_max, lc_path, verbose=True)
    return lc_path


def lightcone_optical_background(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0,
                                  n_points=500):
    """
    Compute the optical/near-IR cosmic background intensity by
    generating FSPS spectra for each lightcone galaxy, redshifting,
    and summing.

    Returns
    -------
    lam_obs   : array (Angstrom)
    intensity : array (erg/s/cm^2/Hz/sr)
    """
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        gal_z = lc["z"][:]
        snap_arr = lc["snap"][:]
        gal_idx = lc["galaxy_index"][:]

    # Common observed wavelength grid: 1000 Å – 50 000 Å
    lam_obs = np.logspace(np.log10(1000), np.log10(50000), n_points)
    omega_sr = area_deg2 * (np.pi / 180.0) ** 2

    total_flux = np.zeros_like(lam_obs)

    # Initialise FSPS stellar population (once)
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, logzsol=0.0,
                                dust_type=2)

    # Cache Caesar objects per snapshot
    caesar_cache = {}

    unique_snaps = np.unique(snap_arr)
    print(f"Processing {len(gal_z)} galaxies across "
          f"{len(unique_snaps)} snapshots …")

    for snap in unique_snaps:
        snap = int(snap)
        if snap in SKIP_SNAPS:
            continue

        smask = snap_arr == snap
        cat_path = cfg.caesar_path(snap)
        if not cat_path.exists():
            print(f"  WARN: missing {cat_path}, skipping snap {snap}")
            continue

        if snap not in caesar_cache:
            caesar_cache[snap] = caesar.load(str(cat_path))
        obj = caesar_cache[snap]

        n_gals = len(obj.galaxies)
        print(f"  snap {snap}: {smask.sum()} lightcone galaxies "
              f"(catalogue has {n_gals})")

        for gi, gz in zip(gal_idx[smask], gal_z[smask]):
            gi = int(gi)
            if gi >= n_gals:
                continue

            gal = obj.galaxies[gi]
            stellar_mass = gal.masses['stellar'].value
            metallicity = gal.metallicities['stellar']

            if stellar_mass <= 0 or metallicity <= 0:
                continue

            age_gyr = cfg.cosmology.age(gz).value
            if age_gyr <= 0:
                continue

            # Set FSPS parameters
            sp.params['logzsol'] = np.log10(metallicity / 0.0142)
            sp.params['tage'] = age_gyr

            wave, spec_Lsun_Hz = sp.get_spectrum(tage=age_gyr)

            # Scale by stellar mass (FSPS normalises to 1 Msun)
            spec_Lsun_Hz *= stellar_mass

            # Convert Lsun/Hz → erg/s/cm²/Hz at observer
            d_L = cfg.cosmology.luminosity_distance(gz).to(u.cm).value
            spec_flux = spec_Lsun_Hz * LSUN_ERG_S / (4.0 * np.pi * d_L ** 2)

            # Redshift the wavelength grid
            wave_obs = wave * (1.0 + gz)

            # Interpolate onto common grid and accumulate
            flux_interp = np.interp(lam_obs, wave_obs, spec_flux,
                                    left=0.0, right=0.0)
            if np.all(np.isfinite(flux_interp)):
                total_flux += flux_interp

    # Convert summed flux to surface brightness
    intensity = total_flux / omega_sr
    print("Done.")
    return lam_obs, intensity