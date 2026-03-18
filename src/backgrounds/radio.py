from pathlib import Path
import numpy as np
import h5py
import astropy.units as u
import caesar

from src.config import SimConfig
from src.utils import get_redshift
from src.physics.radio import radio_luminosity_sf, agn_radio_luminosity, CHABRIER_FRAC_M5
from src.lightcone.generate import generate_lightcone

LIGHTCONE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lightcones"

def _redshift_for_snap(cfg, snap):
    """Get redshift for a snapshot, trying HDF5 attrs then Caesar."""
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


def lightcone_radio_background(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0,
                                n_points=500, galaxy_mask=None):
    """
    Compute the radio cosmic background intensity from star formation
    (Condon 1992 / Thomas+2021) **and** AGN accretion.

    For each lightcone galaxy
    -------------------------
    SF  :  P_ν(ν_rest) via Condon (1992) eqs 10+11          [W Hz⁻¹]
    AGN :  P_rad(ν_rest) from Mdot_BH relation + ν^{-0.7}   [erg s⁻¹ Hz⁻¹]

    Observed flux:  F_ν = (1+z) P_ν / (4π d_L²)

    Parameters
    ----------
    galaxy_mask : array-like, optional
        Boolean mask of same length as lightcone galaxies. If provided,
        only galaxies where mask is True are included. Used for jackknife.

    Returns
    -------
    nu_obs    : array (Hz)               – observed frequency grid
    intensity : array (erg/s/cm²/Hz/sr)  – total specific intensity
    intensity_sf  : array                – SF-only component
    intensity_agn : array                – AGN-only component
    """
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        gal_z    = lc["z"][:]
        snap_arr = lc["snap"][:]
        gal_idx  = lc["galaxy_index"][:]

    # Apply galaxy mask if provided
    if galaxy_mask is not None:
        galaxy_mask = np.asarray(galaxy_mask)
        if len(galaxy_mask) != len(gal_z):
            raise ValueError(f"galaxy_mask length ({len(galaxy_mask)}) != "
                           f"lightcone length ({len(gal_z)})")
    else:
        galaxy_mask = np.ones(len(gal_z), dtype=bool)

    # Observed frequency grid: 10 MHz  →  100 GHz  (radio regime)
    nu_obs_hz = np.logspace(np.log10(1e7), np.log10(1e11), n_points)  # Hz
    omega_sr  = area_deg2 * (np.pi / 180.0) ** 2

    total_flux_sf  = np.zeros_like(nu_obs_hz)   # erg/s/cm²/Hz
    total_flux_agn = np.zeros_like(nu_obs_hz)
    cache = {}

    unique_snaps = np.unique(snap_arr)
    print(f"Processing {galaxy_mask.sum()} galaxies across "
          f"{len(unique_snaps)} snapshots (radio) …")

    for snap in unique_snaps:
        snap  = int(snap)
        smask = (snap_arr == snap) & galaxy_mask

        if snap not in cache:
            hdf5 = cfg.hdf5_path(snap)
            if not hdf5.exists():
                print(f"  WARN: missing {hdf5}, skipping snap {snap}")
                continue
            with h5py.File(hdf5, "r") as f:
                if "galaxy_data/sfr" not in f:
                    print(f"  WARN: SFR missing in snap {snap}, skipping")
                    continue
                sfr   = f["galaxy_data/sfr"][:]
                bhmdot = (f["galaxy_data/bhmdot"][:]
                          if "galaxy_data/bhmdot" in f
                          else np.zeros_like(sfr))
            cache[snap] = (sfr, bhmdot)

        sfr, bhmdot = cache[snap]

        for gi, gz in zip(gal_idx[smask], gal_z[smask]):
            gi = int(gi)
            if gi >= len(sfr):
                continue

            sfr_gal   = sfr[gi]
            bhmdot_gal = bhmdot[gi]

            # Rest-frame frequencies for observed grid
            nu_rest_ghz = nu_obs_hz * (1.0 + gz) / 1e9

            # Luminosity distance  (cm)
            d_L = cfg.cosmology.luminosity_distance(gz).to(u.cm).value
            prefactor = (1.0 + gz) / (4.0 * np.pi * d_L ** 2)

            # ── SF contribution ──────────────────────────────
            if np.isfinite(sfr_gal) and sfr_gal > 0:
                P_nu_sf = radio_luminosity_sf(sfr_gal, nu_rest_ghz)  # W/Hz
                P_nu_sf_cgs = P_nu_sf * 1e7                          # erg/s/Hz
                flux_sf = prefactor * P_nu_sf_cgs
                if np.all(np.isfinite(flux_sf)):
                    total_flux_sf += flux_sf

            # ── AGN contribution ─────────────────────────────
            if np.isfinite(bhmdot_gal) and bhmdot_gal > 0:
                P_agn = agn_radio_luminosity(bhmdot_gal, nu_rest_ghz)  # erg/s/Hz
                flux_agn = prefactor * P_agn
                if np.all(np.isfinite(flux_agn)):
                    total_flux_agn += flux_agn

    # Convert summed flux to surface brightness
    intensity_sf  = total_flux_sf  / omega_sr
    intensity_agn = total_flux_agn / omega_sr
    intensity     = intensity_sf + intensity_agn
    print("Done.")
    return nu_obs_hz, intensity, intensity_sf, intensity_agn