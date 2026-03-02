from pathlib import Path
import numpy as np
import h5py
import astropy.units as u
import caesar

from src.config import SimConfig
from src.utils import get_redshift
from src.physics.radio import radio_luminosity_sf, CHABRIER_FRAC_M5
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
                                n_points=500):
    """
    Compute the radio cosmic background intensity from star formation
    using the Condon (1992) / Thomas+2021 prescription.

    For each lightcone galaxy
    -------------------------
    1. Read total SFR from the CAESAR HDF5 catalogue.
    2. Compute  SFR[M >= 5 Msun] = f_IMF × SFR_total   (Chabrier IMF).
    3. At each *rest-frame* frequency  ν_rest = ν_obs (1+z),
       evaluate P_ν(ν_rest) via Condon (1992) eqs 10+11.
    4. Convert to observed flux  F_ν = (1+z) P_ν / (4π d_L²).

    Returns
    -------
    nu_obs   : array (Hz)               – observed frequency grid
    intensity : array (erg/s/cm²/Hz/sr) – specific intensity
    """
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        gal_z    = lc["z"][:]
        snap_arr = lc["snap"][:]
        gal_idx  = lc["galaxy_index"][:]

    # Observed frequency grid: 10 MHz  →  100 GHz  (radio regime)
    nu_obs_hz = np.logspace(np.log10(1e7), np.log10(1e11), n_points)  # Hz
    omega_sr  = area_deg2 * (np.pi / 180.0) ** 2

    total_flux = np.zeros_like(nu_obs_hz)   # erg/s/cm²/Hz
    cache = {}

    unique_snaps = np.unique(snap_arr)
    print(f"Processing {len(gal_z)} galaxies across "
          f"{len(unique_snaps)} snapshots (radio) …")

    for snap in unique_snaps:
        snap  = int(snap)
        smask = snap_arr == snap

        if snap not in cache:
            hdf5 = cfg.hdf5_path(snap)
            if not hdf5.exists():
                print(f"  WARN: missing {hdf5}, skipping snap {snap}")
                continue
            with h5py.File(hdf5, "r") as f:
                if "galaxy_data/sfr" not in f:
                    print(f"  WARN: SFR missing in snap {snap}, skipping")
                    continue
                sfr = f["galaxy_data/sfr"][:]
            cache[snap] = sfr

        sfr = cache[snap]

        for gi, gz in zip(gal_idx[smask], gal_z[smask]):
            gi = int(gi)
            if gi >= len(sfr):
                continue

            sfr_gal = sfr[gi]
            if not (np.isfinite(sfr_gal) and sfr_gal > 0):
                continue

            # Rest-frame frequencies for observed grid
            nu_rest_ghz = nu_obs_hz * (1.0 + gz) / 1e9

            # Rest-frame radio luminosity at each frequency (W/Hz)
            P_nu = radio_luminosity_sf(sfr_gal, nu_rest_ghz)

            # Convert  W/Hz → erg/s/Hz  (1 W = 1e7 erg/s)
            P_nu_cgs = P_nu * 1e7

            # Observed flux density:  F_ν = (1+z) L_ν / (4π d_L²)
            d_L  = cfg.cosmology.luminosity_distance(gz).to(u.cm).value
            flux = (1.0 + gz) * P_nu_cgs / (4.0 * np.pi * d_L ** 2)

            if np.all(np.isfinite(flux)):
                total_flux += flux

    # Convert summed flux to surface brightness
    intensity = total_flux / omega_sr
    print("Done.")
    return nu_obs_hz, intensity