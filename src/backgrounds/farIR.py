"""Far-IR cosmic background pipeline."""

from pathlib import Path
import numpy as np
import h5py
import caesar
import astropy.units as u

from src.config import SimConfig
from src.utils import get_redshift
from src.physics.dust import equivalent_dust_temperature
from src.physics.sed import mbb, normalised_mbb
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


def summed_mbb_single(cfg, snap, beta=2.0, n_points=500, a_dust=-0.05):
    """
    Rest-frame summed MBB SED for one snapshot.
    Returns wavelength (Angstrom), total SED.
    """
    hdf5 = cfg.hdf5_path(snap)
    z = _redshift_for_snap(cfg, snap)

    with h5py.File(hdf5, "r") as f:
        if "galaxy_data/L_FIR" not in f:
            print(f"  WARN: L_FIR missing in snap {snap}, skipping")
            lam = np.logspace(3.5, 5, n_points)
            return lam, np.zeros(n_points)
        lfir = f["galaxy_data/L_FIR"][:]

    T_eqv, mask = equivalent_dust_temperature(hdf5, z, a=a_dust)
    lam = np.logspace(4, 5, n_points)

    total_sed = np.zeros_like(lam)
    for i in range(len(lfir)):
        if not mask[i]:
            continue
        sed = normalised_mbb(lam, lfir[i], T_eqv[i], beta)
        if sed is not None:
            total_sed += sed

    return lam, total_sed


def build_lightcone(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0):
    """Generate or load a cached lightcone."""
    LIGHTCONE_DIR.mkdir(parents=True, exist_ok=True)
    lc_path = LIGHTCONE_DIR / f"lc_{cfg.name}_a{area_deg2}_z{z_min}-{z_max}.h5"

    if lc_path.exists():
        print(f"Lightcone cached: {lc_path}")
        return lc_path

    generate_lightcone(cfg, area_deg2, z_min, z_max, lc_path, verbose=True)
    return lc_path


def lightcone_farIR_background(cfg, area_deg2=1.0, z_min=0.0, z_max=3.0,
                                beta=2.0, n_points=500, a_dust=-0.05,
                                return_dust_temps=False):
    """
    Compute the far-IR cosmic background intensity by summing
    redshifted MBB SEDs from all lightcone galaxies.

    Parameters
    ----------
    a_dust : float
        Leading normalisation parameter 'a' in the Liang+19 T_eqv relation.
    return_dust_temps : bool
        If True, also return arrays of per-galaxy dust temperatures and
        redshifts (useful for diagnostic plots).

    Returns
    -------
    lam_obs   : array (Angstrom)
    intensity : array (erg/s/cm^2/sr/AA)
    [dust_temps, dust_redshifts] : returned only when return_dust_temps=True
    """
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        gal_z = lc["z"][:]
        snap_arr = lc["snap"][:]
        gal_idx = lc["galaxy_index"][:]

    # ── wavelength grid: 8 µm  →  10 mm ─────────────────
    lam_obs = np.logspace(np.log10(1.5e5), np.log10(1e8), n_points)  # Å
    omega_sr = area_deg2 * (np.pi / 180.0) ** 2

    total_intensity = np.zeros_like(lam_obs)
    cache = {}

    # Collectors for dust-temperature diagnostics
    all_temps = [] if return_dust_temps else None
    all_zs    = [] if return_dust_temps else None

    unique_snaps = np.unique(snap_arr)
    print(f"Processing {len(gal_z)} galaxies across "
          f"{len(unique_snaps)} snapshots …")

    for snap in unique_snaps:
        snap = int(snap)
        smask = snap_arr == snap

        if snap not in cache:
            hdf5 = cfg.hdf5_path(snap)
            if not hdf5.exists():
                print(f"  WARN: missing {hdf5}, skipping snap {snap}")
                continue
            z = _redshift_for_snap(cfg, snap)
            with h5py.File(hdf5, "r") as f:
                if "galaxy_data/L_FIR" not in f:
                    print(f"  WARN: L_FIR missing in snap {snap}, skipping")
                    continue
                lfir = f["galaxy_data/L_FIR"][:]
            T_eqv, vmask = equivalent_dust_temperature(hdf5, z, a=a_dust)
            cache[snap] = (lfir, T_eqv, vmask)

        lfir, T_eqv, vmask = cache[snap]

        for gi, gz in zip(gal_idx[smask], gal_z[smask]):
            gi = int(gi)
            if gi >= len(lfir) or not vmask[gi]:
                continue
            L, T = lfir[gi], T_eqv[gi]
            if not (np.isfinite(L) and np.isfinite(T) and L > 0 and T > 0):
                continue

            if return_dust_temps:
                all_temps.append(T)
                all_zs.append(gz)

            lam_rest = lam_obs / (1.0 + gz)
            sed = normalised_mbb(lam_rest, L, T, beta)
            if sed is None:
                continue

            d_L = cfg.cosmology.luminosity_distance(gz).to(u.cm).value
            LSUN_ERG_S = 3.828e33

            flux = sed * LSUN_ERG_S / (4.0 * np.pi * d_L ** 2 * (1.0 + gz))

            if np.all(np.isfinite(flux)):
                total_intensity += flux

    total_intensity /= omega_sr
    print("Done.")

    if return_dust_temps:
        return (lam_obs, total_intensity,
                np.asarray(all_temps), np.asarray(all_zs))
    return lam_obs, total_intensity