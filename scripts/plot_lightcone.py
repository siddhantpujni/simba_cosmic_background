"""
Visualise the SIMBA reconstructed light-cone.

Produces three figures:
  1. Number of galaxies per redshift bin  (N vs z histogram).
  2. 2-D light-cone slice (comoving radial vs transverse distance),
     coloured by redshift and by L_FIR.
  3. Sky-plane projected map with PSF convolution.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.ndimage import gaussian_filter

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LC_FILE = ROOT / "data" / "lightcones" / "lc_m100n1024_a0.5_z0.0-7.0.h5"
FIG_DIR = ROOT / "figures" / "lightcone"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# HDF5 snapshot directory (same layout used by src/config)
HDF5_DIR = Path("/home/spujni/sim/m100n1024/s50/Groups/")
SNAP_PREFIX = "m100n1024"


def load_lightcone(path):
    """Load lightcone data from HDF5."""
    with h5py.File(path, "r") as f:
        data = {
            "ra": f["RA"][:],
            "dec": f["DEC"][:],
            "z": f["z"][:],
            "stellar_mass": f["stellar_mass"][:],
            "snap": f["snap"][:],
            "galaxy_index": f["galaxy_index"][:],
            "area_deg2": f.attrs["area_deg2"],
        }
    return data

def load_app_mag(data, filter_name="g"):
    """
    Load apparent magnitude for each galaxy from Caesar snapshot files for a given filter.
    filter_name: e.g. 'g', 'r', etc. (for 'appmag.g', 'appmag.r', ...)
    Returns an array of apparent magnitudes (same length as lightcone).
    """
    n_gal = len(data["z"])
    app_mag = np.full(n_gal, np.nan)
    snap_arr = data["snap"]
    gal_idx = data["galaxy_index"]
    key_name = f"appmag.{filter_name}"

    cache = {}
    for snap in np.unique(snap_arr):
        snap = int(snap)
        hdf5 = HDF5_DIR / f"{SNAP_PREFIX}_{snap:03d}.hdf5"
        if not hdf5.exists():
            hdf5 = HDF5_DIR / f"{SNAP_PREFIX}_{snap}.hdf5"
        if not hdf5.exists():
            print(f"  WARN: missing {hdf5}, skipping snap {snap}")
            continue
        with h5py.File(hdf5, "r") as f:
            group = f["galaxy_data/dicts"]
            if key_name not in group:
                print(f"  WARN: {key_name} missing in snap {snap}, skipping")
                continue
            cache[snap] = group[key_name][:]

    for i in range(n_gal):
        s = int(snap_arr[i])
        gi = int(gal_idx[i])
        if s in cache and gi < len(cache[s]):
            val = cache[s][gi]
            if np.isfinite(val):
                app_mag[i] = val

    return app_mag

def load_lfir(data):
    """
    Cross-reference snapshot catalogues to retrieve per-galaxy L_FIR.

    Returns an array of L_FIR values (L_sun) with the same length as the
    lightcone.  Galaxies for which L_FIR is unavailable are set to NaN.
    """
    n_gal = len(data["z"])
    lfir = np.full(n_gal, np.nan)

    snap_arr = data["snap"]
    gal_idx = data["galaxy_index"]

    cache = {}
    for snap in np.unique(snap_arr):
        snap = int(snap)
        # Try zero-padded then unpadded filename
        hdf5 = HDF5_DIR / f"{SNAP_PREFIX}_{snap:03d}.hdf5"
        if not hdf5.exists():
            hdf5 = HDF5_DIR / f"{SNAP_PREFIX}_{snap}.hdf5"
        if not hdf5.exists():
            print(f"  WARN: missing {hdf5}, skipping snap {snap}")
            continue
        with h5py.File(hdf5, "r") as f:
            if "galaxy_data/L_FIR" not in f:
                print(f"  WARN: L_FIR missing in snap {snap}, skipping")
                continue
            cache[snap] = f["galaxy_data/L_FIR"][:]

    for i in range(n_gal):
        s = int(snap_arr[i])
        gi = int(gal_idx[i])
        if s in cache and gi < len(cache[s]):
            val = cache[s][gi]
            if np.isfinite(val) and val > 0:
                lfir[i] = val

    return lfir

def plot_lightcone_slice_appmag(data, app_mag, outpath, absmag_key="absmag.g"):
    """
    2D light-cone slice colored by apparent magnitude.
    """
    z = data["z"]
    ra = data["ra"]
    d_radial = cosmo.comoving_distance(z).to("Mpc").value
    ra_centre = np.median(ra)
    delta_ra_rad = np.deg2rad(ra - ra_centre)
    d_transverse = delta_ra_rad * d_radial
    y_lim = np.nanpercentile(np.abs(d_transverse), 99.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.scatter(
        d_radial, d_transverse, s=0.5, alpha=0.6,
        c=app_mag, cmap="viridis_r",  # reverse so bright = yellow
        vmin=np.nanpercentile(app_mag, 1),
        vmax=np.nanpercentile(app_mag, 99),
        rasterized=True,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f"Apparent magnitude ({absmag_key})", fontsize=11)
    ax.set_xlabel("Comoving radial distance [Mpc]", fontsize=11)
    ax.set_ylabel("Comoving transverse distance [Mpc]", fontsize=11)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title("Galaxies colored by apparent magnitude", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# ── Plot 2: 2-D light-cone slice ────────────────────────────────────────
def plot_lightcone_slice(data, lfir, outpath):
    """
    Two-panel 2-D light-cone slice (comoving radial vs transverse distance).

    Left  panel: points coloured by redshift.
    Right panel: points coloured by L_FIR (galaxies without L_FIR are grey).
    """
    z = data["z"]
    ra = data["ra"]

    d_radial = cosmo.comoving_distance(z).to("Mpc").value
    ra_centre = np.median(ra)
    delta_ra_rad = np.deg2rad(ra - ra_centre)
    d_transverse = delta_ra_rad * d_radial

    y_lim = np.nanpercentile(np.abs(d_transverse), 99.5)

    fig, axes = plt.subplots(1, 1, figsize=(16, 6))

    # ── left: coloured by redshift ───────────────────────────────────────
    im0 = axes.scatter(
        d_radial, d_transverse, s=0.5, alpha=0.6,
        c=z, cmap="plasma", vmin=0, vmax=z.max(),
        rasterized=True,
    )
    cb0 = fig.colorbar(im0, ax=axes, pad=0.02)
    cb0.set_label("Redshift $z$", fontsize=11)
    axes.set_xlabel("Comoving radial distance [Mpc]", fontsize=11)
    axes.set_ylabel("Comoving transverse distance [Mpc]", fontsize=11)
    axes.set_ylim(-y_lim, y_lim)
    axes.set_title("Number of Galaxies Coloured by Redshift", fontsize=12)

    fig.suptitle(
        f"SIMBA light cone 2-D slice  "
        f"($N_{{\\mathrm{{gal}}}} = {len(z):,}$)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_lightcone(LC_FILE)
    print("Loading apparent magnitude from snapshot catalogues …")
    app_mag = load_app_mag(data, filter_name="g")  # or "r", "i", etc.
    plot_lightcone_slice_appmag(
        data, app_mag,
        FIG_DIR / "lightcone_slice_appmag_g.png",
        absmag_key="appmag.g"
    )