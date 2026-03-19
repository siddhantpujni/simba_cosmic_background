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

def load_appmag_v(data):
    """
    Cross-reference snapshot catalogues to retrieve per-galaxy apparent magnitude in the 'v' filter.

    Returns an array of apparent magnitudes (same length as the lightcone).
    Galaxies for which appmag.v is unavailable are set to NaN.
    """
    n_gal = len(data["z"])
    appmag_v = np.full(n_gal, np.nan)

    snap_arr = data["snap"]
    gal_idx = data["galaxy_index"]
    key_name = "appmag.v"

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
                appmag_v[i] = val

    return appmag_v

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

def plot_two_panel_wedge_with_redshift(data, appmag_v, lfir, outpath, random_seed=42, max_points=100000):
    """
    Create a two-panel wedge diagram for the SIMBA light cone.
    Panel 1: v-band apparent magnitude (optical)
    Panel 2: log10(L_FIR / L_sun) (far-IR)
    Both panels: x-axis is comoving radial distance [Mpc], secondary x-axis is redshift.
    """
    plt.style.use('dark_background')

    # Geometry
    z = data["z"]
    ra = data["ra"]
    d_radial = cosmo.comoving_distance(z).to("Mpc").value
    ra_centre = np.median(ra)
    delta_ra_rad = np.deg2rad(ra - ra_centre)
    d_transverse = delta_ra_rad * d_radial

    n_gal = len(z)
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(n_gal, size=min(max_points, n_gal), replace=False)

    # Subsampled arrays
    d_radial_sub = d_radial[idx]
    d_transverse_sub = d_transverse[idx]
    appmag_v_sub = appmag_v[idx]
    lfir_sub = lfir[idx]
    z_sub = z[idx]

    # Panel 1: v-band apparent magnitude
    vmin_v = np.nanpercentile(appmag_v_sub, 1)
    vmax_v = np.nanpercentile(appmag_v_sub, 99)

    # Panel 2: log10(L_FIR / L_sun)
    lfir_log = np.log10(lfir_sub)
    lfir_log[~np.isfinite(lfir_log)] = np.nan
    vmin_lfir = np.nanpercentile(lfir_log, 2)
    vmax_lfir = np.nanpercentile(lfir_log, 98)

    # Shared axis limits
    y_lim = np.nanpercentile(np.abs(d_transverse_sub), 99.5)
    x_lim = [np.nanmin(d_radial_sub), np.nanmax(d_radial_sub)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
    panels = [
        {
            "c": appmag_v_sub,
            "cmap": "viridis_r",
            "vmin": vmin_v,
            "vmax": vmax_v,
            "label": "v-band apparent magnitude",
            "panel_label": "(a)"
        },
        {
            "c": lfir_log,
            "cmap": "inferno",
            "vmin": vmin_lfir,
            "vmax": vmax_lfir,
            "label": r"$\log_{10}(L_\mathrm{FIR} / L_\odot)$",
            "panel_label": "(b)"
        }
    ]

    for i, ax in enumerate(axes):
        im = ax.scatter(
            d_radial_sub, d_transverse_sub, s=0.5, alpha=0.3,
            c=panels[i]["c"], cmap=panels[i]["cmap"],
            vmin=panels[i]["vmin"], vmax=panels[i]["vmax"],
            rasterized=True
        )
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(panels[i]["label"], fontsize=11)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlim(x_lim)
        ax.text(0.01, 0.95, panels[i]["panel_label"], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")
        if i == 0:
            ax.set_ylabel("Comoving transverse distance [Mpc]", fontsize=12)
        if i == 1:
            ax.set_xlabel("Comoving radial distance [Mpc]", fontsize=12)
            # Add secondary x-axis for redshift
            secax = ax.secondary_xaxis('top')
            # Map comoving distance to redshift using interpolation
            sorted_idx = np.argsort(d_radial)
            d_radial_sorted = d_radial[sorted_idx]
            z_sorted = z[sorted_idx]
            def comoving_to_z(x):
                return np.interp(x, d_radial_sorted, z_sorted)
            def z_to_comoving(zval):
                return np.interp(zval, z_sorted, d_radial_sorted)
            
            secax.set_functions((comoving_to_z, z_to_comoving))
            secax.set_xlabel("Redshift $z$", fontsize=12)
            secax.set_xlim(x_lim)
            z_ticks = [0.5, 1, 2, 3, 4, 5, 6, 7]
            comoving_ticks = [z_to_comoving(z) for z in z_ticks]
            secax.set_xticks(comoving_ticks)
            secax.set_xticklabels([f"{z:.1f}" for z in z_ticks])
            
    fig.suptitle(
        f"SIMBA light cone — $N_\\text{{gal}} = {n_gal:,}$",
        fontsize=16, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

data = load_lightcone(LC_FILE)
appmag_v = load_appmag_v(data)
lfir = load_lfir(data)
plot_two_panel_wedge_with_redshift(
    data, appmag_v, lfir,
    FIG_DIR / "lightcone_wedge_two_panel.png"
)