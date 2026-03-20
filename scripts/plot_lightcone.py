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

import matplotlib as mpl

mpl.rcParams.update({
    # Requires a LaTeX install (TeX Live / MiKTeX). Without one, set
    # 'text.usetex': False and 'mathtext.fontset': 'cm' instead.
    'text.usetex'         : True,
    'text.latex.preamble' : r'\usepackage{amsmath}',
    'font.family'         : 'serif',   # Computer Modern = default LaTeX font

    # Match your document's font sizes (most journals: 10 pt)
    'font.size'           : 10,
    'axes.labelsize'      : 10,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 9,

    # Okabe–Ito palette — colorblind-safe, one line to replace the default cycle
    'axes.prop_cycle': mpl.cycler('color', [
        '#0072B2', '#D55E00', '#009E73',
        '#E69F00', '#CC79A7', '#56B4E9',
    ]),

    'lines.linewidth'  : 1.5,
    'axes.linewidth'   : 0.8,
})


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

with h5py.File(ROOT / "data" / "results" / "radio_flux_1p4GHz_m100n1024.h5", "r") as f:
    radio_flux = f["flux_total"][:]

def plot_three_panel_wedge_with_radio(data, appmag_v, lfir, radio_flux, outpath, random_seed=42, max_points=240883):
    """
    Three-panel wedge diagram for the SIMBA light cone.
    Panel 1: v-band apparent magnitude (optical)
    Panel 2: log10(L_FIR / L_sun) (far-IR)
    Panel 3: log10(radio flux at 1.4 GHz) [erg/s/cm^2/Hz]
    All: x-axis is comoving radial distance [Mpc], y is transverse.
    """
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
    radio_flux_sub = radio_flux[idx]
    z_sub = z[idx]

    # Panel 1: v-band apparent magnitude
    vmin_v = np.nanpercentile(appmag_v_sub, 1)
    vmax_v = np.nanpercentile(appmag_v_sub, 99)

    # Panel 2: log10(L_FIR / L_sun) 
    lfir_log = np.log10(lfir)
    lfir_log[~np.isfinite(lfir_log)] = np.nan
    lfir_log_sub = lfir_log[idx]
    vmin_lfir = np.nanpercentile(lfir_log_sub, 2)
    vmax_lfir = np.nanpercentile(lfir_log_sub, 98)

    # Panel 3: log10(radio flux)
    radio_log = np.log10(radio_flux_sub)
    radio_log[~np.isfinite(radio_log)] = np.nan
    vmin_radio = np.nanpercentile(radio_log, 2)
    vmax_radio = np.nanpercentile(radio_log, 98)

    # Shared axis limits
    y_lim = np.nanpercentile(np.abs(d_transverse_sub), 99.5)
    x_lim = [np.nanmin(d_radial_sub), np.nanmax(d_radial_sub)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    panels = [
        {
            "c": appmag_v_sub,
            "cmap": "viridis_r",
            "vmin": vmin_v,
            "vmax": vmax_v,
            "label": r"$m_V$",
            "panel_label": "(a)"
        },
        {
            "c": lfir_log_sub,
            "cmap": "inferno",
            "vmin": vmin_lfir,
            "vmax": vmax_lfir,
            "label": r"$\log_{10}(L_\mathrm{FIR} / L_\odot)$",
            "panel_label": "(b)"
        },
        {
            "c": radio_log,
            "cmap": "plasma",
            "vmin": vmin_radio,
            "vmax": vmax_radio,
            "label": r"$\log_{10}(S_{1.4\,\mathrm{GHz}})\ \mathrm{[erg\,s^{-1}\,cm^{-2}\,Hz^{-1}]}$",            "panel_label": "(c)"
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
        if i == 2:
            ax.set_xlabel("Comoving Radial Distance [Mpc]", fontsize=12)
        if i == 1:
            ax.set_ylabel("Comoving Transverse Distance [Mpc]", fontsize=12)
    fig.suptitle(
        f"SIMBA Light Cone Coloured Wedges",
    fontsize=16, y=0.90
)
# ── Redshift Axis Placement ─────────────────────────────────────────
    # Target the top panel so the redshift axis frames the entire figure
    ax_top_panel = axes[0]
    x_min, x_max = axes[-1].get_xlim()
    
    # Choose redshift ticks (e.g., z = 0, 1, 2, 3, 4, 5, 6, 7)
    z_ticks = np.arange(0, 8, 1)
    d_ticks = cosmo.comoving_distance(z_ticks).to("Mpc").value
    
    # Only keep ticks within the current x-limits
    mask = (d_ticks >= x_min) & (d_ticks <= x_max)
    z_ticks = z_ticks[mask]
    d_ticks = d_ticks[mask]
    
    # Add the twin axis
    ax_z = ax_top_panel.twiny()
    ax_z.set_xlim(x_min, x_max)
    ax_z.set_xticks(d_ticks)
    ax_z.set_xticklabels([f"{z:.0f}" for z in z_ticks])
    ax_z.set_xlabel(r"Redshift $z$", fontsize=12, labelpad=2)

    # ── Final Layout Adjustments ────────────────────────────────────────
    # Run tight_layout first to handle the internal panel spacing
    fig.tight_layout()
    
    # Adjust the top margin manually to leave room for the twin axis and the suptitle
    fig.subplots_adjust(top=0.88)
    fig.suptitle("SIMBA Light Cone Coloured Wedges", fontsize=16, y=0.96)
    
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# --- Usage ---
data = load_lightcone(LC_FILE)
appmag_v = load_appmag_v(data)
lfir = load_lfir(data)
with h5py.File(ROOT / "data" / "results" / "radio_flux_1p4GHz_m100n1024.h5", "r") as f:
    radio_flux = f["flux_total"][:]

plot_three_panel_wedge_with_radio(
    data, appmag_v, lfir, radio_flux,
    FIG_DIR / "lightcone_wedge_three_panel.png"
)

