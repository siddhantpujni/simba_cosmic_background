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


# ── Plot 1: N(z) histogram ──────────────────────────────────────────────
def plot_ngal_vs_redshift(data, outpath, n_bins=50):
    """Histogram of galaxy counts per redshift bin."""
    z = data["z"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z, bins=n_bins, range=(0, z.max()), color="steelblue",
            edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Redshift $z$", fontsize=12)
    ax.set_ylabel("$N_{\\mathrm{gal}}$ per bin", fontsize=12)
    ax.set_title(
        f"SIMBA light-cone redshift distribution  "
        f"($N_{{\\mathrm{{total}}}} = {len(z):,}$)",
        fontsize=13,
    )
    ax.tick_params(labelsize=10)
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

# ────────────────────────────────────────────────────────────────
def plot_projected_map(data, outpath, pixel_arcmin=0.25, psf_fwhm_arcmin=0.2466667):
    """
    Build a 2-D projected sky map from all light-cone galaxies,
    summing stellar mass into each pixel, then convolve with a
    Gaussian PSF to produce the 'observed' realisation.

    Left  panel: raw projected SIMBA map (all galaxies, all redshifts)
    Right panel: PSF-convolved 'observed' map

    RA and DEC stored in the HDF5 file are in radians (astropy
    decompose() artefact); multiply by (180/pi * 60) to get arcminutes.
    """
    # ── coordinate conversion: radians → arcminutes ──────────────────────
    ra_am  = data["ra"]  * np.degrees(1.0) * 60.0   # arcmin
    dec_am = data["dec"] * np.degrees(1.0) * 60.0   # arcmin
    mass   = data["stellar_mass"]                    # M_sun

    # Field side in arcminutes derived from stored area attribute
    field_am = np.sqrt(data["area_deg2"]) * 60.0    # arcmin
    n_pix    = int(np.ceil(field_am / pixel_arcmin))
    edges    = np.linspace(0.0, n_pix * pixel_arcmin, n_pix + 1)

    # ── 2-D stellar-mass map ─────────────────────────────────────────────
    map_raw, _, _ = np.histogram2d(
        ra_am, dec_am, bins=[edges, edges], weights=mass,
    )

    # ── Gaussian PSF convolution ─────────────────────────────────────────
    sigma_pix = (psf_fwhm_arcmin / pixel_arcmin) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    map_conv  = gaussian_filter(map_raw, sigma=sigma_pix)

    # Surface brightness: M_sun / arcmin^2
    pix_area = pixel_arcmin ** 2
    sb_raw  = map_raw  / pix_area
    sb_conv = map_conv / pix_area

    # ── colour scale (shared vmax, individual vmins for log scale) ───────
    vmax      = sb_raw.max()
    vmin_raw  = sb_raw[sb_raw   > 0].min() if (sb_raw   > 0).any() else 1.0
    vmin_conv = sb_conv[sb_conv > 0].min() if (sb_conv  > 0).any() else 1.0

    extent = [0.0, n_pix * pixel_arcmin, 0.0, n_pix * pixel_arcmin]

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # -- left: raw projected map --
    im0 = axes[0].imshow(
        sb_raw.T, origin="lower", extent=extent,
        cmap="inferno", norm=LogNorm(vmin=vmin_raw, vmax=vmax),
        aspect="equal",
    )
    axes[0].set_xlabel("RA [arcmin]", fontsize=12)
    axes[0].set_ylabel("DEC [arcmin]", fontsize=12)
    axes[0].set_title(
        "Projected SIMBA map\n"
        r"(all objects, all $z$, summed $M_\star$)",
        fontsize=11,
    )
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(
        r"$\Sigma_\star\;[\mathrm{M}_\odot\,\mathrm{arcmin}^{-2}]$",
        fontsize=10,
    )

    # -- right: PSF-convolved "observed" map --
    im1 = axes[1].imshow(
        sb_conv.T, origin="lower", extent=extent,
        cmap="inferno", norm=LogNorm(vmin=vmin_conv, vmax=vmax),
        aspect="equal",
    )
    axes[1].set_xlabel("RA [arcmin]", fontsize=12)
    axes[1].set_ylabel("DEC [arcmin]", fontsize=12)
    axes[1].set_title(
        f"Observed map  (PSF FWHM = {psf_fwhm_arcmin:.1f} arcmin)\n"
        "Gaussian beam convolution",
        fontsize=11,
    )
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(
        r"$\Sigma_\star\;[\mathrm{M}_\odot\,\mathrm{arcmin}^{-2}]$",
        fontsize=10,
    )

    fig.suptitle(
        rf"SIMBA EBL light-cone  "
        rf"($0.5\,\mathrm{{deg}}^2$, $0 < z < 7$)  —  "
        rf"$N_\mathrm{{gal}} = {len(mass):,}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_lightcone(LC_FILE)

    # 1. N(z) histogram
    plot_ngal_vs_redshift(
        data,
        FIG_DIR / "lightcone_ngal_vs_redshift.png",
    )

    # 2. 2-D light-cone slice (z-coloured + L_FIR-coloured)
    print("Loading L_FIR from snapshot catalogues …")
    lfir = load_lfir(data)
    n_valid = np.isfinite(lfir).sum()
    print(f"  L_FIR available for {n_valid:,} / {len(lfir):,} galaxies")

    plot_lightcone_slice(
        data, lfir,
        FIG_DIR / "lightcone_slice_z_lfir.png",
    )

    # 3. Projected sky map
    plot_projected_map(
        data,
        FIG_DIR / "lightcone_projected_map.png",
        pixel_arcmin=0.25,
        psf_fwhm_arcmin=0.2466667,
    )