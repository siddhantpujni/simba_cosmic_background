"""
Visualise the SIMBA reconstructed light-cone.

Produces two figures:
  1. Radial (comoving) vs transverse (comoving) distance slice.
  2. Sky-plane map with blended line-of-sight sources marked.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from pathlib import Path
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.ndimage import gaussian_filter

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LC_FILE = ROOT / "data" / "lightcones" / "lc_m100n1024_a0.5_z0.0-7.0.h5"
FIG_DIR = ROOT / "figures" / "lightcone"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_lightcone(path):
    """Load lightcone data from HDF5."""
    with h5py.File(path, "r") as f:
        data = {
            "ra": f["RA"][:],
            "dec": f["DEC"][:],
            "z": f["z"][:],
            "stellar_mass": f["stellar_mass"][:],
            "area_deg2": f.attrs["area_deg2"],
        }
    return data


def plot_lightcone_slice(data, outpath):
    """
    2x2 radial-vs-transverse number-density maps in redshift bins:
    z = [0,1), [1,2), [2,3), [4,7].

    This highlights the rise in galaxy counts up to z~2 and decline at higher z.
    """
    z = data["z"]
    ra = data["ra"]  # degrees

    # Comoving coordinates
    d_radial = cosmo.comoving_distance(z).to("Mpc").value
    ra_centre = np.median(ra)
    delta_ra_rad = np.deg2rad(ra - ra_centre)
    d_transverse = delta_ra_rad * d_radial

    # Requested redshift bins
    z_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (4.0, 7.0)]

    # Shared plotting grid so panels are directly comparable
    x_max = np.nanmax(d_radial)
    y_lim = np.nanpercentile(np.abs(d_transverse), 99.5)
    x_edges = np.linspace(0.0, x_max, 220)
    y_edges = np.linspace(-y_lim, y_lim, 220)

    # Precompute densities to share one color scale
    densities = []
    counts = []
    vmax = 0.0
    vmin_pos = np.inf

    for z0, z1 in z_bins:
        m = (z >= z0) & (z < z1)
        counts.append(int(m.sum()))

        H, _, _ = np.histogram2d(d_radial[m], d_transverse[m], bins=[x_edges, y_edges])

        # Number density in Mpc^-2 (count per pixel area in comoving coords)
        dx = np.diff(x_edges)[:, None]
        dy = np.diff(y_edges)[None, :]
        density = H / (dx * dy)

        densities.append(density)

        if np.any(density > 0):
            vmax = max(vmax, float(np.nanmax(density)))
            vmin_pos = min(vmin_pos, float(np.nanmin(density[density > 0])))

    if not np.isfinite(vmin_pos):
        vmin_pos = 1e-6
    if vmax <= vmin_pos:
        vmax = vmin_pos * 10.0

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (z0, z1), density, n_gal in zip(axes, z_bins, densities, counts):
        if np.any(density > 0):
            im = ax.pcolormesh(
                x_edges, y_edges, density.T,
                cmap="magma", norm=LogNorm(vmin=vmin_pos, vmax=vmax), shading="auto"
            )
        else:
            im = None
            ax.text(0.5, 0.5, "No galaxies", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(f"z = {z0:.0f}-{z1:.0f}   (N = {n_gal:,})", fontsize=11)
        ax.tick_params(labelsize=9)

    # Axis labels
    axes[2].set_xlabel(r"Comoving radial distance [Mpc]", fontsize=11)
    axes[3].set_xlabel(r"Comoving radial distance [Mpc]", fontsize=11)
    axes[0].set_ylabel(r"Comoving transverse distance [Mpc]", fontsize=11)
    axes[2].set_ylabel(r"Comoving transverse distance [Mpc]", fontsize=11)

    # Shared colorbar
    if 'im' in locals() and im is not None:
        cbar = fig.colorbar(im, ax=axes, pad=0.01, aspect=40)
        cbar.set_label(r"Galaxy number density [Mpc$^{-2}$]", fontsize=11)

    fig.suptitle(
        r"SIMBA light-cone number density in radial-transverse slices by redshift",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
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

    plot_lightcone_slice(
        data,
        FIG_DIR / "lightcone_radial_transverse_density_zbins.png",
    )

    # plot_sky_plane(...)  # delete this entire call

    plot_projected_map(
        data,
        FIG_DIR / "lightcone_projected_map.png",
        pixel_arcmin=0.25,
        psf_fwhm_arcmin=0.2466667,
    )