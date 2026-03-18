"""
Jackknife error estimation for the SIMBA light cone.

Splits the field into n_side x n_side spatial sub-regions (default 4x4 = 16),
then uses delete-one jackknife resampling to estimate uncertainties on:
  1. N(z) — galaxy counts per redshift bin
  2. Projected stellar-mass surface-brightness map
  3. EBL/SED background calculations (optical, far-IR, radio)
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.constants import c as c_light


# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LC_FILE = ROOT / "data" / "lightcones" / "lc_m100n1024_a0.5_z0.0-7.0.h5"
FIG_DIR = ROOT / "figures" / "lightcone"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BG_FIG_DIR = ROOT / "figures" / "combined" / "jackknife"
BG_FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SIDE = 4  # 4x4 = 16 jackknife sub-regions


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

def assign_subregions(ra, dec, n_side):
    """
    Assign each galaxy to one of n_side x n_side spatial sub-regions
    based on RA/DEC quantile bins (equal-area tiling).

    Returns an integer label array of shape (N_gal,) with values in
    [0, n_side^2).
    """
    ra_edges = np.quantile(ra, np.linspace(0, 1, n_side + 1))
    dec_edges = np.quantile(dec, np.linspace(0, 1, n_side + 1))

    # digitize returns 1-based bin indices; clip to [0, n_side-1]
    i_ra = np.clip(np.digitize(ra, ra_edges[1:-1]), 0, n_side - 1)
    i_dec = np.clip(np.digitize(dec, dec_edges[1:-1]), 0, n_side - 1)

    return i_ra * n_side + i_dec

def compute_backgrounds_masked(cfg, area_deg2, z_min, z_max, a_dust, galaxy_mask=None, verbose=True):
    """Compute optical, far-IR, and radio backgrounds using the same flow as run_combined.compute_backgrounds."""
    if verbose:
        print("=== Optical/NIR background ===")
    lam_opt, I_nu_opt, I_nu_opt_nodust = lightcone_optical_background(
        cfg, area_deg2=area_deg2, z_min=z_min, z_max=z_max, galaxy_mask=galaxy_mask
    )
    nu_opt = (c_light / (lam_opt * u.AA)).to_value(u.Hz)
    nuInu_opt = nu_opt * I_nu_opt
    nuInu_opt_nodust = nu_opt * I_nu_opt_nodust

    if verbose:
        print("\n=== Far-IR background ===")
    lam_fir, I_lam_fir = lightcone_farIR_background(
        cfg, area_deg2=area_deg2, z_min=z_min, z_max=z_max,
        a_dust=a_dust, galaxy_mask=galaxy_mask
    )
    nuInu_fir = lam_fir * I_lam_fir

    if verbose:
        print("\n=== Radio background (SF + AGN) ===")
    nu_radio, I_nu_radio, _, _ = lightcone_radio_background(
        cfg, area_deg2=area_deg2, z_min=z_min, z_max=z_max, galaxy_mask=galaxy_mask
    )
    lam_radio_um = (c_light / (nu_radio * u.Hz)).to_value(u.AA) * 1e-4
    nuInu_radio = nu_radio * I_nu_radio

    cgs_to_nWm2 = 1e6
    return {
        "optical": {
            "lam_um": lam_opt * 1e-4,
            "nuInu_nW": nuInu_opt * cgs_to_nWm2,
            "nuInu_nodust_nW": nuInu_opt_nodust * cgs_to_nWm2,
        },
        "farIR": {
            "lam_um": lam_fir * 1e-4,
            "nuInu_nW": nuInu_fir * cgs_to_nWm2,
        },
        "radio": {
            "lam_um": lam_radio_um,
            "nuInu_nW": nuInu_radio * cgs_to_nWm2,
        },
    }

# ─────────────────────────────────────────────────────────────────────────
# EBL / Background Jackknife Functions
# Uses existing background functions from src/backgrounds/ with galaxy_mask
# ─────────────────────────────────────────────────────────────────────────

from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background
from src.backgrounds.farIR import lightcone_farIR_background, build_lightcone
from src.backgrounds.radio import lightcone_radio_background


def load_lightcone_labels(lc_path, n_side=N_SIDE):
    """
    Load lightcone and assign jackknife sub-region labels based on RA/DEC.

    Returns
    -------
    labels : array of sub-region labels (0 to n_side^2 - 1)
    n_gal  : number of galaxies in lightcone
    """
    with h5py.File(lc_path, "r") as f:
        ra = f["RA"][:]
        dec = f["DEC"][:]
    labels = assign_subregions(ra, dec, n_side)
    return labels, len(ra)


def jackknife_ebl(cfg, area_deg2=0.5, z_min=0.0, z_max=7.0,
                  n_side=N_SIDE, a_dust=-0.017341, verbose=True):
    """
    Compute combined EBL (optical + far-IR + radio) with jackknife errors.

    Uses the existing lightcone_*_background functions with the new
    galaxy_mask parameter to compute leave-one-out jackknife realizations.

    Parameters
    ----------
    cfg : SimConfig
    area_deg2 : float
    z_min, z_max : float
    n_side : int
        Number of sub-regions per side (n_side^2 total quadrants)
    a_dust : float
        Dust temperature parameter for far-IR

    Returns
    -------
    results : dict with keys:
        'optical'  : {lam_um, nuInu_nW, nuInu_err_nW, jackknife}
        'farIR'    : {lam_um, nuInu_nW, nuInu_err_nW, jackknife}
        'radio'    : {lam_um, nuInu_nW, nuInu_err_nW, jackknife}
        'metadata' : simulation parameters
    """
    n_sub = n_side ** 2

    # Build/load lightcone and get sub-region labels
    lc_path = build_lightcone(cfg, area_deg2, z_min, z_max)
    labels, n_gal = load_lightcone_labels(lc_path, n_side)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Jackknife EBL: {n_side}x{n_side} = {n_sub} sub-regions")
        print(f"Lightcone: {n_gal:,} galaxies, {area_deg2} deg²")
        print(f"{'='*60}\n")

    results = {"metadata": {
        "sim": cfg.name,
        "area_deg2": area_deg2,
        "z_min": z_min,
        "z_max": z_max,
        "n_side": n_side,
        "n_jackknife": n_sub,
    }}

    # Full sample
    full = compute_backgrounds_masked(
        cfg, area_deg2, z_min, z_max, a_dust, galaxy_mask=None, verbose=verbose
    )

    # Prepare jackknife arrays
    lam_fir = full["farIR"]["lam_um"]
    lam_radio = full["radio"]["lam_um"]
    lam_opt = full["optical"]["lam_um"]

    nuInu_fir_jack = np.zeros((n_sub, len(lam_fir)))
    nuInu_radio_jack = np.zeros((n_sub, len(lam_radio)))
    nuInu_opt_jack = np.zeros((n_sub, len(lam_opt)))
    nuInu_opt_nodust_jack = np.zeros((n_sub, len(lam_opt)))

    # Jackknife realizations
    for k in range(n_sub):
        if verbose:
            print(f"  Jackknife {k+1}/{n_sub} ...")
        jk_mask = labels != k
        jk = compute_backgrounds_masked(
            cfg, area_deg2, z_min, z_max, a_dust, galaxy_mask=jk_mask, verbose=False
        )
        nuInu_fir_jack[k] = jk["farIR"]["nuInu_nW"]
        nuInu_radio_jack[k] = jk["radio"]["nuInu_nW"]
        nuInu_opt_jack[k] = jk["optical"]["nuInu_nW"]
        nuInu_opt_nodust_jack[k] = jk["optical"]["nuInu_nodust_nW"]

    # Jackknife errors
    nuInu_fir_err = np.sqrt(
        (n_sub - 1) / n_sub * np.sum((nuInu_fir_jack - nuInu_fir_jack.mean(axis=0)) ** 2, axis=0)
    )
    nuInu_radio_err = np.sqrt(
        (n_sub - 1) / n_sub * np.sum((nuInu_radio_jack - nuInu_radio_jack.mean(axis=0)) ** 2, axis=0)
    )
    nuInu_opt_err = np.sqrt(
        (n_sub - 1) / n_sub * np.sum((nuInu_opt_jack - nuInu_opt_jack.mean(axis=0)) ** 2, axis=0)
    )
    nuInu_opt_nodust_err = np.sqrt(
        (n_sub - 1) / n_sub * np.sum((nuInu_opt_nodust_jack - nuInu_opt_nodust_jack.mean(axis=0)) ** 2, axis=0)
    )

    results["farIR"] = {
        "lam_um": lam_fir,
        "nuInu_nW": full["farIR"]["nuInu_nW"],
        "nuInu_err_nW": nuInu_fir_err,
        "jackknife": nuInu_fir_jack,
    }

    results["radio"] = {
        "lam_um": lam_radio,
        "nuInu_nW": full["radio"]["nuInu_nW"],
        "nuInu_err_nW": nuInu_radio_err,
        "jackknife": nuInu_radio_jack,
    }

    results["optical"] = {
        "lam_um": lam_opt,
        "nuInu_nW": full["optical"]["nuInu_nW"],
        "nuInu_err_nW": nuInu_opt_err,
        "nuInu_nodust_nW": full["optical"]["nuInu_nodust_nW"],
        "nuInu_nodust_err_nW": nuInu_opt_nodust_err,
        "jackknife": nuInu_opt_jack,
        "jackknife_nodust": nuInu_opt_nodust_jack,
    }

    return results


def plot_jackknife_ebl(results, outpath=None, show_realizations=False):
    """
    Plot combined EBL with jackknife error bands.

    Parameters
    ----------
    results : dict from jackknife_ebl
    outpath : Path or str, optional
    show_realizations : bool
        If True, also plot individual jackknife realizations
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    floor = 1e-6

    # Optical
    if results.get("optical") is not None:
        opt = results["optical"]
        lam = opt["lam_um"]
        nuInu = opt["nuInu_nW"]
        err = opt["nuInu_err_nW"]

        valid = nuInu > floor
        ax.loglog(lam[valid], nuInu[valid], 'o-', color='steelblue',
                  lw=2, ms=6, label='Optical/NIR')
        ax.fill_between(lam[valid],
                        np.maximum(nuInu[valid] - err[valid], floor),
                        nuInu[valid] + err[valid],
                        color='steelblue', alpha=0.3)

        if show_realizations:
            for jk in opt["jackknife"]:
                ax.loglog(lam[valid], np.maximum(jk[valid], floor),
                          color='steelblue', alpha=0.1, lw=0.5)

    # Far-IR
    fir = results["farIR"]
    lam = fir["lam_um"]
    nuInu = fir["nuInu_nW"]
    err = fir["nuInu_err_nW"]

    valid = nuInu > floor
    ax.loglog(lam[valid], nuInu[valid], '-', color='firebrick',
              lw=2, label='Far-IR (dust)')
    ax.fill_between(lam[valid],
                    np.maximum(nuInu[valid] - err[valid], floor),
                    nuInu[valid] + err[valid],
                    color='firebrick', alpha=0.3)

    if show_realizations:
        for jk in fir["jackknife"]:
            ax.loglog(lam[valid], np.maximum(jk[valid], floor),
                      color='firebrick', alpha=0.1, lw=0.5)

    # Radio
    rad = results["radio"]
    lam = rad["lam_um"]
    nuInu = rad["nuInu_nW"]
    err = rad["nuInu_err_nW"]

    valid = nuInu > floor
    ax.loglog(lam[valid], nuInu[valid], '-', color='forestgreen',
              lw=2, label='Radio (SF+AGN)')
    ax.fill_between(lam[valid],
                    np.maximum(nuInu[valid] - err[valid], floor),
                    nuInu[valid] + err[valid],
                    color='forestgreen', alpha=0.3)

    if show_realizations:
        for jk in rad["jackknife"]:
            ax.loglog(lam[valid], np.maximum(jk[valid], floor),
                      color='forestgreen', alpha=0.1, lw=0.5)

    meta = results["metadata"]
    ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=13)
    ax.set_ylabel(r'$\nu\, I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
    ax.set_title(
        f"Cosmic background — {meta['sim']} "
        f"($z = {meta['z_min']}$–${meta['z_max']}$)\n"
        f"Jackknife $1\\sigma$ ({meta['n_side']}$\\times${meta['n_side']} sub-regions)",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_xlim(0.1, 3e7)
    ax.set_ylim(1e-4, 1e3)

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved: {outpath}")

    return fig, ax


def save_jackknife_results(results, outpath):
    """Save jackknife EBL results to HDF5."""
    from datetime import datetime

    with h5py.File(outpath, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        for k, v in results["metadata"].items():
            meta.attrs[k] = v
        meta.attrs["created"] = datetime.now().isoformat()

        # Far-IR
        fir = f.create_group("farIR")
        fir.create_dataset("lam_um", data=results["farIR"]["lam_um"])
        fir.create_dataset("nuInu_nW", data=results["farIR"]["nuInu_nW"])
        fir.create_dataset("nuInu_err_nW", data=results["farIR"]["nuInu_err_nW"])
        fir.create_dataset("jackknife", data=results["farIR"]["jackknife"])

        # Radio
        rad = f.create_group("radio")
        rad.create_dataset("lam_um", data=results["radio"]["lam_um"])
        rad.create_dataset("nuInu_nW", data=results["radio"]["nuInu_nW"])
        rad.create_dataset("nuInu_err_nW", data=results["radio"]["nuInu_err_nW"])
        rad.create_dataset("jackknife", data=results["radio"]["jackknife"])

        # Optical (if available)
        if results.get("optical") is not None:
            opt = f.create_group("optical")
            opt.create_dataset("lam_um", data=results["optical"]["lam_um"])
            opt.create_dataset("nuInu_nW", data=results["optical"]["nuInu_nW"])
            opt.create_dataset("nuInu_err_nW", data=results["optical"]["nuInu_err_nW"])
            opt.create_dataset("jackknife", data=results["optical"]["jackknife"])
            opt.create_dataset("nuInu_nodust_nW", data=results["optical"]["nuInu_nodust_nW"])
            opt.create_dataset("nuInu_nodust_err_nW", data=results["optical"]["nuInu_nodust_err_nW"])
            opt.create_dataset("jackknife_nodust", data=results["optical"]["jackknife_nodust"])

    print(f"Results saved → {outpath}")
    return outpath


def load_jackknife_results(path):
    """Load jackknife EBL results from HDF5."""
    results = {}

    with h5py.File(path, "r") as f:
        results["metadata"] = dict(f["metadata"].attrs)

        results["farIR"] = {
            "lam_um": f["farIR/lam_um"][:],
            "nuInu_nW": f["farIR/nuInu_nW"][:],
            "nuInu_err_nW": f["farIR/nuInu_err_nW"][:],
            "jackknife": f["farIR/jackknife"][:],
        }

        results["radio"] = {
            "lam_um": f["radio/lam_um"][:],
            "nuInu_nW": f["radio/nuInu_nW"][:],
            "nuInu_err_nW": f["radio/nuInu_err_nW"][:],
            "jackknife": f["radio/jackknife"][:],
        }

        results["optical"] = {
            "lam_um": f["optical/lam_um"][:],
            "nuInu_nW": f["optical/nuInu_nW"][:],
            "nuInu_err_nW": f["optical/nuInu_err_nW"][:],
            "nuInu_nodust_nW": f["optical/nuInu_nodust_nW"][:],
            "nuInu_nodust_err_nW": f["optical/nuInu_nodust_err_nW"][:],
            "jackknife": f["optical/jackknife"][:],
            "jackknife_nodust": f["optical/jackknife_nodust"][:],
        }

    print(f"Loaded jackknife results from {path}")
    return results


def run_jackknife_ebl(sim="m100n1024", area=0.5, z_min=0.0, z_max=7.0, n_side=4):
    """
    Run the full jackknife EBL calculation.

    Example
    -------
    >> results = run_jackknife_ebl("m100n1024", area=0.5)
    >> plot_jackknife_ebl(results, "figures/combined/jackknife/ebl_m100n1024.png")
    """
    cfg = load_config(sim)
    print(f"Running jackknife EBL on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)\n")

    results = jackknife_ebl(
        cfg, area_deg2=area, z_min=z_min, z_max=z_max, n_side=n_side
    )

    # Save results
    outpath = ROOT / "data" / "results" / f"bg_jackknife_{sim}_a{area}_z{z_min}-{z_max}.h5"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_jackknife_results(results, outpath)

    # Plot
    figpath = BG_FIG_DIR / f"ebl_jackknife_{sim}.png"
    plot_jackknife_ebl(results, figpath, show_realizations=True)

    return results


def print_jackknife_summary(results):
    """Print summary statistics from jackknife results."""
    meta = results["metadata"]
    n_jk = meta["n_jackknife"]

    print(f"\nJackknife EBL Summary ({n_jk} sub-regions)")
    print("=" * 50)

    # Far-IR peak
    fir = results["farIR"]
    peak_idx = np.argmax(fir["nuInu_nW"])
    peak_lam = fir["lam_um"][peak_idx]
    peak_val = fir["nuInu_nW"][peak_idx]
    peak_err = fir["nuInu_err_nW"][peak_idx]
    frac_err = peak_err / peak_val * 100 if peak_val > 0 else 0
    print(f"Far-IR peak at {peak_lam:.1f} µm: {peak_val:.2e} ± {peak_err:.2e} nW/m²/sr ({frac_err:.1f}%)")

    rad = results["radio"]
    lam_1p4ghz = (c_light / (1.4e9 * u.Hz)).to_value(u.um)  # ~21 cm
    idx_1p4 = np.argmin(np.abs(rad["lam_um"] - lam_1p4ghz))
    val_1p4 = rad["nuInu_nW"][idx_1p4]
    err_1p4 = rad["nuInu_err_nW"][idx_1p4]
    frac_err = err_1p4 / val_1p4 * 100 if val_1p4 > 0 else 0
    print(f"Radio at 1.4 GHz: {val_1p4:.2e} ± {err_1p4:.2e} nW/m²/sr ({frac_err:.1f}%)")

    if results.get("optical") is not None:
        opt = results["optical"]
        peak_idx = np.argmax(opt["nuInu_nW"])
        peak_lam = opt["lam_um"][peak_idx]
        peak_val = opt["nuInu_nW"][peak_idx]
        peak_err = opt["nuInu_err_nW"][peak_idx]
        frac_err = peak_err / peak_val * 100 if peak_val > 0 else 0
        print(f"Optical peak at {peak_lam:.2f} µm: {peak_val:.2e} ± {peak_err:.2e} nW/m²/sr ({frac_err:.1f}%)")

    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Jackknife error estimation for SIMBA EBL"
    )
    parser.add_argument("--sim", default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=0.5,
                        help="Lightcone area in deg²")
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--n_side", type=int, default=4,
                        help="Number of sub-regions per side (default 4 = 16 quadrants)")
    parser.add_argument("--show_realizations", action="store_true",
                        help="Plot individual jackknife realizations")
    args = parser.parse_args()

    print("=" * 60)
    print("JACKKNIFE EBL CALCULATION")
    print("=" * 60)

    results = run_jackknife_ebl(
        sim=args.sim, area=args.area,
        z_min=args.z_min, z_max=args.z_max,
        n_side=args.n_side
    )
    print_jackknife_summary(results)
