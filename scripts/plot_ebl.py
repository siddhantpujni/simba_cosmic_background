"""
Plot the final EBL with jackknife uncertainties and redshift-bin decomposition.

Figure 1 — Full EBL with jackknife 1-sigma envelope + observed data
Figure 2 — EBL decomposed by redshift bins (0-1.9, 2-3.9, 4-7.0)

Usage:
    # Plot from saved data only (no recomputation):
    python plot_ebl_final.py --sim m100n1024 --area 0.5 --z_min 0 --z_max 7

    # Also compute and cache redshift-bin decomposition (slow, run once):
    python plot_ebl_final.py --sim m100n1024 --area 0.5 --z_min 0 --z_max 7 --compute_bins
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SPS_HOME", "/home/spujni/fsps")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import astropy.units as u
from astropy.constants import c as c_light
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background
from src.backgrounds.farIR import lightcone_farIR_background
from src.backgrounds.radio import lightcone_radio_background

# ── Paths ──────────────────────────────────────────────────────────────────
EBL_PATH = Path("/home/spujni/simba_cosmic_background/data/results/"
                "bg_m100n1024_a0.5_z0.0-7.0.h5")
JK_PATH  = Path("/home/spujni/simba_cosmic_background/data/jackknife/"
                "jackknife_m100n1024_a0.5_z0.0-7.0.h5")
OBS_PATH = Path("/home/spujni/simba_cosmic_background/data/ebl/ebldata.csv")
BIN_CACHE = Path("/home/spujni/simba_cosmic_background/data/results/"
                 "ebl_redshift_bins_m100n1024.h5")
FIG_DIR   = Path("figures/ebl_final")

# ── Redshift bins ──────────────────────────────────────────────────────────
ZBINS = [
    (0.0, 1.9, "z = 0.0–1.9", "#4C9BE8"),
    (2.0, 3.9, "z = 2.0–3.9", "#E8834C"),
    (4.0, 7.0, "z = 4.0–7.0", "#6ABF69"),
]

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "legend.fontsize":   10,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth":    1.2,
})

FLOOR = 1e-8   # Lowered significantly so faint radio signals aren't dropped
CGS_TO_NW = 1e6

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


# ══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════

def load_ebl(path: Path) -> dict:
    """Load full EBL from HDF5."""
    with h5py.File(path, "r") as f:
        opt_lam  = f["optical/lam_AA"][:]  * 1e-4        # → µm
        opt_nW   = f["optical/nuInu_nW"][:]
        fir_lam  = f["farIR/lam_AA"][:]   * 1e-4
        fir_nW   = f["farIR/nuInu_nW"][:]
        rad_lam  = f["radio/lam_um"][:]
        rad_nW   = f["radio/nuInu_nW"][:]
    return dict(opt_lam=opt_lam, opt_nW=opt_nW,
                fir_lam=fir_lam, fir_nW=fir_nW,
                rad_lam=rad_lam, rad_nW=rad_nW)


def load_jackknife(path: Path) -> dict:
    """Load jackknife mean and std from HDF5."""
    with h5py.File(path, "r") as f:
        data = {}
        for key in ("optical", "farIR", "radio"):
            data[key] = {
                "lam_um": f[f"{key}/lam_um"][:],
                "mean":   f[f"{key}/mean"][:],
                "std":    f[f"{key}/std"][:],
            }
    return data


def load_observed(path: Path) -> pd.DataFrame:
    """
    Load observed EBL data from ebldata.csv.
    Expected columns: wave (µm), ebl (nW m⁻² sr⁻¹), debl (uncertainty).
    An 'instrument' column is used for grouping if present, otherwise all
    points are labelled 'Observed'.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={"wave": "lam_um", "ebl": "nuInu_nW", "debl": "err_nW"})

    if "instrument" not in df.columns:
        df["instrument"] = "Observed"

    return df


def compute_redshift_bins(cfg, args, a_dust=-0.017341) -> dict:
    """
    Compute optical + far-IR + radio backgrounds for each redshift bin.
    Returns dict keyed by bin label.
    """
    results = {}
    for z_lo, z_hi, label, _ in ZBINS:
        print(f"\n── Computing bin {label} ──")
        # Optical
        lam_o, I_nu_o, _ = lightcone_optical_background(
            cfg, area_deg2=args.area, z_min=z_lo, z_max=z_hi)
        valid_o = np.isfinite(lam_o) & (lam_o > 0)
        lam_o, I_nu_o = lam_o[valid_o], I_nu_o[valid_o]
        nu_o = (c_light / (lam_o * u.AA)).to_value(u.Hz)
        nuInu_o = nu_o * I_nu_o * CGS_TO_NW

        # Far-IR
        lam_f, I_lam_f, _, _ = lightcone_farIR_background(
            cfg, area_deg2=args.area, z_min=z_lo, z_max=z_hi,
            a_dust=a_dust, return_dust_temps=True)
        nuInu_f = lam_f * I_lam_f * CGS_TO_NW

        # Radio
        nu_r, I_nu_r, _, _ = lightcone_radio_background(
            cfg, area_deg2=args.area, z_min=z_lo, z_max=z_hi)
        nuInu_r = nu_r * I_nu_r * CGS_TO_NW
        lam_r = (c_light / (nu_r * u.Hz)).to_value(u.AA) * 1e-4

        results[label] = dict(
            opt_lam=lam_o * 1e-4, opt_nW=nuInu_o,
            fir_lam=lam_f,        fir_nW=nuInu_f,
            rad_lam=lam_r,        rad_nW=nuInu_r,
        )

    # Cache to HDF5
    BIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(BIN_CACHE, "w") as f:
        for label, d in results.items():
            grp = f.create_group(label.replace(" ", "_"))
            for k, v in d.items():
                grp.create_dataset(k, data=v)
    print(f"\nRedshift-bin data cached → {BIN_CACHE}")
    return results


def load_bin_cache(path: Path) -> dict:
    results = {}
    with h5py.File(path, "r") as f:
        for grp_name in f:
            label = grp_name.replace("_", " ")
            results[label] = {k: f[grp_name][k][:] for k in f[grp_name]}
    return results


# ══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════

def _mask(lam, val, std=None, floor=FLOOR):
    """
    Return arrays with invalid entries removed.
    Drops NaN/non-finite wavelengths and sub-floor values so matplotlib
    never draws connecting lines across bad data points.
    std is filtered with the same valid mask if provided.
    """
    l = np.asarray(lam, dtype=float)
    v = np.asarray(val, dtype=float)
    valid = np.isfinite(l) & (l > 0) & np.isfinite(v) & (v >= floor)
    if std is not None:
        s = np.asarray(std, dtype=float)
        return l[valid], v[valid], s[valid]
    return l[valid], v[valid]


def _plot_component(ax, lam, val, std=None, color="steelblue",
                    label="", ls="-", alpha_fill=0.25, zorder=2):
    if std is not None:
        lam, v, s = _mask(lam, val, std=std)
        
        # Sort by wavelength to prevent lines drawing backwards
        idx = np.argsort(lam)
        lam, v, s = lam[idx], v[idx], s[idx]
        
        # Use np.maximum to ensure the lower bound doesn't drop below the log floor
        lo = np.maximum(v - (s), FLOOR)
        hi = v + (s)
        ax.fill_between(lam, lo, hi, color=color,
                        alpha=alpha_fill, lw=0, zorder=zorder - 1)
    else:
        lam, v = _mask(lam, val)
        
        idx = np.argsort(lam)
        lam, v = lam[idx], v[idx]
        
    ax.plot(lam, v, color=color, lw=1.8, ls=ls, label=label, zorder=zorder)

def _obs_scatter(ax, df):
    """Scatter observed EBL measurements, grouped by instrument."""
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    instruments = df["instrument"].unique()
    cmap = plt.colormaps["tab10"].resampled(len(instruments))
    for i, inst in enumerate(instruments):
        sub = df[df["instrument"] == inst].dropna(subset=["lam_um", "nuInu_nW"])
        has_err = sub["err_nW"].notna().any()
        ax.errorbar(
            sub["lam_um"], sub["nuInu_nW"],
            yerr=sub["err_nW"] if has_err else None,
            fmt=markers[i % len(markers)], color=cmap(i),
            ms=5, lw=1.0, capsize=2, label=inst,
            zorder=5, alpha=0.85,
        )


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Full EBL with jackknife errors
# ══════════════════════════════════════════════════════════════════════════

def plot_full_ebl(ebl: dict, jk: dict, obs: pd.DataFrame,
                 save_dir: Path) -> None:

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ── Three components plotted independently — no interpolation ──
    # Use jackknife lam_um + mean directly.
    _plot_component(ax,
                    jk["optical"]["lam_um"], jk["optical"]["mean"],
                    std=jk["optical"]["std"],
                    color="#4C9BE8", label="Optical / NIR")

    _plot_component(ax,
                    jk["farIR"]["lam_um"], jk["farIR"]["mean"],
                    std=jk["farIR"]["std"],
                    color="#E85C4C", label="Far-IR (dust MBB)")

    _plot_component(ax,
                    jk["radio"]["lam_um"], jk["radio"]["mean"],
                    std=jk["radio"]["std"],
                    color="#6ABF69", label="Radio (SF + AGN)")

    # ── Observed data ─────────────────────────────────────────────
    _obs_scatter(ax, obs)

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_ylim(1e-7, 1e3) 
    
    ax.set_xlabel(r"$\lambda_{\rm obs}\ [\mu\mathrm{m}]$")
    ax.set_ylabel(r"$\nu I_\nu\ [\mathrm{nW\,m^{-2}\,sr^{-1}}]$")
    ax.set_title("SIMBA extragalactic background light  ($z = 0$–$7$)",
                 pad=8)

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8",
              ncol=2, fontsize=9)
    ax.grid(True, which="major", ls=":", alpha=0.35, color="0.6")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "ebl_full_jackknife.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — EBL decomposed by redshift bins
# ══════════════════════════════════════════════════════════════════════════

def plot_redshift_binned_ebl(jk_paths, obs, save_dir):
    """
    Plot EBL for three jackknife bins (0–1, 1–3, 3–7) in a 3-row, 1-column layout.
    Each panel shows optical, far-IR, and radio with jackknife errors, plus observed data.
    """
    bin_labels = [
        "z = 0.0–1.0",
        "z = 1.0–3.0",
        "z = 3.0–7.0"
    ]
    bin_colors = [
        ("#4C9BE8", "#E85C4C", "#6ABF69"),  # optical, farIR, radio
        ("#4C9BE8", "#E85C4C", "#6ABF69"),
        ("#4C9BE8", "#E85C4C", "#6ABF69"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    fig.suptitle("SIMBA extragalactic background light by redshift bin", fontsize=14, y=0.99)

    for i, (jk_path, label, colors) in enumerate(zip(jk_paths, bin_labels, bin_colors)):
        jk = load_jackknife(jk_path)
        ax = axes[i]
        # Optical
        _plot_component(ax,
                        jk["optical"]["lam_um"], jk["optical"]["mean"],
                        std=jk["optical"]["std"],
                        color=colors[0], label="Optical / NIR")
        # Far-IR
        _plot_component(ax,
                        jk["farIR"]["lam_um"], jk["farIR"]["mean"],
                        std=jk["farIR"]["std"],
                        color=colors[1], label="Far-IR (dust MBB)")
        # Radio
        _plot_component(ax,
                        jk["radio"]["lam_um"], jk["radio"]["mean"],
                        std=jk["radio"]["std"],
                        color=colors[2], label="Radio (SF + AGN)")
        # Observed
        _obs_scatter(ax, obs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e3)
        ax.set_ylabel(r"$\nu I_\nu\ [\mathrm{nW\,m^{-2}\,sr^{-1}}]$")
        ax.set_title(label, pad=8)
        ax.grid(True, which="major", ls=":", alpha=0.35, color="0.6")
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8", ncol=2, fontsize=9)
    axes[-1].set_xlabel(r"$\lambda_{\rm obs}\ [\mu\mathrm{m}]$")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "ebl_jackknife_bins.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",   default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area",  type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--compute_bins", action="store_true",
                        help="Recompute redshift-bin decomposition "
                             "(slow — run once, then cached)")
    parser.add_argument("--a_dust", type=float, default=-0.017341)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading EBL results …")
    ebl = load_ebl(EBL_PATH)

    print("Loading jackknife results …")
    jk = load_jackknife(JK_PATH)

    print("Loading observed EBL data …")
    obs = load_observed(OBS_PATH)

    # ── Figure 1 ──────────────────────────────────────────────────
    print("\n── Figure 1: Full EBL with jackknife uncertainty ──")
    plot_full_ebl(ebl, jk, obs, FIG_DIR)

    # ── Figure 2 ──────────────────────────────────────────────────
    print("\n── Figure 2: EBL by redshift bin ──")
    if args.compute_bins:
        cfg = load_config(args.sim)
        bins = compute_redshift_bins(cfg, args, a_dust=args.a_dust)
    elif BIN_CACHE.exists():
        print(f"Loading cached bin data from {BIN_CACHE} …")
        bins = load_bin_cache(BIN_CACHE)
    else:
        print("No cached bin data found. Run with --compute_bins to generate it.")
        return

    jk_bin_paths = [
        Path("data/jackknife/jackknife_m100n1024_a0.5_z0.0-1.0.h5"),
        Path("data/jackknife/jackknife_m100n1024_a0.5_z1.0-3.0.h5"),
        Path("data/jackknife/jackknife_m100n1024_a0.5_z3.0-7.0.h5"),
    ]
    plot_redshift_binned_ebl(jk_bin_paths, obs, FIG_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()