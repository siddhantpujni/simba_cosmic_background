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
        opt_nodust_nW = f["optical/nuInu_nodust_nW"][:]
        fir_lam  = f["farIR/lam_AA"][:]   * 1e-4
        fir_nW   = f["farIR/nuInu_nW"][:]
        rad_lam  = f["radio/lam_um"][:]
        rad_nW   = f["radio/nuInu_nW"][:]

    return dict(opt_lam=opt_lam, opt_nW=opt_nW,
                fir_lam=fir_lam, fir_nW=fir_nW,
                rad_lam=rad_lam, rad_nW=rad_nW,
                opt_nodust_nW=opt_nodust_nW)

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
            if key == "optical":
                data[key]["mean_nodust"] = f[f"{key}/mean_nodust"][:]
                data[key]["std_nodust"] = f[f"{key}/std_nodust"][:]
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
            fmt=markers[i % len(markers)], color='k', alpha = 0.7, ecolor='k',
            ms=5, lw=1.0, capsize=2, label='Observational EBL Data',
            zorder=5,
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
    
    # Optical (no dust)
    _plot_component(ax, jk["optical"]["lam_um"], jk["optical"]["mean_nodust"],
                        std=jk["optical"]["std_nodust"],
                        color="#024588", label="Optical (no dust)")

    # ── Observed data ─────────────────────────────────────────────
    _obs_scatter(ax, obs)

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_ylim(1e-7, 1e3) 
    
    ax.set_xlabel(r"$\lambda_{\rm obs}\ [\mu\mathrm{m}]$")
    ax.set_ylabel(r"$\nu I_\nu\ [\mathrm{nW\,m^{-2}\,sr^{-1}}]$")

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8",
              ncol=2, fontsize=9)
    ax.grid(True, which="major", ls=":", alpha=0.35, color="0.6")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "ebl_full_jackknife.pdf"
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
        ("#4C9BE8", "#024588", "#E85C4C", "#6ABF69"),  # optical, optical no dust, farIR, radio
        ("#4C9BE8", "#024588", "#E85C4C", "#6ABF69"),
        ("#4C9BE8", "#024588", "#E85C4C", "#6ABF69"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    #fig.suptitle("SIMBA Extragalactic Background Light by Redshift Bin", fontsize=14, y=0.99)

    for i, (jk_path, label, colors) in enumerate(zip(jk_paths, bin_labels, bin_colors)):
        jk = load_jackknife(jk_path)
        ax = axes[i]
        # Optical
        _plot_component(ax,
                        jk["optical"]["lam_um"], jk["optical"]["mean"],
                        std=jk["optical"]["std"],
                        color=colors[0], label="Optical / NIR")
        # Optical (no dust)
        _plot_component(ax,
                        jk["optical"]["lam_um"], jk["optical"]["mean_nodust"],
                        std=jk["optical"]["std_nodust"],
                        color=colors[1], label="Optical (no dust)")
        # Far-IR
        _plot_component(ax,
                        jk["farIR"]["lam_um"], jk["farIR"]["mean"],
                        std=jk["farIR"]["std"],
                        color=colors[2], label="Far-IR (dust MBB)")
        # Radio
        _plot_component(ax,
                        jk["radio"]["lam_um"], jk["radio"]["mean"],
                        std=jk["radio"]["std"],
                        color=colors[3], label="Radio (SF + AGN)")
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
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — EBL decomposed by redshift bins (Single Axis)
# ══════════════════════════════════════════════════════════════════════════

def plot_redshift_binned_single_ax(jk_paths: list, obs: pd.DataFrame, save_dir: Path) -> None:
    """
    Plot EBL for three jackknife bins on a single graph.
    Each redshift bin is assigned a unique colour applied to its optical, far-IR, and radio curves.
    """
    bin_labels = [
        "z = 0.0–1.0",
        "z = 1.0–3.0",
        "z = 3.0–7.0"
    ]
    
    # Use distinct, colourblind-safe colours for each redshift bin
    bin_colours = ["#4C9BE8", "#E8834C", "#6ABF69"]  

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for jk_path, label, colour in zip(jk_paths, bin_labels, bin_colours):
        if not jk_path.exists():
            print(f"  WARN: {jk_path} not found, skipping {label}...")
            continue
            
        jk = load_jackknife(jk_path)
        
        # Plot Optical (with label for the legend)
        _plot_component(ax,
                        jk["optical"]["lam_um"], jk["optical"]["mean"],
                        std=jk["optical"]["std"],
                        color=colour, label=label)
        
        # Plot Far-IR and Radio (no label, so the legend doesn't duplicate)
        _plot_component(ax,
                        jk["farIR"]["lam_um"], jk["farIR"]["mean"],
                        std=jk["farIR"]["std"],
                        color=colour, label="")
        
        _plot_component(ax,
                        jk["radio"]["lam_um"], jk["radio"]["mean"],
                        std=jk["radio"]["std"],
                        color=colour, label="")

    # Observed data
    _obs_scatter(ax, obs)

    # Axes and formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e3) 
    ax.set_xlabel(r"$\lambda_{\rm obs}\ [\mu\mathrm{m}]$")
    ax.set_ylabel(r"$\nu I_\nu\ [\mathrm{nW\,m^{-2}\,sr^{-1}}]$")
    # ax.set_title("SIMBA EBL Contributions by Redshift", pad=8)

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8", fontsize=9)
    ax.grid(True, which="major", ls=":", alpha=0.35, color="0.6")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "ebl_jackknife_bins_single.pdf"
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
    jk_bin_paths = [
        Path("data/jackknife/jackknife_m100n1024_a0.5_z0.0-1.0.h5"),
        Path("data/jackknife/jackknife_m100n1024_a0.5_z1.0-3.0.h5"),
        Path("data/jackknife/jackknife_m100n1024_a0.5_z3.0-7.0.h5"),
    ]
    plot_redshift_binned_ebl(jk_bin_paths, obs, FIG_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()