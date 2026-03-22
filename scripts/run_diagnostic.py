"""
Far-IR cosmic background with dust-temperature diagnostics.

Usage:
    # Single run (default a = -0.05):
    python scripts/run_combined_diagnostic.py --sim m25n256

    # Sweep several values of 'a' to compare far-IR curves:
    python scripts/run_combined_diagnostic.py --sim m25n256 --a_dust -0.10 -0.05 0.00 0.05 0.10
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append("/home/spujni/simba_cosmic_background")
os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.farIR import lightcone_farIR_background

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


# ── unit helpers ──────────────────────────────────────────────────
CGS_TO_NW_M2 = 1e6   # 1 erg/s/cm² → 1e6 nW/m²
FLOOR = 1e-6         # nW/m²/sr  – plotting noise floor

def _to_nW(nuInu):
    """erg/s/cm²/sr  →  nW/m²/sr, NaN below noise floor."""
    v = nuInu * CGS_TO_NW_M2
    return np.where(v > FLOOR, v, np.nan)


def main():
    parser = argparse.ArgumentParser(
        description="Far-IR cosmic background with dust-temperature diagnostics"
    )
    parser.add_argument("--sim", default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument(
        "--a_dust", type=float, nargs="+", default=[-0.05],
        help="One or more values of the normalisation parameter 'a' "
             "in the Liang+19 T_eqv relation."
    )
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)\n")

    df = pd.read_csv(r"/home/spujni/simba_cosmic_background/data/ebl/ebldata.csv")
    df = df.sort_values(by = "wave")


    # ── Far-IR for each value of 'a' ─────────────────────────────
    fir_results = {}          # a_val -> (lam_um, nuInu_nW)
    dust_temp_results = {}    # a_val -> (temps_array, z_array)

    for a_val in args.a_dust:
        tag = f"a = {a_val:+.2f}"
        print(f"=== Far-IR background ({tag}) ===")

        lam_fir, I_lam_fir, temps, zs = lightcone_farIR_background(
            cfg, area_deg2=args.area,
            z_min=args.z_min, z_max=args.z_max,
            a_dust=a_val, return_dust_temps=True
        )
        nuInu_fir = _to_nW(lam_fir * I_lam_fir)  # λI_λ = νI_ν
        lam_fir_um = lam_fir * 1e-4              # Å -> µm

        fir_results[a_val] = (lam_fir_um, nuInu_fir)
        dust_temp_results[a_val] = (temps, zs)
        print()

    # ── Figure layout: 2 rows ────────────────────────────────────
    #   top  : full-width far-IR SED overlays
    #   bot  : dust temperature histogram
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.30)
    ax_sed = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    cmap = plt.cm.autumn_r
    a_vals = sorted(fir_results.keys())
    colours = [cmap(i / max(len(a_vals) - 1, 1)) for i in range(len(a_vals))]

    # ── Top panel: Far-IR SED ────────────────────────────────────
    for a_val, col in zip(a_vals, colours):
        lam_um, nuInu = fir_results[a_val]
        ax_sed.loglog(lam_um, nuInu, lw=2, color=col,
                      label=f'$a_{{\\rm dust}} = {a_val:+.2f}$')

    ax_sed.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=13)
    ax_sed.set_ylabel(r'$\nu\, I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)

    ax_sed.scatter(df["wave"], df["ebl"], s=20, color='k', alpha=0.7, label='Observational EBL data')
    ax_sed.errorbar(df["wave"], df["ebl"], yerr=df["debl"], fmt='none', ecolor='k', alpha=0.5, capsize=3)
    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")

    #ax_sed.set_title(f'Far-IR Cosmic Background Diagnostic Plot', fontsize=14)
    ax_sed.legend(fontsize=10, ncol=2)
    ax_sed.grid(True, which='both', ls=':', alpha=0.4)
    ax_sed.set_xlim(8, 1e4)  # ~8 µm to 10 mm

    # ── Bottom: dust temperature distribution ────────────────
    for a_val, col in zip(a_vals, colours):
        temps, _ = dust_temp_results[a_val]
        if len(temps) == 0:
            continue
        ax_hist.hist(temps, bins=60, range=(5, 80), alpha=0.55,
                     color=col, edgecolor='k', linewidth=0.3,
                     label=f'$a_{{\\rm dust}} = {a_val:+.4f}$'
                           f'  (med = {np.nanmedian(temps):.3f} K)')
    ax_hist.set_xlabel(r'$T_{\rm eqv}$ [K]', fontsize=12)
    ax_hist.set_ylabel('Number of galaxies', fontsize=12)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, ls=':', alpha=0.4)

    # ── Save ──────────────────────────────────────────────────────
    out = Path("figures/farIR") / f"farIR_bg_{cfg.name}_diagnostic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()