"""
Combined optical + far-IR cosmic background with dust-temperature diagnostics.

Usage:
    # Single run (default a = -0.05):
    python scripts/run_combined.py --sim m25n256

    # Sweep several values of 'a' to find the best peak position:
    python scripts/run_combined.py --sim m25n256 --a_dust -0.10 -0.05 0.00 0.05 0.10
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
from astropy.constants import c as c_light

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background
from src.backgrounds.farIR import lightcone_farIR_background


# ── unit helpers ──────────────────────────────────────────────────
CGS_TO_NW_M2 = 1e6          # 1 erg/s/cm² → 1e6 nW/m²
FLOOR        = 1e-6          # nW/m²/sr  – plotting noise floor


def _to_nW(nuInu):
    """erg/s/cm²/sr  →  nW/m²/sr, NaN below noise floor."""
    v = nuInu * CGS_TO_NW_M2
    return np.where(v > FLOOR, v, np.nan)


def main():
    parser = argparse.ArgumentParser(
        description="Combined optical + far-IR cosmic background "
                    "with dust-temperature diagnostics")
    parser.add_argument("--sim", default="m25n256",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument(
        "--a_dust", type=float, nargs="+", default=[-0.05],
        help="One or more values of the normalisation parameter 'a' "
             "in the Liang+19 T_eqv relation.  "
             "Pass several values to overlay far-IR curves for comparison.")
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)\n")

    # ── Optical / near-IR (computed once) ─────────────────────────
    print("=== Optical/NIR background ===")
    lam_opt, I_nu_opt = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    nu_opt      = (c_light / (lam_opt * u.AA)).to_value(u.Hz)
    nuInu_opt   = _to_nW(nu_opt * I_nu_opt)
    lam_opt_um  = lam_opt * 1e-4                           # Å → µm

    # ── Far-IR for each value of 'a' ─────────────────────────────
    fir_results = {}          # a_val → (lam_µm, nuInu_nW)
    dust_temp_results = {}    # a_val → (temps_array, z_array)

    for a_val in args.a_dust:
        tag = f"a = {a_val:+.2f}"
        print(f"\n=== Far-IR background ({tag}) ===")

        lam_fir, I_lam_fir, temps, zs = lightcone_farIR_background(
            cfg, area_deg2=args.area,
            z_min=args.z_min, z_max=args.z_max,
            a_dust=a_val, return_dust_temps=True
        )
        nuInu_fir = _to_nW(lam_fir * I_lam_fir)
        lam_fir_um = lam_fir * 1e-4

        fir_results[a_val]       = (lam_fir_um, nuInu_fir)
        dust_temp_results[a_val] = (temps, zs)

    # ── Figure layout: 2 rows ────────────────────────────────────
    #   top  : full-width SED  (optical + far-IR overlays)
    #   bot-L: dust temperature histogram
    #   bot-R: peak wavelength vs 'a'
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1],
                            hspace=0.30, wspace=0.30)
    ax_sed   = fig.add_subplot(gs[0, :])
    ax_hist  = fig.add_subplot(gs[1, 0])
    ax_peak  = fig.add_subplot(gs[1, 1])

    # ── Top panel: SED ────────────────────────────────────────────
    ax_sed.loglog(lam_opt_um, nuInu_opt, lw=2, color='steelblue',
                  label='Optical / NIR (stellar)')

    cmap = plt.cm.autumn_r
    a_vals = sorted(fir_results.keys())
    colours = [cmap(i / max(len(a_vals) - 1, 1)) for i in range(len(a_vals))]

    for a_val, col in zip(a_vals, colours):
        lam_um, nuInu = fir_results[a_val]
        ax_sed.loglog(lam_um, nuInu, lw=2, color=col,
                      label=f'Far-IR  $a = {a_val:+.2f}$')

    ax_sed.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=13)
    ax_sed.set_ylabel(r'$\nu\, I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
    ax_sed.set_title(f'Cosmic background — {cfg.name}  '
                     f'($z = {args.z_min}$–${args.z_max}$)', fontsize=14)
    ax_sed.legend(fontsize=10, ncol=2)
    ax_sed.grid(True, which='both', ls=':', alpha=0.4)
    ax_sed.set_xlim(0.1, 1e4)   # extended to 10^4 µm = 10 mm

    # ── Bottom-left: dust temperature distribution ────────────────
    for a_val, col in zip(a_vals, colours):
        temps, _ = dust_temp_results[a_val]
        if len(temps) == 0:
            continue
        ax_hist.hist(temps, bins=60, range=(5, 80), alpha=0.55,
                     color=col, edgecolor='k', linewidth=0.3,
                     label=f'$a = {a_val:+.2f}$'
                           f'  (med = {np.nanmedian(temps):.1f} K)')
    ax_hist.set_xlabel(r'$T_{\rm eqv}$ [K]', fontsize=12)
    ax_hist.set_ylabel('Number of galaxies', fontsize=12)
    ax_hist.set_title('Dust temperature distribution', fontsize=13)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, ls=':', alpha=0.4)

    # ── Bottom-right: peak wavelength vs 'a' ─────────────────────
    peaks_um = []
    for a_val in a_vals:
        lam_um, nuInu = fir_results[a_val]
        valid = np.isfinite(nuInu) & (nuInu > 0)
        if valid.any():
            peaks_um.append(lam_um[valid][np.argmax(nuInu[valid])])
        else:
            peaks_um.append(np.nan)

    ax_peak.plot(a_vals, peaks_um, 'o-', color='firebrick', lw=2, ms=8)
    ax_peak.set_xlabel("Normalisation parameter  $a$", fontsize=12)
    ax_peak.set_ylabel(r'$\lambda_{\rm peak}$ [$\mu$m]', fontsize=12)
    ax_peak.set_title('Far-IR peak wavelength vs $a$', fontsize=13)
    ax_peak.grid(True, ls=':', alpha=0.4)

    # ── Save ──────────────────────────────────────────────────────
    out = Path("figures/combined/diagnostic") / f"combined_bg_{cfg.name}_diagnostic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()