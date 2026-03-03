"""
Combined optical + far-IR cosmic background.

Usage:
    python scripts/run_combined.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c as c_light

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background
from src.backgrounds.farIR import lightcone_farIR_background
from src.backgrounds.radio import lightcone_radio_background


def main():
    parser = argparse.ArgumentParser(
        description="Combined optical + far-IR cosmic background")
    parser.add_argument("--sim", default="m25n256",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)\n")

    # ── Optical / near-IR ─────────────────────────────────────────
    print("=== Optical/NIR background ===")
    lam_opt, I_nu_opt = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    nu_opt = (c_light / (lam_opt * u.AA)).to_value(u.Hz)
    nuInu_opt = nu_opt * I_nu_opt          # erg/s/cm²/sr

    # ── Far-IR ────────────────────────────────────────────────────
    print("\n=== Far-IR background ===")
    lam_fir, I_lam_fir = lightcone_farIR_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    nuInu_fir = lam_fir * I_lam_fir        # λ I_λ = ν I_ν  (erg/s/cm²/sr)

    # ── Radio (star formation) ────────────────────────────────────
    print("\n=== Radio background (SF, Condon 1992 / Thomas+2021) ===")
    nu_radio, I_nu_radio = lightcone_radio_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    lam_radio_um = (c_light / (nu_radio * u.Hz)).to_value(u.AA) * 1e-4  # → µm
    nuInu_radio  = nu_radio * I_nu_radio


    # ── Convert to nW m⁻² sr⁻¹ (standard CIB unit) ──────────────
    # 1 erg/s/cm² = 1e-3 W/m² = 1e6 nW/m²
    cgs_to_nWm2 = 1e6
    nuInu_opt_nW = nuInu_opt * cgs_to_nWm2
    nuInu_fir_nW = nuInu_fir * cgs_to_nWm2

    # ── Convert wavelength to µm for readability ─────────────────
    lam_opt_um = lam_opt * 1e-4
    lam_fir_um = lam_fir * 1e-4

    # ── Mask out negligible values so the log plot isn't wrecked ──
    floor = 1e-6  # nW/m²/sr — anything below this is numerical noise
    opt_plot = np.where(nuInu_opt_nW > floor, nuInu_opt_nW, np.nan)
    fir_plot = np.where(nuInu_fir_nW > floor, nuInu_fir_nW, np.nan)

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(lam_opt_um, opt_plot, lw=2, color='steelblue',
              label='Optical / NIR (stellar)')
    ax.loglog(lam_fir_um, fir_plot, lw=2, color='firebrick',
              label='Far-IR (dust MBB)')
    ax.loglog(lam_radio_um, nuInu_radio, lw=2, color='forestgreen',
                  label='Radio (SF)')

    # Mark crossover if it exists within overlapping range
    lam_min = max(lam_opt_um.min(), lam_fir_um.min())
    lam_max = min(lam_opt_um.max(), lam_fir_um.max())
    if lam_min < lam_max:
        lam_common = np.logspace(np.log10(lam_min), np.log10(lam_max), 2000)
        opt_interp = np.interp(lam_common, lam_opt_um, nuInu_opt_nW,
                               left=0, right=0)
        fir_interp = np.interp(lam_common, lam_fir_um, nuInu_fir_nW,
                               left=0, right=0)
        # Only look for crossovers where both are above the floor
        both_valid = (opt_interp > floor) & (fir_interp > floor)
        diff = np.where(both_valid, opt_interp - fir_interp, np.nan)
        sign_changes = np.where(np.diff(np.sign(diff[~np.isnan(diff)])))[0]
        for idx in sign_changes:
            valid_idx = np.where(both_valid)[0][idx]
            lam_cross = lam_common[valid_idx]
            ax.axvline(lam_cross, ls='--', color='grey', alpha=0.6)
            ax.annotate(f'crossover\n{lam_cross:.0f} µm',
                        xy=(lam_cross, opt_interp[valid_idx]),
                        fontsize=9, ha='center',
                        xytext=(0, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='grey'))

    ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=13)
    ax.set_ylabel(r'$\nu\, I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
    ax.set_title(f'Cosmic background — {cfg.name}  '
                 f'($z = {args.z_min}$–${args.z_max}$)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    ax.set_xlim(0.1, 3e7)   # 0.1 µm → 30 m

    out = Path("figures") / f"combined_bg_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()