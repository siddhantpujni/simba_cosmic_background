"""
Combined optical + far-IR + radio cosmic background.

Usage:
    python scripts/run_combined.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3
    
    # Load cached results instead of recomputing:
    python scripts/run_combined.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3 --load
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
from src.utils import save_background_results, load_background_results


def compute_backgrounds(cfg, args, a_dust=-0.017341):
    """Compute all background components from scratch."""
    
    # ── Optical / near-IR ─────────────────────────────────────────
    print("=== Optical/NIR background ===")
    lam_opt, I_nu_opt, I_nu_opt_nodust = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    nu_opt = (c_light / (lam_opt * u.AA)).to_value(u.Hz)
    nuInu_opt        = nu_opt * I_nu_opt          # erg/s/cm²/sr
    nuInu_opt_nodust = nu_opt * I_nu_opt_nodust

    # ── Far-IR ────────────────────────────────────────────────────
    print("\n=== Far-IR background ===")
    lam_fir, I_lam_fir, dust_temps, dust_zs = lightcone_farIR_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
        a_dust=a_dust, return_dust_temps=True
    )
    nuInu_fir = lam_fir * I_lam_fir

    # ── Radio (SF + AGN) ──────────────────────────────────────────
    print("\n=== Radio background (SF + AGN) ===")
    nu_radio, I_nu_radio, _, _ = lightcone_radio_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )
    lam_radio_um = (c_light / (nu_radio * u.Hz)).to_value(u.AA) * 1e-4
    nuInu_radio  = nu_radio * I_nu_radio

    # ── Convert to nW m⁻² sr⁻¹ ───────────────────────────────────
    cgs_to_nWm2 = 1e6
    nuInu_opt_nW        = nuInu_opt * cgs_to_nWm2
    nuInu_opt_nodust_nW = nuInu_opt_nodust * cgs_to_nWm2
    nuInu_fir_nW        = nuInu_fir * cgs_to_nWm2
    nuInu_radio_nW      = nuInu_radio * cgs_to_nWm2

    lam_opt_um = lam_opt * 1e-4
    lam_fir_um = lam_fir * 1e-4

    # ── Save results ──────────────────────────────────────────────
    save_background_results(
        cfg, args,
        lam_opt, I_nu_opt, nuInu_opt_nW,
        I_nu_opt_nodust, nuInu_opt_nodust_nW,
        lam_fir, I_lam_fir, nuInu_fir_nW,
        nu_radio, I_nu_radio, lam_radio_um, nuInu_radio_nW,
        dust_temps=dust_temps, dust_redshifts=dust_zs,
        a_dust=a_dust,
    )

    return {
        "optical": {"lam_um": lam_opt_um, "nuInu_nW": nuInu_opt_nW,
                     "nuInu_nodust_nW": nuInu_opt_nodust_nW},
        "farIR":   {"lam_um": lam_fir_um, "nuInu_nW": nuInu_fir_nW},
        "radio":   {"lam_um": lam_radio_um, "nuInu_nW": nuInu_radio_nW},
    }

def load_cached(cfg, args):
    """Load pre-computed backgrounds from cache."""
    data = load_background_results(cfg.name, args.area, args.z_min, args.z_max)
    
    lam_opt_um = data["optical"]["lam_AA"] * 1e-4
    lam_fir_um = data["farIR"]["lam_AA"] * 1e-4
    
    return {
        "optical": {"lam_um": lam_opt_um,
                     "nuInu_nW": data["optical"]["nuInu_nW"],
                     "nuInu_nodust_nW": data["optical"].get("nuInu_nodust_nW")},
        "farIR":   {"lam_um": lam_fir_um, "nuInu_nW": data["farIR"]["nuInu_nW"]},
        "radio":   {"lam_um": data["radio"]["lam_um"], "nuInu_nW": data["radio"]["nuInu_nW"]},
    }


def plot_combined(cfg, args, results):
    """Create the combined background plot."""
    
    lam_opt_um          = results["optical"]["lam_um"]
    nuInu_opt_nW        = results["optical"]["nuInu_nW"]
    nuInu_opt_nodust_nW = results["optical"].get("nuInu_nodust_nW")
    lam_fir_um          = results["farIR"]["lam_um"]
    nuInu_fir_nW        = results["farIR"]["nuInu_nW"]
    lam_radio_um        = results["radio"]["lam_um"]
    nuInu_radio_nW      = results["radio"]["nuInu_nW"]

    floor = 1e-6
    opt_plot = np.where(nuInu_opt_nW > floor, nuInu_opt_nW, np.nan)
    fir_plot = np.where(nuInu_fir_nW > floor, nuInu_fir_nW, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(lam_opt_um, opt_plot, lw=2, color='steelblue',
              label='Optical / NIR (with dust)')

    if nuInu_opt_nodust_nW is not None:
        opt_nodust_plot = np.where(nuInu_opt_nodust_nW > floor,
                                   nuInu_opt_nodust_nW, np.nan)
        ax.loglog(lam_opt_um, opt_nodust_plot, lw=2, ls='--',
                  color='cornflowerblue', label='Optical / NIR (no dust)')

    ax.loglog(lam_fir_um, fir_plot, lw=2, color='firebrick',
              label='Far-IR (dust MBB)')
    ax.loglog(lam_radio_um, nuInu_radio_nW, lw=2, color='forestgreen',
              label='Radio (SF + AGN)')

    # Mark crossover if it exists within overlapping range
    lam_min = max(lam_opt_um.min(), lam_fir_um.min())
    lam_max = min(lam_opt_um.max(), lam_fir_um.max())
    if lam_min < lam_max:
        lam_common = np.logspace(np.log10(lam_min), np.log10(lam_max), 2000)
        opt_interp = np.interp(lam_common, lam_opt_um, nuInu_opt_nW,
                               left=0, right=0)
        fir_interp = np.interp(lam_common, lam_fir_um, nuInu_fir_nW,
                               left=0, right=0)
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

    ax.set_xlim(0.1, 3e7)

    out = Path("figures/combined/main") / f"combined_bg_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure → {out}")
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(
        description="Combined optical + far-IR + radio cosmic background")
    parser.add_argument("--sim", default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--load", action="store_true",
                        help="Load cached results instead of recomputing")
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)\n")

    # ── Either load cached or compute fresh ───────────────────────
    if args.load:
        print("Loading cached results...")
        try:
            results = load_cached(cfg, args)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Run without --load first to compute and cache results.")
            return
    else:
        results = compute_backgrounds(cfg, args)

    # ── Plot ──────────────────────────────────────────────────────
    plot_combined(cfg, args, results)


if __name__ == "__main__":
    main()