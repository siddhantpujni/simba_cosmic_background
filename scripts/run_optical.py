"""
Usage:
    python scripts/run_optical.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')
import fsps

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background


def main():
    parser = argparse.ArgumentParser(description="Optical cosmic background")
    parser.add_argument("--sim", default="m25n256",
                        choices=["m25n256", "m50n512", "m200n2048"])
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)")

    lam, intensity = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    # intensity is per Hz; convert to ν I_ν
    from astropy.constants import c as c_light
    import astropy.units as u
    nu = (c_light / (lam * u.AA)).to_value(u.Hz)
    nu_I_nu = nu * intensity

    ax.loglog(lam, nu_I_nu, lw=1.5)
    ax.set_xlabel(r"$\lambda_\mathrm{obs}$ [Å]")
    ax.set_ylabel(r"$\nu\,I_\nu$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")
    ax.set_title(f"Optical/NIR background — {cfg.name}")
    ax.set_xlim(1000, 50000)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    out = Path("figures/optical") / f"optical_bg_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()