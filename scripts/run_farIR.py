"""
Usage:
    python scripts/run_farIR.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.farIR import lightcone_farIR_background


def main():
    parser = argparse.ArgumentParser(description="Far-IR cosmic background")
    parser.add_argument("--sim", default="m25n256",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)")

    lam, intensity = lightcone_farIR_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    nu_I_nu = intensity * lam  # λ I_λ = ν I_ν
    ax.loglog(lam * 1e-4, nu_I_nu, lw=1.5)
    ax.set_xlabel(r"$\lambda_\mathrm{obs}$ [$\mu$m]")
    ax.set_ylabel(r"$\nu\,I_\nu$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")
    ax.set_title(f"Far-IR background — {cfg.name}")

    out = Path("figures/farIR") / f"farIR_bg_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()