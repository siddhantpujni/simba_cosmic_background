import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.farIR import lightcone_farIR_background
from src.utils import save_farIR_parameter_sweep


def main():
    parser = argparse.ArgumentParser(description="Far-IR a_dust sweep")
    parser.add_argument("--sim", default="m100n1024")
    parser.add_argument("--area", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--a_min", type=float, default=-0.5)
    parser.add_argument("--a_max", type=float, default=0.5)
    parser.add_argument("--n_a", type=int, default=90) 
    args = parser.parse_args()

    cfg = load_config(args.sim)
    a_values = np.linspace(args.a_min, args.a_max, args.n_a)

    print(f"Running far-IR sweep: {cfg.name}, area={args.area}, "
          f"z=[{args.z_min}, {args.z_max}], {len(a_values)} a_dust values")

    results_dict = {}
    for i, a in enumerate(a_values):
        print(f"  [{i+1}/{len(a_values)}] a_dust = {a:.4f}")
        lam_fir, I_lam_fir = lightcone_farIR_background(
            cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
            a_dust=a
        )
        nuInu_fir = lam_fir * I_lam_fir
        nuInu_nW = nuInu_fir * 1e6
        results_dict[a] = (lam_fir, nuInu_nW)

    save_farIR_parameter_sweep(cfg.name, args.z_min, args.z_max,
                               a_values, results_dict)
    print("Done.")


if __name__ == "__main__":
    main()