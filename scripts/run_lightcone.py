"""
Usage:
    python scripts/run_lightcone.py --sim m100n1024 --area 0.5 --z_min 0 --z_max 7
    python scripts/run_lightcone.py --sim m100n1024 --area 0.5 --z_min 0 --z_max 7 --midsnap
    python scripts/run_lightcone.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3 --snap_step 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.lightcone.generate import generate_lightcone


def main():
    parser = argparse.ArgumentParser(description="Generate lightcone")
    parser.add_argument("--sim", default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--snap_step", type=int, default=2,
                        help="Use every Nth snapshot. For the 100 Mpc/h box "
                             "the comoving size is 2x the inter-snapshot path "
                             "length, so snap_step=2 avoids double-counting.")
    parser.add_argument("--midsnap", action="store_true",
                        help="Use even snapshot set (0,2,4,...) and centre "
                             "the comoving offset on the intermediate snap. "
                             "Default uses the odd set (1,3,5,...).")
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Generating lightcone for {cfg.name}")

    generate_lightcone(cfg, args.area, args.z_min, args.z_max,
                       snap_step=args.snap_step, midsnap=args.midsnap)


if __name__ == "__main__":
    main()