"""
Usage:
    python scripts/run_lightcone.py --sim m25n256 --area 1.0 --z_min 0 --z_max 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.lightcone.generate import generate_lightcone


def main():
    parser = argparse.ArgumentParser(description="Generate lightcone")
    parser.add_argument("--sim", default="m25n256",
                        choices=["m25n256", "m50n512", "m200n2048"])
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Generating lightcone for {cfg.name}")

    generate_lightcone(cfg, args.area, args.z_min, args.z_max)


if __name__ == "__main__":
    main()