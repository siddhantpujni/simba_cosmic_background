"""
Jackknife error estimation for cosmic background SEDs.

Splits the lightcone into 16 spatial regions (4x4 grid in RA/DEC),
then computes backgrounds leaving out one region at a time.
Uses jackknife variance formula to estimate errors.

Usage:
    python scripts/run_jackknife.py --sim m100n1024 --area 0.5 --z_min 0 --z_max 7
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SPS_HOME', '/home/spujni/fsps')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import astropy.units as u
from astropy.constants import c as c_light

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.backgrounds.optical import lightcone_optical_background, build_lightcone as build_lc_optical
from src.backgrounds.farIR import lightcone_farIR_background, build_lightcone as build_lc_farIR
from src.backgrounds.radio import lightcone_radio_background, build_lightcone as build_lc_radio


def load_lightcone_coords(cfg, area_deg2, z_min, z_max):
    """Load RA/DEC from the lightcone file."""
    # Build/load the lightcone (uses optical's build_lightcone)
    lc_path = build_lc_optical(cfg, area_deg2, z_min, z_max)

    with h5py.File(lc_path, "r") as lc:
        ra = lc["RA"][:]
        dec = lc["DEC"][:]
        n_gal = len(ra)

    return ra, dec, n_gal


def create_spatial_regions(ra, dec, n_regions_per_side=4):
    """
    Split the lightcone into n_regions_per_side^2 spatial regions.

    Returns
    -------
    region_masks : list of bool arrays
        Each mask is True for galaxies in that region.
    """
    n_regions = n_regions_per_side ** 2

    # Create bins in RA and DEC
    ra_edges = np.linspace(ra.min(), ra.max(), n_regions_per_side + 1)
    dec_edges = np.linspace(dec.min(), dec.max(), n_regions_per_side + 1)

    # Handle edge case where all galaxies have same RA or DEC
    if ra.max() == ra.min():
        ra_edges = np.array([ra.min() - 0.1, ra.max() + 0.1])
    if dec.max() == dec.min():
        dec_edges = np.array([dec.min() - 0.1, dec.max() + 0.1])

    region_masks = []
    for i in range(n_regions_per_side):
        for j in range(n_regions_per_side):
            # Include right edge for last bin
            if i == n_regions_per_side - 1:
                ra_mask = (ra >= ra_edges[i]) & (ra <= ra_edges[i + 1])
            else:
                ra_mask = (ra >= ra_edges[i]) & (ra < ra_edges[i + 1])

            if j == n_regions_per_side - 1:
                dec_mask = (dec >= dec_edges[j]) & (dec <= dec_edges[j + 1])
            else:
                dec_mask = (dec >= dec_edges[j]) & (dec < dec_edges[j + 1])

            region_masks.append(ra_mask & dec_mask)

    return region_masks


def jackknife_variance(samples):
    """
    Compute jackknife variance estimate.

    samples : array of shape (n_jackknife, n_wavelengths)
        Each row is a jackknife sample (computed with one region removed).

    Returns
    -------
    mean : array of shape (n_wavelengths,)
    variance : array of shape (n_wavelengths,)
    std : array of shape (n_wavelengths,)
    """
    n = samples.shape[0]
    mean = np.mean(samples, axis=0)

    # Jackknife variance formula: Var = (n-1)/n * sum((x_i - mean)^2)
    variance = (n - 1) / n * np.sum((samples - mean) ** 2, axis=0)
    std = np.sqrt(variance)

    return mean, variance, std


def run_jackknife(cfg, args, n_regions_per_side=4, a_dust=-0.017341):
    """
    Run jackknife error estimation for all three backgrounds.
    """
    n_regions = n_regions_per_side ** 2
    print(f"\n=== Jackknife Error Estimation ({n_regions} regions) ===\n")

    # Load lightcone coordinates
    print("Loading lightcone coordinates...")
    ra, dec, n_gal = load_lightcone_coords(cfg, args.area, args.z_min, args.z_max)
    print(f"Total galaxies: {n_gal}")

    # Create spatial regions
    print(f"Splitting into {n_regions} spatial regions...")
    region_masks = create_spatial_regions(ra, dec, n_regions_per_side)

    region_counts = [mask.sum() for mask in region_masks]
    print(f"Galaxies per region: min={min(region_counts)}, max={max(region_counts)}, "
          f"mean={np.mean(region_counts):.1f}")

    # Storage for jackknife samples
    optical_samples = []
    optical_samples_nodust = []
    farIR_samples = []
    radio_samples = []

    # Reference wavelength grids (will be set from first run)
    lam_opt = None
    lam_fir = None
    nu_radio = None

    # Run backgrounds leaving out each region
    for i in range(n_regions):
        print(f"\n--- Jackknife sample {i + 1}/{n_regions} (excluding region {i}) ---")

        # Create mask: include all galaxies EXCEPT those in region i
        jackknife_mask = ~region_masks[i]
        n_included = jackknife_mask.sum()
        print(f"Including {n_included} galaxies ({100 * n_included / n_gal:.1f}%)")

        # Optical/NIR
        print("  Computing optical/NIR background...")
        lam, I_nu, I_nu_nodust = lightcone_optical_background(
            cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
            galaxy_mask=jackknife_mask
        )

        valid = np.isfinite(lam) & np.isfinite(I_nu) & (lam > 0)
        lam   = lam[valid]
        I_nu  = I_nu[valid]
        I_nu_nodust = I_nu_nodust[valid]

        if lam_opt is None:
            lam_opt = lam
        nu_opt = (c_light / (lam * u.AA)).to_value(u.Hz)
        nuInu = nu_opt * I_nu * 1e6  # nW m^-2 sr^-1
        nuInu_nodust = nu_opt * I_nu_nodust * 1e6  # nW m^-2 sr^-1

        optical_samples.append(nuInu)
        optical_samples_nodust.append(nuInu_nodust)

        # Far-IR
        print("  Computing far-IR background...")
        lam_f, I_lam_f, _, _ = lightcone_farIR_background(
            cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
            a_dust=a_dust, return_dust_temps=True, galaxy_mask=jackknife_mask
        )
        if lam_fir is None:
            lam_fir = lam_f
        nuInu_fir = lam_f * I_lam_f * 1e6  # nW m^-2 sr^-1
        farIR_samples.append(nuInu_fir)

        # Radio
        print("  Computing radio background...")
        nu_r, I_nu_r, _, _ = lightcone_radio_background(
            cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
            galaxy_mask=jackknife_mask
        )
        if nu_radio is None:
            nu_radio = nu_r
        nuInu_radio = nu_r * I_nu_r * 1e6  # nW m^-2 sr^-1
        radio_samples.append(nuInu_radio)

    # Convert to arrays
    optical_samples = np.array(optical_samples)
    optical_samples_nodust = np.array(optical_samples_nodust)
    farIR_samples = np.array(farIR_samples)
    radio_samples = np.array(radio_samples)

    # Compute jackknife statistics
    print("\n=== Computing jackknife statistics ===")

    opt_mean, opt_var, opt_std = jackknife_variance(optical_samples)
    opt_nodust_mean, opt_nodust_var, opt_nodust_std = jackknife_variance(optical_samples_nodust)
    fir_mean, fir_var, fir_std = jackknife_variance(farIR_samples)
    radio_mean, radio_var, radio_std = jackknife_variance(radio_samples)

    # Convert wavelengths to microns
    lam_opt_um = lam_opt * 1e-4
    lam_fir_um = lam_fir * 1e-4
    lam_radio_um = (c_light / (nu_radio * u.Hz)).to_value(u.AA) * 1e-4

    results = {
        "optical": {
            "lam_um": lam_opt_um,
            "mean": opt_mean,
            "std": opt_std,
            "samples": optical_samples,
            "mean_nodust": opt_nodust_mean,
            "std_nodust": opt_nodust_std,
            "samples_nodust": optical_samples_nodust,
        },
        "farIR": {
            "lam_um": lam_fir_um,
            "mean": fir_mean,
            "std": fir_std,
            "samples": farIR_samples,
        },
        "radio": {
            "lam_um": lam_radio_um,
            "mean": radio_mean,
            "std": radio_std,
            "samples": radio_samples,
        },
        "n_regions": n_regions,
        "region_counts": region_counts,
    }

    return results


def save_results(cfg, args, results):
    """Save jackknife results to HDF5."""
    out_dir = Path("data/jackknife")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"jackknife_{cfg.name}_a{args.area}_z{args.z_min}-{args.z_max}.h5"

    with h5py.File(out_file, "w") as f:
        f.attrs["simulation"] = cfg.name
        f.attrs["area_deg2"] = args.area
        f.attrs["z_min"] = args.z_min
        f.attrs["z_max"] = args.z_max
        f.attrs["n_regions"] = results["n_regions"]

        for band in ["optical", "farIR", "radio"]:
            grp = f.create_group(band)
            grp.create_dataset("lam_um", data=results[band]["lam_um"])
            grp.create_dataset("mean", data=results[band]["mean"])
            grp.create_dataset("std", data=results[band]["std"])
            grp.create_dataset("samples", data=results[band]["samples"])
            if band == "optical":
                grp.create_dataset("mean_nodust", data=results[band]["mean_nodust"])
                grp.create_dataset("std_nodust", data=results[band]["std_nodust"])
                grp.create_dataset("samples_nodust", data=results[band]["samples_nodust"])

    print(f"\nSaved results → {out_file}")
    return out_file


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("JACKKNIFE ERROR SUMMARY")
    print("=" * 60)

    for band in ["optical", "farIR", "radio"]:
        data = results[band]
        mean = data["mean"]
        std = data["std"]

        # Find peak
        valid = mean > 0
        if valid.any():
            peak_idx = np.argmax(mean[valid])
            peak_lam = data["lam_um"][valid][peak_idx]
            peak_val = mean[valid][peak_idx]
            peak_err = std[valid][peak_idx]
            rel_err = 100 * peak_err / peak_val if peak_val > 0 else 0

            print(f"\n{band.upper()}:")
            print(f"  Peak at λ = {peak_lam:.2f} µm")
            print(f"  νIν = {peak_val:.4f} ± {peak_err:.4f} nW m⁻² sr⁻¹")
            print(f"  Relative error: {rel_err:.1f}%")

        if band == "optical":
            mean_nodust = data["mean_nodust"]
            std_nodust = data["std_nodust"]
            valid_nodust = mean_nodust > 0
            if valid_nodust.any():
                peak_idx = np.argmax(mean_nodust[valid_nodust])
                peak_lam = data["lam_um"][valid_nodust][peak_idx]
                peak_val = mean_nodust[valid_nodust][peak_idx]
                peak_err = std_nodust[valid_nodust][peak_idx]
                rel_err = 100 * peak_err / peak_val if peak_val > 0 else 0

                print(f"  (no dust) Peak at λ = {peak_lam:.2f} µm")
                print(f"    νIν = {peak_val:.4f} ± {peak_err:.4f} nW m⁻² sr⁻¹")
                print(f"    Relative error: {rel_err:.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Jackknife error estimation for cosmic backgrounds")
    parser.add_argument("--sim", default="m100n1024",
                        choices=["m25n256", "m50n512", "m100n1024"])
    parser.add_argument("--area", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=7.0)
    parser.add_argument("--n_regions", type=int, default=4,
                        help="Number of regions per side (default: 4 → 16 total)")
    args = parser.parse_args()

    cfg = load_config(args.sim)
    print(f"Running jackknife on {cfg.name} (box={cfg.box_size_mpc_h} Mpc/h)")

    results = run_jackknife(cfg, args, n_regions_per_side=args.n_regions)

    save_results(cfg, args, results)

    print_summary(results)

if __name__ == "__main__":
    main()