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
    farIR_samples = []
    radio_samples = []

    # Reference wavelength grids (will be set from first run)
    lam_opt = None
    lam_fir = None
    nu_radio = None

    # DEBUG: Run once without mask to check baseline
    print("\n=== DEBUG: Running optical baseline (no mask) ===")
    lam_base, I_nu_base, _ = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
        galaxy_mask=None
    )
    nu_base = (c_light / (lam_base * u.AA)).to_value(u.Hz)
    nuInu_base = nu_base * I_nu_base * 1e6
    print(f"Baseline (no mask): nuInu max={np.max(nuInu_base):.3e}, peak at {lam_base[np.argmax(nuInu_base)]*1e-4:.2f} um")

    # DEBUG: Run with all-True mask to isolate mask handling
    print("\n=== DEBUG: Running optical with all-True mask ===")
    all_true_mask = np.ones(n_gal, dtype=bool)
    lam_test, I_nu_test, _ = lightcone_optical_background(
        cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
        galaxy_mask=all_true_mask
    )
    nu_test = (c_light / (lam_test * u.AA)).to_value(u.Hz)
    nuInu_test = nu_test * I_nu_test * 1e6
    print(f"All-True mask: nuInu max={np.max(nuInu_test):.3e}, peak at {lam_test[np.argmax(nuInu_test)]*1e-4:.2f} um")

    # Run backgrounds leaving out each region
    for i in range(n_regions):
        print(f"\n--- Jackknife sample {i + 1}/{n_regions} (excluding region {i}) ---")

        # Create mask: include all galaxies EXCEPT those in region i
        jackknife_mask = ~region_masks[i]
        n_included = jackknife_mask.sum()
        print(f"Including {n_included} galaxies ({100 * n_included / n_gal:.1f}%)")

        # Optical/NIR
        print("  Computing optical/NIR background...")
        lam, I_nu, _ = lightcone_optical_background(
            cfg, area_deg2=args.area, z_min=args.z_min, z_max=args.z_max,
            galaxy_mask=jackknife_mask
        )

        valid = np.isfinite(lam) & np.isfinite(I_nu) & (lam > 0)
        lam   = lam[valid]
        I_nu  = I_nu[valid]

        if lam_opt is None:
            lam_opt = lam
        nu_opt = (c_light / (lam * u.AA)).to_value(u.Hz)
        nuInu = nu_opt * I_nu * 1e6  # nW m^-2 sr^-1

        # Debug: print intermediate values for first iteration
        if i == 0:
            print(f"  DEBUG optical: lam={lam[:5]}... (len={len(lam)}, any nan={np.any(np.isnan(lam))})")
            print(f"  DEBUG optical: I_nu max={np.max(I_nu):.3e}, min={np.min(I_nu):.3e}")
            print(f"  DEBUG optical: nu max={np.max(nu_opt):.3e}, min={np.min(nu_opt):.3e}")
            print(f"  DEBUG optical: nuInu max={np.max(nuInu):.3e}, min={np.min(nuInu):.3e}")

        optical_samples.append(nuInu)

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
    farIR_samples = np.array(farIR_samples)
    radio_samples = np.array(radio_samples)

    # Compute jackknife statistics
    print("\n=== Computing jackknife statistics ===")

    opt_mean, opt_var, opt_std = jackknife_variance(optical_samples)
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

    print(f"\nSaved results → {out_file}")
    return out_file


def plot_results(cfg, args, results):
    """Create plot with jackknife error bands."""
    fig, ax = plt.subplots(figsize=(12, 7))

    floor = 1e-6

    # Optical with dust
    lam = results["optical"]["lam_um"]
    mean = results["optical"]["mean"]
    std = results["optical"]["std"]
    valid = mean > floor
    ax.fill_between(lam[valid], (mean - std)[valid], (mean + std)[valid],
                    alpha=0.3, color='steelblue')
    ax.loglog(lam[valid], mean[valid], lw=2, color='steelblue',
              label='Optical/NIR')

    # Far-IR
    lam_fir = results["farIR"]["lam_um"]
    mean_fir = results["farIR"]["mean"]
    std_fir = results["farIR"]["std"]
    valid_fir = mean_fir > floor
    ax.fill_between(lam_fir[valid_fir], (mean_fir - std_fir)[valid_fir],
                    (mean_fir + std_fir)[valid_fir], alpha=0.3, color='firebrick')
    ax.loglog(lam_fir[valid_fir], mean_fir[valid_fir], lw=2, color='firebrick',
              label='Far-IR (dust MBB)')

    # Radio
    lam_radio = results["radio"]["lam_um"]
    mean_radio = results["radio"]["mean"]
    std_radio = results["radio"]["std"]
    # Sort by wavelength for proper plotting
    order = np.argsort(lam_radio)
    lam_radio = lam_radio[order]
    mean_radio = mean_radio[order]
    std_radio = std_radio[order]
    valid_radio = mean_radio > floor
    ax.fill_between(lam_radio[valid_radio], (mean_radio - std_radio)[valid_radio],
                    (mean_radio + std_radio)[valid_radio], alpha=0.3, color='forestgreen')
    ax.loglog(lam_radio[valid_radio], mean_radio[valid_radio], lw=2, color='forestgreen',
              label='Radio (SF + AGN)')

    ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=13)
    ax.set_ylabel(r'$\nu\, I_\nu$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
    ax.set_title(f'Cosmic background with jackknife errors — {cfg.name}\n'
                 f'($z = {args.z_min}$–${args.z_max}$, {results["n_regions"]} regions)',
                 fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_xlim(0.1, 3e7)

    out = Path("figures/jackknife") / f"jackknife_bg_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved figure → {out}")

    return fig, ax


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

    # Run jackknife
    results = run_jackknife(cfg, args, n_regions_per_side=args.n_regions)

    # Save results
    save_results(cfg, args, results)

    # Print summary
    print_summary(results)

    # Plot
    plot_results(cfg, args, results)


if __name__ == "__main__":
    main()
