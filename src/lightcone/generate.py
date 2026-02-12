"""
Generate a lightcone from Simba snapshots.

Usage (via scripts/run_lightcone.py):
    python scripts/run_lightcone.py --sim m25n256 --area 1.0 --z_min 0.0 --z_max 3.0
"""

import numpy as np
import h5py
from pathlib import Path
from astropy.cosmology import z_at_value
import astropy.units as u
import caesar

from src.config import SimConfig
from src.utils import get_redshift

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lightcones" 

def list_snapshots(cfg):
    """List available snapshots sorted by number."""
    snapshots = []
    for f in cfg.catalogue_dir.glob(f"{cfg.snapshot_prefix}_*.hdf5"):
        snap_num = int(f.stem.split('_')[1])
        snapshots.append((snap_num, f))
    return sorted(snapshots, key=lambda x: x[0], reverse=True)

def get_snapshot_info(path):
    """Get redshift and box size from a Caesar catalogue."""
    try:
        obj = caesar.load(str(path))
    except Exception:
        return None, None, None

    z = obj.simulation.redshift

    boxsize = obj.simulation.boxsize
    if hasattr(boxsize, 'value'):
        L = boxsize.value
        if L > 1000:
            L = L / 1000.0
    else:
        L = float(boxsize)

    return z, L, obj


def generate_lightcone(cfg, area_deg2, z_min, z_max, output_file=None,
                       verbose=True):
    """
    Generate a lightcone catalogue for any simulation.

    Parameters
    ----------
    cfg        : SimConfig
    area_deg2  : float
    z_min      : float
    z_max      : float
    output_file: Path or None (auto-named)
    verbose    : bool
    """
    if output_file is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / (
            f"lc_{cfg.name}_a{area_deg2}_z{z_min}-{z_max}.h5"
        )

    snaps = list_snapshots(cfg)

    snap_data = []
    for snap_num, path in snaps:
        z, L, _ = get_snapshot_info(path)
        if z is None:
            if verbose:
                print(f"Skipping snap {snap_num} (no halo data)")
            continue
        snap_data.append((snap_num, path, z, L))

    snap_data = sorted(snap_data, key=lambda x: x[2])
    snap_data = [(s, p, z, L) for s, p, z, L in snap_data
                 if z_min - 0.5 <= z <= z_max + 0.5]

    if verbose:
        print(f"\nUsing {len(snap_data)} snapshots")
        if snap_data:
            print(f"Redshift range: {snap_data[0][2]:.2f} to "
                  f"{snap_data[-1][2]:.2f}")

    if len(snap_data) == 0:
        raise ValueError("No valid snapshots found in redshift range!")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ra, all_dec, all_z = [], [], []
    all_snap, all_idx, all_stellar_mass = [], [], []

    cosmo = cfg.cosmology

    for snap_num, path, z_snap, L in snap_data:
        if verbose:
            print(f"\nProcessing snap {snap_num}, z={z_snap:.3f}")

        obj = caesar.load(str(path))

        coods = []
        for g in obj.galaxies:
            try:
                pos = g.pos.to('Mpc').value
            except Exception:
                try:
                    pos = g.pos.value
                except Exception:
                    pos = np.array(g.pos)
            coods.append(pos)
        coods = np.array(coods)

        stellar_mass = np.array(
            [g.masses['stellar'].value for g in obj.galaxies]
        )

        if len(coods) == 0:
            continue

        L_unit = cosmo.kpc_comoving_per_arcmin(z_snap).to('Mpc / degree')
        A = (L_unit * area_deg2 ** 0.5).value

        xmin, ymin = np.random.rand(2) * (L - A)

        mask = (
            (coods[:, 0] > xmin) & (coods[:, 0] < xmin + A) &
            (coods[:, 1] > ymin) & (coods[:, 1] < ymin + A)
        )

        if verbose:
            print(f"  Selected {mask.sum()} of {len(coods)} galaxies")

        selected_coods = coods[mask]
        selected_mass = stellar_mass[mask]
        selected_idx = np.where(mask)[0]

        ra = ((selected_coods[:, 0] - xmin - A / 2)
              / A * area_deg2 ** 0.5)
        dec = ((selected_coods[:, 1] - ymin - A / 2)
               / A * area_deg2 ** 0.5)

        z_offset = cosmo.comoving_distance(z_snap).value
        galaxy_z = np.array([
            z_at_value(
                cosmo.comoving_distance,
                (z_offset + coord) * u.Mpc
            ).value
            for coord in (selected_coods[:, 2] - L / 2)
        ])

        z_mask = (galaxy_z >= z_min) & (galaxy_z <= z_max)

        all_ra.extend(ra[z_mask])
        all_dec.extend(dec[z_mask])
        all_z.extend(galaxy_z[z_mask])
        all_snap.extend([snap_num] * z_mask.sum())
        all_idx.extend(selected_idx[z_mask])
        all_stellar_mass.extend(selected_mass[z_mask])

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('RA', data=np.array(all_ra))
        f.create_dataset('DEC', data=np.array(all_dec))
        f.create_dataset('z', data=np.array(all_z))
        f.create_dataset('snap', data=np.array(all_snap))
        f.create_dataset('galaxy_index', data=np.array(all_idx))
        f.create_dataset('stellar_mass', data=np.array(all_stellar_mass))
        f.attrs['area_deg2'] = area_deg2
        f.attrs['z_min'] = z_min
        f.attrs['z_max'] = z_max
        f.attrs['n_galaxies'] = len(all_ra)
        f.attrs['simulation'] = cfg.name

    if verbose:
        print(f"\n=== Lightcone saved to {output_file} ===")
        print(f"Total galaxies: {len(all_ra)}")

    return output_file