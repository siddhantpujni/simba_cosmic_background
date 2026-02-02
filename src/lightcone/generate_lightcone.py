"""
Generate a lightcone from m25n256 Simba snapshots.

Usage:
    python generate_lightcone_m25n256.py --area 1.0 --z_min 0.0 --z_max 3.0
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import caesar

# === Configuration ===
CAT_DIR = Path("/home/spujni/sim/m50n512/s50/Groups/")
OUTPUT_DIR = Path("/home/spujni/simba_cosmic_background/data/lightcones")


def list_snapshots():
    """List available snapshots sorted by number."""
    snapshots = []
    for f in CAT_DIR.glob("m50n512_*.hdf5"):
        snap_num = int(f.stem.split('_')[1])
        snapshots.append((snap_num, f))
    return sorted(snapshots, key=lambda x: x[0], reverse=True)  # highest snap (z=0) first


def get_snapshot_info(path):
    """Get redshift and box size from a Caesar catalogue."""
    try:
        obj = caesar.load(str(path))
    except Exception as e:
        # Skip files with no halos (very high z, early universe)
        return None, None, None
    
    z = obj.simulation.redshift
    
    # Get box size value directly
    boxsize = obj.simulation.boxsize
    if hasattr(boxsize, 'value'):
        L = boxsize.value
        if L > 1000:  # Likely kpc
            L = L / 1000.0
    else:
        L = float(boxsize)
    
    return z, L, obj


def generate_lightcone(area_deg2, z_min, z_max, output_file, verbose=True):
    """
    Generate a lightcone catalogue.
    """
    snaps = list_snapshots()
    
    # Filter snapshots to redshift range (with some buffer)
    snap_data = []
    for snap_num, path in snaps:
        z, L, _ = get_snapshot_info(path)
        
        # Skip files that failed to load (no halos)
        if z is None:
            if verbose:
                print(f"Skipping snap {snap_num} (no halo data)")
            continue
        
        snap_data.append((snap_num, path, z, L))
    
    # Sort by redshift
    snap_data = sorted(snap_data, key=lambda x: x[2])
    
    # Filter to z range
    snap_data = [(s, p, z, L) for s, p, z, L in snap_data 
                 if z_min - 0.5 <= z <= z_max + 0.5]
    
    if verbose:
        print(f"\nUsing {len(snap_data)} snapshots")
        if len(snap_data) > 0:
            print(f"Redshift range: {snap_data[0][2]:.2f} to {snap_data[-1][2]:.2f}")
    
    if len(snap_data) == 0:
        raise ValueError("No valid snapshots found in redshift range!")
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to collect all galaxies
    all_ra = []
    all_dec = []
    all_z = []
    all_snap = []
    all_idx = []
    all_stellar_mass = []
    
    for snap_num, path, z_snap, L in snap_data:
        if verbose:
            print(f"\nProcessing snap {snap_num}, z={z_snap:.3f}")
        
        obj = caesar.load(str(path))
        
        # Get galaxy positions - handle units flexibly
        coods = []
        for g in obj.galaxies:
            try:
                pos = g.pos.to('Mpc').value
            except:
                try:
                    pos = g.pos.value
                except:
                    pos = np.array(g.pos)
            coods.append(pos)
        coods = np.array(coods)
        
        stellar_mass = np.array([g.masses['stellar'].value for g in obj.galaxies])
        
        if len(coods) == 0:
            continue
        
        # Angular size of box at this redshift
        L_unit = cosmo.kpc_comoving_per_arcmin(z_snap).to('Mpc / degree')
        A = (L_unit * area_deg2**0.5).value
        
        # Random offset within box
        xmin, ymin = np.random.rand(2) * (L - A)
        
        # Select galaxies in the lightcone area
        # Use x, y for RA/DEC and z for distance
        mask = (
            (coods[:, 0] > xmin) & (coods[:, 0] < xmin + A) &
            (coods[:, 1] > ymin) & (coods[:, 1] < ymin + A)
        )
        
        if verbose:
            print(f"  Selected {mask.sum()} of {len(coods)} galaxies")
        
        selected_coods = coods[mask]
        selected_mass = stellar_mass[mask]
        selected_idx = np.where(mask)[0]
        
        # Convert to RA, DEC (degrees from center of field)
        ra = (selected_coods[:, 0] - xmin - A/2) / A * area_deg2**0.5
        dec = (selected_coods[:, 1] - ymin - A/2) / A * area_deg2**0.5
        
        # Assign redshift based on z-coordinate + offset
        z_offset = cosmo.comoving_distance(z_snap).value
        galaxy_z = np.array([
            z_at_value(cosmo.comoving_distance, (z_offset + coord) * u.Mpc).value
            for coord in (selected_coods[:, 2] - L/2)  # center around middle of box
        ])
        
        # Filter to actual redshift range
        z_mask = (galaxy_z >= z_min) & (galaxy_z <= z_max)
        
        all_ra.extend(ra[z_mask])
        all_dec.extend(dec[z_mask])
        all_z.extend(galaxy_z[z_mask])
        all_snap.extend([snap_num] * z_mask.sum())
        all_idx.extend(selected_idx[z_mask])
        all_stellar_mass.extend(selected_mass[z_mask])
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('RA', data=np.array(all_ra))
        f.create_dataset('DEC', data=np.array(all_dec))
        f.create_dataset('z', data=np.array(all_z))
        f.create_dataset('snap', data=np.array(all_snap))
        f.create_dataset('galaxy_index', data=np.array(all_idx))
        f.create_dataset('stellar_mass', data=np.array(all_stellar_mass))
        
        # Metadata
        f.attrs['area_deg2'] = area_deg2
        f.attrs['z_min'] = z_min
        f.attrs['z_max'] = z_max
        f.attrs['n_galaxies'] = len(all_ra)
    
    if verbose:
        print(f"\n=== Lightcone saved to {output_file} ===")
        print(f"Total galaxies: {len(all_ra)}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a lightcone from m50n512')
    parser.add_argument('--area', type=float, default=1.0, help='Area in deg^2')
    parser.add_argument('--z_min', type=float, default=0.0, help='Min redshift')
    parser.add_argument('--z_max', type=float, default=3.0, help='Max redshift')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    
    if args.output is None:
        output_file = OUTPUT_DIR / f"lightcone_area{args.area}_z{args.z_min}-{args.z_max}.h5"
    else:
        output_file = Path(args.output)
    
    generate_lightcone(args.area, args.z_min, args.z_max, output_file)