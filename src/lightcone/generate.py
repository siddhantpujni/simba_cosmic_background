"""
Generate a lightcone from Simba snapshots.

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
    """Get redshift and box size (comoving Mpc) from a Caesar catalogue."""
    try:
        obj = caesar.load(str(path))
    except Exception:
        return None, None, None

    z = obj.simulation.redshift
    L = obj.simulation.boxsize.to('kpccm').value / 1000.0 # convert to Mpc

    return z, L, obj


def generate_lightcone(cfg, area_deg2, z_min, z_max, output_file=None,
                       snap_step=2, midsnap=False, verbose=True):
    """
    Generate a lightcone catalogue for any simulation.

    Uses the frustum-based method to ensure box-size independence.

    Parameters
    ----------
    cfg        : SimConfig
    area_deg2  : float
    z_min      : float
    z_max      : float
    output_file: Path or None (auto-named)
    snap_step  : int
        Use every snap_step'th snapshot.  For the 100 Mpc/h box the
        comoving box length is ~2× the path length between consecutive
        outputs, so snap_step=2 gives back-to-back shells with no
        double-counting (matching the original lightcone.py behaviour).
        Set to 1 to use every snapshot (appropriate for smaller boxes).
    midsnap    : bool
        If True, use even-numbered snapshots (0, 2, 4, …) and centre
        the comoving offset on the intermediate (skipped) snapshot.
        If False (default), use odd-numbered snapshots (1, 3, 5, …)
        with the offset at each snapshot's own redshift.
    verbose    : bool
    """
    if output_file is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / (
            f"lc_{cfg.name}_a{area_deg2}_z{z_min}-{z_max}.h5"
        )

    snaps = list_snapshots(cfg)

    # ------------------------------------------------------------------
    # Load info for ALL snapshots first so that the midsnap z_offset
    # lookup can reference the intermediate (skipped) snapshot later.
    # ------------------------------------------------------------------
    all_snap_info = {}          # snap_num -> (z, L)
    snap_data = []
    for snap_num, path in snaps:
        z, L, _ = get_snapshot_info(path)
        if z is None:
            if verbose:
                print(f"Skipping snap {snap_num} (no halo data)")
            continue
        all_snap_info[snap_num] = (z, L)
        snap_data.append((snap_num, path, z, L))

    # ------------------------------------------------------------------
    # Select every snap_step'th snapshot to avoid double-counting.
    # For the 100 Mpc/h box the comoving size ≈ 2× the inter-snapshot
    # path length, so stacking every other snap (e.g. 1, 3, 5, …)
    # gives exactly back-to-back shells in redshift space.
    #   midsnap=False (default) → odd  set: 1, 3, 5, …, 149
    #   midsnap=True            → even set: 0, 2, 4, …, 150
    # This mirrors the scheme in the original lightcone.py.
    # ------------------------------------------------------------------
    if snap_step > 1:
        if midsnap:
            valid_snaps = set(range(0, cfg.n_snapshots, snap_step))
        else:
            valid_snaps = set(range(1, cfg.n_snapshots + 1, snap_step))
        snap_data = [s for s in snap_data if s[0] in valid_snaps]

    # Sort by redshift (ascending)
    snap_data = sorted(snap_data, key=lambda x: x[2])

    # Build redshift mask with one-snapshot buffer on each side
    zeds = np.array([s[2] for s in snap_data])
    zeds_mask = np.where((zeds >= z_min) & (zeds <= z_max))[0]

    if len(zeds_mask) == 0:
        raise ValueError("No valid snapshots found in redshift range!")

    if np.min(zeds_mask) > 0:
        zeds_mask = np.insert(zeds_mask, 0, np.min(zeds_mask) - 1)
    if np.max(zeds_mask) < len(snap_data) - 2:
        zeds_mask = np.append(zeds_mask, np.max(zeds_mask) + 1)
    if np.max(zeds_mask) > len(snap_data) - 2:
        zeds_mask = zeds_mask[zeds_mask < len(snap_data) - 1]

    snap_data = [snap_data[i] for i in zeds_mask]

    if verbose:
        print(f"\nUsing {len(snap_data)} snapshots (snap_step={snap_step}, "
              f"midsnap={midsnap})")
        if snap_data:
            print(f"Redshift range: {snap_data[0][2]:.2f} to "
                  f"{snap_data[-1][2]:.2f}")

    # Check area can be covered by the box at all redshifts
    cosmo = cfg.cosmology
    L = snap_data[0][3]

    for _, _, z_s, L_s in snap_data:
        L_unit = cosmo.kpc_comoving_per_arcmin(z_s).to('Mpc / degree')
        A_check = (L_unit * area_deg2 ** 0.5).value
        if A_check > L_s:
            raise ValueError(
                'Specified area too large for simulation box '
                '(lateral tiling not yet implemented)'
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ra, all_dec, all_z = [], [], []
    all_snap, all_idx, all_stellar_mass = [], [], []

    A_A = 0.0  # previous snapshot's A for frustum continuity

    for idx, (snap_num, path, z_snap, L) in enumerate(snap_data):
        if verbose:
            print(f"\nProcessing snap {snap_num}, z={z_snap:.3f}")

        obj = caesar.load(str(path))

        # Comoving coordinates
        coods = np.array([g.pos.to('kpccm').value / 1000.0 for g in obj.galaxies])
        stellar_mass = np.array(
            [g.masses['stellar'].value for g in obj.galaxies]
        )

        if len(coods) == 0:
            continue

        # Use next snapshot's redshift for the far edge of the shell
        if idx + 1 < len(snap_data):
            z_B = snap_data[idx + 1][2]
        else:
            z_B = z_snap

        L_unit = cosmo.kpc_comoving_per_arcmin(z_B).to('Mpc / degree')
        A = (L_unit * area_deg2 ** 0.5).value

        # ---- comoving offset along the line of sight ----
        z_offset = cosmo.comoving_distance(z_snap).value
        if midsnap:
            # Use the intermediate (skipped) snapshot's redshift as the
            # shell offset — centres the snapshot in its redshift range,
            # matching the original lightcone.py behaviour.
            mid_snap_num = snap_num + 1
            if mid_snap_num in all_snap_info:
                z_mid = all_snap_info[mid_snap_num][0]
                z_offset = cosmo.comoving_distance(z_mid).value

        if verbose:
            print(f"  z_offset: {z_offset:.2f}")

        # Randomly choose axes — matches lightcone.py
        i_ax = np.random.randint(0, 3)
        j_ax = i_ax
        while j_ax == i_ax:
            j_ax = np.random.randint(0, 3)
        k_ax = np.where(
            (np.arange(0, 3) != i_ax) & (np.arange(0, 3) != j_ax)
        )[0][0]

        xmin, ymin = np.random.rand(2) * (L - A)

        if verbose:
            print(f"  xmin: {xmin:.2f}, ymin: {ymin:.2f}, A: {A:.2f}, L: {L:.2f}")

        # Frustum geometry — the key to box-size independence
        theta = np.arctan((A - A_A) / (2 * L))
        dx = np.abs(L - coods[:, k_ax]) * np.tan(theta)

        # Frustum selection
        mask = (
            (coods[:, i_ax] > (xmin + dx)) &
            (coods[:, i_ax] < ((xmin + A) - dx)) &
            (coods[:, j_ax] > (ymin + dx)) &
            (coods[:, j_ax] < ((ymin + A) - dx))
        )

        if verbose:
            print(f"  N(lightcone cut): {mask.sum()}")

        lc_idx_arr = np.where(mask)[0]
        selected_coods = coods[lc_idx_arr]
        selected_mass = stellar_mass[lc_idx_arr]

        # RA/DEC with frustum correction — matches lightcone.py exactly
        _frac_ra = (np.abs(selected_coods[:, i_ax] - xmin - (A / 2))
                    / ((A / 2) - dx[lc_idx_arr]))
        ra = _frac_ra * ((A * u.Mpc) / L_unit).decompose().value

        _frac_dec = (np.abs(selected_coods[:, j_ax] - ymin - (A / 2))
                     / ((A / 2) - dx[lc_idx_arr]))
        dec = _frac_dec * ((A * u.Mpc) / L_unit).decompose().value

        # Redshift from depth axis — no L/2 centring, matches lightcone.py
        galaxy_z = np.array([
            float(z_at_value(cosmo.comoving_distance, c * u.Mpc))
            for c in (selected_coods[:, k_ax] + z_offset)
        ])

        # Strict inequality — matches lightcone.py
        z_mask = (galaxy_z > z_min) & (galaxy_z < z_max)

        all_ra.extend(ra[z_mask])
        all_dec.extend(dec[z_mask])
        all_z.extend(galaxy_z[z_mask])
        all_snap.extend([snap_num] * z_mask.sum())
        all_idx.extend(lc_idx_arr[z_mask])
        all_stellar_mass.extend(selected_mass[z_mask])

        # Update A_A for next iteration's frustum
        A_A = A

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
        f.attrs['snap_step'] = snap_step
        f.attrs['midsnap'] = midsnap

    if verbose:
        print(f"\n=== Lightcone saved to {output_file} ===")
        print(f"Total galaxies: {len(all_ra)}")

    return output_file