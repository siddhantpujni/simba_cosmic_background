import re
import numpy as np
import caesar


def get_redshift(obj):
    """Get redshift from a Caesar object."""
    for attr in ["redshift", "z"]:
        if hasattr(obj.simulation, attr):
            return float(getattr(obj.simulation, attr))


def list_snapshots(cfg):
    """
    List available snapshots sorted by snap number (descending).
    
    Parameters
    ----------
    cfg : SimConfig
    
    Returns
    -------
    list of (snap_number, Path)
    """
    files = sorted(cfg.catalogue_dir.glob(f"{cfg.snapshot_prefix}_*.hdf5"))
    snaps = []
    for f in files:
        m = re.search(r"_(\d+)\.hdf5$", f.name)
        if m:
            snaps.append((int(m.group(1)), f))
    snaps.sort(key=lambda t: t[0], reverse=True)
    return snaps