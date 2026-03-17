import re
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
import caesar

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

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

def save_background_results(
    cfg, args,
    # Optical
    lam_opt, I_nu_opt, nuInu_opt_nW,
    I_nu_opt_nodust, nuInu_opt_nodust_nW,
    # Far-IR
    lam_fir, I_lam_fir, nuInu_fir_nW,
    # Radio
    nu_radio, I_nu_radio, lam_radio_um, nuInu_radio_nW,
    # Optional diagnostics
    dust_temps=None, dust_redshifts=None,
    # Physics parameters
    a_dust=0.0807, beta=2.0,
):
    """
    Save all computed background spectra to HDF5 for later analysis.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    args : argparse.Namespace
        Command-line arguments (area, z_min, z_max).
    lam_opt, I_nu_opt, nuInu_opt_nW : arrays
        Optical/NIR wavelength (AA), intensity (erg/s/cm²/Hz/sr), νIν (nW/m²/sr).
    lam_fir, I_lam_fir, nuInu_fir_nW : arrays
        Far-IR wavelength (AA), intensity (erg/s/cm²/sr/AA), νIν (nW/m²/sr).
    nu_radio, I_nu_radio, lam_radio_um, nuInu_radio_nW : arrays
        Radio frequency (Hz), intensity, wavelength (µm), νIν (nW/m²/sr).
    dust_temps, dust_redshifts : arrays, optional
        Per-galaxy dust temperatures and redshifts for diagnostics.
    a_dust, beta : float
        Dust model parameters.
    
    Returns
    -------
    Path : path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fname = f"bg_{cfg.name}_a{args.area}_z{args.z_min}-{args.z_max}.h5"
    outpath = RESULTS_DIR / fname
    
    with h5py.File(outpath, "w") as f:
        # ── Metadata ──
        meta = f.create_group("metadata")
        meta.attrs["sim"] = cfg.name
        meta.attrs["area_deg2"] = args.area
        meta.attrs["z_min"] = args.z_min
        meta.attrs["z_max"] = args.z_max
        meta.attrs["a_dust"] = a_dust
        meta.attrs["beta"] = beta
        meta.attrs["created"] = datetime.now().isoformat()
        
        # ── Optical / NIR ──
        opt = f.create_group("optical")
        opt.create_dataset("lam_AA", data=lam_opt)
        opt.create_dataset("I_nu", data=I_nu_opt)
        opt.create_dataset("nuInu_nW", data=nuInu_opt_nW)
        opt.create_dataset("I_nu_nodust", data=I_nu_opt_nodust)
        opt.create_dataset("nuInu_nodust_nW", data=nuInu_opt_nodust_nW)

        # ── Far-IR ──
        fir = f.create_group("farIR")
        fir.create_dataset("lam_AA", data=lam_fir)           # Angstrom
        fir.create_dataset("I_lam", data=I_lam_fir)          # erg/s/cm²/sr/AA
        fir.create_dataset("nuInu_nW", data=nuInu_fir_nW)    # nW/m²/sr
        
        # ── Radio ──
        rad = f.create_group("radio")
        rad.create_dataset("nu_Hz", data=nu_radio)
        rad.create_dataset("I_nu", data=I_nu_radio)
        rad.create_dataset("lam_um", data=lam_radio_um)
        rad.create_dataset("nuInu", data=nuInu_radio)
        
        # ── Dust diagnostics (for T vs z plots) ──
        if dust_temps is not None and len(dust_temps) > 0:
            diag = f.create_group("diagnostics")
            diag.create_dataset("dust_temps", data=np.asarray(dust_temps))
            diag.create_dataset("dust_redshifts", data=np.asarray(dust_redshifts))
    
    print(f"Results saved → {outpath}")
    return outpath


def load_background_results(cfg_name, area, z_min, z_max):
    """
    Load previously computed background spectra.

    Parameters
    ----------
    cfg_name : str
        Simulation name (e.g., "m25n256").
    area, z_min, z_max : float
        Lightcone parameters.

    Returns
    -------
    dict : nested dictionary with all saved data
    """
    fname = f"bg_{cfg_name}_a{area}_z{z_min}-{z_max}.h5"
    path = RESULTS_DIR / fname

    if not path.exists():
        raise FileNotFoundError(f"No cached results at {path}")

    data = {}
    with h5py.File(path, "r") as f:
        # Metadata
        data["metadata"] = dict(f["metadata"].attrs)

        # Optical
        data["optical"] = {
            "lam_AA": f["optical/lam_AA"][:],
            "I_nu": f["optical/I_nu"][:],
            "nuInu_nW": f["optical/nuInu_nW"][:],
            "I_nu_nodust": f["optical/I_nu_nodust"][:],
            "nuInu_nodust_nW": f["optical/nuInu_nodust_nW"][:],
        }

        # Far-IR
        data["farIR"] = {
            "lam_AA": f["farIR/lam_AA"][:],
            "I_lam": f["farIR/I_lam"][:],
            "nuInu_nW": f["farIR/nuInu_nW"][:]
        }

        # Radio (handle old cache files that used "nuInu" instead of "nuInu_nW")
        radio_grp = f["radio"]
        if "nuInu_nW" in radio_grp:
            radio_nuInu = radio_grp["nuInu_nW"][:]
        elif "nuInu" in radio_grp:
            radio_nuInu = radio_grp["nuInu"][:] * 1e6   # erg/s/cm²/sr → nW/m²/sr
        else:
            # Recompute from raw I_nu if neither key exists
            nu_hz = radio_grp["nu_Hz"][:]
            I_nu  = radio_grp["I_nu"][:]
            radio_nuInu = nu_hz * I_nu * 1e6             # erg/s/cm²/sr → nW/m²/sr
        data["radio"] = {
            "nu_Hz": radio_grp["nu_Hz"][:],
            "I_nu": radio_grp["I_nu"][:],
            "lam_um": radio_grp["lam_um"][:],
            "nuInu_nW": radio_nuInu
        }

        # Diagnostics if present
        if "diagnostics" in f:
            data["diagnostics"] = {
                "dust_temps": f["diagnostics/dust_temps"][:],
                "dust_redshifts": f["diagnostics/dust_redshifts"][:]
            }

    print(f"Loaded results from {path}")
    return data


def list_cached_results():
    """List all cached background result files."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("bg_*.h5"))


def save_farIR_parameter_sweep(cfg_name, z_min, z_max, a_values, results_dict):
    """
    Save far-IR results for multiple a_dust values (for parameter optimization).
    
    Parameters
    ----------
    cfg_name : str
        Simulation name.
    z_min, z_max : float
        Redshift range.
    a_values : array-like
        List of a_dust values tested.
    results_dict : dict
        {a_dust: (lam_fir, nuInu_fir_nW), ...}
    
    Returns
    -------
    Path : path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"farIR_sweep_{cfg_name}_z{z_min}-{z_max}.h5"
    outpath = RESULTS_DIR / fname
    
    with h5py.File(outpath, "w") as f:
        f.attrs["a_values"] = np.array(a_values)
        f.attrs["sim"] = cfg_name
        f.attrs["z_min"] = z_min
        f.attrs["z_max"] = z_max
        f.attrs["created"] = datetime.now().isoformat()
        
        for a in a_values:
            grp = f.create_group(f"a_{a:.4f}".replace("-", "m").replace(".", "p"))
            lam, nuInu = results_dict[a]
            grp.create_dataset("lam_AA", data=lam)
            grp.create_dataset("nuInu_nW", data=nuInu)
            grp.attrs["a_dust"] = a
    
    print(f"Parameter sweep saved → {outpath}")
    return outpath


def load_farIR_parameter_sweep(cfg_name, z_min, z_max):
    """Load a far-IR parameter sweep."""
    fname = f"farIR_sweep_{cfg_name}_z{z_min}-{z_max}.h5"
    path = RESULTS_DIR / fname
    
    if not path.exists():
        raise FileNotFoundError(f"No sweep file at {path}")
    
    data = {"a_values": [], "spectra": {}}
    with h5py.File(path, "r") as f:
        data["a_values"] = f.attrs["a_values"][:]
        data["z_min"] = f.attrs["z_min"]
        data["z_max"] = f.attrs["z_max"]
        
        for key in f.keys():
            if key.startswith("a_"):
                a = f[key].attrs["a_dust"]
                data["spectra"][a] = {
                    "lam_AA": f[key]["lam_AA"][:],
                    "nuInu_nW": f[key]["nuInu_nW"][:]
                }
    
    print(f"Loaded sweep from {path}")
    return data