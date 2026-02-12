from pathlib import Path
import yaml
from dataclasses import dataclass
from astropy.cosmology import Planck15

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Map string names to astropy cosmology objects
_COSMOLOGIES = {
    "Planck15": Planck15,
}


@dataclass
class SimConfig:
    name: str
    box_size_mpc_h: float
    n_particles: int
    catalogue_dir: Path
    hdf5_dir: Path
    snapshot_prefix: str
    n_snapshots: int
    cosmology: object

    def caesar_path(self, snap: int) -> Path:
        return self.catalogue_dir / f"{self.snapshot_prefix}_{snap:03d}.hdf5"

    def hdf5_path(self, snap: int) -> Path:
        """Try zero-padded first, then unpadded."""
        p = self.hdf5_dir / f"{self.snapshot_prefix}_{snap:03d}.hdf5"
        if p.exists():
            return p
        return self.hdf5_dir / f"{self.snapshot_prefix}_{snap}.hdf5"


def load_config(sim_name: str = "m25n256") -> SimConfig:
    """Load a simulation config by name."""
    cfg_path = CONFIG_DIR / f"{sim_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config found at {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    return SimConfig(
        name=raw["name"],
        box_size_mpc_h=raw["box_size_mpc_h"],
        n_particles=raw["n_particles"],
        catalogue_dir=Path(raw["catalogue_dir"]),
        hdf5_dir=Path(raw["hdf5_dir"]),
        snapshot_prefix=raw["snapshot_prefix"],
        n_snapshots=raw["n_snapshots"],
        cosmology=_COSMOLOGIES[raw["cosmology"]],
    )