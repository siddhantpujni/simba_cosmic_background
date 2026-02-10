import numpy as np
import h5py
import caesar
from astropy.cosmology import Planck15 as cosmo


class simba_m25n256:
    """Adapted simba class for m25n256 simulation."""
    
    def __init__(self):
        self.cs_directory = '/home/spujni/sim/m25n256/s50/Groups/'
        self.cosmo = cosmo
        
        # Box size for m25n256 is 25 Mpc/h
        self.box_size = 25.0 
    
    def get_caesar(self, snap, verbose=False):
        """Load a Caesar catalogue by snapshot number."""
        fname = self.cs_directory + f'm25n256_{snap}.hdf5'
        return caesar.load(fname)
    
    def get_snapshot_redshifts(self):
        """Return dict mapping snapshot number to redshift."""
        redshifts = {}
        from pathlib import Path
        for f in Path(self.cs_directory).glob('m25n256_*.hdf5'):
            snap = int(f.stem.split('_')[1])
            obj = caesar.load(str(f))
            redshifts[snap] = obj.simulation.redshift
        return redshifts
    
    #HDF5 helper methods from original
    def _check_hdf5(self, fname, obj_str):
        with h5py.File(fname, 'a') as h5file:
            return obj_str in h5file

    def create_dataset(self, fname, values, name, group='/', overwrite=False,
                       dtype=np.float64, desc=None, unit=None, verbose=False):
        shape = np.shape(values)

        if self._check_hdf5(fname, group) is False:
            raise ValueError("Group does not exist")

        with h5py.File(fname, mode='a') as h5f:
            if overwrite and self._check_hdf5(fname, f"{group}/{name}"):
                if verbose:
                    print(f'Overwriting data in {group}/{name}')
                del h5f[group][name]

            dset = h5f.create_dataset(
                f"{group}/{name}", 
                shape=shape,
                maxshape=(None,) + shape[1:],
                dtype=dtype,
                data=values
            )
            if desc is not None:
                dset.attrs['Description'] = desc
            if unit is not None:
                dset.attrs['Units'] = unit