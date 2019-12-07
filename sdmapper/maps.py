import numpy as np


class Maps(object):

    def __init__(self, nside=256, dtype=np.float64):
        self.nside = nside
        self.dtype = dtype

    @property
    def m(self):
        """The clean map."""
        # cm = (A^T A)^-1 A^T d
        try:
            return self._cm
        except AttributeError:
            print 'WARN: No map yet, cal `tod2map` to generate dirty map first'
            raise

    def set_map(self, m, pix):
        """Set the map and its corresponding pixels."""
        unique_pix = np.unique(pix)
        if len(m) == len(unique_pix):
            self._cm = m
            self._pix = unique_pix
        else:
            raise ValueError("Map and pixel don't mach")

    def clear_map(self):
        """Clear the map."""
        del self._cm

    @property
    def full_map(self):
        """The full healpix map."""
        temp_m = np.full(12*self.nside**2, np.nan, self.dtype)
        try:
            temp_m[self._pix] = self.m
        except AttributeError:
            pass

        return temp_m

    def tod2map(self, tod, tod_mask, pix):
        """Make map from the given time ordered data."""
        self._pix, pix_inverse = np.unique(pix, return_inverse=True)

        npix = len(self._pix)
        # to save the dirty map dm = A^T d
        dm = np.zeros(npix, dtype=self.dtype)
        # to save the count of each pixel
        c = np.zeros(npix, dtype=np.int)

        for i in range(len(pix)):
            if (tod_mask is None) or (not tod_mask[i]):
                pi = pix_inverse[i]
                dm[pi] += tod[i]
                c[pi] += 1

        # the clean map
        self._cm = np.where(c > 0, dm / c, np.nan)

    def map2tod(self, pix):
        """Resample the map to time ordered data, d = A m."""
        return self.full_map[pix]
