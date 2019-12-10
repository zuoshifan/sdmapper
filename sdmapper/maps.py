import numpy as np
import cg
# from scipy.sparse import linalg as sla


class Maps(object):

    def __init__(self, nside=256, dtype=np.float64):
        if isinstance(nside, int) or (isinstance(nside, tuple) and len(nside) == 2):
            self.nside = nside
        else:
            raise ValueError('nside must be an integer or a tuple of two integers')
        self.dtype = dtype

    @property
    def m(self):
        """The clean map."""
        # cm = (A^T A)^-1 A^T d
        try:
            return self._m
        except AttributeError:
            print 'WARN: No map yet, cal `tod2map` to generate dirty map first'
            raise

    def set_map(self, m, pix):
        """Set the map and its corresponding pixels."""
        unique_pix = np.unique(pix)
        if len(m) == len(unique_pix):
            self._m = m
            self._pix = unique_pix
        else:
            raise ValueError("Map and pixel don't mach")

    def clear_map(self):
        """Clear the map."""
        del self._m

    @property
    def full_map(self):
        """The full healpix map."""
        if isinstance(self.nside, tuple):
            temp_m = np.full(self.nside, np.nan, self.dtype)
        else:
            temp_m = np.full(12*self.nside**2, np.nan, self.dtype)
        try:
            temp_m.flat[self._pix] = self.m
        except AttributeError:
            pass

        return temp_m

    def tod2map(self, tod, A, pix, normalize=True):
        """Make map from the given time ordered data."""
        self._pix = pix
        npix = len(self._pix)

        # dirty map: dm = A^T d
        self._m = A.T.dot(tod)

        if normalize:
            # clean map: cm = (A^T A)^-1 A^T d
            # self._m[:], _ = sla.cg(A.T.dot(A), self._m, self._m)
            self._m[:] = cg.cg_solver(np.zeros_like(self._m), Ax_func, b_func, args=(A, self._m), max_iter=10000, tol=0.0001)

    def map2tod(self, A, pix):
        """Resample the map to time ordered data, d = A m."""
        return A.dot(self.full_map.flat[pix])


def Ax_func(x, Ax, A, b):
    # (A^T A) x
    Ax[:] = A.T.dot(A.dot(x))

def b_func(x, A, b):
    # b = A^T d
    return b