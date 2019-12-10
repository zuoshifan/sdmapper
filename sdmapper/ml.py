import numpy as np
import scipy.interpolate as interp
from scipy import signal
import cg
import maps


def Ax_func(m0, Am, tod, psd, A, pix, nside):
    """Return Am = A^T N^-1 A m0."""
    m = maps.Maps(nside)
    m.set_map(m0, pix)
    # d = A m
    d = m.map2tod(A, pix)

    # N^-1 d = IF(F d / psd)
    Nid = np.fft.ifft(np.fft.fft(d) / psd).real

    # get A^T N^-1 d
    m.tod2map(Nid, A, pix, normalize=False)
    Am[:] = m.m


def b_func(x0, tod, psd, A, pix, nside):
    """Return b = A^T N^-1 d."""
    # N^-1 d = IF(F d / psd)
    Nid = np.fft.ifft(np.fft.fft(tod) / psd).real

    m = maps.Maps(nside)
    # get A^T N^-1 d
    m.tod2map(Nid, A, pix, normalize=False)

    return m.m


def maker(tod, tod_mask, A, pix, nside, ml_iter=3, max_iter=200, tol=1.0e-6, verbose=False):
    tod = tod.reshape(-1)

    if tod_mask is not None:
        tod_mask = tod_mask.reshape(-1)
        # drop masked values
        tod = tod[~tod_mask]
        A = A[~tod_mask]
        tod_mask = None

    m = maps.Maps(nside)
    # m = (A^T A)^-1 A^T d
    m.tod2map(tod, A, pix)

    # tod1 = A m
    tod1 = m.map2tod(A, pix)

    # inital estimate of the noise
    n = tod - tod1

    # power spectral density of the noise
    f = np.fft.fftfreq(len(n))
    # f1, p1 = signal.welch(n, nperseg=2**12, return_onesided=False)
    f1, p1 = signal.welch(n, return_onesided=False)
    psd = interp.interp1d(f1, p1, bounds_error=False, fill_value='extrapolate', kind='linear')(f)

    # outer iteration
    for i in range(ml_iter):
        if verbose:
            print 'Outer iteration %d of %d...\n' % (i+1, ml_iter)
        # solve for m = (A^T N^-1 A)^-1 A^T N^-1 d
        m_est = cg.cg_solver(m.m, Ax_func, b_func, args=(tod, psd, A, pix, nside), max_iter=max_iter, tol=tol, verbose=verbose)
        m.set_map(m_est, pix)
        # tod1 = A m
        tod1 = m.map2tod(A, pix)
        # update the estimate of n
        n = tod - tod1

        # f1, p1 = signal.welch(n, nperseg=2**12, return_onesided=False)
        f1, p1 = signal.welch(n, return_onesided=False)
        psd = interp.interp1d(f1, p1, bounds_error=False, fill_value='extrapolate', kind='linear')(f)

    return m