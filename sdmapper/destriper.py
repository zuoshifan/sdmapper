import numpy as np
import cg
import maps


def Ax_func(a0, Aa, tod, tod_mask, A, pix, nside, bl):
    """Return Aa = A a0, where A = F^T Z F + 1 1^T."""
    nt, nb = tod.shape
    na  = np.int(np.ceil(nt / np.float(bl)))
    # d = Fa
    a0 = a0.reshape(-1, nb)
    Aa = Aa.reshape(-1, nb)
    d = np.repeat(a0, bl, axis=0)[:nt]

    m = maps.Maps(nside)
    # m = (A^T A)^-1 A^T d
    m.tod2map(d.reshape(-1), A, pix)
    # d1 = A m
    d1 = m.map2tod(A, pix).reshape(d.shape)

    for i in range(na):
        # F^T (d - d1) + 1 1^T a
        Aa[i, :] = np.sum(d[i*bl:(i+1)*bl] - d1[i*bl:(i+1)*bl], axis=0) + np.sum(a0, axis=0)
        # F^T (d - d1)
        # Aa[i] = np.sum(d[i*bl:(i+1)*bl] - d1[i*bl:(i+1)*bl], axis=0)

    Aa = Aa.reshape(-1)


def b_func(x0, tod, tod_mask, A, pix, nside, bl):
    """Return b = F^T Z d, where Z = I - A (A^T A)^-1 A^T."""
    nt, nb = tod.shape
    na  = np.int(np.ceil(nt / np.float(bl)))
    b = np.zeros((na, nb), dtype=tod.dtype)

    m = maps.Maps(nside)
    # m = (A^T A)^-1 A^T d
    if tod_mask is None:
        m.tod2map(tod.reshape(-1), A, pix)
    else:
        m.tod2map(tod.reshape(-1)[~tod_mask.reshape(-1)], A[~tod_mask.reshape(-1)], pix)
    # d1 = A m
    tod1 = m.map2tod(A, pix).reshape(tod.shape)

    for i in range(na):
        # F^T (d - d1)
        if tod_mask is not None:
            b[i, :] = np.ma.sum(np.ma.array(tod[i*bl:(i+1)*bl] - tod1[i*bl:(i+1)*bl], mask=tod_mask[i*bl:(i+1)*bl]), axis=0).filled(0)
        else:
            b[i, :] = np.sum(tod[i*bl:(i+1)*bl] - tod1[i*bl:(i+1)*bl], axis=0)

    return b.reshape(-1)


def maker(tod, tod_mask, A, pix, nside, bl, max_iter=200, tol=1.0e-6, verbose=False):
    if len(tod.shape) == 1:
        tod = tod.reshape(-1, 1)
        if tod_mask is not None:
            tod_mask = tod_mask.reshape(-1, 1)
        # pix = pix.reshape(-1, 1)
    elif len(tod.shape) == 2:
        pass
    else:
        raise ValueError('tod must be a 1D or 2D array')
    nt, nb = tod.shape
    na  = np.int(np.ceil(nt / np.float(bl)))
    a0 = np.zeros((na, nb), dtype=tod.dtype)
    for i in range(na):
        # initial guess of a
        if tod_mask is not None:
            a0[i, :] = np.ma.median(np.ma.array(tod[i*bl:(i+1)*bl], mask=tod_mask[i*bl:(i+1)*bl]), axis=0).filled(0)
        else:
            a0[i, :] = np.median(tod[i*bl:(i+1)*bl], axis=0)
    # solve for a = (F^T Z F + 1 1^T)^-1 F^T Z d
    a = cg.cg_solver(a0.reshape(-1), Ax_func, b_func, args=(tod, tod_mask, A, pix, nside, bl), max_iter=max_iter, tol=tol, verbose=verbose)
    a = a.reshape(a0.shape)
    # get F a
    Fa = np.repeat(a, bl, axis=0)[:nt]
    m = maps.Maps(nside)
    # m = (A^T A)^-1 A^T (d - F a)
    if tod_mask is None:
        m.tod2map((tod - Fa).reshape(-1), A, pix)
    else:
        m.tod2map((tod - Fa).reshape(-1)[~tod_mask.reshape(-1)], A[~tod_mask.reshape(-1)], pix)

    return m