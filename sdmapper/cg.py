"""
Implementation of the Conjugate Gradients algorithm.

Solve x for a linear equation Ax = b.

Inputs:

A function which describes Ax <-- Modify this vector -in place-.
An array which describes b <-- Returns a column vector.
An initial guess for x.

"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def cg_solver(x0, Ax_func, b_func, args=(), max_iter=200, tol=1.0e-6, verbose=False):
    """
    Returns x for a linear system Ax = b where A is a symmetric, positive-definite matrix.

    Parameters
    ----------
    x0 : np.ndarray
        Initial guess of the solution `x`.
    Ax_func : function
        Function returns the product of `A` and `x`, has form Ax_func(x, Ax, *args),
        the result is returned via the inplace modification of `Ax`.
    b_func : function
        Function returns the right hand side vector `b`, has form b_func(x, *args),
    args : tuple, optional
        Arguments that will be pass to `Ax_func` and `b_func`.
    max_iter : interger, optional
        Maximum number of iteration.
    tol : float, optional
        Tolerance of the solution error.
    verbose : boolean, optional
        Output verbose information.

    Returns
    -------
    x : np.ndarray
        The solution `x`.
    """

    # get value of b
    b = b_func(x0, *args)

    # initial guess of Ax
    Ax = np.zeros_like(b)
    # Ad = np.zeros_like(b)
    Ax_func(x0, Ax, *args)

    x = x0 # re-use x0
    r = b - Ax
    d = r.copy()

    delta_new = np.dot(r, r)
    delta0 = delta_new

    it = 0
    while it < max_iter and delta_new > tol**2 * delta0:
        Ax_func(d, Ax, *args) # re-use Ax for Ad
        alpha = delta_new / np.dot(d, Ax) # re-use Ax for Ad
        x += alpha * d

        if (it + 1) % max(50, np.int(it**0.5)) == 0:
            Ax_func(x, Ax, *args)
            r[:] = b - Ax # re-use the existing r
        else:
            r -= alpha * Ax # re-use Ax for Ad

        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        # d[:] = r + beta * d
        d *= beta
        d += r

        it += 1

        if verbose:
            sgn = '>' if delta_new > tol**2 * delta0 else '<'
            print 'Iteration %d of %d, %g %s %g...' % (it, max_iter, (delta_new/delta0)**0.5, sgn, tol)

    if delta_new > tol**2 * delta0:
        print 'Iteration %d of %d, %g > %g...' % (it, max_iter, (delta_new/delta0)**0.5, tol)

    return x
