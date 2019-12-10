import numpy as np
from scipy import sparse
import pyfits
import healpy as hp
from sdmapper import ml
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# load the test data
hdu = pyfits.open('./test_data.fits')
tod = hdu[1].data['TOD'][0,:,0]
pix = hdu[1].data['PIX'][0,:,0].astype(np.int)

# construct A matrix
nt = len(tod)
pix, pix_inv = np.unique(pix, return_inverse=True)
npix = len(pix)
A = sparse.csr_matrix((np.ones(nt), (np.arange(nt), pix_inv)), shape=(nt, npix), dtype=tod.dtype)

# healpix nside
nside = 256

# map-making
tod -= np.median(tod)
tod_mask = None
m = ml.maker(tod, tod_mask, A, pix, nside, ml_iter=3, max_iter=200, verbose=True)


# plot the map
fig = plt.figure(1, figsize=(13, 5))
hp.mollview(m.full_map, fig=1, min=-10, max=20, title='Maximum-Likelihood Map')
hp.graticule()
fig.savefig('ml_map1.png')
fig.clf()