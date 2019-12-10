import numpy as np
from scipy import sparse
import pyfits
import healpy as hp
from sdmapper import destriper
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

# destripping baseline length
bl = 250

# healpix nside
nside = 256

# map-making
tod -= np.median(tod)
tod_mask = None
m = destriper.maker(tod, tod_mask, A, pix, nside, bl, max_iter=200, verbose=True)


# plot the map
fig = plt.figure(1, figsize=(13, 5))
hp.mollview(m.full_map, fig=1, min=-10, max=20, title='Destriped Map')
hp.graticule()
fig.savefig('destriped_map.png')
fig.clf()