import numpy as np
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

# healpix nside
nside = 256

tod -= np.median(tod)
tod_mask = None
m = ml.maker(tod, tod_mask, pix, nside, ml_iter=6, max_iter=200, verbose=True)


# plot the map
fig = plt.figure(1, figsize=(13, 5))
hp.mollview(m.full_map, fig=1, min=-10, max=20, title='Maximum-Likelihood Map')
hp.graticule()
fig.savefig('ml_map.png')
fig.clf()