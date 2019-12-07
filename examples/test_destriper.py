import numpy as np
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

# destripping baseline length
bl = 250

# healpix nside
nside = 256

tod -= np.median(tod)
tod_mask = None
m = destriper.maker(tod, tod_mask, pix, nside, bl, max_iter=200, verbose=True)


# plot the map
fig = plt.figure(1, figsize=(13, 5))
hp.mollview(m.full_map, fig=1, min=-10, max=20, title='Destriped Map')
hp.graticule()
fig.savefig('destriped_map.png')
fig.clf()