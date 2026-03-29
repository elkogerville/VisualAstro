from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest


@pytest.fixture
def generate_test_cube() -> fits.PrimaryHDU:
    """
    Generate a small 3D test cube along along with
    a valid WCS. The 3rd dimension is a spectral axis.

    Returns
    -------
    hdu : astropy.io.fits.PrimaryHDU
        HDU object containing the data and header extensions.

    """
    data = np.random.randn(10, 50, 50) + 5

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [25, 25, 1]
    wcs.wcs.cdelt = [1e-5, 1e-5, 0.01]
    wcs.wcs.crval = [83.866, -69.270, 6.5]
    wcs.wcs.cunit = ['deg', 'deg', 'um']
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']

    header = wcs.to_header()
    header['BUNIT'] = 'MJy / sr'

    hdu = fits.PrimaryHDU(data=data, header=header)

    data = hdu.data
    header = hdu.header

    assert isinstance(data, np.ndarray)
    assert isinstance(header, fits.Header)

    return hdu


@pytest.fixture
def generate_test_image():
    """Generate a 2D test image."""
    data = np.random.randn(100, 100) + 10

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.cdelt = [1e-5, 1e-5]
    wcs.wcs.crval = [83.866, -69.270]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = wcs.to_header()
    header['BUNIT'] = 'Jy/beam'

    hdu = fits.PrimaryHDU(data=data, header=header)
    return hdu
