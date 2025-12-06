'''
Author: Elko Gerville-Reache
Date Created: 2025-12-06
Date Modified: 2025-12-06
Description:
    WCS utility functions.
Dependencies:
    - astropy
    - numpy
Module Structure:
    - Data Transformations
        Lightweight data class for fits files.
'''

from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.units import Quantity, Unit
from astropy.wcs import WCS
import numpy as np
from .FitsFile import FitsFile


# Data Transformations
# ––––––––––––––––––––
def crop(self, size, position=None, mode='trim', frame='icrs', origin_idx=0):
    '''
    Crop the FitsFile object around a given position using WCS.
    This method creates a `Cutout2D` from the data, centered on a
    specified position in either pixel or world (RA/Dec) coordinates.
    It automatically handles cases where the WCS axes have been swapped
    due to a data transpose, and applies the same cropping to the
    associated error map if available.

    Parameters
    ––––––––––
    size : `~astropy.units.Quantity`, float, int, or tuple
        The size of the cutout region. Can be a single
        `Quantity` or a tuple specifying height and width.
        If a float or int, will interpret as number of pixels
        from center. If float, will round to nearest int.
        Ex:
            - 6 * u.arcsec
            - (6*u.deg, 4*u.deg)
            - (7, 8)
    position : array-like, `~astropy.coordinates.SkyCoord`, optional, default=None
        The center of the cutout region. Accepted formats are:
        - `(x, y)` : pixel coordinates (integers or floats)
        - `(ra, dec)` : sky coordinates as `~astropy.units.Quantity` in angular units
        - `~astropy.coordinates.SkyCoord` : directly specify a coordinate object
        - If None, defaults to the center of the image.
    mode : {'trim', 'partial', 'strict'}, default='trim'
        Defines how the function handles edges that fall outside the image:
        - 'trim': Trim the cutout to fit within the image bounds.
        - 'partial': Include all pixels that overlap the image, padded with NaNs.
        - 'strict': Raise an error if any part of the cutout is outside the image.
    frame : str, default='icrs'
        Coordinate frame for interpreting RA/Dec values when creating the `SkyCoord`.
    origin_idx : int, default=0
        Origin index for pixel-to-world conversion (0 for 0-based, 1 for 1-based).

    Returns
    –––––––
    cropped : FitsFile
        A new `FitsFile` instance containing:
        - data : Cropped image as a `np.ndarray`
        - header : Original FITS header
        - error : Cropped error array (if available)
        - wcs : Updated WCS corresponding to the cutout region

    Raises
    ––––––
    ValueError
        If the WCS is missing (None) or cutout creation fails.
    TypeError
        If the position is not one of the accepted types.

    Notes
    –––––
    - If the data were transposed and the WCS was swapped via `wcs.swapaxes(0, 1)`,
        the method will automatically attempt to correct for inverted RA/Dec axes.
    - The same cutout region is applied to the error array if present.

    Examples
    ––––––––
    Crop by pixel coordinates:
        >>> cube.crop(size=100, position=(250, 300))

    Crop by pixel coordinates:
        >>> cube.crop(size=6*u.arcsec, position=(250, 300))

    Crop by sky coordinates:
        >>> from astropy import units as u
        >>> cube.crop(size=6*u.arcsec, position=(83.8667*u.deg, -69.2697*u.deg))

    Crop using a SkyCoord object:
        >>> from astropy.coordinates import SkyCoord
        >>> c = SkyCoord(ra=83.8667*u.deg, dec=-69.2697*u.deg, frame='icrs')
        >>> cube.crop(size=6*u.arcsec, position=c)
    '''
    data = self.data
    error = self.error
    wcs = self.wcs

    if wcs is None:
        raise ValueError (
            'WCS is None. Please crop using normal array indexing or enter WCS.'
        )
    # if no position passed in use center of image
    if position is None:
        ny, nx = data.shape
        position = [nx//2, ny//2]
    # if position is passed in as integers, assume pixel indeces
    # and convert to world coordinates
    if isinstance(position[0], (float, int)) and isinstance(position[1], (float, int)):
        ra, dec = wcs.wcs_pix2world(position[0], position[1], origin_idx)
        ra *= u.deg
        dec *= u.deg
    # if position passed in as Quantity, convert to degrees
    elif (
        len(position) == 2 and
        all(isinstance(p, Quantity) for p in position)
    ):
        ra = position[0].to(u.deg)
        dec = position[1].to(u.deg)
    # if position passed in as SkyCoord, use that
    elif isinstance(position, SkyCoord):
        center = position
    else:
        raise TypeError(
            'Position must be a (x, y) pixel tuple, (ra, dec) in degrees, or a SkyCoord.'
        )
    # crop image
    if not isinstance(position, SkyCoord):
        # try with ra=ra and dec=dec and if it fails; swap
        # this is because the user maybe performed wcs.swapaxes(0,1)
        try:
            center = SkyCoord(ra=ra, dec=dec, frame=frame)
            cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
        except ValueError:
            # fallback if WCS RA/Dec swapped
            center = SkyCoord(ra=dec, dec=ra, frame=frame)
            cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
    else:
        cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
    # also crop the errors if available
    if error is not None:
        error = Cutout2D(error, position=center, size=size, wcs=wcs, mode=mode).data

    return FitsFile(cutout.data, header=self.header, error=error, wcs=cutout.wcs)
