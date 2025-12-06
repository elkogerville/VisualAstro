'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-07
Description:
    Utility functions for DataCube manipulations.
Dependencies:
    - astropy
    - numpy
    - regions
Module Structure:
    - Cube Manipulation Functions
        Utility functions used when manipulating datacubes numerically.
    - Cube/Image Masking Functions
        Utility functions used when masking datacubes.
'''

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from regions import PixCoord, EllipsePixelRegion, EllipseAnnulusPixelRegion
from .numerical_utils import get_data, get_units
from .DataCube import DataCube
from .FitsFile import FitsFile


# Cube Manipulation Functions
# –––––––––––––––––––––––––––
def slice_cube(cube, idx):
    '''
    Return a slice of a data cube along the first axis.
    Parameters
    ––––––––––
    cube : np.ndarray
        Input data cube, typically with shape (T, N, ...) where T is the first axis.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    Returns
    –––––––
    cube : np.ndarray
        Sliced cube with shape (N, ...).
    '''
    cube = get_data(cube)
    # if index is integer
    if isinstance(idx, int):
        return cube[idx]
    # if index is list of integers
    elif isinstance(idx, list):
        # list of len 1
        if len(idx) == 1:
            return cube[idx[0]]
        # list of len 2
        elif len(idx) == 2:
            start, end = idx
            return cube[start:end+1].sum(axis=0)

    raise ValueError("'idx' must be an int or a list of one or two integers")


def get_spectral_slice_value(spectral_axis, idx):
    '''
    Return a representative value from a spectral axis
    given an index or index range.
    Parameters
    ––––––––––
    spectral_axis : Quantity
        The spectral axis (e.g., wavelength, frequency, or
        velocity) as an 'astropy.units.Quantity' array.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'spectral_axis[i]'
        - [i] -> returns 'spectral_axis[i]'
        - [i, j] -> returns '(spectral_axis[i] + spectral_axis[j+1])/2'
    Returns
    –––––––
    spectral_value : float
        The spectral value at the specified index or index
        range, in the units of 'spectral_axis'.
    '''
    if isinstance(idx, int):
        return spectral_axis[idx].value
    elif isinstance(idx, list):
        if len(idx) == 1:
            return spectral_axis[idx[0]].value
        elif len(idx) == 2:
            return (spectral_axis[idx[0]].value + spectral_axis[idx[1]+1].value)/2

    raise ValueError("'idx' must be an int or a list of one or two integers")


# Cube/Image Masking Functions
# ––––––––––––––––––––––––––––
def mask_image(image, ellipse_region=None, region=None,
               line_points=None, invert_region=False, above_line=True,
               preserve_shape=True, existing_mask=None, **kwargs):
    '''
    Mask an image with modular filters.
    Supports applying an elliptical or annular region mask, an optional
    line cut (upper or lower half-plane), and combining with an existing mask.
    Parameters
    ––––––––––
    image : array-like, DataCube, FitsFile, or SpectralCube
        Input image or cube. If higher-dimensional, the mask is applied
        to the last two axes.
    ellipse_region : `EllipsePixelRegion` or `EllipseAnnulusPixelRegion`, optional, default=None
        Region object specifying an ellipse or annulus.
    region : str {'annulus', 'ellipse'}, optional, default=None
        Type of region to apply. Ignored if `ellipse_region` is provided.
    line_points : array-like, shape (2, 2), optional, default=None
        Two (x, y) points defining a line for masking above/below.
        Ex: [[0,2], [20,10]]
    invert_region : bool, default=False
        If True, invert the region mask.
    above_line : bool, default=True
        If True, keep the region above the line. If False, keep below.
    preserve_shape : bool, default=True
        If True, return an array of the same shape with masked values set to NaN.
        If False, return only the unmasked pixels.
    existing_mask : ndarray of bool, optional, default=None
        An existing mask to combine (union) with the new mask.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - center : tuple of float, optional, default=None
            Center coordinates (x, y).
        - w : float, optional, default=None
            Width of ellipse.
        - h : float, optional, default=None
            Height of ellipse.
        - angle : float, optional, default=0
            Rotation angle in degrees.
        - tolerance : float, optional, default=2
            Tolerance for annulus inner/outer radii
    Returns
    –––––––
    masked_image : ndarray or SpectralCube
        Image with mask applied. Type matches input.
    masks : ndarray of bool or list
        If multiple masks are combined, returns a list containing the
        master mask followed by individual masks. Otherwise returns a single mask.
    '''
    # –––– Kwargs ––––
    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)
    tolerance = kwargs.get('tolerance', 2)

    # extract units
    unit = get_units(image)

    # ensure working with array
    if isinstance(image, (DataCube, FitsFile)):
        image = image.data
    else:
        image = np.asarray(image)

    # determine image shape
    N, M = image.shape[-2:]
    y, x = np.indices((N, M))
    # empty list to hold all masks
    masks = []

    # early return if just applying an existing mask
    if ellipse_region is None and region is None and line_points is None and existing_mask is not None:
        if existing_mask.shape != image.shape[-2:]:
            raise ValueError("existing_mask must have same shape as image")

        if isinstance(image, np.ndarray):
            if preserve_shape:
                masked_image = np.full_like(image, np.nan, dtype=float)
                masked_image[..., existing_mask] = image[..., existing_mask]
            else:
                masked_image = image[..., existing_mask]

            if isinstance(unit, u.UnitBase) and not isinstance(image, Quantity):
                masked_image *= unit
        else:
            # if spectral cube or similar object
            masked_image = image.with_mask(existing_mask)

        return masked_image

    # –––– Region Mask ––––
    # if ellipse region is passed in use those values
    if ellipse_region is not None:
        center = ellipse_region.center
        a = ellipse_region.width / 2
        b = ellipse_region.height / 2
        angle = ellipse_region.angle if ellipse_region.angle is not None else 0
    # accept user defined center, w, and h values if used
    elif None not in (center, w, h):
        a = w / 2
        b = h / 2
    # stop program if attempting to plot a region without necessary data
    elif region is not None:
        raise ValueError("Either 'ellipse_region' or 'center', 'w', 'h' must be provided.")

    # construct region
    if region is not None:
        if region.lower() == 'annulus':
            region_obj = EllipseAnnulusPixelRegion(
                center=PixCoord(center[0], center[1]), # type: ignore
                inner_width=2*(a - tolerance),
                inner_height=2*(b - tolerance),
                outer_width=2*(a + tolerance),
                outer_height=2*(b + tolerance),
                angle=angle * u.deg
            )
        elif region.lower() == 'ellipse':
            region_obj = EllipsePixelRegion(
                center=PixCoord(center[0], center[1]), # type: ignore
                width=2*a,
                height=2*b,
                angle=angle * u.deg
            )
        else:
            raise ValueError("region must be 'annulus' or 'ellipse'")

        # filter by region mask
        region_mask = region_obj.to_mask(mode='center').to_image((N, M)).astype(bool)
        if invert_region:
            region_mask = ~region_mask
        masks.append(region_mask.copy())
    else:
        # empty mask if no region
        region_mask = np.ones((N, M), dtype=bool)

    # –––– Line Mask ––––
    if line_points is not None:
        # start from previous mask
        line_mask = region_mask.copy()
        # compute slope and intercept of line
        m, b_line = compute_line(line_points)
        # filter out points above/below line
        line_mask &= (y >= m*x + b_line) if above_line else (y <= m*x + b_line)
        # add line region to mask array
        masks.append(line_mask.copy())
    else:
        # empty mask if no region
        line_mask = region_mask.copy()

    # –––– Combine Masks ––––
    # start master mask with line_mask (or region if no line)
    mask = line_mask.copy()

    # union with existing mask if provided
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError("existing_mask must have the same shape as the image")
        mask |= existing_mask

    # –––– Apply Mask ––––
    # if numpy array:
    if isinstance(image, np.ndarray):
        if preserve_shape:
            masked_image = np.full_like(image, np.nan, dtype=float)
            masked_image[..., mask] = image[..., mask]
        else:
            masked_image = image[..., mask]
        if isinstance(unit, u.UnitBase) and not isinstance(image, Quantity):
            masked_image *= unit
    # if spectral cube object
    else:
        masked_image = image.with_mask(mask)

    # ––––– Final Mask List –––––
    # Return master mask as first element
    masks = [mask] + masks if len(masks) > 1 else mask

    return masked_image, masks


def compute_line(points):
    '''
    Compute the slope and intercept of a line passing through two points.
    Parameters
    ––––––––––
    points : list or tuple of tuples
        A sequence containing exactly two points, each as (x, y), e.g.,
        [(x0, y0), (x1, y1)].
    Returns
    –––––––
    m : float
        Slope of the line.
    b : float
        Intercept of the line (y = m*x + b).
    Notes
    –––––
    - The function assumes the two points have different x-coordinates.
    - If the x-coordinates are equal, a ZeroDivisionError will be raised.
    '''
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    b = points[0][1] - m*points[0][0]

    return m, b
