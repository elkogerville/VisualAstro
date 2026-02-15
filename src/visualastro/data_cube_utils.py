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
from spectral_cube import SpectralCube
from .config import get_config_value
from .DataCube import DataCube
from .FitsFile import FitsFile
from .numerical_utils import get_data
from .units import get_unit


# Cube Manipulation Functions
# ---------------------------

_STACK_METHODS = {
    'mean': np.nanmean,
    'median': np.nanmedian,
    'sum': np.nansum,
    'max': np.nanmax,
    'min': np.nanmin,
    'std': np.nanstd,
}

def stack_cube(cube, *, idx=None, method=None, axis=0):
    """
    Stack or extract slices from a data cube.

    Parameters
    ----------
    cube : ndarray, Quantity, SpectralCube, or DataCube
        3D data cube to stack.
    idx : int, list of int, or None, optional, default=None
        Index specification:
        - int: extract single slice
        - [start, end]: extract range (inclusive)
        - None: use entire cube
    method : {'mean', 'median', 'sum', 'max', 'min', 'std'}, default=None
        Stacking method. If None, uses the default value set
        by ``config.stack_cube_method``.
    axis : int, optional, default=0
        Axis along which to stack.

    Returns
    -------
    ndarray, Quantity, or SpectralCube
        Stacked result (2D if axis=0) or extracted slice.
    """
    method = get_config_value(method, 'stack_cube_method')

    if method not in _STACK_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Options: {', '.join(_STACK_METHODS.keys())}"
        )

    cube = get_data(cube)

    if idx is not None:
        if isinstance(idx, int):
            return cube[idx]

        if isinstance(idx, list):
            if len(idx) == 1:
                return cube[idx[0]]
            if len(idx) != 2:
                raise ValueError('idx must be an int, [start, end], or None')

            start, end = idx
            cube = cube[start:end+1]
        else:
            raise TypeError(f'idx must be int, list, or None; got {type(idx)}')

    if isinstance(cube, SpectralCube):
        stack_func = getattr(cube, method)
        return stack_func(axis=axis)

    return _STACK_METHODS[method](cube, axis=axis)


# Cube/Image Masking Functions
# ----------------------------
def mask_image(
    image, ellipse_region=None, region=None,
    line_points=None, invert_region=False,
    above_line=True, preserve_shape=True,
    existing_mask=None, combine_method='union', **kwargs):
    '''
    Mask an image with modular filters.
    Supports applying an elliptical or annular region mask, an optional
    line cut (upper or lower half-plane), and combining with an existing mask.

    Parameters
    ----------
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
    combine_method : {'union', 'intersect'}, optional, default=None
        If 'union', combine masks with `|`. If 'intersect', use `&`.

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
        - tolerance : float, list, or tuple, optional, default=2
            ±Tolerance (distance from radius) for annulus inner/outer radii.
            If array-like, uses the first element as the minus bound,
            and the second element as the positive.

    Returns
    -------
    masked_image : ndarray or SpectralCube
        Image with mask applied. Type matches input.
    masks : ndarray of bool or list
        If multiple masks are combined, returns a list containing the
        master mask followed by individual masks. Otherwise returns a single mask.
    '''
    # ---- Kwargs ----
    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)
    tolerance = kwargs.get('tolerance', 2)
    if (isinstance(tolerance, (list, tuple))
        and len(tolerance) == 2):
        mtol = tolerance[0]
        ptol = tolerance[1]
    else:
        mtol = tolerance
        ptol = tolerance

    # extract units
    unit = get_unit(image)

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
    if (ellipse_region is None and region is None
        and line_points is None and existing_mask is not None):
        if existing_mask.shape != image.shape[-2:]:
            raise ValueError('existing_mask must have same shape as image')

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

    # ---- Region Mask ----
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
                inner_width=2*(a - mtol),
                inner_height=2*(b - mtol),
                outer_width=2*(a + ptol),
                outer_height=2*(b + ptol),
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

    # ---- Line Mask ----
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

    # ---- Combine Masks ----
    # start master mask with line_mask (or region if no line)
    mask = line_mask.copy()

    # union with existing mask if provided
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError('existing_mask must have the same shape as the image')
        if combine_method == 'union':
            mask |= existing_mask
        elif combine_method == 'intersect':
            mask &= existing_mask
        else:
            raise ValueError(
                f"`combine_method` has to be 'union' or 'intersect'! "
                f'Got {combine_method}.'
            )

    # ---- Apply Mask ----
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

    # ---- Final Mask List ----
    # Return master mask as first element
    masks = [mask] + masks if len(masks) > 1 else mask

    return masked_image, masks


def compute_line(points):
    '''
    Compute the slope and intercept of a line passing through two points.
    Parameters
    ----------
    points : list or tuple of tuples
        A sequence containing exactly two points, each as (x, y), e.g.,
        [(x0, y0), (x1, y1)].
    Returns
    -------
    m : float
        Slope of the line.
    b : float
        Intercept of the line (y = m*x + b).
    Notes
    -----
    - The function assumes the two points have different x-coordinates.
    - If the x-coordinates are equal, a ZeroDivisionError will be raised.
    '''
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    b = points[0][1] - m*points[0][0]

    return m, b
