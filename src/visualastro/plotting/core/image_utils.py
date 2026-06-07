"""
Author: Elko Gerville-Reache
Date Created: 2026-06-07
Date Modified: 2026-06-07
Description:
    Image utility functions for plotting.
"""

from typing import Literal

import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm, TwoSlopeNorm
import numpy as np
from numpy.typing import NDArray
from spectral_cube import SpectralCube

from visualastro.analysis.image_utils import stack_cube
from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
)
from visualastro.core.numerical_utils import get_data, get_value, to_array
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile


def get_imshow_norm(
    norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope'] | None,
    vmin: float | np.floating | None,
    vmax: float | np.floating | None,
    **kwargs
) -> ImageNormalize | AsinhNorm | LogNorm | PowerNorm | TwoSlopeNorm | None:
    """
    Return a matplotlib or astropy normalization object for image display.

    Returns `None` if `norm=None`.

    Parameters
    ----------
    norm : {'asinh', 'asinhnorm', 'log', 'power', 'twoslope'} | None
        Normalization algorithm for colormap scaling.

        * `'asinh'` -> asinh stretch using `ImageNormalize`
        * `'asinhnorm'` -> asinh stretch using `AsinhNorm`
        * `'log'` -> logarithmic scaling using `LogNorm`
        * `'power'` -> power-law normalization using `PowerNorm`
        * `'twoslope'` -> normalize centered around `vcenter` using `TwoSlopeNorm`

    vmin, vmax : float | None
        Minimum and maximum value for normalization.
    linear_width : float, optional, default=`config.linear_width`
        The effective width of the linear region, beyond
        which the transformation becomes asymptotically logarithmic.
        Used for `norm='asinhnorm'`.
    gamma : float, optional, default=`config.gamma`
    Power law exponent. Used for `norm='power'`.
    vcenter : float, optional, default=None
        Center point of normalization. Must be in between
        `vmin` and `vmax`. If `None`, is the midpoint between
        `vmin` and `vmax`.

    Returns
    -------
    norm_obj : ImageNormalize | AsinhNorm | LogNorm | PowerNorm | None
        Normalization object to pass to `imshow`. `None` if `norm=None`.
    """
    linear_width: float = kwargs.pop('linear_width', config.linear_width)
    gamma: float = kwargs.pop('gamma', config.gamma)
    vcenter: float | None = kwargs.pop('vcenter', None)

    # use linear stretch if plotting boolean array
    if vmin == 0 and vmax == 1:
        return None
    elif norm is None:
        return None
    else:
        if vmin is None or vmax is None:
            raise ValueError(
                'vmin and vmax must not be None if norm is not None! '
                f'got: vmin: {vmin}, vmax: {vmax}'
            )
    vmin = float(vmin)
    vmax = float(vmax)
    vcenter = (vmax + vmin)/2 if vcenter is None else vcenter

    norm_str = norm.lower()

    # dict containing possible stretch algorithms
    norm_map = {
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
        'asinhnorm': AsinhNorm(vmin=vmin, vmax=vmax, linear_width=linear_width),
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'power': PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
        'twoslope': TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    }
    if norm_str not in norm_map:
        raise ValueError(
            f'ERROR: unsupported norm: {norm_str}. '
            f'\nsupported norms are {list(norm_map.keys())}'
        )

    return norm_map[norm_str]


def get_vmin_vmax(
    data: NDArray | u.Quantity | DataCube | FitsFile | SpectralCube,
    percentile: tuple[float, float] | _Unset = _UNSET,
    vmin: float | np.floating | None = None,
    vmax: float | np.floating | None = None
) -> tuple[float | np.floating | int, float | np.floating | int]:
    """
    Compute vmin and vmax for image display. By default uses the
    data nanpercentile set by `percentile`, but optionally `vmin`
    and/or `vmax` can be set by the user.

    Passing in a boolean array returns `vmin=0`, `vmax=1`.
    This function is used internally by  `compute_imshow_scale`.

    Parameters
    ----------
    data : NDArray | u.Quantity | DataCube | FitsFile | SpectralCube
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
        Must be convertable to an array via `to_array`.
    percentile : tuple[float, float] |  _Unset, optional, default=_UNSET
        Percentile range `[pmin, pmax]` to compute vmin and vmax.
        If None, sets vmin and vmax to None. If `_UNSET`, uses
        default value from `config.percentile`.
    vmin, vmax : float | None, optional, default=None
        If provided, overrides the computed vmin and vmax.

    Returns
    -------
    vmin, vmax : float | int | None
        Minimum and maximum values for image scaling.
    """
    percentile_range = config.percentile if percentile is _UNSET else percentile
    if percentile_range is None:
        raise ValueError(
            'get_vmin_vmax requires a valid percentile range. '
            'Received None. This function should only be called '
            'when percentile-based scaling is enabled.'
        )

    # check if data is an array
    data = to_array(data, keep_unit=False)
    # check if data is boolean
    if data.dtype == bool:
        return 0, 1

    if vmin is None:
        vmin = np.nanpercentile(data, percentile_range[0])
    if vmax is None:
        vmax = np.nanpercentile(data, percentile_range[1])

    return vmin, vmax


def compute_imshow_scale(
    data: NDArray | u.Quantity | DataCube | FitsFile | SpectralCube,
    norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None,
    vmin: float | np.floating | None,
    vmax: float | np.floating | None,
    percentile: tuple[float, float] | None,
    **kwargs
) -> tuple[
    ImageNormalize | AsinhNorm | LogNorm | PowerNorm | TwoSlopeNorm | None,
    float | np.floating | None,
    float | np.floating | None
]:
    """
    Compute normalization and intensity scaling parameters for `plt.imshow`.

    Resolves interaction between `norm`, `vmin`, `vmax`, and
    `percentile` into consistent display parameters.

    Modes
    -----
    1. Normalized scaling (`norm` is not None):
    - `vmin`/`vmax` from `percentile` or explicit values (required)
    - Returns `(norm_obj, vmin, vmax)`

    2. Linear with percentile (`norm` is None, `percentile` provided):
    - `vmin`/`vmax` from `percentile` (unless explicit)
    - Returns `(None, vmin, vmax)`

    3. Linear with explicit limits (`norm` is None, `vmin` or `vmax` set):
    - Missing bounds from `percentile` if available, else None
    - Returns `(None, vmin, vmax)`

    4. Matplotlib autoscaling (all None), or norm='linear':
    - Returns `(None, None, None)`

    5. Boolean data (dtype is bool):
    - Forces `vmin=0`, `vmax=1`
    - Returns `(None, 0, 1)`

    Parameters
    ----------
    data : ndarray | Quantity | DataCube | FitsFile | SpectralCube
        Input image data. Must be convertible to a NumPy array via `to_array`.
    norm : {'asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'} | None
        Normalization algorithm for colormap scaling.
        If `None`, linear scaling is used.
    vmin, vmax : float | None
        Lower and upper bounds for intensity scaling. If `None`, may be computed
        from `percentile` or left unset depending on mode.
    percentile : tuple[float, float] | None
        Percentile range `(pmin, pmax)` used to compute `vmin` and
        `vmax`. If `None`, no automatic clipping is applied.
    linear_width : float, optional, default=config.linear_width
        The effective width of the linear region, beyond which the
        transformation becomes asymptotically logarithmic.
        Used for `norm='asinhnorm'`.
    gamma : float, optional, default=config.gamma
        Power law exponent. Used for `norm='power'`.
    vcenter : float, optional, default=None
        Center point of normalization. Must be in between `vmin` and `vmax`.
        If `None`, is the midpoint between `vmin` and `vmax`.

    Returns
    -------
    norm_obj : ImageNormalize | AsinhNorm | LogNorm | PowerNorm | None
        Normalization object for `imshow`. None indicates linear scaling.
    vmin, vmax : float | None
        Lower and upper intensity bounds for `imshow`.
    """
    data = to_array(data)

    if data.dtype == bool:
        return None, 0, 1

    if norm == 'linear':
        return None, None, None

    if norm is None:
        if percentile is not None:
            vmin, vmax = get_vmin_vmax(data, percentile, vmin, vmax)

        return None, vmin, vmax

    if percentile is not None:
        vmin, vmax = get_vmin_vmax(data, percentile, vmin, vmax)

    elif vmin is None or vmax is None:
        raise ValueError(
            'vmin and vmax must not be None if norm is not None! '
            f'got: norm: {norm}, vmin: {vmin}, vmax: {vmax}'
        )
    img_norm = get_imshow_norm(norm, vmin, vmax, **kwargs)

    return img_norm, vmin, vmax


def nanpercentile_limits(
    data,
    vmin,
    vmax,
    *,
    slice_idx=None,
    stack_method=None,
    axis=0,
):
    """
    Compute NaN-safe percentile-based color limits.

    If input is 3D, it is reduced to 2D before computing limits.

    Parameters
    ----------
    data : array-like, SpectralCube, or DataCube
        Input image or cube.
    vmin, vmax : float
        Lower and upper percentiles (0–100).
    slice_idx : int, list of int, or None, optional, default=None
        Optional slicing for cube-like inputs. If None,
        is ignored.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'} or None, default=None
        Reduction method if stacking is required. If None,
        uses `config.stack_cube_method`.
    axis : int, default=0
        Axis to flatten if data.ndim > 2 and no slice_idx is given.

    Returns
    -------
    vmin_val, vmax_val : float
        Lower and upper intensity bounds.
    """
    stack_method = get_config_value(stack_method, 'stack_cube_method')

    data = get_data(data)

    if slice_idx is not None:
        data = stack_cube(
            data, idx=slice_idx, method=stack_method, axis=axis
        )

    arr = to_array(get_value(data))

    if arr.ndim > 2:
        arr = getattr(np, f'nan{stack_method}')(arr, axis=axis)

    return (
        np.nanpercentile(arr, vmin),
        np.nanpercentile(arr, vmax),
    )
