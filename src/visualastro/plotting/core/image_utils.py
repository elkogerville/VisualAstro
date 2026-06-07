"""
Author: Elko Gerville-Reache
Date Created: 2026-06-07
Date Modified: 2026-06-07
Description:
    Image utility functions for plotting.
"""

from typing import Literal

from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm, TwoSlopeNorm
import numpy as np

from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
    _resolve_default
)


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

    vmin : float | None
        Minimum value for normalization.
    vmax : float | None
        Maximum value for normalization.
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
