"""
Author: Elko Gerville-Reache
Date Created: 2026-07-04
Date Modified: 2026-07-04
Description:
    Functions related to colormaps in plotting.
"""

import cmasher
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import tol_colors as tc


def get_cmap(
    cmap: mcolors.Colormap | str | int,
    cmap_range: tuple[float, float] = (0, 1),
    bad_color: ColorType | None = None,
    N: int | None = None
) -> mcolors.Colormap:
    """
    Retrieve a colormap by name or return the input colormap.

    Parameters
    ----------
    cmap : mcolors.Colormap | str | int
        Colormap object or string name. If a string, attempts lookup in VISUALASTRO_CMAPS
        registry before falling back to matplotlib's colormap registry.
        If an `int`, returns `tol_colors.rainbow_discrete(colors)`.
    cmap_range : tuple[float, float], optional, default=(0, 1)
        The normalized range of the colormap. By default, is `(0,1)`,
        meaning the returned colormap has its entire range. Ignored
        if `cmap` is an `int`.
    bad_color : ColorType | None, optional, default=None
        Bad data color (`bad_color`). If None, leaves the colormap unchanged.
    N : int | None, optional, default=None
        Number of discrete color segments to sample from the sub-range
        defined by `cmap_range`. If None, retains all colors within that
        range (continuous colormap). Ignored if `cmap_range` is `(0, 1)`
        or if `cmap` is an `int`.

    Returns
    -------
    mcolors.LinearSegmentedColormap
        The requested colormap.
    """
    def set_bad_color(
        cmap: mcolors.Colormap,
        color: ColorType | None
    ) -> mcolors.Colormap:
        """
        Return a copy of the cmap with a new `bad_color`, or unchanged if
        `color` is None.
        """
        if color is None:
            return cmap
        new_cmap = cmap.copy()
        new_cmap.set_bad(color=color)
        return new_cmap

    if len(cmap_range) != 2:
        raise ValueError(
            'cmap_range must be a tuple[min, max]!'
        )

    if isinstance(cmap, int):
        return set_bad_color(tc.rainbow_discrete(cmap), bad_color)

    out_cmap = set_bad_color(plt.get_cmap(cmap), bad_color)
    if cmap_range[0] == 0 and cmap_range[1] == 1:
        return out_cmap

    return cmasher.get_sub_cmap(
        out_cmap, cmap_range[0], cmap_range[1], N=N
    )
