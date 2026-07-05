"""
Author: Elko Gerville-Reache
Date Created: 2026-07-04
Date Modified: 2026-07-04
Description:
    Functions related to colormaps in plotting.
    To define custom colormaps, define them at
    the bottom of this file in `VISUALASTRO_CMAPS`.
"""

import cmasher
from colorspacious import cspace_converter
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
import tol_colors as tc

from visualastro.core.config import config
from visualastro.core.numerical_utils import to_list


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


def create_cmap(
    colors: list[ColorType] | int,
    positions: list[float] | None = None,
    name: str = 'continous_cmap'
) -> mcolors.LinearSegmentedColormap | mcolors.ListedColormap:
    """
    Creates a colormap from colors with optional position control.

    Parameters
    ----------
    colors : list[ColorType] | int
        Color specifications (hex, named colors, RGB tuples, etc.).
        The cmap will be created from these colors. If `colors` is
        an `int`, the function returns `tol_colors.rainbow_discrete(colors)`.
    positions : list[float] | None, optional
        Positions in [0, 1] for each color. Must start with 0 and end with 1.
        If None, colors are evenly spaced.

    Returns
    -------
    LinearSegmentedColormap
    """
    if isinstance(colors, int):
        return tc.rainbow_discrete(colors)

    rgb_list = [mcolors.to_rgb(color) for color in colors]

    if positions is None:
        positions = list(np.linspace(0, 1, len(rgb_list)))

    cdict = {channel: [[positions[i], rgb_list[i][idx], rgb_list[i][idx]]
                       for i in range(len(positions))]
             for idx, channel in enumerate(['red', 'green', 'blue'])}

    return mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=256)


def plot_cmap_lightness(
    cmap: str | mcolors.Colormap | list[str | mcolors.Colormap],
    ax: Axes | None = None,
    s: float = 300,
    xtick_labels: bool = True,
    xticks: bool = True,
    **kwargs
) ->  list[PathCollection]:
    """
    Plot L* (CAM02-UCS lightness) as a function of colormap index.

    Parameters
    ---------------
    cmap : str | matplotlib.colors.Colormap
        Colormap name or instance.
    ax : matplotlib.axes.Axes, optional, default=None
        Target axes. Created via `plt.subplots` if None.
    s : float, optional, default=300
        Marker size passed to `ax.scatter`.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.scatter`.

    Returns
    ---------------
    ax : PathCollection
        Scatter artist.
    """
    from visualastro.plotting.core.utils import legend

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)

    x = np.linspace(0.0, 1.0, 1000)
    n = 256
    scatters = []

    cmaps = to_list(cmap)
    for cmap in cmaps:
        cmap = get_cmap(cmap)

        rgb = cmap(x)[np.newaxis, :, :3]
        lab = cspace_converter('sRGB1', 'CAM02-UCS')(rgb)
        L = lab[0, :, 0]

        scatter = ax.scatter(x, L, s=s, c=x, cmap=cmap, **kwargs)

        scatters.append(scatter)
        cmasher.set_cmap_legend_entry(scatter, cmap.name)

    legend(ax=ax)
    ax.set_ylabel(r'L$^*$', fontsize=config.axes.label_fontsize)
    if not xtick_labels:
        ax.set_xticklabels([])
    if not xticks:
        ax.set_xticks([])

    return scatters



# VISUALASTRO COLOR MAPS
# ----------------------
iridescent = plt.get_cmap('tol.iridescent').copy()
iridescent.set_bad(color='white')
BuWhRd = create_cmap(
    ['#191970', '#0000FF', '#FFFFFF', '#FF0000', '#8b0000'],
    [0, 0.25, 0.5, 0.75, 1],
    'BuWhRd'
)
tol_rainbow = plt.get_cmap('tol.rainbow').copy()
tol_rainbow.set_bad(color='white')

VISUALASTRO_CMAPS: dict[str, mcolors.Colormap] = {
    'iridescent': iridescent,
    'BuWhRd': BuWhRd,
    'tol_rainbow': tol_rainbow,
    'nuclear_waste': create_cmap(
        ['#1CFF00', '#A7FF63', '#D1E61C', '#A2A838', '#6CA838'],
        name='nuclear_waste'
    ),
    'shrek': create_cmap(
        ['#6CA838', '#1CFF00', '#A7FF63', '#D1E61C', '#A2A838', '#7E8140', '#575931'],
        name='shrek'
    ),
}
CMAPNAMES = [key for key in VISUALASTRO_CMAPS.keys()]
