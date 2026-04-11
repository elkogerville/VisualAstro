"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-04-10
Description:
    Functions related to colors and colormaps in plotting.
Dependencies:
    - matplotlib
    - numpy
"""

import colorsys
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import config, resolve_default, _Unset, _UNSET


def create_colormap(
    colors: list[ColorType],
    positions: list[float] | None = None,
    name: str = 'continous_cmap'
) -> mcolors.LinearSegmentedColormap:
    """
    Creates a colormap from colors with optional position control.

    Parameters
    ----------
    colors : list[ColorType]
        Color specifications (hex, named colors, RGB tuples, etc.).
    positions : list[float] | None, optional
        Positions in [0, 1] for each color. Must start with 0 and end with 1.
        If None, colors are evenly spaced.

    Returns
    -------
    LinearSegmentedColormap
    """
    rgb_list = [mcolors.to_rgb(color) for color in colors]

    if positions is None:
        positions = list(np.linspace(0, 1, len(rgb_list)))

    cdict = {channel: [[positions[i], rgb_list[i][idx], rgb_list[i][idx]]
                       for i in range(len(positions))]
             for idx, channel in enumerate(['red', 'green', 'blue'])}

    return LinearSegmentedColormap(name, segmentdata=cdict, N=256) # type: ignore


def sample_cmap(
    N: int,
    cmap: str | mcolors.Colormap | _Unset = _UNSET,
    as_hex: bool = False
) -> NDArray:
    """
    Sample N distinct colors from a given matplotlib colormap
    returned as RGBA tuples in an array of shape (N,4).

    Parameters
    ----------
    N : int
        Number of colors to sample.
    cmap : str | Colormap | _Unset, optional, default=_UNSET
        Name of the matplotlib colormap or Colormap object. If
        ``_UNSET`` uses the default value in ``config.cmap``.
    as_hex : bool, optional, default=False
        If True, return colors as hex strings.

    Returns
    -------
    NDArray
        Array of shape (N, 4) containing RGBA colors if ``as_hex=False``,
        or array of shape (N,) containing hex strings if ``as_hex=True``.
    """
    cmap = resolve_default(cmap, config.cmap)

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))
    if as_hex:
        colors = np.array([mcolors.to_hex(c) for c in colors])

    return colors


def get_colors(user_colors=None, cmap=_UNSET):
    '''
    Returns plot and model colors based on predefined palettes or user input.

    Parameters
    ----------
    user_colors : str, list, or None, optional, default=None
        - None: returns the default palette (`config.default_palette`).
        - str:
            * If the string matches a palette name, returns that palette.
            * If the string ends with '_r', returns the reversed version of the palette.
            * If the string is a single color (hex or matplotlib color name), returns
              that color and a lighter version for the model.
        - list:
            * A list of colors (hex or matplotlib color names). Returns the list
              for plotting and lighter versions for models.
        - int:
            * An integer specifying how many colors to sample from a matplolib cmap
              using sample_cmap(). By default uses 'turbo'.
    cmap : str, list of str, or None, default=_UNSET
        Matplotlib colormap name. If ``_UNSET``, uses
        the default value in ``config.cmap``.

    Returns
    -------
    plot_colors : list of str
        Colors for plotting the data.
    model_colors : list of str
        Colors for plotting the model (contrasting or lighter versions).
    '''
    palettes = {
        'visualastro': {
            'plot':  ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#E45756', '#17BECF', '#9467BD']
        },
        'va': {
            'plot':  ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#E45756', '#17BECF', '#9467BD']
        },
        'ibm_contrast': {
            'plot':  ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#1F77B4', '#8C564B']
        },
        'astro': {
            'plot':  ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#9467BD', '#2CA02C', '#E45756', '#17BECF', '#8C564B', '#FFD700']
        },
        'MSG': {
            'plot':  ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
            'model': ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#17BECF']
        },
        'ibm': {
            'plot':  ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#E45756']
        },
        'smplot': {
            'plot': ['k', '#FF0000', '#0000FF', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00'],
            'model': ['#808080', '#FF6B6B', '#6B6BFF', '#6BFF6B', '#6BFFFF', '#FF6BFF', '#FFFF6B']
        }
    }
    # get default config values
    cmap = resolve_default(cmap, config.cmap)
    default_palette = config.default_palette

    # default case
    if user_colors is None:
        palette = palettes[default_palette]
        return palette['plot'], palette['model']
    # if user passes a color string
    if isinstance(user_colors, str):
        # if palette in visualastro palettes
        # return a reversed palette if palette
        # ends with '_r'
        if user_colors.rstrip('_r') in palettes:
            base_name = user_colors.rstrip('_r')
            palette = palettes[base_name]
            plot_colors = palette['plot']
            model_colors = palette['model']
            # if '_r', reverse palette
            if user_colors.endswith('_r'):
                plot_colors = plot_colors[::-1]
                model_colors = model_colors[::-1]
            return plot_colors, model_colors
        else:
            return [user_colors], [_lighten_color(user_colors)]
    # if user passes a list or array of colors
    if isinstance(user_colors, (list, np.ndarray)):
        return user_colors, [_lighten_color(c) for c in user_colors]
    # if user passes an integer N, sample a cmap for N colors
    if isinstance(user_colors, int):
        colors = sample_cmap(user_colors, cmap=cmap)
        return colors, [_lighten_color(c) for c in colors]
    raise ValueError(
        'user_colors must be None, a str palette name, a str color, a list of colors, or an integer'
    )


def _lighten_color(color: ColorType, mix: float = 0.5) -> 'str':
    """
    Lightens the given matplotlib color by mixing it with white.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, html color or rgb tuple.
    mix : float or int
        Ratio of color to white in mix.
        ``mix=0`` returns the original color,
        ``mix=1`` returns pure white.
    """

    # convert to rgb
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    # mix color with white
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(tuple(mixed))


def _desaturate_color(color: ColorType, factor: float = 0.5) -> str:
    """
    Desaturate a color by moving it toward gray.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, html color or rgb tuple.
    factor : float
        Desaturation amount. 0=original, 1=full gray.

    Returns
    -------
    str
        Hex color string.
    """
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s_new = s * (1 - factor)
    rgb_new = colorsys.hls_to_rgb(h, l, s_new)

    return mcolors.to_hex(rgb_new)


palettes = {
    'visualastro': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
    'va': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
    'ibm_contrast': ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
    'astro': ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
    'MSG': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
    'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
    'smplot': ['k', '#FF0000', '#0000FF', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00']
}
