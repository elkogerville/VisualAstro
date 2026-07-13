"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-07-01
Description:
    Functions related to colors in plotting.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
import operator as op

from colorspacious import cspace_convert
import colorsys
from typing import Literal, TypeAlias
import colorspacious
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import (
    TABLEAU_COLORS,
    Colormap,
    LogNorm,
    Normalize
)
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray
import tol_colors as tc

from visualastro.core.config import (
    config, _resolve_default, _Unset, _UNSET
)
from visualastro.core.numerical_utils import (
    as_list, to_list, _unwrap_if_single
)
from visualastro.plotting.core.colormaps import get_cmap


RGBTuple: TypeAlias = tuple[float, float, float]
RGBATuple: TypeAlias = tuple[float, float, float, float]


# VISUALASTRO COLOR PALETTES
# --------------------------
COLORSETS: dict[str, list[ColorType]] = {
    'visualastro': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
    'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
    'ibm_contrast': [
        '#648FFF', '#DC267F', '#785EF0',
        '#26DCBA', '#FFB000', '#FE6100'
    ],
    'astro_seq': [
        '#9FB7FF', '#648FFF', '#785EF0', '#DC267F',
        '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'
    ],
    'astro': [
        '#785EF0', '#26DCBA', '#DC267F', '#648FFF',
        '#FFB000', '#9FB7FF', '#CFE23C',
    ],
    'astro_contrast': [
        '#aed1ff', '#8f8ce7', '#5a06ef', '#dc267f',
        '#6c7a0e', '#cfe23c', '#26dcba'
    ],
    'turbo5': ['#35359A', '#4F65FF', '#5BFFD9', '#C4FF05', '#FF7D3C', '#AB0449'],
    'MSG': ['#483D8B', '#D81B60', '#DBB0FF', '#26DCBA', '#7D7FF3', '#CFE23C'],
    'MSGII': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3', '#CFE23C'],
    'MSG_seq': ['#483d8b', '#7d7ff3', '#dbb0ff', '#D81B60', '#26dcba', '#cfe23c'],
    'cardstock_dark': ['#000080', '#668035', '#187218', '#991D1B', '#992391', '#4E6767'],
    'cardstock_light': ['#9FD8FB', '#AED75B', '#BDDCBD', '#FB998E', '#E177AB', '#CDCDCD'],
    'toad': ['#BFDBE8', '#867E09', '#93CB59', '#34E693', '#97968B'],
    'default': list(TABLEAU_COLORS.values()),
    'temple_os': ['#555555', '#5555FF', '#55FF55', '#55FFFF', '#FF5555', '#FF55FF', '#FFFF55'],
    'smplot': [
        'k', '#FF0000', '#0000FF', '#00FF00',
        '#00FFFF', '#FF00FF', '#FFFF00'
    ],
    'set2': mpl.color_sequences['Set2'],
    'dark2': mpl.color_sequences['Dark2'],
    'cornfield': ['#FFF662', '#CDE47D', '#F1BE95', '#70813F', '#DD716B', '#E4AC2C'],
    'tmrw_night': ['#719C95', '#9CD6CF', '#FFDA81', '#F3A169', '#8FB3D3', '#6B859C', '#435561'],
    'tmrw_night_seq': ['#8FB3D3', '#6B859C', '#719C95', '#9CD6CF', '#FFDA81', '#F3A169'],
    '2mrw_nite': ['#9EACD2', '#7C859D', '#719C95', '#9CD6CF', '#FFDA81', '#F3A169'],
    'NGC6818': ['#5AC3BE', '#E770A2', '#4165C0', '#696969'],
    'rgb': ['#FF000F', '#007C6C', '#006B96'],
    'evenaworm': ['#008b8b', '#98fb98', '#ff81c0', '#ceaefa', '#d81b60'],
    'ocean_seq': ['#253494', '#2c7fb8', '#41b6c4', '#a1dab4', '#ffffcc'],
    'forest_seq': ['#006837', '#31a354', '#78c679', '#c2e699', '#ffffcc'],
    'jade_seq': ['#006d2c', '#2ca25f', '#66c2a4', '#b2e2e2', '#edf8fb'],
    'violetred_seq': ['#980043', '#dd1c77', '#df65b0', '#d7b5d8', '#f1eef6'],
    'PiGn_div': ['#d01c8b', '#f1b6da', '#f7f7f7', '#b8e186', '#4dac26'],
    'BrBG_div': ['#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571'],
    'pastel5': ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0'],
    'high_vis': ['#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd'],
    'retro': ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd'],
    'bright': [mcolors.to_hex(c) for c in tc.bright],
    'vibrant': [mcolors.to_hex(c) for c in tc.vibrant],
    'muted': [mcolors.to_hex(c) for c in tc.muted],
    'light': [mcolors.to_hex(c) for c in tc.light],
    'dark': [mcolors.to_hex(c) for c in tc.dark],
    'medium_contrast': [mcolors.to_hex(c) for c in tc.medium_contrast[1:]],
    'high_contrast': [mcolors.to_hex(c) for c in tc.high_contrast],
    'land_cover': [mcolors.to_hex(c) for c in tc.land_cover],
    'okabe_ito': [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000'
    ],
    'oit': ['#E69F00', '#56B4E9', '#D55E00', '#009E73', '#CC79A7'],
    'tab20b': mpl.color_sequences['tab20b'],
}
# COLORSETS ALIASES
# -----------------
COLORSETS['va'] = COLORSETS['visualastro']

COLORSET_NAMES = [key for key in COLORSETS.keys()]


# VISUALASTRO COLOR ALIASES
# -------------------------
@dataclass(frozen=True)
class Color:
    dsb: ColorType = 'darkslateblue'
    msb: ColorType = 'mediumslateblue'
    sb: ColorType = 'slateblue'
    mvr: ColorType = 'mediumvioletred'
    pvr: ColorType = 'palevioletred'
    violetred: ColorType = '#D81B60'
    mam: ColorType = 'mediumaquamarine'
    msg: ColorType = 'mediumseagreen'
    jade: ColorType = '#26DCBA'
    nebula: ColorType = '#9FB7FF'
    unicorn: ColorType = '#DBB0FF'
    pondwater: ColorType = '#CFE23C'
    ibmpur: ColorType = '#785EF0'
    ibmpnk: ColorType = '#DC267F'
    ibmblu: ColorType = '#648FFF'
    ibmylw: ColorType = '#FFB000'
    ibmorg: ColorType = '#FE6100'

    @classmethod
    def all(cls) -> list[ColorType]:
        """Return all defined colors as a list of colors"""
        return [getattr(cls, f.name) for f in fields(cls)]

    def plot(self, ncols: int = 4, sort_colors: bool = True) -> None:
        """
        Display color swatches in a grid with labels.

        Parameters
        ----------
        ncols : int, optional, default=4
            Number of columns to plot.
        sort_colors : bool, optional, default=True
            If `True`, sort colors by hsv value.
        """
        plot_colortable(
            colors='visualastro', ncols=ncols, sort_colors=sort_colors
        )

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + ':']
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f'  {field.name}: {value!r}')
        return '\n'.join(lines)

_color = Color()
VISUALASTRO_NAMED_COLORS = vars(_color)


def get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    transform: Literal['lighten', 'saturate', 'desaturate'] | None = None,
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100
) -> list[str | RGBTuple | RGBATuple] | list[str | None]:
    """
    Get colors from colorset name, colormap sampling, or explicit colors.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType] | _Unset, default=_UNSET

        * `UNSET`: Use default colorset
        * `str`:  visualastro colorset name (with optional '_r' suffix) or single color
        * `ColorType`: Explicit color
        * `int`: Number of colors to sample from cmap
        * `Sequence[ColorType]`: Explicit list of colors
        * `random`: Random sequence of colors

        If `_UNSET`, uses `config.default_colorset`.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap for sampling when colors is int. If `_UNSET`,
        uses `config.cmap`.
    cmap_range : tuple[float, float], optional, default=(0, 1)
        The normalized range of the colormap. By default, is `(0,1)`,
        meaning the returned colormap has its entire range. Ignored
        if `cmap` is an `int`.
    transform : {'lighten', 'desaturate'} | None, optional, default='lighten'
        Method to modify the color. If `None`, returns `color` unchanged.
    factor : float or int
        Modification strength.

        * If `transform='lighten'`: Blending ratio with white.

            * `factor=0`: Original color
            * `factor=1`: Pure white

        * If `transform='saturate'`: Saturation level in hsl space.

            * `factor=1`: Maximum saturation for each given color
            * `factor=0`: Grayscale

        * If `transform='desaturate'`: Desaturation amount.

            * `factor=0`: Original color
            * `factor=1`: Grayscale

    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output format.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not None, return the list of colors with a colorblind simulation applied.
    severity : int, optional, default=100
        Severity level (0-100). 100 = complete colorblindness.
        Only used if `cvd_type` is not None.

    Returns
    -------
    list[str] :
        If `fmt='hex'`.
    list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    list[str | None]:
        If `colors` is either `None`, `'face'`, or `'none'`.
    """
    if colors is None or isinstance(colors, str) and colors in {'face', 'none'}:
         return [colors]
    else:
        colors = _get_colors(colors, cmap, fmt=fmt, cmap_range=cmap_range)
        colors = as_list(
            get_complimentary_colors(
                colors,
                transform=transform,
                factor=factor,
                fmt=fmt
            )
        )
    if cvd_type is not None:
        colors = simulate_colorblindness(
            colors,
            cvd_type=cvd_type,
            severity=severity,
            fmt=fmt
        )

    modulo_idx = config.color_cycle_idx % len(colors)
    return colors[modulo_idx:] + colors[:modulo_idx]


def _get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """Helper function for `get_colors`"""
    cmap = get_cmap(
        _resolve_default(cmap, config.sample_cmap), cmap_range=cmap_range
    )

    if colors is _UNSET:
        colorset = COLORSETS.get(config.default_colorset, COLORSETS['visualastro'])
        return as_list(as_color(colorset, fmt=fmt))

    if colors == 'random':
        return random_colors(
            int(np.random.randint(1, config.random_colors_max_N, 1)[0])
        )

    if isinstance(colors, str):
        # if colorset in visualastro colorsets
        # return a reversed colorset if colorset
        # ends with '_r'
        if colors.removesuffix('_r') in COLORSETS:
            base_name = colors.removesuffix('_r')
            colorset = COLORSETS[base_name]
            # if '_r', reverse colorset
            if colors.endswith('_r'):
                colorset = colorset[::-1]
            return as_list(as_color(colorset, fmt))

        if colors in VISUALASTRO_NAMED_COLORS:
            return as_list(as_color(VISUALASTRO_NAMED_COLORS[colors], fmt))

        if (
            'xkcd:' + colors in mcolors.XKCD_COLORS and
            colors not in mcolors.CSS4_COLORS
        ):
            return as_list(as_color('xkcd:' + colors, fmt))

        else:
            return as_list(as_color(colors, fmt))

    if isinstance(colors, (np.ndarray, list)):
        return [_get_colors(c, fmt=fmt)[0] for c in colors]

    if isinstance(colors, tuple):
        return as_list(as_color(colors, fmt))

    # if user passes an integer N, sample a cmap for N colors
    if isinstance(colors, int):
        return as_list(sample_cmap(colors, cmap=cmap, fmt=fmt))

    raise TypeError(
        'colors must be None, a str colorset name, a str color, '
        f'a list of colors, or an integer! got {type(colors).__name__}'
    )


def sample_cmap(
    N: int,
    cmap: str | mcolors.Colormap | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
) -> list[str | RGBTuple | RGBATuple]:
    """
    Sample N distinct colors from a given matplotlib colormap
    returned as a list of colors in a specified format.

    Parameters
    ----------
    N : int
        Number of colors to sample.
    cmap : str | Colormap | _Unset, optional, default=_UNSET
        Name of the matplotlib colormap or Colormap object. If
        `_UNSET` uses `config.cmap`.
    cmap_range : tuple[float, float], optional, default=(0,1)
        The normalized value range in the colormap from which colors
        should be taken. By default, the entire colormap is used.
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    list[str] :
        If `fmt='hex'`.
    list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    cmap = _resolve_default(cmap, config.cmap)
    colors = plt.get_cmap(cmap)(np.linspace(cmap_range[0], cmap_range[1], N))

    return [_convert_color(c, fmt) for c in colors]


def simulate_colorblindness(
    colors: ColorType | list[ColorType] | list[str | RGBTuple | RGBATuple],
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] = 'deuteranomaly',
    severity: int = 100,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Simulate colorblindness perception of a color palette.

    Parameters
    ----------
    colors : ColorType | list[ColorType]
        Color or list of colors recognized by matplotlib.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'}, optional, default='deuteranomaly'
        Type of colorblindness to simulate. Can be shorthanded to {'d', 'p', 't'}.
    severity : int, optional, default=100
        Severity level (0-100). 100 = complete colorblindness.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    list of ColorType
        List of ColorType as perceived by colorblind vision.
    """
    if not 0 <= severity <= 100:
        raise ValueError(
            'severity must be >= 0 and <= 100!'
        )

    aliases = {
        'd': 'deuteranomaly',
        'p': 'protanomaly',
        't': 'tritanomaly'
    }
    colorblind_type = aliases.get(cvd_type, cvd_type)

    cvd_space = {
        'name': 'sRGB1+CVD',
        'cvd_type': colorblind_type,
        'severity': severity
    }

    # convert to RGB [0, 1]
    rgb = np.array(as_list(as_color(colors, fmt='rgb')))

    cvd_rgb = cspace_convert(rgb, cvd_space, 'sRGB1')
    cvd_rgb = np.clip(cvd_rgb, 0, 1)

    return [_convert_color(tuple(row), fmt=fmt) for row in cvd_rgb]


def as_color(
    c: ColorType | Sequence[ColorType],
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> (
    str
    | RGBTuple
    | RGBATuple
    | list[str | RGBTuple | RGBATuple]
):
    """
    Convert a matplotlib `ColorType` or a `list[ColorType]` into
    one of the following formats: `'hex'`, `'rgb'`, or `'rgba'`.

    Parameters
    ----------
    c : ColorType | List[ColorType]
        Matplotlib color(s). Can be named colors, rgb/rgba, hex, etc...
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    color_list = as_list(c)
    color_list = [_convert_color(c, fmt=fmt) for c in color_list]

    return _unwrap_if_single(color_list)


def _convert_color(
    c: ColorType,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
)-> str | tuple[float, float, float] | tuple[float, float, float, float]:
    """
    Convert a matplotlib `ColorType` into one of the following
    formats: `'hex`'', `'rgb'`, or `'rgba'`.
    """
    return getattr(mcolors, f'to_{fmt}')(c)


def get_complimentary_colors(
    color: ColorType | Sequence[ColorType],
    transform: Literal['lighten', 'saturate', 'desaturate'] | None = 'lighten',
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> (
    str
    | RGBTuple
    | RGBATuple
    | list[str | RGBTuple | RGBATuple]
):
    """
    Lightens, saturates, or desaturates a color or list of colors.
    Mixes colors with white to lighten, and moves colors
    towards grey to desaturate.

    `transform=None` returns color unchanged.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, HTML color, or RGB tuple.
    transform : {'lighten', 'desaturate'} | None, optional, default='lighten'
        Method to modify the color. If `None`, returns `color` unchanged.
    factor : float or int
        Modification strength.

        * If `transform='lighten'`: Blending ratio with white.

            * `factor=0`: Original color
            * `factor=1`: Pure white

        * If `transform='saturate'`: Saturation level in hsl space.

            * `factor=1`: Maximum saturation for each given color
            * `factor=0`: Grayscale

        * If `transform='desaturate'`: Desaturation amount.

            * `factor=0`: Original color
            * `factor=1`: Grayscale

    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    if transform is None:
        return as_color(color, fmt=fmt)
    method = {
        'lighten': _lighten_color,
        'saturate': _saturate_color,
        'desaturate': _desaturate_color,
    }.get(transform, _lighten_color)

    colors = to_list(color)
    colors = [method(c, factor) for c in colors]

    return as_color(colors, fmt=fmt)


def lighten_colors(
    color: ColorType | Sequence[ColorType],
    mix: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Lighten a set of color(s) by mixing the each color with white.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to lighten.
    mix : float, optional, default=0.5
        Mixing factor. `1` results in all white, while `0`
        leaves `color` unchanged.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    colors = to_list(color)
    return as_color([_lighten_color(c, mix=mix) for c in colors], fmt=fmt)


def _lighten_color(color: ColorType, mix: float = 0.5) -> 'str':
    """Lightens the given matplotlib color by mixing it with white."""
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(tuple(mixed))


def saturate_colors(
    color: ColorType | Sequence[ColorType],
    factor: float = 1,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Saturate a set of color(s) by modifying the saturation level
    of each color in hls space.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to saturate.
    factor : float, optional, default=1
        Saturation level. `1` results in maximum saturation, while `0`
        returns `color` in grayscale.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    colors = to_list(color)
    return as_color(
        [_saturate_color(c, factor=factor) for c in colors], fmt=fmt
    )


def _saturate_color(color: ColorType, factor: float = 1) -> str:
    """Saturate a color by shifting the saturation in hls space."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    rgb_new = colorsys.hls_to_rgb(h, l, factor)

    return mcolors.to_hex(rgb_new)


def desaturate_colors(
    color: ColorType | Sequence[ColorType],
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Desaturate a set of color(s) by moving each color towards gray
    in hsl space.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to desaturate.
    factor : float, optional, default=0.5
        Desaturation level. `1` returns `color` in grayscale while
        `0` returns the colors unchanged.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    colors = to_list(color)
    return as_color(
        [_desaturate_color(c, factor=factor) for c in colors], fmt=fmt
    )


def _desaturate_color(color: ColorType, factor: float = 0.5) -> str:
    """Desaturate a color by moving it toward gray."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s_new = s * (1 - factor)
    rgb_new = colorsys.hls_to_rgb(h, l, s_new)

    return mcolors.to_hex(rgb_new)


def random_colors(
    N: int,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Generate N random colors

    Parameters
    ----------
    N : int
        Number of colors to generate

    Returns
    -------
    colors : list[tuple[float, float, float]]
    """
    random_colors = np.random.rand(N, 3)
    return as_color(
        [tuple([float(c[0]), float(c[1]), float(c[2])]) for c in random_colors],  # type: ignore
        fmt=fmt
    )


def _resolve_color_kwargs(
    color: ColorType,
    c: NDArray | float | int | None,
    kwargs: dict,
    cmap: Colormap | str | None = None,
    norm: Normalize | None = None
) -> dict:
    """Resolve `color` and `c` kwargs, giving priority to `c`"""
    scatter_kwargs = dict(kwargs)
    if c is not None:
        scatter_kwargs.pop('color', None)
        scatter_kwargs['c'] = c
        if cmap is not None:
            scatter_kwargs['cmap'] = cmap
        if norm is not None:
            scatter_kwargs['norm'] = norm

    elif color is not None:
        scatter_kwargs.pop('c', None)
        scatter_kwargs['color'] = color

    else:
        scatter_kwargs.pop('c', None)
        scatter_kwargs.pop('color', None)

    return scatter_kwargs


def plot_colorset(
    colors: ColorType | int | Sequence[ColorType] = 'astro_seq',
    style: str | list[str] = 'full',
    legend: bool = True
) -> list[list[Line2D] | list[PatchCollection]]:
    """
    Plot a sample figure demonstrating a VisualAstro color set.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType | int], optional
        Color set to visualize. Passed to `get_colors`.
    style : str | list[str], optional
        Matplotlib style(s) applied while plotting. See `plt.style.available`
        or `va.style.available`.
    legend : bool, optional
        Whether to include labels for the plotted color examples.

    Returns
    -------
    list[list[Line2D | PatchCollection]]
        Artists returned by the plotting functions, grouped by plot element.
    """
    from visualastro.plotting.base.plots import plot, scatter
    colorset = get_colors(colors)
    N = len(colorset)

    r_p = 1.0
    theta = np.linspace(0, 2 * np.pi, 500)
    if N < 6:
        e_vals = np.logspace(-.9, -0.1, N)
    else:
        e_vals = np.logspace(-.9, 0.2, N)

    with np.errstate(invalid='ignore', divide='ignore'):
        a_vals = [r_p / (1 - e) for e in e_vals]
        r_vals = [a * (1 - e**2) / (1 + e * np.cos(theta)) for (a, e) in zip(a_vals, e_vals)]
    x_vals = [r * np.cos(theta) for r in r_vals]
    y_vals = [r * np.sin(theta) for r in r_vals]

    labels = [f'e={e:.1f}' for e in e_vals]

    artists = []

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(6,6))
        labels = labels if legend else None
        pl = plot(
            x_vals[:N], y_vals[:N],
            label=labels, color=colorset, lw=1,
            xlim=(-5, 3), ylim=(-4, 4),
            xlabel='X', ylabel='Y',
        )

        sc = scatter(
            0, 0,
            color='k', fc='none',
            s=55, label='star',
            compute_limits=False,
            legend_loc='upper right',
            legend_title='Eccentricity',
            legend_frameon=True
        )
        artists.extend([pl, sc])
        plt.show()

        return artists


def plot_color_deltaE(
    colorset: ColorType | int | Sequence[ColorType],
    ax: Axes | None = None,
    cmap: str | Colormap = 'viridis',
    uniform_space: str = 'CIELab',
    cvd_type: Literal['all', 'deuteranomaly', 'protanomaly', 'tritanomaly'] | None = 'all',
    normalize: bool = False,
    wspace: float = 0.4,
    hspace: float = 0.0
):
    """
    Plot pairwise CAM/CIE color-difference (deltaE) matrices for a colorset,
    optionally alongside CVD-simulated deltaE ratios showing distinguishability
    loss under color vision deficiency. The higher the value, the farther apart
    two colors are from each other.

    If `normalize=True`, the cvd ratios compare the color-differences as calculated
    in each colorspace. The higher the value, the more consistent the two color
    differences, i.e. a ratio of 1 means the pair's perceptual distance under CVD
    equals the distance under normal vision (no distinguishability loss). Ratios
    below 1 indicate a loss of distinguishability under the simulated CVD.

    Parameters
    ---------------
    colorset : ColorType | int | Sequence[ColorType]
        Colors to compare, or colormap/count reference resolvable via `get_colors`.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Target axes. Ignored if `None`; axes are created via `gridspec` (when
        `cvd_type='all'`) or `plt.subplots` (otherwise). If `cvd_type='all'`,
        `ax` should be an array-like of 4 `Axes`, ie `list[Axes, Axes, Axes, Axes]`.
    cmap : str | matplotlib.colors.Colormap, optional, default='viridis'
        Colormap passed to `imshow` for the deltaE / ratio matrices.
        It is recommended to use perceptually uniform sequential colormaps
        such as `'viridis'`, `'cividis'`, `'plasma'`, `'inferno'`, or `'magma'`.
    uniform_space : str, optional, default='CIELab'
        Perceptual uniform color space passed to `colorspacious.deltaE`.
    cvd_type : {'all', 'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default='all'
        CVD condition(s) to simulate. `'all'` plots normal deltaE plus all three
        deficiencies in a 2x2 grid. A single condition plots normal deltaE
        alongside that one condition. `None` plots only the normal deltaE matrix.
    normalize : bool, optional, default=False
        If `True`:

        * Normal deltaE matrix is scaled by its own max value (relative distance).
        * CVD matrices show the ratio CVD deltaE / normal deltaE (distinguishability
        retention). A ratio of 1 means the color pair remains equally
        distinguishable under CVD as under normal vision; lower values indicate
        greater loss of distinguishability.

    wspace : float, optional, default=0.4
        Horizontal spacing between subplots, passed to `gridspec`. Only used
        when `cvd_type='all'`.
    hspace : float, optional, default=0.0
        Vertical spacing between subplots, passed to `gridspec`. Only used
        when `cvd_type='all'`.

    Returns
    ---------------
    imgs : list[matplotlib.image.AxesImage]
        Image artists, one per plotted matrix.
    """
    from visualastro.plotting.core.axes import gridspec
    from visualastro.plotting.science.wcs_plots import imshow

    colors = np.asarray(get_colors(colorset, fmt='rgb'))

    cvds = [None, 'deuteranomaly', 'protanomaly', 'tritanomaly']

    if ax is None:
        if cvd_type == 'all':
            fig, axs = gridspec(2, 2, figsize=(10,10), hspace=hspace, wspace=wspace)
        else:
            fig, axs = plt.subplots(figsize=config.figsize)
            axs = to_list(axs)
    else:
        axs = to_list(ax)

    for axis in axs:
        axis.set_xticks(np.arange(0, len(colors), 1))
        axis.set_yticks(np.arange(0, len(colors), 1))
        for i, c in enumerate(colors):
            axis.get_xticklabels()[i].set_color(c)
            axis.get_yticklabels()[i].set_color(c)

    c1 = colors[:, np.newaxis, :]
    c2 = colors[np.newaxis, :, :]
    deltaE = colorspacious.deltaE(c1, c2, uniform_space=uniform_space)

    label = r'$\Delta E^*$'
    imgs = []

    if cvd_type == 'all' or cvd_type is None:
        deltaE_plot = deltaE / np.nanmax(deltaE) if normalize else deltaE
        label_plot = 'normalized ' + label if normalize else label
        img = imshow(
            deltaE_plot,
            ax=axs[0],
            cmap=cmap,
            vmin=0,
            cbar_width=0.02,
            cbar_label=label_plot,
            norm=None
        )
        imgs.append(img)

    for i, ax in enumerate(axs):
        cvd = cvds[i] if cvd_type == 'all' else cvd_type
        if cvd is None:
            continue

        colors_cvd = np.asarray(get_colors(
            colorset, fmt='rgb', cvd_type=cvd)
        )

        c1_cvd = colors_cvd[:, np.newaxis, :]
        c2_cvd = colors_cvd[np.newaxis, :, :]
        deltaE_cvd = colorspacious.deltaE(c1_cvd, c2_cvd, uniform_space=uniform_space)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(deltaE > 0, deltaE_cvd / deltaE, np.nan)

        cvd_plot = ratio if normalize else deltaE_cvd
        cvd_label = fr'$\Delta E^*_{{{cvd}}}$'
        cbar_label = cvd_label+r'/'+label if normalize else cvd_label
        vmax = 1 if normalize else None

        img = imshow(
            cvd_plot,
            ax=axs[i],
            cmap=cmap,
            vmax=vmax,
            vmin=0,
            cbar_width=0.02,
            cbar_label=cbar_label,
            norm=None
        )

        imgs.append(img)

    return imgs


def plot_colortable(
    colors: dict[str, ColorType] | str | None = None,
    *,
    ncols: int = 4,
    sort_colors: bool = True,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100
) -> None:
    """
    Adapted from matplotlib gallery example:
    https://matplotlib.org/stable/gallery/color/named_colors.html

    Copyright (c) 2012-2023 Matplotlib Development Team
    Licensed under the Matplotlib License (BSD-compatible)
    https://matplotlib.org/stable/users/project/license.html

    Plot all the named colors in matplotlib!

    Parameters
    ----------
    colors : dict[str, ColorType] | str | None, optional, default=None
        Dictionary containing colors to plot. If str, should be one
        of the following: `'named_colors'`, `'xkcd'`, `'base'`, or
        `'tableau'`. If `None`, plots `mcolors.CSS4_COLORS`.
    ncols : int, optional, default=4
        Number of columns to plot.
    sort_colors : bool, optional, default=True
        If `True`, sort colors by hsv value.
    """
    if isinstance(colors, str) or colors is None:
        colors = str(colors).lower() if isinstance(colors, str) else None
        if colors == 'named_colors' or colors == 'css4' or colors is None:
            colors = mcolors.CSS4_COLORS
        elif colors == 'xkcd' or colors == 'xkcd_colors':
            colors = mcolors.XKCD_COLORS
        elif colors == 'base' or colors == 'base_colors':
            colors = mcolors.BASE_COLORS
        elif colors == 'tableau' or colors == 'tableau_colors':
            colors = mcolors.TABLEAU_COLORS
        elif colors == 'visualastro' or colors == 'va':
            colors = VISUALASTRO_NAMED_COLORS
        else:
            raise ValueError(
                "colors must be either 'named_colors', 'xkcd', 'base', or 'tableau'! "
                f'got {colors}'
            )

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(colors[c])))
        )
    else:
        names = list(colors)

    color_names = [n.split('xkcd:')[1] if op.contains(n, 'xkcd:') else n for n in names]
    if cvd_type is not None:
        facecolors = {
            name:simulate_colorblindness(
                color, cvd_type=cvd_type, severity=severity
            )[0] for name, color in colors.items()
        }
    else:
        facecolors = colors

    n = len(names)
    nrows = np.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin/width,
        margin/height,
        (width-margin)/width,
        (height-margin)/height
    )

    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x, y, color_names[i],
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='center'
        )

        ax.add_patch(
            mpatches.Rectangle(
                xy=(swatch_start_x, y-9),
                width=swatch_width,
                height=18,
                facecolor=facecolors[name],
                edgecolor='0.7'
            )
        )

    plt.show()


def _has_color_mapping(mappable: ScalarMappable) -> bool:
    """Check that a ScalarMappable instance has valid data for a colormap"""
    return (
        mappable is not None
        and isinstance(mappable, ScalarMappable)
        and mappable.get_array() is not None
    )


def _resolve_scatter_norm(c_list, norm_method, log_floor=1e-10):
    """
    Resolve a matplotlib normalization object for scatter color mapping.

    Parameters
    ----------
    c_list : list of array-like | None
        List of color value arrays, one per population. If `None`, returns `None`.
    norm_method : {'log', 'global'} | None
        Normalization method.

        * `'log'` -> logarithmic scaling using `LogNorm` with global min/max.
        * `'global'` -> linear scaling using `Normalize` with global min/max.
        * `None` -> per-population normalization (matplotlib default).

    log_floor : float, optional, default=1e-10
        Minimum value clamp for `vmin` when `norm_method='log'`, to avoid
        `log(0)` errors.

    Returns
    -------
    norm : LogNorm | Normalize | None
        Normalization object to pass to `scatter`, or `None` for
        per-population default scaling.
    """
    if c_list is not None:
        global_min = min(np.nanmin(c) for c in c_list)
        global_max = max(np.nanmax(c) for c in c_list)
        if norm_method == 'log':
            norm = LogNorm(vmin=max(global_min, log_floor), vmax=global_max)
        elif norm_method == 'global':
            norm = Normalize(vmin=global_min, vmax=global_max)
        else:
            norm = None
    else:
        norm = None

    return norm
