"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-05-11
Description:
    Functions related to colors and colormaps in plotting.
Dependencies:
    - matplotlib
    - numpy
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

from colorspacious import cspace_convert
import colorsys
from typing import Literal, TypeAlias
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib.colors import TABLEAU_COLORS, Colormap, ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray
import tol_colors as tc

from visualastro.core.config import config, _resolve_default, _Unset, _UNSET
from visualastro.core.numerical_utils import as_list, to_list, _unwrap_if_single


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
        '#FFB000', '#9FB7FF', '#CFE23C', '#FE6100'
    ],
    'MSG': ['#483D8B', '#D81B60', '#DBB0FF', '#26DCBA', '#7D7FF3', '#CFE23C'],
    'MSGII': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3', '#CFE23C'],
    'MSG_seq': ['#483d8b', '#7d7ff3', '#dbb0ff', '#D81B60', '#26dcba', '#cfe23c'],
    'default': list(TABLEAU_COLORS.values()),
    'smplot': [
        'k', '#FF0000', '#0000FF', '#00FF00',
        '#00FFFF', '#FF00FF', '#FFFF00'
    ],
    'set2': mpl.color_sequences['Set2'],
    'dark2': mpl.color_sequences['Dark2'],
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
    'tab20b': mpl.color_sequences['tab20b'],
}
# COLORSETS ALIASES
# -----------------
COLORSETS['va'] = COLORSETS['visualastro']


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
    bbypnk: ColorType = '#DBB0FF'
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

    def plot(self, cols: int = 4) -> None:
        """
        Display color swatches in a grid with labels.

        Parameters
        ----------
        cols : int, default 4
            Number of columns in grid
        """
        color_list = [(f.name, getattr(self, f.name)) for f in fields(self)]
        rows = (len(color_list) + cols - 1) // cols
        fig, ax = plt.subplots(figsize=(cols * 0.9, rows * 1.2), tight_layout=True)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        ax.axis('off')

        for idx, (name, color) in enumerate(color_list):
            row, col = divmod(idx, cols)
            x, y = col, row

            rect = mpatches.FancyBboxPatch(
                (x + 0.25, y + 0.1), 0.5, 0.4,
                boxstyle='round,pad=0.02',
                facecolor=color,
                edgecolor='#333',
                linewidth=0.5
            )
            ax.add_patch(rect)

            ax.text(
                x + 0.5, y + 0.52,
                name,
                ha='center', va='top',
                fontsize=7,
                fontweight='normal'
            )

        plt.show()

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + ':']
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f'  {field.name}: {value!r}')
        return '\n'.join(lines)

_NAMED_COLORS = vars(Color())


def get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    transform: Literal['lighten', 'desaturate'] | None = None,
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100
) -> list[str | RGBTuple | RGBATuple]:
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

        If `_UNSET`, uses `config.default_colorset`.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap for sampling when colors is int. If `_UNSET`,
        uses `config.cmap`.
    transform : {'lighten', 'desaturate'} | None, optional, default='lighten'
        Method to modify the color. If `None`, returns `color` unchanged.
    factor : float or int
        Modification strength.

        - If `transform='lighten'`: Blending ratio with white.

            * `factor=0`: Original color
            * `factor=1`: Pure white

        - If `transform='desaturate'`: Desaturation amount.

            * `factor=0`: Original color
            * `factor=1`: Full gray

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
    """
    colors = _get_colors(colors, cmap, transform=transform, factor=factor, fmt=fmt)
    if cvd_type is not None:
        colors = simulate_colorblindness(colors, cvd_type=cvd_type, severity=severity)
    return colors


def _get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    transform: Literal['lighten', 'desaturate'] | None = None,
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """Helper function for `get_colors`"""
    cmap = _resolve_default(cmap, config.cmap)

    if colors is _UNSET:
        colorset = COLORSETS.get(config.default_colorset, COLORSETS['visualastro'])
        return as_list(
            get_complimentary_colors(
                as_color(colorset, fmt=fmt),
                transform=transform,
                factor=factor
            )
        )

    if colors is None or isinstance(colors, str) and colors in {'face', 'none'}:
        return [colors] # type: ignore

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
            return as_list(
                get_complimentary_colors(
                    as_color(colorset, fmt),
                    transform=transform,
                    factor=factor
                )
            )
        if colors in _NAMED_COLORS:
            return as_list(
                get_complimentary_colors(
                    as_color(_NAMED_COLORS[colors], fmt),
                    transform=transform,
                    factor=factor
                )
            )

        else:
            return as_list(
                get_complimentary_colors(
                    as_color(colors, fmt),
                    transform=transform,
                    factor=factor
                )
            )

    if isinstance(colors, (np.ndarray, list)):
        return [
            get_colors(c, fmt=fmt, transform=transform, factor=factor)[0]
            for c in colors
        ]

    if isinstance(colors, tuple):
        return as_list(
            get_complimentary_colors(
                as_color(colors, fmt),
                transform=transform,
                factor=factor
            )
        )

    # if user passes an integer N, sample a cmap for N colors
    if isinstance(colors, int):
        return as_list(
            get_complimentary_colors(
                sample_cmap(colors, cmap=cmap, fmt=fmt),
                transform=transform,
                factor=factor
            )
        )

    raise TypeError(
        'colors must be None, a str colorset name, a str color, '
        f'a list of colors, or an integer! got {type(colors).__name__}'
    )


def get_cmap(
    cmap: mcolors.Colormap | str | int,
    bad_color: ColorType | None = None
) -> mcolors.Colormap:
    """
    Retrieve a colormap by name or return the input colormap.

    Parameters
    ----------
    cmap : mcolors.Colormap | str | int
        Colormap object or string name. If a string, attempts lookup in CMAPS
        registry before falling back to matplotlib's colormap registry.
        If an int, returns `tol_colors.rainbow_discrete(colors)`.
    bad_color : ColorType | None, optional, default=None
        Bad data color (`bad_color`). If None, leaves the colormap unchanged.

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

    if isinstance(cmap, str):
        cmap_name = cmap.removesuffix('_r')
        cm = CMAPS.get(cmap_name, None)
        if cm is not None:
            cm = cm.reversed() if cmap.endswith('_r') else cm
            return set_bad_color(cm, bad_color)

    if isinstance(cmap, int):
        return set_bad_color(tc.rainbow_discrete(cmap), bad_color)

    return set_bad_color(plt.get_cmap(cmap), bad_color)


def create_cmap(
    colors: list[ColorType] | int,
    positions: list[float] | None = None,
    name: str = 'continous_cmap'
) -> mcolors.LinearSegmentedColormap | ListedColormap:
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


# VISUALASTRO COLOR MAPS
# ----------------------
iridescent = plt.get_cmap('tol.iridescent').copy()
iridescent.set_bad(color='white')
BuWhRd = create_cmap(
    ['#191970', '#0000FF', '#FFFFFF', '#FF0000', '#8b0000'],
    [0, 0.25, 0.5, 0.75, 1],
    'BuWhRd'
)
nuclear_waste = create_cmap(
    ['#1CFF00', '#A7FF63', '#D1E61C', '#A2A838', '#6CA838'],
    name='nuclear_waste'
)
tol_rainbow = plt.get_cmap('tol.rainbow').copy()
tol_rainbow.set_bad(color='white')

CMAPS: dict[str, mcolors.Colormap] = {
    'iridescent': iridescent,
    'BuWhRd': BuWhRd,
    'tol_rainbow': tol_rainbow,
    'nuclear_waste': nuclear_waste,
}
CMAPNAMES = [key for key in CMAPS.keys()]


def sample_cmap(
    N: int,
    cmap: str | mcolors.Colormap | _Unset = _UNSET,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
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
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))

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
        Type of colorblindness to simulate.
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
    cvd_space = {
        'name': 'sRGB1+CVD',
        'cvd_type': cvd_type,
        'severity': severity
    }

    # convert to RGB [0, 1]
    rgb = np.array(as_color(colors, fmt='rgb'))

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
    transform: Literal['lighten', 'desaturate'] | None = 'lighten',
    factor: float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> (
    str
    | RGBTuple
    | RGBATuple
    | list[str | RGBTuple | RGBATuple]
):
    """
    Lightens or desaturates a color or list of colors.
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

        - If `transform='lighten'`: Blending ratio with white.

            * `factor=0`: Original color
            * `factor=1`: Pure white

        - If `transform='desaturate'`: Desaturation amount.

            * `factor=0`: Original color
            * `factor=1`: Full gray

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
        'desaturate': _desaturate_color
    }.get(transform, _lighten_color)

    colors = to_list(color)
    colors = [method(c, factor) for c in colors]

    return as_color(colors, fmt=fmt)


def _lighten_color(color: ColorType, mix: float = 0.5) -> 'str':
    """Lightens the given matplotlib color by mixing it with white"""
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(tuple(mixed))


def _desaturate_color(color: ColorType, factor: float = 0.5) -> str:
    """Desaturate a color by moving it toward gray"""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s_new = s * (1 - factor)
    rgb_new = colorsys.hls_to_rgb(h, l, s_new)

    return mcolors.to_hex(rgb_new)


def random_colors(N: int, fmt: Literal['hex', 'rgb', 'rgba'] = 'hex') -> list[ColorType]:
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


COLORSETS['random'] = random_colors(int(np.random.randint(0, 20, 1)))


def _resolve_color_kwargs(
    color: ColorType,
    c: NDArray | float | int | None,
    kwargs: dict,
    cmap: Colormap | str | None = None
) -> dict:
    """Resolve `color` and `c` kwargs, giving priority to `c`"""
    scatter_kwargs = dict(kwargs)
    if c is not None:
        scatter_kwargs.pop('color', None)
        scatter_kwargs['c'] = c
        if cmap is not None:
            scatter_kwargs['cmap'] = cmap
    elif color is not None:
        scatter_kwargs.pop('c', None)
        scatter_kwargs['color'] = color
    else:
        scatter_kwargs.pop('c', None)
        scatter_kwargs.pop('color', None)

    return scatter_kwargs

COLORNAMES = [key for key in COLORSETS.keys()]
