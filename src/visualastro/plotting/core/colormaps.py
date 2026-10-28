"""
Author: Elko Gerville-Reache
Date Created: 2026-07-04
Date Modified: 2026-10-06
Description:
    Functions related to colormaps in plotting.
    To define custom colormaps, define them at
    the bottom of this file in `VISUALASTRO_CMAPS`.
"""

from collections.abc import Sequence
from typing import Literal
import warnings

from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import config, _Unset, _UNSET
from visualastro.core.data import to_list
from visualastro.core.kwargs import _resolve_default
from visualastro.optional_dependencies.register import (
    _offer_dependency, _require_dependency
)
from visualastro.optional_dependencies._cmasher import cmasher, _HAS_CMASHER
from visualastro.optional_dependencies._colorspacious import cspace_converter
from visualastro.optional_dependencies._tol_colors import _HAS_TOLCOLORS
from visualastro.plotting.core.colors.utils import _is_colorset


def get_cmap(
    cmap: mcolors.Colormap | str,
    cmap_range: tuple[float, float] = (0, 1),
    N: int = 256,
    bad_color: ColorType | None = None,
    under_color: ColorType | None = None,
    over_color: ColorType | None = None
) -> mcolors.Colormap:
    """
    Retrieve a colormap by name or object.

    Parameters
    ----------
    cmap : mcolors.Colormap | str
        Colormap object or string name, looked up via Matplotlib's
        colormap registry.
    cmap_range : tuple[float, float], optional, default=(0, 1)
        Normalized sub-range to extract from `cmap`.
    N : int, optional, default=256
        Resolution of the resampled sub-range colormap. Ignored if
        `cmap_range == (0, 1)`.
    bad_color : ColorType | None, optional, default=None
        Color for masked/invalid (NaN) data. If `None`, leaves unchanged.
    under_color : ColorType | None, optional, default=None
        Color for values below the normalization range. If `None`, leaves
        unchanged.
    over_color : ColorType | None, optional, default=None
        Color for values above the normalization range. If `None`, leaves
        unchanged.

    Returns
    -------
    mcolors.Colormap
        The requested colormap.

    Raises
    ------
    ValueError
        If `cmap_range` does not have exactly 2 elements.
    """
    cm_range = tuple(cmap_range)
    if len(cmap_range) != 2:
        raise ValueError(
            'cmap_range must be a tuple[min, max]!'
        )

    if isinstance(cmap, str) and _is_colorset(cmap) and not _is_valid_cmap(cmap):
        out_cmap = create_cmap(cmap)
    else:
        out_cmap = plt.get_cmap(cmap)
    if cm_range != (0, 1):
        colors = out_cmap(np.linspace(cm_range[0], cm_range[1], N))
        out_cmap = create_cmap(colors, N=N, name=out_cmap.name + '_sub')

    return out_cmap.with_extremes(
        bad=bad_color,
        under=under_color,
        over=over_color,
    )


def create_cmap(
    colors: Sequence[ColorType] | str,
    kind: Literal['continuous', 'discrete'] = 'continuous',
    positions: Sequence[float] | None = None,
    N: int = 256,
    name: str = 'custom_cmap'
) -> mcolors.LinearSegmentedColormap | mcolors.ListedColormap:
    """
    Creates a colormap from a color sequence,
    with continuous or discrete interpolation.

    Parameters
    ----------
    colors : Sequence[ColorType] | str
        Color specifications (hex, named colors, RGB tuples, etc.).
        The cmap will be created from these colors.
    kind : {'continuous', 'discrete'}, optional, default='continuous'
        `'continuous'` returns a `LinearSegmentedColormap` (interpolated).
        `'discrete'` returns a `ListedColormap` (stepwise, no interpolation).
    positions : list[float] | None, optional, default=None
        Positions in [0, 1] for each color, monotonically increasing,
        and must start with 0 and end with 1. Only used when `kind='continuous'`.
        If `None`, colors are evenly spaced. Ignored when `kind='discrete'`.
    N : int, optional, default=256
        The number of RGB quantization levels. Valid for `kind='continuous'`.
        Ignored for `kind='discrete'`.
    name : str, optional, default='custom_cmap'
        Name assigned to the colormap.

    Returns
    -------
    LinearSegmentedColormap | ListedColormap
        Colormap object corresponding to `kind`.

    Notes
    -----
    `N` has no effect for `ListedColormap`: its resolution is fixed to
    `len(colors)` by construction. Passing `N` with `kind='discrete'`
    triggers a warning, not an error.
    """
    if isinstance(colors, str) and colors in mpl.color_sequences:
        name = colors
        colors = mpl.color_sequences[colors]

    rgba_list = [mcolors.to_rgba(color) for color in colors]

    if kind == 'continuous':
        if positions is None:
            positions = list(np.linspace(0, 1, len(rgba_list)))
        return mcolors.LinearSegmentedColormap.from_list(
            name, list(zip(positions, rgba_list)), N=N
        )

    elif kind == 'discrete':
        if N != 256:
            warnings.warn("N is ignored for kind='discrete'", stacklevel=2)
        return mcolors.ListedColormap(rgba_list, name=name)

    else:
        raise ValueError(f"Invalid kind: {kind!r}. Expected 'continuous' or 'discrete'.")


def plot_cmap_lightness(
    cmap: str | mcolors.Colormap | list[str | mcolors.Colormap],
    ax: Axes | None = None,
    s: float = 300,
    offset: float = 0,
    ncols: int = 1,
    legend_label: bool = True,
    inline_label: bool = False,
    inline_label_offset: float = 0,
    xticks: bool = True,
    xtick_labels: bool = True,
    **kwargs
) -> list[PathCollection]:
    """
    Plot L* (CAM02-UCS lightness) as a function of colormap index.

    Parameters
    ----------
    cmap : str | mcolors.Colormap | list[str | mcolors.Colormap]
        Colormap name(s) or instance(s).
    ax : matplotlib.axes.Axes, optional, default=None
        Target axes. Created via `plt.subplots` if None.
    s : float, optional, default=300
        Marker size passed to `ax.scatter`.
    offset : float, optional, default=0
        Gap inserted between adjacent columns along x. Column `col` spans
        `[col*(1+offset), col*(1+offset)+1]`.
    ncols : int, optional, default=1
        Number of columns before wrapping to a new row.
    legend_label : bool, optional, default=True
        If `True`, add each colormap to the axes legend.
    inline_label : bool, optional, default=False
        If `True`, annotate the colormap name directly above its column
        instead of relying on the legend. Recommended when `ncols > 1`.
    inline_label_offset : float, optional, default=0
        Additional vertical stagger applied to inline labels on odd-indexed
        columns (`col % 2 == 1`) to reduce label collisions when adjacent
        columns are closely spaced. Has no effect if `inline_label=False`.
    xticks : bool, optional, default=True
        If `True`, plot xticks.
    xtick_labels : bool, optional, default=True
        If `True`, plot xtick labels.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.scatter`.

    Returns
    -------
    scatters : list[matplotlib.collections.PathCollection]
        Scatter artists, one per colormap.
    """
    _require_dependency('colorspacious')
    from visualastro.plotting.core.legend import legend as _legend

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)

    ncols = int(ncols)
    if ncols < 1:
        raise ValueError('ncols must be >= 1!')

    cmaps = to_list(cmap)
    samples = np.linspace(0.0, 1.0, 1000)
    row_spacing = 200
    col_stride = 1 + offset
    label_pad = 20
    scatters = []

    for i, c in enumerate(cmaps):
        row, col = divmod(i, ncols)
        x_start = col * col_stride
        x = np.linspace(x_start, x_start + 1, 1000)
        y_offset = -row * row_spacing if ncols != 1 else 0

        c = get_cmap(c)
        rgb = c(samples)[np.newaxis, :, :3]
        lab = cspace_converter('sRGB1', 'CAM02-UCS')(rgb)
        L = lab[0, :, 0]

        scatter = ax.scatter(x, L + y_offset, s=s, c=x, cmap=c, **kwargs)
        scatters.append(scatter)

        if _HAS_CMASHER and legend_label:
            _offer_dependency('cmasher')
            cmasher.set_cmap_legend_entry(scatter, c.name)

        if inline_label:
            stagger = inline_label_offset if col % 2 else 0
            ax.text(
                x_start + 0.5, y_offset + 100 + label_pad + stagger, c.name,
                ha='center', va='bottom',
                fontsize=config.fontsizes.text
            )

    if legend_label:
        _legend(ax=ax)

    ax.set_ylabel(r'L$^*$', fontsize=config.axes.label_fontsize)

    if ncols != 1:
        n_rows = -(-len(cmaps) // ncols)
        yticks = [r * -row_spacing + v for r in range(n_rows) for v in (0, 50, 100)]
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: f'{y % row_spacing:.0f}')
        )

    if not xtick_labels:
        ax.set_xticklabels([])
    if not xticks:
        ax.set_xticks([])

    return scatters


def get_norm(
    norm: str | mcolors.Normalize | None,
    vmin: float | np.floating | None = None,
    vmax: float | np.floating | None = None,
    *,
    linear_width: float | _Unset = _UNSET,
    gamma: float | _Unset = _UNSET,
    vcenter: float | None = None,
) -> mcolors.Normalize | None:
    """
    Return a Matplotlib or Astropy normalization object.

    Parameters
    ----------
    norm : {'asinh', 'asinhnorm', 'log', 'power', 'twoslope'} | Normalize | None
        Normalization algorithm. A `Normalize` instance is returned unchanged.
        `None` returns `None`.
    vmin, vmax : float | np.floating | None, optional, default=None
        Normalization limits. `None` autoscales. Both are required for
        `'twoslope'` unless `vcenter` is given.
    linear_width : float | None, optional, default=None
        Linear region width for `'asinhnorm'`. `None` uses `config.linear_width`.
    gamma : float | None, optional, default=None
        Exponent for `'power'`. `None` uses `config.gamma`.
    vcenter : float | None, optional, default=None
        Center for `'twoslope'`. `None` uses the midpoint of `vmin` and `vmax`.

    Returns
    -------
    Normalize | None
        Normalization object, or `None` if `norm` is `None`.

    Raises
    ------
    ValueError
        If `norm` is unsupported, or `'twoslope'` lacks the limits it needs.
    """
    if norm is None or isinstance(norm, mcolors.Normalize):
        return norm

    linear_width = _resolve_default(linear_width, config.linear_width)
    gamma = _resolve_default(gamma, config.gamma)
    if vmin is not None: vmin = float(vmin)
    if vmax is not None: vmax = float(vmax)

    def _twoslope() -> mcolors.TwoSlopeNorm:
        if vcenter is None and (vmin is None or vmax is None):
            raise ValueError("'twoslope' requires vmin and vmax, or vcenter.")
        center = (vmin + vmax) / 2 if vcenter is None else vcenter
        return mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

    factories = {
        'asinh': lambda: ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
        'asinhnorm': lambda: mcolors.AsinhNorm(vmin=vmin, vmax=vmax, linear_width=linear_width),
        'log': lambda: mcolors.LogNorm(vmin=vmin, vmax=vmax),
        'power': lambda: mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
        'twoslope': _twoslope,
    }
    key = norm.lower()
    if key not in factories:
        raise ValueError(f'Unsupported norm {norm!r}. Supported: {list(factories)}')
    return factories[key]()


# VISUALASTRO COLOR MAPS
# ----------------------
BuWhRd = create_cmap(
    ['#191970', '#0000FF', '#FFFFFF', '#FF0000', '#8b0000'],
    positions=[0, 0.25, 0.5, 0.75, 1],
    name='BuWhRd'
)

VISUALASTRO_CMAPS: dict[str, mcolors.Colormap] = {
    'BuWhRd': BuWhRd,
    'nuclear_waste': create_cmap(
        ['#1CFF00', '#A7FF63', '#D1E61C', '#A2A838', '#6CA838'],
        name='nuclear_waste'
    ),
    'shrek': create_cmap(
        ['#6CA838', '#1CFF00', '#A7FF63', '#D1E61C', '#A2A838', '#7E8140', '#575931'],
        name='shrek'
    ),
    'crayons_neon': create_cmap(
        ['#FF1DCE', '#CCFF00', '#00B9FB'],
        name='crayons_neon'
    ),
    'debos': create_cmap(
        ['#3464F5', '#93BFE6', '#8FE3BC', '#F4C572', '#F56D53', '#D3153A', '#9C0569'],
        name='debos'
    ),
}

if _HAS_TOLCOLORS:
    iridescent = plt.get_cmap('tol.iridescent').copy()
    iridescent.set_bad(color='white')
    tol_rainbow = plt.get_cmap('tol.rainbow').copy()
    tol_rainbow.set_bad(color='white')

    VISUALASTRO_CMAPS['iridescent'] = iridescent
    VISUALASTRO_CMAPS['tol_rainbow'] = tol_rainbow

CMAPNAMES = [key for key in VISUALASTRO_CMAPS.keys()]


def _is_valid_cmap(cmap: str | mcolors.Colormap) -> bool:
    """
    Check whether `cmap` is a usable colormap.

    Parameters
    ----------
    cmap : str | Colormap
        Registered colormap name or `Colormap` instance.

    Returns
    -------
    bool
        `True` if `cmap` is a `Colormap` instance or a name in `mpl.colormaps`.
    """
    if isinstance(cmap, mcolors.Colormap):
        return True
    return isinstance(cmap, str) and cmap in mpl.colormaps
