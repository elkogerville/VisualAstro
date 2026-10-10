"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-10-06
Description:
    Color plotting functions.
"""

from collections.abc import Sequence

from typing import Literal
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    config, _resolve_default, _Unset, _UNSET
)
from visualastro.core.data import as_list, to_list
from visualastro.core.sequences import _unwrap_if_single
from visualastro.optional_dependencies._colorspacious import deltaE as cs_deltaE
from visualastro.plotting.core.colors.colors import get_color, get_colors
from visualastro.plotting.core.colors.definitions import (
    COLORSET_NAMES,
    VISUALASTRO_NAMED_COLORS,
    RGBATuple,
    RGBTuple,
)
from visualastro.plotting.core.colors.transforms import (
    simulate_colorblindness,
    _transform_colors
)


def plot_colors(
    color: ColorType | int | Sequence[ColorType] | None = None,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None = None,
    severity: int = 100,
    show_color_name: bool = True
) -> None:
    """
    Visualize one or multiple colorsets.

    Parameters
    ----------
    color : ColorType | int | Sequence[ColorType] | None, optional, default=None
        Plot each sequence of colors as a set of colored rectangle patches.
        If `None`, plots each colorset in VisualAstro.
    cvd_type : str | None, optional, default=None
        Type of colorblindness to simulate. Can be shorthanded to
        `'d'`, `'p'`, `'t'`. If `'all'`, simulates all cvd types.
    severity : float, optional, default=100
        Severity of colorblindness. Must be < 100.
    show_color_name : bool, optional, default=True
        If `True`, also plots the colorset name. Only applicable to VisualAstro
        colorsets (as opposed to a sequence of colors).

    Examples
    --------
    Display all default VisualAstro color colorsets:
    >>> plot_colors()

    Display the 'astro' colorset as perceived by protanomaly:
    >>> plot_colors('astro', cvd_type='protanomaly')

    Display the 'astro' colorset with all colorblindness simulations:
    >>> plot_colors('astro', cvd_type='all')
    """
    cvd_types = (
        ['deuteranomaly', 'protanomaly', 'tritanomaly'] if cvd_type == 'all'
        else ([cvd_type] if cvd_type else [])
    )
    if color is None:
        color_names = COLORSET_NAMES + ['random']
        colorsets = [get_colors(c) for c in color_names]
    else:
        colors = as_list(color)
        if all(c in COLORSET_NAMES for c in colors):
            color_names = colors
            colorsets = [get_colors(c) for c in colors]
        else:
            colorsets = [get_colors(color)]
            color_names = ['']*len(colorsets)
    pad = 0.1
    n_rows = len(colorsets) * (1 + len(cvd_types))
    factor = 0.3 if n_rows > 10 else 1
    fig, ax = plt.subplots(figsize=(8, n_rows*factor), layout='constrained')
    ax.axis('off')
    row = 0
    for i, colorset in enumerate(colorsets):
        for j, c in enumerate(colorset):
            ax.add_patch(mpatches.Rectangle((j, -row), 1, 1, color=c, ec='black'))

        if show_color_name and color_names[i] != '':
            ax.text(-pad, -row + 0.5, f'{color_names[i]}', va='center', ha='right')
        row += 1

        # CVD simulations
        for cvd in cvd_types:
            cvd_colors = simulate_colorblindness(colorset, cvd, severity) # type: ignore
            for j, c in enumerate(cvd_colors):
                ax.add_patch(mpatches.Rectangle((j, -row), 1, 1, color=c, ec='black'))
            label = f'({cvd})' if color_names[i] == '' else f'{color_names[i]} ({cvd})'
            ax.text(-pad, -row + 0.5, label, va='center', ha='right', fontsize=9)
            row += 1

    ax.set_xlim(-pad, max(len(get_colors(c)) for c in colorsets) + pad)
    ax.set_ylim(-n_rows + 1 - pad, 1 + pad)

    plt.show()


def plot_colorset(
    colors: ColorType | int | Sequence[ColorType] = 'astro_seq',
    ax: Axes | None = None,
    lw: float = 2,
    legend: bool = True
) -> list[list[Line2D] | list[PatchCollection]]:
    """
    Plot a sample figure demonstrating a VisualAstro color set.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType | int], optional, default='astro_seq'
        Color set to visualize. Passed to `get_colors`.
    ax : matplotlib.axes.Axes | None, optional, default=None
        The Axes object on which to plot the histogram. If `None`,
        uses `plt.gca()`.
    lw : float, optional, default=2
        Linewidth of each line.
    legend : bool, optional, default=True
        If `True`, plots the legend.

    Returns
    -------
    list[list[Line2D | PatchCollection]]
        Artists returned by the plotting functions, grouped by plot element.
    """
    from visualastro.plotting.base.plots import plot, scatter
    from visualastro.plotting.core.axes import get_ax, set_title

    ax = get_ax(ax)
    if isinstance(colors, str) and colors in mpl.color_sequences:
        set_title(f'colorset: {colors}', ax=ax)
    colorset = get_colors(colors)
    N = len(colorset)

    r_p = 1.0
    theta = np.linspace(0, 2 * np.pi, 500)
    e_vals = np.linspace(0.1, 0.99, N)

    with np.errstate(invalid='ignore', divide='ignore'):
        a_vals = [r_p / (1 - e) for e in e_vals]
        r_vals = [a * (1 - e**2) / (1 + e * np.cos(theta)) for (a, e) in zip(a_vals, e_vals)]
    x_vals = [r * np.cos(theta) for r in r_vals]
    y_vals = [r * np.sin(theta) for r in r_vals]

    labels = [f'e={e:.1f}' for e in e_vals]

    artists = []

    labels = labels if legend else None
    pl = plot(
        x_vals[:N], y_vals[:N],
        ax=ax,
        label=labels, color=colorset, lw=lw,
        xlim=(-5, 3), ylim=(-4, 4),
        xlabel='X', ylabel='Y',
    )

    sc = scatter(
        0, 0,
        ax=ax,
        color='k', fc='none',
        s=55, label='star' if legend else None,
        compute_limits=False,
        legend_loc='upper right',
        legend_title='Eccentricity',
        legend_frameon=True
    )
    artists.extend([pl, sc])

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
    ----------
    colorset : ColorType | int | Sequence[ColorType]
        Colors to compare, or colormap/count reference resolvable via `get_colors`.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Target axes. Ignored if `None`; axes are created via `gridspec` (when
        `cvd_type='all'`) or `plt.subplots` (otherwise). If `cvd_type='all'`,
        `ax` should be an ArrayLike of 4 `Axes`, ie `list[Axes, Axes, Axes, Axes]`.
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
    -------
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
    deltaE = cs_deltaE(c1, c2, uniform_space=uniform_space)

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
        deltaE_cvd = cs_deltaE(c1_cvd, c2_cvd, uniform_space=uniform_space)

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


# ----------------------------------------------------------------------------
# The following function is adapted from matplotlib:
# https://github.com/matplotlib/matplotlib
#
# License agreement for matplotlib versions 1.3.0 and later
# =========================================================

# 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
# ("MDT"), and the Individual or Organization ("Licensee") accessing and
# otherwise using matplotlib software in source or binary form and its
# associated documentation.

# 2. Subject to the terms and conditions of this License Agreement, MDT
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that MDT's
# License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
# 2012- Matplotlib Development Team; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib .

# 4. MDT is making matplotlib available to Licensee on an "AS
# IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between MDT and
# Licensee.  This License Agreement does not grant permission to use MDT
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using matplotlib ,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.

# License agreement for matplotlib versions prior to 1.3.0
# ========================================================

# 1. This LICENSE AGREEMENT is between John D. Hunter ("JDH"), and the
# Individual or Organization ("Licensee") accessing and otherwise using
# matplotlib software in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, JDH
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that JDH's
# License Agreement and JDH's notice of copyright, i.e., "Copyright (c)
# 2002-2011 John D. Hunter; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib  or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib.

# 4. JDH is making matplotlib  available to Licensee on an "AS
# IS" basis.  JDH MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, JDH MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. JDH SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between JDH and
# Licensee.  This License Agreement does not grant permission to use JDH
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using matplotlib,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.
# ----------------------------------------------------------------------------

def plot_colortable(
    colors: dict[str, ColorType] | str | None = None,
    *,
    ncols: int = 4,
    sort_colors: bool = True,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100
) -> None:
    """
    Plot a grid of colors with their names.

    This function is adapted from a Matplotlib gallery example:
    https://matplotlib.org/stable/gallery/color/named_colors.html

    Copyright (c) 2012-2023 Matplotlib Development Team.
    Licensed under the Matplotlib License (BSD-compatible)
    https://matplotlib.org/stable/users/project/license.html

    Parameters
    ----------
    colors : dict[str, ColorType] | str | None, optional, default=None
        Dictionary containing colors to plot, or one of the following:

        * `'named_colors'` or `None`: VisualAstro and Matplotlib named colors
        * `'mpl'` or `'matplotlib'` or `'mpl_colors'` or `'matplotlib_colors'` or `'css4'`: Matplotlib named colors
        * `'xkcd'` or `'xkcd_colors'`: XKCD named colors
        * `'visualastro'` or `'va'`: VisualAstro named colors
        * `'base'` or `'base_colors'`: Matplotlib base colors
        * `'tableau'` or `'tableau_colors'`: Matplotlib tableau colors
        * `'all'` or `'all_colors'`: All of the above

    ncols : int, optional, default=4
        Number of columns to plot.
    sort_colors : bool, optional, default=True
        If `True`, sort colors by hsv value.
    """
    if isinstance(colors, str) or colors is None:
        colors = str(colors).lower() if isinstance(colors, str) else None
        if colors == 'named_colors' or colors is None:
            colors = mcolors.CSS4_COLORS | VISUALASTRO_NAMED_COLORS
        elif colors in {
            'mpl', 'matplotlib', 'mpl_colors', 'matplotlib_colors', 'css4'
        }:
            colors = mcolors.CSS4_COLORS
        elif colors in {'visualastro', 'va'}:
            colors = VISUALASTRO_NAMED_COLORS
        elif colors in {'base', 'base_colors'}:
            colors = mcolors.BASE_COLORS
        elif colors in {'tableau', 'tableau_colors'}:
            colors = mcolors.TABLEAU_COLORS
        else:
            all_colors = (
                mcolors.CSS4_COLORS |
                mcolors.BASE_COLORS |
                mcolors.TABLEAU_COLORS |
                VISUALASTRO_NAMED_COLORS
            )
            xkcd_stripped = {
                k.replace('xkcd:', ''): v \
                    for k, v in mcolors.XKCD_COLORS.items()
            }
            overlap_stripped_names = all_colors.keys() & xkcd_stripped.keys()
            xkcd_resolved = {
                ('xkcd:' if k in overlap_stripped_names else '') + k: v \
                    for k, v in xkcd_stripped.items()
            }
            if colors in {'xkcd', 'xkcd_colors'}:
                colors = xkcd_resolved
            elif colors in {'all', 'all_colors'}:
                colors = all_colors | xkcd_resolved
            else:
                raise ValueError(
                    "colors must be a dictionary or one of the following: "
                    "'named_colors', 'mpl', 'matplotlib', 'mpl_colors', "
                    "'matplotlib_colors', 'css4', 'visualastro', 'va', "
                    "'base', 'base_colors', 'tableau', 'tableau_colors', "
                    "'xkcd', 'xkcd_colors', 'all', 'all_colors'. "
                    f"Got '{colors}'."
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
            text_pos_x, y, names[i],
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


def _get_plot_color(
    color,
    *,
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    cvd_type,
    severity,
    fmt
) -> str | RGBTuple | RGBATuple | None:
    transform = _resolve_default(transform, config.color_transform)
    factor = _resolve_default(factor, config.color_transform_factor)

    if color is None or isinstance(color, str) and color in {'face', 'none'}:
         return color

    color = _transform_colors(
        get_color(color),
        transform=transform,
        factor=factor,
        fmt=fmt
    )

    if cvd_type is not None:
        color = simulate_colorblindness(
            color,
            cvd_type=cvd_type,
            severity=severity,
            fmt=fmt
        )

    return _unwrap_if_single(color)


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


def _has_color_mapping(mappable: ScalarMappable) -> bool:
    """Check that a ScalarMappable instance has valid data for a colormap"""
    return (
        mappable is not None
        and isinstance(mappable, ScalarMappable)
        and mappable.get_array() is not None
    )


def _resolve_scatter_norm(c_list, norm_method, log_floor=1e-10):
    """
    Resolve a Matplotlib normalization object for scatter color mapping.

    Parameters
    ----------
    c_list : list[ArrayLike] | None
        List of color value arrays, one per population. If `None`, returns `None`.
    norm_method : {'log', 'global'} | None
        Normalization method.

        - `'log'` -> logarithmic scaling using `LogNorm` with global min/max.
        - `'global'` -> linear scaling using `Normalize` with global min/max.
        - `None` -> per-population normalization (Matplotlib default).

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
