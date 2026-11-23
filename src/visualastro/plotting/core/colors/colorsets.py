"""
Author: Elko Gerville-Reache
Date Created: 2026-11-02
Date Modified: 2026-11-02
Description:
    VisualAstro colorset registry.
"""

from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType, LineStyleType

from visualastro.core.configuration import _UNSET, _Unset
from visualastro.plotting.core.colors._registry import _ColorRegistry
from visualastro.plotting.core.colors.colors import get_color, get_colorset
from visualastro.plotting.core.colors.definitions import (
    COLORSET_ALIASES,
    COLORSETS,
    NAMED_COLOR_ALIASES,
    VISUALASTRO_NAMED_COLORS,
)
from visualastro.plotting.core.colors.plots import (
    plot_colors,
    plot_colortable,
    preview_colorset,
)

__all__ = ['Colorsets', 'Colors']


class Colorsets(_ColorRegistry[list[ColorType]]):
    """
    Query and plot registered VisualAstro colorsets.

    See :ref:`colorset guide <colorset-guide>` for usage documentation.
    """

    _SORT_KEYS: dict[str, Callable[[str, list], object]] = {
        'name': lambda name, colors: name.lower(),
        'length': lambda name, colors: len(colors),
    }

    _names = COLORSETS
    _alias_names = COLORSET_ALIASES

    def __init__(self):
        self._registry = mpl.color_sequences

    @property
    def available(self) -> list[str]:
        """
        Canonical names of the registered VisualAstro colorsets.

        Aliases and Matplotlib built-in colorsets (e.g. ``'tab10'``) are
        excluded. Use :attr:`aliases` to list alias names.

        Returns
        -------
        list of str
            Colorset names present in both ``COLORSETS`` and the Matplotlib
            color sequence registry, in registration order.

        See Also
        --------
        aliases : Mapping of alias names to canonical names.
        query : Filter and sort colorset names.
        """
        return self._available

    @property
    def aliases(self) -> dict[str, str]:
        """
        Mapping of colorset aliases to canonical names.

        Only aliases registered in the Matplotlib color sequence registry
        are included.

        Returns
        -------
        dict of str to str
            Keys are alias names, values are the canonical names in
            :attr:`available` they resolve to.

        See Also
        --------
        available : Canonical colorset names.
        """
        return self._aliases

    def get(
        self,
        name: str,
        *,
        fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
        transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
        factor: float | _Unset = _UNSET,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100,
    ):
        """
        Get a colorset by name or alias.

        Thin wrapper around :func:`~visualastro.get_colorset`. See that
        function for the full description of the options.

        Parameters
        ----------
        name : str
            Colorset name or alias.
        fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
            Output color format.
        transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None, optional
            Color transform to apply. Default is set by ``get_colorset``.
        factor : float, optional
            Transform strength. Default is set by ``get_colorset``.
        cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
            Color vision deficiency to simulate.
        severity : int, optional, default=100
            Severity of the simulated deficiency.

        Returns
        -------
        list
            Colors in the requested format.

        See Also
        --------
        visualastro.get_colorset : Underlying implementation.
        """
        return get_colorset(
            name,
            fmt=fmt,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity,
        )

    def query(
        self,
        *,
        length : int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sort_by: Literal['name', 'length'] | None = None,
        reverse: bool = False,
    ) -> list[str]:
        """
        Filter and sort colorset names.

        Parameters
        ----------
        length : int | None, optional, default=None
            Absolute length of colors.
        min_length : int | None, optional, default=None
            Minimum number of colors (inclusive).
        max_length : int | None, optional, default=None
            Maximum number of colors (inclusive).
        sort_by : {'name', 'length'} | None, optional, default=None
            Sort criterion. If ``None``, registration order is kept.
        reverse : bool, optional, default=False
            Reverse the sort order.

        Returns
        -------
        names : list[str]
            Matching canonical colorset names.

        Raises
        ------
        ValueError
            If ``sort_by`` is not a valid criterion.
        """
        if length is not None and (min_length is not None or max_length is not None):
            raise ValueError(
                "'length' is mutually exclusive with 'min_length' and 'max_length'."
            )
        if (
            min_length is not None and max_length is not None
            and min_length > max_length
        ):
            raise ValueError(
                f'min_length ({min_length}) cannot exceed max_length ({max_length}).'
            )
        if sort_by is not None and sort_by not in self._SORT_KEYS:
            raise ValueError(
                f'sort_by must be one of {list(self._SORT_KEYS)}, got {sort_by!r}'
            )

        names = []
        for name in self.available:
            n = len(self._registry[name])
            if length is not None and n != length:
                continue
            if min_length is not None and n < min_length:
                continue
            if max_length is not None and n > max_length:
                continue
            names.append(name)

        if sort_by is not None:
            key = self._SORT_KEYS[sort_by]
            names.sort(key=lambda n: key(n, self._registry[n]), reverse=reverse)
        return names

    def plot(
        self,
        colorset: ColorType | Sequence[ColorType] | None = None,
        *,
        show_color_name: bool = True,
        length : int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sort_by: Literal['name', 'length'] | None = None,
        reverse: bool = False,
        transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
        factor: float | _Unset = _UNSET,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None = None,
        severity: int = 100,
        width: float | int | None = 8,
        savefig: bool = False
    ) -> None:
        """
        Plot the colorset(s), optionally with additional
        filters.

        Any of the filters in :meth:`query` are valid arguments
        if ``colorset`` is ``None``.

        Parameters
        ----------
        colorset : str | None, optional, default=None
            Colorset name or alias, or a color or sequence of colors. If
            ``None``, plots all colorsets unless one or more of the filters are used.
        show_color_name : bool, optional, default=True
            If `True`, also plots the colorset name. Only applicable to VisualAstro
            colorsets (as opposed to a sequence of colors).
        length : int | None, optional, default=None
            Absolute length of colors.
        min_length : int | None, optional, default=None
            Minimum number of colors (inclusive).
        max_length : int | None, optional, default=None
            Maximum number of colors (inclusive).
        sort_by : {'name', 'length'} | None, optional, default=None
            Sort criterion. If ``None``, registration order is kept.
        reverse : bool, optional, default=False
            Reverse the sort order.
        transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
            Method to modify the colors. If `None`, colors are unchanged.
            If `_UNSET`, uses `config.color_transform`.
        factor : float | _Unset, optional, default=_UNSET
            Modification strength, see `get_colors`. If `_UNSET`, uses
            `config.color_transform_factor`.
        cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly', 'all'} | None, optional, default=None
            Type of colorblindness to simulate. Can be shorthanded to
            `'d'`, `'p'`, `'t'`. If `'all'`, simulates all cvd types.
            If `None`, no colorblindness simulation is applied.
        severity : float, optional, default=100
            Severity of colorblindness. Must be < 100.
        width : float | int | None, optional, default=8
            Width of the colorset plot.
        savefig:
            If `True`, save the figure to disk with `savefig`.
            Opens a prompt to set the file name and extention.

        Returns
        -------
        None
        """
        if colorset is not None:
            if isinstance(colorset, str):
                colors = self.get(colorset, transform=None, cvd_type=None)
            else:
                colors = colorset
            plot_colors(
                color=colors,
                show_color_name=show_color_name,
                transform=transform,
                factor=factor,
                cvd_type=cvd_type,
                severity=severity,
                width=width,
                savefig=savefig,
            )
            return

        query = self.query(
            length=length,
            min_length=min_length,
            max_length=max_length,
            sort_by=sort_by,
            reverse=reverse,
        )
        if len(query) == 0:
            raise ValueError("Query returned zero results!")
        plot_colors(
            color=query,
            show_color_name=show_color_name,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity,
            width=width,
            savefig=savefig,
        )

    def preview(
        self,
        colorset: int | ColorType | Sequence[ColorType],
        *,
        cmap: Colormap | _Unset = _UNSET,
        linestyle: LineStyleType = 'solid',
        linewidth: float = 2,
        transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
        legend: bool = True,
        loc: str = 'bottom center',
        ncols: int = 4,
        factor: float | _Unset = _UNSET,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None = None,
        severity: int = 100,
    ) -> None:
        """
        Plot a sample figure demonstrating a VisualAstro color set.

        Parameters
        ----------
        colors : ColorType | int | Sequence[ColorType | int], optional, default='astro_seq'
            Color set to visualize. Can be either a VisualAstro
            colorset name, a color, an integer, or a sequence of colors.
            If an ``int``, generates a sequence of colors by sampling
            ``cmap``.
        ax : matplotlib.axes.Axes | None, optional, default=None
            The Axes object on which to plot the demo orbits. If `None`,
            uses `plt.gca()`.
        cmap : Colormap | _Unset, optional, default=_UNSET
            Colormap used to generate a sequence of colors if
            ``colors`` is an int. If ``_UNSET`` uses ``config.colors``.
        linestyle : LineStyleType, optional, default='solid'
            Line style of each line.
        linewidth : float, optional, default=2
            Line width of each line.
        legend : bool, optional, default=True
            If `True`, plots the legend.
        loc : str, optional, default='bottom center'
            Legend location.
        ncols : int, optional, default=4
            Number of columns in the legend.
        transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
            Method to modify the colors. If `None`, colors are unchanged.
            If `_UNSET`, uses `config.color_transform`.
        factor : float | _Unset, optional, default=_UNSET
            Modification strength, see `get_colors`. If `_UNSET`, uses
            `config.color_transform_factor`.
        cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly', 'all'} | None, optional, default=None
            Type of colorblindness to simulate. Can be shorthanded to
            `'d'`, `'p'`, `'t'`. If `'all'`, simulates all cvd types.
            If `None`, no colorblindness simulation is applied.
        severity : float, optional, default=100
            Severity of colorblindness. Must be < 100.

        Returns
        -------
        None
        """
        preview_colorset(
            colors=colorset,
            cmap=cmap,
            linestyle=linestyle,
            linewidth=linewidth,
            legend=legend,
            loc=loc,
            ncols=ncols,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity,
        )
        plt.show()


class Colors(_ColorRegistry[ColorType]):
    """
    Query and plot registered VisualAstro named colors.

    See :ref:`color guide <color-guide>` for usage documentation.
    """

    _names = VISUALASTRO_NAMED_COLORS
    _alias_names = NAMED_COLOR_ALIASES

    def __init__(self):
        self._registry = mcolors.get_named_colors_mapping()

    @property
    def available(self) -> list[str]:
        """
        Canonical names of the registered VisualAstro named colors.

        Aliases and Matplotlib named colors are excluded.
        Use :attr:`aliases` to list alias names.

        Returns
        -------
        list of str
            Colorset names present in both ``COLORSETS`` and the Matplotlib
            color sequence registry, in registration order.

        See Also
        --------
        aliases : Mapping of alias names to canonical names.
        """
        return self._available

    @property
    def aliases(self) -> dict[str, str]:
        """
        Mapping of named color aliases to canonical names.

        Only aliases registered in the Matplotlib color sequence registry
        are included.

        Returns
        -------
        dict of str to str
            Keys are alias names, values are the canonical names in
            :attr:`available` they resolve to.

        See Also
        --------
        available : Canonical colorset names.
        """
        return self._aliases

    def get(
        self,
        name: str,
        *,
        fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
        transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
        factor: float | _Unset = _UNSET,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100,
    ):
        """
        Retrieve a single color to the requested format, optionally applying a
        color transform and a color vision deficiency (CVD) simulation.

        Parameters
        ----------
        color : str
            A single color name.
        fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
            Output color format.
        transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
            Transform applied to the color.

            - `None`: no transform.
            - `_UNSET`: use `config.color_transform`.

        factor : float | _Unset, optional, default=_UNSET
            Transform strength, see `get_colors`. Ignored if `transform` is `None`.
            If `_UNSET`, use `config.color_transform_factor`.
        cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
            CVD type to simulate, applied after `transform`. Shorthands
            `'d'`, `'p'`, `'t'` are accepted.

            - `'all'`: simulate every CVD type.
            - `None`: no simulation.

        severity : int, optional, default=100
            CVD severity in [0, 100], where 100 is complete color blindness.
            Ignored if `cvd_type` is `None`.

        Returns
        -------
        str | RGBTuple | RGBATuple
            `str` for `fmt='hex'`, `RGBTuple` for `fmt='rgb'`,
            `RGBATuple` for `fmt='rgba'`.
        """
        return get_color(
            self._resolve(name),
            fmt=fmt,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity,
        )

    def plot(
        self,
        family = None,
        *,
        ncols: int = 4,
        sort_colors: bool = True,
        transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None = None,
        factor: int | float = 0.5,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ):
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

            - `'named_colors'` or `None`: VisualAstro and Matplotlib named colors
            - `'mpl'` or `'matplotlib'` or `'mpl_colors'` or `'matplotlib_colors'` or `'css4'`: Matplotlib named colors
            - `'xkcd'` or `'xkcd_colors'`: XKCD named colors
            - `'visualastro'` or `'va'`: VisualAstro named colors
            - `'base'` or `'base_colors'`: Matplotlib base colors
            - `'tableau'` or `'tableau_colors'`: Matplotlib tableau colors
            - `'all'` or `'all_colors'`: All of the above

        ncols : int, optional, default=4
            Number of columns to plot.
        sort_colors : bool, optional, default=True
            If `True`, sort colors by hsv value.
        transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None, optional, default=None
            Transform applied to every color before sorting and CVD simulation.
            If `None`, colors are unchanged.
        factor : int | float, optional, default=0.5
            Transform strength. Ignored if `transform` is `None`.
        cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
            CVD type to simulate, applied after `transform`. If `None`,
            no simulation is applied.
        severity : int, optional, default=100
            CVD severity in [0, 100], where 100 is complete color blindness.
            Ignored if `cvd_type` is `None`.
        savefig:
            If `True`, save the figure to disk with `savefig`.
            Opens a prompt to set the file name and extention.
        """
        plot_colortable(
            colors=family,
            ncols=ncols,
            sort_colors=sort_colors,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity
        )
