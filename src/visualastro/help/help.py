"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-07-19
Description:
    VisualAstro help documentation class.
"""

from collections.abc import Sequence
from glob import glob
from importlib.resources import files
import inspect
import os
from typing import Literal
import warnings

import astropy.units as u
from matplotlib import colors as mcolors, colormaps
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np

from visualastro.analysis.ic import blob
from visualastro.core.config import (
    config, _Unset, _UNSET, _resolve_default
)
from visualastro.core.io import imread, savefig as _savefig
from visualastro.core.numerical import number_density
from visualastro.core.numerical_utils import to_list, _cycle
from visualastro.plotting.ax import ax as _ax
from visualastro.plotting.base.plots import plot
from visualastro.plotting.core.colormaps import get_cmap, plot_cmap_lightness
from visualastro.plotting.core.colors import (
    VISUALASTRO_NAMED_COLORS,
    get_colors,
    plot_colors,
    plot_colortable,
)
from visualastro.plotting.core.style import _style_context
from visualastro.plotting.core.utils import legend


class help:
    @staticmethod
    def color(
        color: ColorType | int | Sequence[ColorType] | None = None,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None = None,
        severity: int = 100,
        show_color_name: bool = True
    ) -> None:
        """
        Display VisualAstro color colorsets with optional colorblindness simulation.

        Parameters
        ----------
        color : str | ColorType | int | Sequence[ColorType] | None, optional, default=None
            Name of a specific VisualAstro colorset to display. If `None`,
            all VisualAstro colorsets are shown.
        cvd_type : Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None, optional, default=None
            Simulate colorblindness. If `'all'`, all three types are simulated.
            If `None`, no simulation is applied.
        severity : int
            Severity level (0-100). 100 = complete colorblindness.

        Examples
        --------
        Display all default VisualAstro color colorsets:
        >>> va.help.colors()

        Display the 'astro' colorset as perceived by protanomaly:
        >>> va.help.colors('astro', cvd_type='protanomaly')

        Display the 'astro' colorset with all colorblindness simulations:
        >>> va.help.colors('astro', cvd_type='all')
        """
        plot_colors(
            color=color,
            cvd_type=cvd_type,
            severity=severity,
            show_color_name=show_color_name
        )


    @staticmethod
    def named_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot all VisualAstro and Matplotlib named colors together."""
        plot_colortable(
            colors='named_colors',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def mpl_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot Matplotlib named colors."""
        plot_colortable(
            colors='mpl_colors',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def va_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot VisualAstro named colors."""
        plot_colortable(
            colors=VISUALASTRO_NAMED_COLORS,
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def xkcd_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot xkcd named colors."""
        msg = (
            'All colors displayed here are recognizable by Matplotlib under the name space '
            "'xkcd:colorname'. \nVisualAstro will also recognize the colors if 'xkcd:' is dropped."
        )
        print(msg)
        plot_colortable(
            colors='xkcd',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def base_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot Matplotlib base named colors."""
        plot_colortable(
            colors='base',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def tableau_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """Plot Matplotlib tableau colors."""
        plot_colortable(
            colors='tableau',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )

    @staticmethod
    def all_colors(
        ncols: int = 4,
        sort_colors: bool = True,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
        severity: int = 100
    ) -> None:
        """
        Plot all named colors recognized by VisualAstro. Includes all named colors from
        Matplotlib (including base and tableau colors), xkcd, and VisualAstro.
        """
        plot_colortable(
            colors='all',
            ncols=ncols,
            sort_colors=sort_colors,
            cvd_type=cvd_type,
            severity=severity
        )


    @staticmethod
    def cmap(
        cmap: str | mcolors.Colormap | list[str | mcolors.Colormap] | None = None
    ) -> None:
        """
        Display a series of colormap(s), or all available colormaps if `cmap=None`.

        Parameters
        ----------
        cmap : str | mcolors.Colormap | list[str | mcolors.Colormap] | None, optional, default=None
            Colormap or list of colormaps to plot. If `None`, plots all colormaps.

        Examples
        --------
        Display all colormaps:
        >>> va.help.cmap()

        Display the 'turbo' and 'PuRd' colormaps:
        >>> va.help.cmap(['turbo', 'PuRd'])
        """
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        if cmap is None:
            cmaps = [c for c in colormaps() if not c.endswith('_r')]
        else:
            cmaps = to_list(cmap)

        n = len(cmaps)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*1.5))
        axes = axes.flatten()

        for ax, cmap_name in zip(axes, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=get_cmap(cmap_name))
            ax.set_title(cmap_name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    @staticmethod
    def cmap_lightness(
        cmap: str | mcolors.Colormap | None = None,
        ncols: int = 1,
        savefig: bool = False
    ):
        if cmap is None:
            cmaps = [c for c in colormaps() if not c.endswith('_r')]
            ncols = 12
            fig, ax = plt.subplots(figsize=(12,30))
            inline_offset = 30
        else:
            cmaps = to_list(cmap)
            fig, ax = plt.subplots(figsize=(5,5))
            inline_offset = 0

        legend_label = True if len(cmaps) <= 5 else False
        inline_label = not legend_label

        plot_cmap_lightness(
            cmaps,
            ax=ax,
            s=150,
            ncols=ncols,
            offset=5,
            legend_label=legend_label,
            inline_label=inline_label,
            inline_label_offset=inline_offset
        )
        if savefig:
            _savefig()
        plt.show()


    @staticmethod
    def style(style_name: str | None = None) -> None:
        """
        Display example plots for one or more available Matplotlib style sheets.

        This method is primarily intended for previewing and comparing the
        visual appearance of built-in style sheets such as `'astro'`,
        `'latex'`, and `'cmu'`.

        Parameters
        ----------
        style_name : str or None, optional
            Name of a specific style to preview. If `None` (default),
            all predefined VisualAstro styles are shown sequentially.

        Examples
        --------
        Display all VisualAstro plotting styles:
        >>> va.help.styles()
        Display a Matplotlib or VisualAstro plotting style:
        >>> va.help.styles('classic')
        """

        if style_name is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            style_files = glob(os.path.join(base_dir, 'stylelib', '*.mplstyle'))
            style_names = np.sort([os.path.splitext(os.path.basename(f))[0] for f in style_files])
        else:
            style_names = to_list(style_name)
        colors = get_colors(len(style_names), cmap=config.cmap)
        ecs = get_colors('astro')
        print(
            'Here are sample plots made with the available VisualAstro plot styles. '
            '\nEach style sets the axes, fonts and font sizes, but leaves the color up to the user.\n'
        )
        for i, style_name in enumerate(style_names):
            with _style_context(style_name):
                if style_name == 'cm10':
                    warnings.warn(
                        "\nWARNING: cm10 style sheet cannont properly render 'º'! "
                        'This is mostly an issue for plots with WCSAxes. Either use '
                        "'cmu' or 'latex' stylesheets to get around this issue.",
                        stacklevel=2
                    )
                fig, ax = plt.subplots(figsize=(7,2))
                ax.set_xscale('log')

                x = np.logspace(1, 9, 100)
                y = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(x)**2
                ax.scatter(
                    x, y,
                    color=colors[i%len(colors)],
                    ec=_cycle(ecs, i),
                    s=15,
                    label=r'${\lambda}$',
                    zorder=config.zorder.plot_data
                )

                ax.set_xlabel(r'Wavelength [$\mu m$]')
                ax.set_ylabel('Counts')
                ax.set_title(fr'Style : {style_name}', fontdict={'fontsize': 13})

                ax.legend(loc='upper left')

                plt.show()


    @staticmethod
    def imshow() -> None:
        foampath = files('visualastro') / 'data' / 'foamrnbw.png'
        foam = imread(foampath)
        _ax.imshow(foam, origin='upper')

    @staticmethod
    def plot() -> None:
        i = np.random.randint(0, 3, 1)
        if i == 0:
            def function(lamda, x):
                return (x/lamda - 1)*np.exp(2*x)
            x = np.linspace(-10,10,1000)
            l0 = function(1.5, x)
            l1 = function(1, x)
            l2 = function(.5, x)
            l3 = function(.25, x)
            l4 = function(.1, x)
            labels = [
                r'$\lambda$ = 1.5',
                r'$\lambda$ = 1.0',
                r'$\lambda$ = 0.5',
                r'$\lambda$ = 0.25',
                r'$\lambda$ = 0.1'
            ]
            with plt.style.context('thorlabs'):
                fig, ax = plt.subplots(figsize=(6,6))
                plot(
                    x, [l0,l1,l2,l3,l4],
                    ax=ax,
                    lw=1,
                    xlim=(-2.6, 2.6), ylim=(-2.6, 2.6),
                    xlabel=r'$ka$',
                    ylabel=r'$[\frac{ka}{\lambda} - 1]e^{2ka}$',
                    color='high_vis',
                    label=labels,
                    hlines=[-1,1]
                )
                plt.show()
        if i == 1:
            t = np.arange(0,2*np.pi,.01)
            x = (8**(1/2))*np.cos(t)
            y = (8**(1/2))*np.sin(t)
            x2 = (4**(1/2))*np.cos(t)
            y2 = (4**(1/2))*np.sin(t)
            x3 = (16**(1/2))*np.cos(t)
            y3 = (16**(1/2))*np.sin(t)
            x4 = ((2*(6/5))**(1/2))*np.cos(t)
            y4 = ((10*(6/5))**(1/2))*np.sin(t)
            labels = [
                'Px=x=2', r'px=x=$\sqrt{2}$', r'px=x=$\sqrt{8}$', r'px=x=$\sqrt{2}$'
            ]

            _ax.plot(
                [x,x2,x3,x4], [y,y2,y3,y4],
                xlim=(-4.5,6), ylim=(-4.5,6),
                color='MSG',
                xlabel='x', ylabel=r'P$_{\mathrm{x}}$',
                point=[0,0],
                label=labels,
                legend_ncols=2
            )
        if i == 2:
            with _style_context(config.style):
                t = np.arange(0,2*np.pi,0.01)
                ft = (4/np.pi)*np.sin(t)+(4/(3*np.pi))*np.sin(3*t) + \
                    (4/(5*np.pi))*np.sin(5*t)+(4/(7*np.pi))*np.sin(7*t)
                fig, ax = plt.subplots(figsize=(6,6))
                plot(
                    t*u.s, ft,
                    ax=ax,
                    color='darkslateblue',
                    xlabel='Time', ylabel='Intensity Value',
                    label='Fourier \nExpansion'
                )
                ax.vlines(x=np.pi, ymin=-1, ymax=1, color='red', label='Square Wave')
                ax.vlines(x=2*np.pi, ymin=-1, ymax=0, color='red')
                ax.vlines(x=0, ymin=0, ymax=1, color='red')
                ax.hlines(y=1, xmin=0, xmax=np.pi, color='red')
                ax.hlines(y=-1, xmin=np.pi, xmax=2*np.pi, color='red')

                legend()

                plt.show()

    @staticmethod
    def scatter(cmap: mcolors.Colormap | str | _Unset = _UNSET) -> None:
        cmap = get_cmap(_resolve_default(cmap, config.cmap))
        i = np.random.randint(0, 2, 1)
        if i == 0:
            data = blob(10000, as_array=True)
            _ax.scatter(data, c=number_density(data[:,0:2], 100), index_spec=[0,1], cmap=cmap)
        if i == 1:
            x=np.linspace(1,300,200) * u.deg
            y=np.sin(x) * u.M_sun
            y2=np.sin(2*x) * u.M_sun
            y3=np.sin(3*x) * u.M_sun
            _ax.scatter(
                x, [y,y2,y3],
                s=100,
                l=['a','b','c'],
                color=get_colors('MSGII', cvd_type='deuteranomaly'),
                style='smplot-og',
                ec='ibm'
            )

    @staticmethod
    def scatter_fit(**kwargs) -> None:
        x=np.linspace(1,300,50) * u.radian
        y=np.sin(x) * u.Lsun
        y2=np.sin(-2*x) * u.Lsun
        _ax.scatter_fit(x, [y, y2], deg=20, **kwargs)

    @staticmethod
    def scatter3D(
        figsize: tuple[float, float] = (6,6),
        plot_contours: Literal['x', 'y', 'z', 'all'] = 'all',
        axis_style='cube',
        elev=30,
        azim=-60,
        roll=0,
        **kwargs
    ) -> None:
        data = blob(7000, as_array=True)
        _ax.scatter3D(
            data[:,0], data[:,1], data[:,2],
            c=number_density(data, 1000),
            figsize=figsize,
            plot_contours=plot_contours,
            axis_style=axis_style,
            elev=elev, azim=azim, roll=roll,
            **kwargs
        )


def getsource(function) -> None:
    """
    Print the source code of a function.

    Equivalent to:
        >>> import inspect
        >>> print(inspect.getsource(function))

    Parameters
    ----------
    function :
        function to inspect.
    """
    print(inspect.getsource(function))
