"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-03-14
Description:
    VisualAstro help documentation class.
Dependencies:
    - matplotlib
    - numpy
Module Structure:
    - Spectra Extraction Functions
        Functions for extracting spectra from data.
    - Spectra Plotting Functions
        Functions for plotting extracted spectra.
    - Spectra Fitting Functions
        Fitting routines for spectra.
"""

from collections.abc import Sequence
from glob import glob
import inspect
import os
from typing import Literal
import warnings
from matplotlib import colors as mcolors, colormaps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import config
from visualastro.core.numerical_utils import to_list
from visualastro.plotting.core.colors import (
    CMAPNAMES,
    COLORNAMES,
    get_cmap,
    get_colors,
    simulate_colorblindness
)
from visualastro.plotting.core.plot_utils import _get_stylepath


class help:
    @staticmethod
    def colors(
        color: str | ColorType | int | Sequence[ColorType] | None = None,
        cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None = None,
        severity: int = 100
    ) -> None:
        """
        Display VisualAstro color colorsets with optional colorblindness simulation.

        Parameters
        ----------
        color : str | ColorType | int | Sequence[ColorType] | None, optional, default=None
            Name of a specific color scheme to display. If `None`,
            all visualastro colorsets are shown.
        cvd_type : Literal['deuteranomaly', 'protanomaly', 'tritanomaly', 'all'] | None, optional, default=None
            Simulate colorblindness. If `'all'`, displays all three types.
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
        cvd_types = (
            ['deuteranomaly', 'protanomaly', 'tritanomaly'] if cvd_type == 'all'
            else ([cvd_type] if cvd_type else [])
        )
        style = _get_stylepath(config.style)

        if color is None:
            with plt.style.context(style):
                n_rows = len(COLORNAMES) * (1 + len(cvd_types))
                fig, ax = plt.subplots(figsize=(8, n_rows * 0.5))
                ax.axis('off')
                row = 0

                for color_name in COLORNAMES:
                    plot_colors = get_colors(color_name)

                    # original
                    for j, c in enumerate(plot_colors):
                        ax.add_patch(Rectangle((j, -row), 1, 1, color=c, ec='black'))
                    ax.text(-0.5, -row + 0.5, color_name, va='center', ha='right')
                    row += 1

                    # CVD simulations
                    for cvd in cvd_types:
                        cvd_colors = simulate_colorblindness(plot_colors, cvd, severity) # type: ignore
                        for j, c in enumerate(cvd_colors):
                            ax.add_patch(Rectangle((j, -row), 1, 1, color=c, ec='black'))
                        ax.text(-0.5, -row + 0.5, f'{color_name} ({cvd})',
                                va='center', ha='right', fontsize=9)
                        row += 1

                ax.set_xlim(-2, max(len(get_colors(c)) for c in COLORNAMES))
                ax.set_ylim(-n_rows, 1)
                plt.tight_layout()
                plt.show()
        else:
            colorset = get_colors(color)
            n_rows = 1 + len(cvd_types)
            fig, ax = plt.subplots(figsize=(8, n_rows + 0.5))
            ax.axis('off')
            row = 0

            # original
            for i, c in enumerate(colorset):
                ax.add_patch(Rectangle((i, -row), 1, 1, color=c, ec='black'))
            ax.text(-0.5, -row + 0.5, color, va='center', ha='right') # type: ignore
            row += 1

            # CVD simulations
            for cvd in cvd_types:
                cvd_colors = simulate_colorblindness(colorset, cvd, severity) # type: ignore
                for i, c in enumerate(cvd_colors):
                    ax.add_patch(Rectangle((i, -row), 1, 1, color=c, ec='black'))
                ax.text(-0.5, -row + 0.5, f'{color} ({cvd})',
                        va='center', ha='right', fontsize=9)
                row += 1

            ax.set_xlim(-2, len(colorset))
            ax.set_ylim(-n_rows, 1)
            plt.tight_layout()
            plt.show()


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
            cmaps += CMAPNAMES
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
    def styles(style_name: str | None = None) -> None:
        """
        Display example plots for one or more available matplotlib style sheets.

        This method is primarily intended for previewing and comparing the
        visual appearance of built-in style sheets such as `'astro'`,
        `'latex'`, and `'cmu'`.

        Parameters
        ----------
        style_name : str or None, optional
            Name of a specific style to preview. If `None` (default),
            all predefined styles visualastro style are shown sequentially.

        Examples
        --------
        Display all visualastro plotting styles:
        >>> va.help.styles()
        Display a matplotlib or visualastro plotting style:
        >>> va.help.styles('classic')
        """

        if style_name is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            style_files = glob(os.path.join(base_dir, 'stylelib', '*.mplstyle'))
            style_names = np.sort([os.path.splitext(os.path.basename(f))[0] for f in style_files])
        else:
            style_names = to_list(style_name)
        colors = get_colors(len(style_names))
        print(
            'Here are sample plot made with the available visualastro plot styles. '
            '\nEach style sets the axes, fonts and font sizes, but leaves the color up to the user.\n'
        )
        for i, style_name in enumerate(style_names):
            style = _get_stylepath(style_name)
            with plt.style.context(style):
                print(fr"Style : '{style_name}'")
                if style_name == 'cm10':
                    warnings.warn(
                        "\nWARNING: cm10 style sheet cannont properly render 'º'! "
                        'This is mostly an issue for plots with WCSAxes. Either use '
                        "'cmu' or 'latex' stylesheets to get around this issue."
                    )
                fig, ax = plt.subplots(figsize=(7,2))
                ax.set_xscale('log')

                x = np.logspace(1, 9, 100)
                y = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(x)**2
                ax.scatter(x, y, color=colors[i%len(colors)], s=8, label=r'${\lambda}$')

                ax.set_xlabel(r'Wavelength [$\mu m$]')
                ax.set_ylabel('Counts')

                ax.legend(loc='upper left')

                plt.show()


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
