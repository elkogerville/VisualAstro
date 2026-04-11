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
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import config
from visualastro.core.numerical_utils import to_list
from visualastro.plotting.colors import COLORNAMES, get_colors
from visualastro.plotting.plot_utils import get_stylepath


class help:
    @staticmethod
    def colors(
        color: str | ColorType | int | Sequence[ColorType] | None = None,
    ) -> None:
        """
        Display VisualAstro color colorsets.

        Displays predefined visualastro colorsets or, a input colorset
        recognized by matplotlib.

        Parameters
        ----------
        color : str | ColorType | int | Sequence[ColorType] | None, optional, default=None
            Name of a specific color scheme to display. If ``None``,
            all visualastro colorsets are shown.

        Examples
        --------
        Display all default VisualAstro color colorsets:
        >>> va.help.colors()
        Display only the 'astro' colorset, including plot and model colors:
        >>> va.help.colors('astro')
        """
        style = get_stylepath(config.style)

        if color is None:
            print(
                'visualastro includes many built-in color colorsets.\n'
                'Many visualastro colorsets are designed to be colorblind friendly. '
                "These include: \n 'ibm' from the ibm colorblind palette, the tol colorsets "
                "'bright', 'vibrant', 'muted', 'light', 'dark', 'medium_contrast', 'high_contrast', "
                "and 'land_cover', the okabe-ito palette 'okabe-ito'"

            )
            with plt.style.context(style):
                fig, ax = plt.subplots(figsize=(8, len(COLORNAMES)))
                ax.axis('off')
                for i, color in enumerate(COLORNAMES):
                    plot_colors = get_colors(color)
                    # plot each color as a tile
                    for j, c in enumerate(plot_colors):
                        ax.add_patch(
                            Rectangle((j, -i), 1, 1, color=c, ec='black')
                        )
                    ax.text(-0.5, -i + 0.5, color, va='center', ha='right')

                ax.set_xlim(-1, max(len(get_colors(c)) for c in COLORNAMES))
                ax.set_ylim(-len(COLORNAMES), 1)
                plt.tight_layout()
                plt.show()

        else:
            colorset = get_colors(color)

            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            for i, color in enumerate(colorset):

                ax.add_patch(
                    Rectangle((i, 0), 1, 1, color=color, ec='black')
                )

            ax.set_xlim(-1, max(len(get_colors(c)[0]) for c in colorset))
            ax.set_ylim(-len(colorset), 1)
            plt.tight_layout()
            plt.show()

        return None


    @staticmethod
    def styles(style_name=None):
        """
        Display example plots for one or more available matplotlib style sheets.

        This method is primarily intended for previewing and comparing the
        visual appearance of built-in style sheets such as ``'astro'``,
        ``'latex'``, and ``'cmu'``.

        Parameters
        ----------
        style_name : str or None, optional
            Name of a specific style to preview. If ``None`` (default),
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
        colors = get_colors(config.default_colorset)
        print(
            'Here are sample plot made with the available visualastro plot styles. '
            '\nEach style sets the axes, fonts and font sizes, but leaves the color up to the user.\n'
        )
        for i, style_name in enumerate(style_names):
            style = get_stylepath(style_name)
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
