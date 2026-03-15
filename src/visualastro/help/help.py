"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-03-14
Description:
    VisualAstro help documentation class.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - scipy
    - spectral-cube
    - specutils
    - tqdm
Module Structure:
    - Spectra Extraction Functions
        Functions for extracting spectra from data.
    - Spectra Plotting Functions
        Functions for plotting extracted spectra.
    - Spectra Fitting Functions
        Fitting routines for spectra.
"""

from glob import glob
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from visualastro.core.numerical_utils import to_list
from visualastro.plotting.plot_utils import get_stylepath, set_plot_colors


class help:
    @staticmethod
    def colors(user_color=None):
        """
        Display VisualAstro color palettes.

        Displays predefined VisualAstro color schemes or, if specified, a custom
        user-provided palette. Each palette is shown as a horizontal row of color
        tiles, labeled by palette name. Two sets of colors are displayed for each
        scheme: 'plot colors' and 'model colors'.

        Parameters
        ----------
        user_color : str or None, optional, default=None
            Name of a specific color scheme to display. If `None`,
            all default VisualAstro palettes are shown.

        Examples
        --------
        Display all default VisualAstro color palettes:
        >>> va.help.colors()
        Display only the 'astro' palette, including plot and model colors:
        >>> va.help.colors('astro')
        """
        style = get_stylepath('astro')
        # visualastro default color schemes
        color_map = ['visualastro', 'ibm_contrast', 'astro', 'MSG', 'ibm', 'ibm_r', 'smplot']
        if user_color is None:
            print(
                'Visualastro includes many built-in color palettes.\n'
                'Each palette also has a matching *model palette* — '
                'a complementary set of colors designed to pair well with the original.'
            )
            with plt.style.context(style):
                fig, ax = plt.subplots(figsize=(8, len(color_map)))
                ax.axis('off')
                print('Default VisualAstro color palettes:')
                # loop through color schemes
                for i, color in enumerate(color_map):
                    plot_colors, _ = set_plot_colors(color)
                    # add color tile for each color in scheme
                    for j, c in enumerate(plot_colors):
                        ax.add_patch(
                            Rectangle((j, -i), 1, 1, color=c, ec='black')
                        )
                    # add color scheme name
                    ax.text(-0.5, -i + 0.5, color, va="center", ha="right")
                # formatting
                ax.set_xlim(-1, max(len(set_plot_colors(c)[0]) for c in color_map))
                ax.set_ylim(-len(color_map), 1)
                plt.tight_layout()
                plt.show()

            with plt.style.context(style):
                fig, ax = plt.subplots(figsize=(8, len(color_map)))
                ax.axis("off")
                print('VisualAstro model color palettes:')
                # loop through color schemes
                for i, color in enumerate(color_map):
                    _, model_colors = set_plot_colors(color)
                    # add color tile for each color in scheme
                    for j, c in enumerate(model_colors):
                        ax.add_patch(
                            Rectangle((j, -i), 1, 1, color=c, ec="black")
                        )
                    # add color scheme name
                    ax.text(-0.5, -i + 0.5, color, va="center", ha="right")
                # formatting
                ax.set_xlim(-1, max(len(set_plot_colors(c)[0]) for c in color_map))
                ax.set_ylim(-len(color_map), 1)
                plt.tight_layout()
                plt.show()
        else:
            print(
                'Visualastro will automatically generate a set of *model colors* from any\n'
                'input color or list of colors. It will take the original color and lighten it.\n'
            )
            color_palettes = set_plot_colors(user_color)
            label = ['plot colors', 'model colors']
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            for i in range(2):
                for j in range(len(color_palettes[i])):
                    ax.add_patch(
                        Rectangle((j, -i), 1, 1, color=color_palettes[i][j], ec="black")
                    )
                # add color scheme name
                ax.text(-0.5, -i + 0.5, label[i], va="center", ha="right")
            # formatting
            ax.set_xlim(-1, max(len(set_plot_colors(c)[0]) for c in color_map))
            ax.set_ylim(-len(color_map), 1)
            plt.tight_layout()
            plt.show()


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
        colors, _ = set_plot_colors('astro')
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
