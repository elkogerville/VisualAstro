"""
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2026-03-14
Description:
    Visualastro user interface for publication ready plots.
"""

from collections.abc import Sequence
from typing import Literal
import warnings
from astropy.io.fits import Header
import astropy.units as u
from astropy.wcs import WCS
from matplotlib.colors import Colormap
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET
)
from visualastro.core.io import savefig
from visualastro.core.numerical_utils import to_list, _cycle
from visualastro.plotting.science.wcs_plots import imshow, plot_spectral_cube
from visualastro.plotting.base.plots import (
    hist,
    plot_density_histogram,
    plot,
    scatter,
    scatter3D,
    scatter_fit
)
from visualastro.plotting.core.utils import apply_style_modifiers, _get_stylepath
from visualastro.plotting.science.spectra_plots import plot_combine_spectrum, plot_spectrum
from visualastro.utils.wcs_utils import get_wcs_celestial


class ax:
    @staticmethod
    def imshow(
        datas,
        idx: int | tuple[int, int] | list[int | tuple[int, int] | None] | None = None,
        *,
        vmin: float | _Unset = _UNSET,
        vmax: float | _Unset = _UNSET,
        norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None | _Unset = _UNSET,
        percentile: tuple[float, float] | _Unset = _UNSET,
        stack_method: Literal['mean', 'median', 'sum', 'max', 'min', 'std'] | _Unset = _UNSET,
        origin: Literal['lower', 'upper'] | _Unset = _UNSET,
        wcs_input: WCS | Header | None = None,
        invert_wcs: bool = False,
        cmap: Colormap | str | _Unset = _UNSET,
        aspect: Literal['auto', 'equal'] | float | None | _Unset = _UNSET,
        mask_non_pos: bool | _Unset = _UNSET,
        axis: int = 0,
        **kwargs
    ) -> None:
        """
        Wrapper for `imshow` with automatic figure creation.

        See `visualastro.plotting.science.wcs_plots.imshow` for full documentation.

        Additional Parameters
        ---------------------
        wcs_input : WCS | Header | None, optional, default=None
            If provided, will be used to create a `WCSAxes` plot.
        invert_wcs : bool, optional, default=False
            If `True`, will swap WCS axes using `wcs.swapaxes(0, 1)`.

        Equivalent to:

            >>> ax = va.add_subplot()
            >>> va.imshow(data, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        # by default plot WCS if available
        wcs = None
        if wcs_input is not False:
            if wcs_input is None:
                # if wcs or header is available, use that
                for attr in ('wcs', 'header'):
                    value = getattr(datas, attr, None)
                    if value is not None:
                        if isinstance(value, (list, np.ndarray, tuple)):
                            value = value[0]
                        wcs_input = value
                        break
                else:
                    # no wcs data; fall back to default axes
                    wcs_input = None

            # create wcs object if provided
            if isinstance(wcs_input, Header):
                try:
                    wcs = WCS(wcs_input)
                except Exception as e:
                    warnings.warn(f'Failed to create WCS from Header: {e}')
                    wcs_input = None

            elif isinstance(wcs_input, (list, np.ndarray, tuple)):
                try:
                    wcs = WCS(wcs_input[0])
                except Exception as e:
                    warnings.warn(f'Failed to create WCS from array-like: {e}')
                    wcs_input = None

            elif isinstance(wcs_input, WCS):
                wcs = wcs_input

            elif wcs_input is not None:
                raise TypeError(f'Unsupported wcs_input type: {type(wcs_input).__name__}')

            if isinstance(wcs, WCS):
                wcs = wcs.celestial
                if invert_wcs:
                    wcs = wcs.swapaxes(0, 1)

        stylepath = _get_stylepath(style)
        with plt.style.context(stylepath):
            plt.figure(figsize=figsize)
            ax = plt.subplot(111) if wcs_input is None else plt.subplot(111, projection=wcs)
            apply_style_modifiers(ax, style)

            imshow(
                datas,
                ax=ax,
                idx=idx,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                percentile=percentile,
                stack_method=stack_method,
                origin=origin,
                cmap=cmap,
                aspect=aspect,
                mask_non_pos=mask_non_pos,
                axis=axis,
                **kwargs
            )

            if savefigure:
                savefig(dpi=dpi)

            plt.show()


    @staticmethod
    def plot_spectral_cube(
        cubes,
        idx: int | tuple[int, int] | None | list[int | tuple[int, int] | None] = None,
        vmin: float | _Unset = _UNSET,
        vmax: float | _Unset = _UNSET,
        norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None | _Unset = _UNSET,
        percentile: tuple[float, float] | None | _Unset = _UNSET,
        stack_method: Literal['mean', 'median', 'sum', 'max', 'min', 'std'] | _Unset = _UNSET,
        radial_vel: float | _Unset = _UNSET,
        spectral_unit: u.UnitBase | None = None,
        cmap: Colormap | str | list[Colormap | str] | _Unset = _UNSET,
        mask_non_pos: bool | _Unset = _UNSET,
        axis: int = 0,
        **kwargs
    ):
        """
        Wrapper for `plot_spectral_cube` with automatic figure creation.

        See `visualastro.plotting.science.wcs_plots.plot_spectral_cube` for full
        documentation.

        Equivalent to:

            >>> fig, ax = plt.subplots(projection=data.wcs)
            >>> va.plot_spectral_cube(data, idx, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        cubes = to_list(cubes)

        # define wcs figure axes
        stylepath = _get_stylepath(style)
        with plt.style.context(stylepath):
            fig = plt.figure(figsize=figsize)
            wcs2d = get_wcs_celestial(_cycle(cubes, config.reference_idx))
            if isinstance(wcs2d, WCS):
                ax = fig.add_subplot(111, projection=wcs2d)
            else:
                ax = fig.add_subplot(111)
            apply_style_modifiers(ax, style)

            _ = plot_spectral_cube(
                cubes, idx,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                percentile=percentile,
                stack_method=stack_method,
                radial_vel=radial_vel,
                spectral_unit=spectral_unit,
                cmap=cmap,
                mask_non_pos=mask_non_pos,
                axis=axis,
                **kwargs
            )
            if savefigure:
                filename = savefigure if isinstance(savefigure, str) else None
                savefig(filename, dpi=dpi)

            plt.show()


    @staticmethod
    def plot_spectrum(extracted_spectrums=None, plot_norm_continuum=None,
                      plot_continuum=None, emission_line=None, wavelength=None,
                      flux=None, continuum=None, colors=None, **kwargs):
        '''
        Convenience wrapper for `plot_spectrum`, which visualizes extracted
        spectra with optional continuum fits and emission-line overlays.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_spectrum` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.

        Parameters
        ----------
        extracted_spectrums : SpectrumPlus or list of SpectrumPlus, optional
            Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
            and `flux` must be given.
        plot_norm_continuum : bool, optional, default=None
            If True, plot normalized flux instead of raw flux.
            If None, uses `plot_normalized_continuum`.
        plot_continuum : bool, optional, default=None
            If True, overplot continuum fit. If None, uses
            the default value set by `config.plot_continuum_fit`.
        emission_line : str, optional, default=None
            Label for an emission line to annotate on the plot.
        wavelength : array-like, optional, default=None
            Wavelength array (required if `extracted_spectrums` is None).
        flux : array-like, optional, default=None
            Flux array (required if `extracted_spectrums` is None).
        continuum : array-like, optional, default=None
            Fitted continuum array.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color colorset from
            `config.default_colorset`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color` or `c` : list of colors or None, optional, default=None
                Aliases for `colors`.
            - `linestyles`, `linestyle`, `ls` : str or list of str, default=`config.linestyle`
                Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`config.linewidth`
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=`config.alpha`
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - `zorders`, `zorder` : float, default=None
                Order of line placement. If None, will increment by 1 for
                each additional line plotted.
            - `cmap` : str, optional, default=`config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple, optional, default=None
                Wavelength range to display.
            - `ylim` : tuple, optional
                Flux range to display.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default=`config.legend.loc`
                Location of legend.
            - `xlabel` : str, optional
                Label for the x-axis.
            - `ylabel` : str, optional
                Label for the y-axis.
            - `text_loc` : list of float, optional, default=`config.text_loc`
                Location for emission line annotation text in axes coordinates.
            - unit_bracket_style : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
                If `'round`' displays `extracted_spectrums` unit as (unit). If `'square`' as [unit].
            - `figsize` : tuple of float, default=`config.figsize`
                Figure size in inches.
            - `style` : str, default=`config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`config.savefig.enable`
                If True, saves the figure to disk using `savefig`.
            - `dpi` : int, default=`config.savefig.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # ---- KWARGS ----
        # figure params
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        # savefig
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        # set plot style
        style = _get_stylepath(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot_spectrum(extracted_spectrums, ax, plot_norm_continuum,
                              plot_continuum, emission_line, wavelength,
                              flux, continuum, colors, **kwargs)

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def plot_combine_spectrum(extracted_spectra, idx=0, wave_cuttofs=None,
                              concatenate=False, return_spectra=False,
                              plot_normalize=False, use_samecolor=True,
                              colors=None, **kwargs):
        '''
        Convenience wrapper for `plot_combine_spectrum`, to facilitate stiching
        spectra together.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_combine_spectrum` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.

        Parameters
        ----------
        extracted_spectra : list of `SpectrumPlus`/`Spectrum`, or list of list of `SpectrumPlus`/`Spectrum`
            List of spectra to plot. Each element should contain wavelength and flux attributes,
            and optionally the normalize attribute.
        idx : int, optional, default=0
            Index to select a specific spectrum if elements of `extracted_spectra` are lists.
            This is useful when extracting spectra from multiple regions at once.
            Ex:
                spec_1 = [spectrum1, spectrum2]
                spec_2 = [spectrum3, spectrum4]
                extracted_spectra = [spec_1[idx], spec_2[idx]]
        wave_cuttofs : list of float, optional, default=None
            Wavelength limits of each spectra used to mask spectra when stiching together.
            If provided, should contain the boundary wavelengths in sequence (e.g., [λ₀, λ₁, λ₂, ...λₙ]).
            Note:
                If N spectra are provided, ensure there are N+1 limits. For each i spectra, the
                program will define the limits as `wave_cuttofs[i]` < `spectra[i]` < `wave_cuttofs[i+1]`.
        concatenate : bool, optional, default=False
            If True, concatenate all spectra and plot as a single continuous curve.
        return_spectra : bool, optional, default=False
            If True, return the concatenated `SpectrumPlus` object instead of only plotting.
            If True, `concatenate` is set to True.
        plot_normalize : bool, optional, default=False
            If True, plot the normalized flux instead of the raw flux.
        use_samecolor : bool, optional, default=True
            If True, use the same color for all spectra. If `concatenate` is True,
            `use_samecolor` is also set to True.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color colorset from
            `config.default_colorset`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - ylim : tuple, optional, default=None
                y-axis limits as (ymin, ymax).
            - `color` or `c` : list of colors or None, optional, default=None
                Aliases for `colors`.
            - `linestyles`, `linestyle`, `ls` : str or list of str, default=`config.linestyle`
                Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`config.linewidth`
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=`config.alpha`
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - cmap : str, optional, default=`config.cmap`
                Colormap name for generating colors.
            - label : str, optional, default=None.
                Label for the plotted spectrum.
            - loc : str, optional, default=`config.legend.loc`
                Legend location (e.g., 'best', 'upper right').
            - xlabel, ylabel : str, optional, default=None
                Axis labels.
            - unit_bracket_style : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
                If `'round`' displays `extracted_spectra` unit as (unit). If `'square`' as [unit].
            - `figsize` : tuple of float, default=`config.figsize`
                Figure size in inches.
            - `style` : str, default=`config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`config.savefig.enable`
                If True, saves the figure to disk using `savefig`.
            - `dpi` : int, default=`config.savefig.dpi`
                Resolution (dots per inch) for saved figure.

        Returns
        -------
        SpectrumPlus or None
            If `return_spectra` is True, returns the concatenated spectrum.
            Otherwise, returns None.

        Notes
        -----
        - If `concatenate` is True, all spectra are merged and plotted as one line.
        - If `wave_cuttofs` is provided, each spectrum is masked to its corresponding
        wavelength interval before plotting.
        '''
        # figure params
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        # savefig
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        # set plot style
        style = _get_stylepath(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
            if return_spectra:
                combined_spectra = plot_combine_spectrum(extracted_spectra, ax, idx,
                                                         wave_cuttofs, concatenate,
                                                         return_spectra, plot_normalize,
                                                         use_samecolor, colors, **kwargs)
            else:
                combined_spectra = plot_combine_spectrum(extracted_spectra, ax, idx,
                                      wave_cuttofs, concatenate,
                                      return_spectra, plot_normalize,
                                      use_samecolor, colors, **kwargs)

            if savefigure:
                savefig(dpi=dpi)
            plt.show()

        if return_spectra:
            return combined_spectra


    @staticmethod
    def plot_density_histogram(X, Y, bins=None, xlog=None, ylog=None,
                               xlog_hist=None, ylog_hist=None, histtype=None,
                               normalize=True, colors=None, **kwargs):
        '''
        Convenience wrapper for `plot_density_histogram`, to plot 2D scatter
        distributions with normalizable histograms of the distributions.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_density_histogram` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.

        Parameters
        ----------
        X : array-like or list of arrays
            The x-axis data or list of data arrays.
        Y : array-like or list of arrays
            The y-axis data or list of data arrays.
        bins : int, sequence, str, or None, optional, default=None
            Histogram bin specification. Passed directly to
            `matplotlib.pyplot.hist`. If None, uses the default
            value from `config.bins`. If `bins` is a str, use
            one of the supported binning strategies 'auto', 'fd',
            'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        xlog : bool or None, optional, default=None
            Whether to use a logarithmic x-axis scale for the scatter plot.
            If None, uses `config.axes.xlog`.
        ylog : bool or None, optional, default=None
            Whether to use a logarithmic y-axis scale for the scatter plot.
            If None, uses `config.axes.ylog`.
        xlog_hist : bool or None, optional, default=None
            Whether to use a logarithmic x-axis scale for the top histogram.
            If None, uses `config.axes.xlog_hist`.
        ylog_hist : bool or None, optional, default=None
            Whether to use a logarithmic y-axis scale for the right histogram.
            If None, uses `config.axes.ylog_hist`.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
            Type of histogram to draw. If None, uses `config.histtype`.
        normalize : bool, optional, default=None
            If True, normalize histograms to a probability density.
            If None, uses `config.normalize_hist`.
        colors : list of colors, str, or None, optional, default=None
            Colors for each dataset. If None, uses the
            default color colorset from `config.default_colorset`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keyword arguments include:

            - `rasterized` : bool, default=`config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : list of colors, str, or None, optional, default=None
                aliases for `colors`.
            - `sizes`, `size`, `s` : float or list, optional, default=`config.scatter_size`
                Marker size(s) for scatter points.
            - `markers`, `marker`, `m` : str or list, optional, default=`config.marker`
                Marker style(s) for scatter points.
            - `alphas`, `alpha`, `a` : float or list, optional, default=`config.alpha`
                Transparency level(s).
            - `edgecolors`, `edgecolor`, `ec` : str or list, optional, default=`config.edgecolor`
                Edge colors for scatter points.
            - `linestyles`, `linestyle`, `ls` : str or list, optional, default=`config.linestyle`
                Line style(s) for histogram edges.
            - `linewidth`, `lw` : float or list, optional, default=`config.linewidth`
                Line width(s) for histogram edges.
            - `zorders`, `zorder` : int or list, optional, default=None
                Z-order(s) for drawing priority.
            - `cmap` : str, optional, default=`config.cmap`
                Colormap name for automatic color assignment.
            - `xlim`, `ylim` : tuple, optional, default=None
                Axis limits for the scatter plot.
            - `labels`, `label`, `l` : list or str, optional, default=None
                Labels for legend entries.
            - `loc` : str, optional, default=`config.legend.loc`
                Legend location.
            - `xlabel`, `ylabel` : str, optional, default=None
                Axis labels for the scatter plot.
            - `figsize` : tuple of float, default=`config.figsize`
                Figure size in inches.
            - `style` : str, default=`config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`config.savefig.enable`
                If True, saves the figure to disk using `savefig`.
            - `dpi` : int, default=`config.savefig.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        # savefig
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        style = _get_stylepath(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # adjust grid layout to prevent overlap
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.6),
                                    height_ratios=(1.6, 4),
                                    left=0.15, right=0.9, bottom=0.15,
                                    top=0.9, wspace=0.09, hspace=0.09)
            # create subplots
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0])
            ax_histy = fig.add_subplot(gs[1, 1])

            _ = plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins,
                                       xlog, ylog, xlog_hist, ylog_hist,
                                       histtype, normalize, colors, **kwargs)

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def hist(
        datas: u.Quantity | NDArray | list[u.Quantity | NDArray],
        bins: int | Sequence[float] | str | _Unset = _UNSET,
        histtype: Literal['bar', 'barstacked', 'step', 'stepfilled'] | _Unset = _UNSET,
        normalize: bool | _Unset = _UNSET,
        align: Literal['left', 'mid', 'right'] = 'mid',
        color: ColorType | list[ColorType] | _Unset = _UNSET,
        xlog: bool | _Unset = _UNSET,
        ylog: bool | _Unset = _UNSET,
        vlines: float | u.Quantity | Sequence[float | u.Quantity] | None = None,
        **kwargs
    ) -> None:
        """
        Wrapper for `hist` with automatic figure creation.

        See `visualastro.plotting.base.plots.hist` for full documentation.

        Equivalent to:

            >>> fig, ax = plt.subplots()
            >>> va.hist(X, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        style = _get_stylepath(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = hist(
                datas, ax,
                bins=bins,
                histtype=histtype,
                normalize=normalize,
                align=align,
                color=color,
                xlog=xlog,
                ylog=ylog,
                vlines=vlines,
                **kwargs
            )

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def plot(
        *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
        color: ColorType | list[ColorType] | _Unset = _UNSET,
        linestyle: Literal['-', '--', '-.', ':', ''] | list[Literal['-', '--', '-.', ':', '']] | _Unset = _UNSET,
        linewidth: float | list[float] | _Unset = _UNSET,
        alpha: float | list[float] | _Unset = _UNSET,
        normalize: bool | _Unset = _UNSET,
        xlog: bool | _Unset = _UNSET,
        ylog: bool | _Unset = _UNSET,
        zorder: float | list[float] | None = None,
        array_order: Literal['c', 'fortran'] | _Unset = _UNSET,
        **kwargs
    ) -> None:
        """
        Wrapper for `plot` with automatic figure creation.

        See `visualastro.plotting.base.plots.plot` for full documentation.

        Equivalent to:

            >>> fig, ax = plt.subplots()
            >>> va.plot(X, Y, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        style = _get_stylepath(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot(
                *data,
                ax=ax,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                normalize=normalize,
                xlog=xlog,
                ylog=ylog,
                zorder=zorder,
                array_order=array_order,
                **kwargs
            )

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def scatter(
        *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
        xerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
        yerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
        normalize: bool | _Unset = _UNSET,
        xlog: bool | _Unset = _UNSET,
        ylog: bool | _Unset = _UNSET,
        color: ColorType | list[ColorType] | _Unset =_UNSET,
        size: float | list[float] | _Unset = _UNSET,
        marker: MarkerStyle | list[MarkerStyle] | _Unset = _UNSET,
        alpha: float | list[float] | _Unset = _UNSET,
        edgecolor: Literal['face', 'none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        facecolor: Literal['none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        array_order: Literal['C', 'c', 'F', 'fortran'] | _Unset = _UNSET,
        **kwargs
    ) ->  None:
        """
        Wrapper for `scatter` with automatic figure creation.

        See `visualastro.plotting.base.plots.scatter` for full documentation.

        Equivalent to:

            >>> fig, ax = plt.subplots()
            >>> va.scatter(X, Y, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)
        style = _get_stylepath(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = scatter(
                *data,
                ax=ax,
                xerr=xerr,
                yerr=yerr,
                normalize=normalize,
                xlog=xlog,
                ylog=ylog,
                color=color,
                size=size,
                marker=marker,
                alpha=alpha,
                edgecolor=edgecolor,
                facecolor=facecolor,
                array_order=array_order,
                **kwargs
            )

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def scatter_fit(
        *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
        xerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
        yerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
        color: ColorType | list[ColorType] | int | _Unset =_UNSET,
        marker: MarkerStyle | list[MarkerStyle] | _Unset = _UNSET,
        size: float | list[float] | _Unset = _UNSET,
        alpha: float | list[float] | _Unset = _UNSET,
        edgecolor: Literal['face', 'none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        facecolor: Literal['none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        normalize: bool | _Unset = _UNSET,
        xlog: bool | _Unset = _UNSET,
        ylog: bool | _Unset = _UNSET,
        array_order: Literal['C', 'c', 'F', 'fortran'] | _Unset = _UNSET,
        **kwargs
    ) -> None:
        """
        Wrapper for `scatter_fit` with automatic figure creation.

        See `visualastro.plotting.base.plots.scatter_fit` for full documentation.

        Equivalent to:

            >>> fig, ax = plt.subplots()
            >>> va.scatter_fit(X, Y, deg=3, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize)
        style = kwargs.pop('style', config.style)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        style = _get_stylepath(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
            _ = scatter_fit(
                *data,
                ax=ax,
                xerr=xerr,
                yerr=yerr,
                normalize=normalize,
                xlog=xlog,
                ylog=ylog,
                color=color,
                size=size,
                marker=marker,
                alpha=alpha,
                edgecolor=edgecolor,
                facecolor=facecolor,
                array_order=array_order,
                **kwargs
            )

            if savefigure:
                savefig(dpi=dpi)
            plt.show()


    @staticmethod
    def scatter3D(
        X,
        Y,
        Z,
        elev: float | _Unset = 30,
        azim: float | _Unset = 45,
        roll: float | _Unset = 0,
        scale: float | None = None,
        axes_off: bool = False,
        grid_lines: bool = False,
        color: ColorType | list[ColorType] | int | _Unset = _UNSET,
        marker: MarkerStyle | list[MarkerStyle] | _Unset = _UNSET,
        size: float | list[float] | _Unset = _UNSET,
        alpha: float | list[float] | _Unset = _UNSET,
        edgecolor: Literal['face', 'none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        facecolor: Literal['none'] | ColorType | list[ColorType] | _Unset = _UNSET,
        plot_contours: Literal['x', 'y', 'z', 'all'] | list[Literal['x', 'y', 'z']] | None = None,
        array_order: Literal['C', 'c', 'F', 'fortran'] | _Unset = _UNSET,
        **kwargs
    ):
        """
        Wrapper for `scatter3D` with automatic figure creation.

        See `visualastro.plotting.base.plots.scatter3D` for full documentation.

        Equivalent to:

            >>> fig = plt.figure(figsize=figsize)
            >>> ax = fig.add_subplot(111, projection='3d')
            >>> va.scatter3D(X, Y, Z, ax=ax, **kwargs)
            >>> plt.show()
        """
        figsize = kwargs.pop('figsize', config.figsize3d)
        style = kwargs.pop('style', config.style)
        tight_layout = kwargs.pop('tight_layout', True)
        savefigure = kwargs.pop('savefig', config.savefig.enable)
        dpi = kwargs.pop('dpi', config.savefig.dpi)

        style = _get_stylepath(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            _ = scatter3D(
                X,
                Y,
                Z,
                ax=ax,
                elev=elev,
                azim=azim,
                roll=roll,
                scale=scale,
                axes_off=axes_off,
                grid_lines=grid_lines,
                color=color,
                marker=marker,
                size=size,
                alpha=alpha,
                edgecolor=edgecolor,
                facecolor=facecolor,
                plot_contours=plot_contours,
                array_order=array_order,
                **kwargs
            )

            if tight_layout:
                plt.tight_layout()

            if savefigure:
                savefig(dpi=dpi)

            plt.show()
