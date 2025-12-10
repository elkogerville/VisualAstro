'''
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2025-10-20
Description:
    Visualastro user interface for publication ready plots.
Dependencies:
    - astropy
    - matplotlib
    - numpy
Module Structure:
    - Plotting Functions
        Publication ready plots.
    - VisualAstro Help
        VisualAstro user help.
'''

from contextlib import contextmanager
import warnings
from astropy.io.fits import Header
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from .data_cube import plot_spectral_cube
from .io import save_figure_2_disk
from .numerical_utils import get_data
from .plotting import (
    imshow, plot_density_histogram,
    plot_histogram, plot_lines,
    plot_scatter, scatter3D
)
from .plot_utils import return_stylename, set_plot_colors
from .spectra import plot_combine_spectrum, plot_spectrum
from .va_config import get_config_value, va_config, _default_flag

class va:
    @contextmanager
    def style(name=None, rc=None, **rc_kwargs):
        '''
        Context manager to temporarily apply a Matplotlib or VisualAstro style,
        with optional rcParams overrides.

        Parameters
        ––––––––––
        name : str or None
            Matplotlib or VisualAstro style name. If None, uses the default
            value from `va_config.style`. Ex: 'astro' or 'latex'.
        rc : dict, optional
            Dictionary of rcParams overrides.
            Ex: {'font.size': 14}
        **rc_kwargs
            Additional rcParams overrides supplied as keyword arguments.
            Use underscores in place of dots: font_size → font.size

        Examples
        ––––––––
        >>> with style('latex', font_size=23, axes_labelsize=40):
        ...     plt.plot(x, y)

        >>> with style('paper', rc={'font.size': 14, 'lines.linewidth': 2}):
        ...     fig, ax = plt.subplots()

        >>> with style('astro', rc={'font.size': 12}, xtick_labelsize=10):
        ...     # rc dict and kwargs are merged (kwargs take precedence)
        ...     plt.plot(x, y)
        '''
       # get visualastro style
        name = get_config_value(name, 'style')
        style_name = return_stylename(name)

        # update rcParams, with priority to kwargs
        rc_combined = {}
        if rc is not None:
            rc_combined.update(rc)
        if rc_kwargs:
            # replace '_' with '.' for rcParams
            rc_combined.update({
                k.replace('_', '.'): v for k, v in rc_kwargs.items()
            })

        context = [style_name, rc_combined] if rc_combined else style_name

        with plt.style.context(context): # type: ignore
            yield


    @staticmethod
    def imshow(datas, idx=None, vmin=_default_flag, vmax=_default_flag,
               norm=_default_flag, percentile=_default_flag, origin=None,
               wcs_input=None, invert_wcs=False, cmap=None, aspect=_default_flag,
               mask_non_pos=None, wcs_grid=None, **kwargs):
        '''
        Convenience wrapper for `imshow`, which displays a
        2D image with optional visual customization.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `imshow` routine with the provided parameters.
        This method is intended for rapid visualization and consistent figure
        formatting, while preserving full configurability through **kwargs.

        Parameters
        ––––––––––
        datas : np.ndarray or list of np.ndarray
            Image array or list of image arrays to plot. Each array should
            be 2D (Ny, Nx) or 3D (Nz, Nx, Ny) if using 'idx' to slice a cube.
        idx : int or list of int, optional, default=None
            Index for slicing along the first axis if 'datas'
            contains a cube.
            - i -> returns cube[i]
            - [i] -> returns cube[i]
            - [i, j] -> returns the sum of cube[i:j+1] along axis 0
            If 'datas' is a list of cubes, you may also pass a list of
            indeces.
            ex: passing indeces for 2 cubes-> [[i,j], k].
        vmin : float or None, optional, default=`_default_flag`
            Lower limit for colormap scaling; overides `percentile[0]`.
            If None, values are determined from `percentile[0]`.
            If `_default_flag`, uses the default value in `va_config.vmin`.
        vmax : float or None, optional, default=`_default_flag`
            Upper limit for colormap scaling; overides `percentile[1]`.
            If None, values are determined from `percentile[1]`.
            If `_default_flag`, uses the default value in `va_config.vmax`.
        norm : str or None, optional, default=`_default_flag`
            Normalization algorithm for colormap scaling.
            - 'asinh' -> asinh stretch using 'ImageNormalize'
            - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
            - 'log' -> logarithmic scaling using 'LogNorm'
            - 'powernorm' -> power-law normalization using 'PowerNorm'
            - 'linear', 'none', or None -> no normalization applied
            If `_default_flag`, uses the default value in `va_config.norm`.
        percentile : list or tuple of two floats, or None, default=`_default_flag`
            Default percentile range used to determine 'vmin' and 'vmax'.
            If `_default_flag`, uses default value from `va_config.percentile`.
            If None, use no percentile stretch.
        origin : {'upper', 'lower'} or None, default=None
            Pixel origin convention for imshow. If None,
            uses the default value from `va_config.origin`.
        wcs_input : `astropy.wcs.WCS`, `astropy.io.fits.Header`, list, tuple, or bool, optional
            World Coordinate System (WCS) definition for the input data. If `None`,
            the method will attempt to infer a WCS from the provided data if it is a
            `DataCube` or `FitsFile` instance. If `False`, no WCS projection is used
            and a standard Matplotlib axis is created.

            Supported types:
                - `WCS` : a pre-constructed WCS object.
                - `Header` : a FITS header from which a WCS can be constructed.
                - `list` or `tuple` : sequence of headers, in which case the first
                    element is used to build the WCS.
                - `None` : attempt automatic inference, or fall back to default axes.
            Invalid types will raise a `TypeError`.
        invert_wcs : bool, optional
            If `True`, swaps the WCS axes (i.e., RA and DEC) using `WCS.swapaxes(0, 1)`.
            Useful for correcting coordinate orientation in cases where the FITS header
            or image orientation is flipped. Ignored if no valid WCS is present.
        cmap : str, list of str or None, default=None
            Matplotlib colormap name or list of colormaps, cycled across images.
            If None, uses the default value from `va_config.cmap`.
            ex: ['turbo', 'RdPu_r']
        aspect : {'auto', 'equal'}, float, or None, optional, default=`_default_flag`
            Aspect ratio passed to imshow, shortcut for `Axes.set_aspect`. 'auto'
            results in fixed axes with the aspect adjusted to fit the axes. 'equal`
            sets an aspect ratio of 1. None defaults to 'equal', however, if the
            image uses a transform that does not contain the axes data transform,
            then None means to not modify the axes aspect at all. If `_default_flag`,
            uses the default value from `va_config.aspect`.
        mask_non_pos : bool or None, optional, default=None
            If True, mask out non-positive data values. Useful for displaying
            log scaling of images with non-positive values. If None, uses the
            default value set by `va_config.mask_non_positive`.
        wcs_grid : bool or None, optional, default=None
            If True, display WCS grid ontop of plot. If None,
            uses the default value set by `va_config.wcs_grid`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `invert_xaxis` : bool, optional, default=False
                Invert the x-axis if True.
            - `invert_yaxis` : bool, optional, default=False
                Invert the y-axis if True.
            - `text_loc` : list of float, optional, default=`va_config.text_loc`
                Relative axes coordinates for text placement when
                plotting interactive ellipses.
            - `text_color` : str, optional, default=`va_config.text_color`
                Color of the ellipse annotation text.
            - `xlabel` : str, optional, default=None
                X-axis label.
            - `ylabel` : str, optional, default=None
                Y-axis label.
            - `colorbar` : bool, optional, default=`va_config.cbar`
                Add colorbar if True.
            - `clabel` : str or bool, optional, default=`va_config.clabel`
                Colorbar label. If True, use default label; if None or False, no label.
            - `cbar_width` : float, optional, default=`va_config.cbar_width`
                Width of the colorbar.
            - `cbar_pad` : float, optional, default=`va_config.cbar_pad`
                Padding between plot and colorbar.
            - `circles` : list, optional, default=None
                List of Circle objects (e.g., `matplotlib.patches.Circle`) to overplot on the axes.
            - `ellipses` : list, optional, default=None
                List of Ellipse objects (e.g., `matplotlib.patches.Ellipse`) to overplot on the axes.
                Single Ellipse objects can also be passed directly.
            - `points` : array-like, shape (2,) or (N, 2), optional, default=None
                Coordinates of points to overplot. Can be a single point `[x, y]`
                or a list/array of points `[[x1, y1], [x2, y2], ...]`.
                Points are plotted as red stars by default.
            - `plot_ellipse` : bool, optional, default=False
                If True, plot an interactive ellipse overlay. Requires an interactive backend.
            - `center` : list of float, optional, default=[Nx//2, Ny//2]
                Center of the default interactive ellipse (x, y).
            - `w` : float, optional, default=X//5
                Width of the default interactive ellipse.
            - `h` : float, optional, default=Y//5
                Height of the default interactive ellipse.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # –––– KWARGS ––––
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

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
                raise TypeError(f'Unsupported wcs_input type: {type(wcs_input)}')

            if invert_wcs and isinstance(wcs, WCS):
                wcs = wcs.swapaxes(0, 1) # type: ignore

        style = return_stylename(style)
        with plt.style.context(style):
            plt.figure(figsize=figsize)
            ax = plt.subplot(111) if wcs_input is None else plt.subplot(111, projection=wcs)

            _ = imshow(datas, ax, idx, vmin, vmax, norm, percentile,
                       origin, cmap, aspect, mask_non_pos, wcs_grid, **kwargs)

            if savefig:
                    save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot_spectral_cube(cubes, idx, vmin=_default_flag,
                           vmax=_default_flag, norm=_default_flag,
                           percentile=_default_flag, radial_vel=None,
                           unit=None, cmap=None, mask_non_pos=None, **kwargs):
        '''
        Convenience wrapper for `plot_spectral_cube`, which plots a `SpectralCube`
        along a given slice.

        Initializes a Matplotlib figure and axis using the specified plotting style,
        then calls the core `plot_spectral_cube` routine with the provided parameters.
        This method is intended for rapid visualization and consistent figure formatting,
        while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        cubes : DataCube, SpectralCube, or list of such
            One or more spectral cubes to plot. All cubes should have consistent units.
        idx : int
            Index along the spectral axis corresponding to the slice to plot.
        vmin : float or None, optional, default=`_default_flag`
            Lower limit for colormap scaling; overides `percentile[0]`.
            If None, values are determined from `percentile[0]`.
            If `_default_flag`, uses the default value in `va_config.vmin`.
        vmax : float or None, optional, default=`_default_flag`
            Upper limit for colormap scaling; overides `percentile[1]`.
            If None, values are determined from `percentile[1]`.
            If `_default_flag`, uses the default value in `va_config.vmax`.
        norm : str or None, optional, default=`_default_flag`
            Normalization algorithm for colormap scaling.
            - 'asinh' -> asinh stretch using 'ImageNormalize'
            - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
            - 'log' -> logarithmic scaling using 'LogNorm'
            - 'powernorm' -> power-law normalization using 'PowerNorm'
            - 'linear', 'none', or None -> no normalization applied
            If `_default_flag`, uses the default value in `va_config.norm`.
        percentile : list or tuple of two floats, or None, default=`_default_flag`
            Default percentile range used to determine `vmin` and `vmax`.
            If None, use no percentile stretch (as long as vmin/vmax are None).
            If `_default_flag`, uses default value from `va_config.percentile`.
        radial_vel : float or None, optional, default=None
            Radial velocity in km/s to shift the spectral axis.
            Astropy units are optional. If None, uses the default
            value set by `va_config.radial_velocity`.
        unit : astropy.units.Unit or str, optional, default=None
            Desired spectral axis unit for labeling.
        cmap : str, list or tuple of str, or None, default=None
            Colormap(s) to use for plotting. If None,
            uses the default value set by `va_config.cmap`.
        mask_non_pos : bool or None, optional, default=None
            If True, mask out non-positive data values. Useful for displaying
            log scaling of images with non-positive values. If None, uses the
            default value set by `va_config.mask_non_positive`.
        wcs_grid : bool or None, optional, default=None
            If True, display WCS grid ontop of plot. If None,
            uses the default value set by `va_config.wcs_grid`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `title` : bool, default=False
                If True, display spectral slice label as plot title.
            - `emission_line` : str or None, default=None
                Optional emission line label to display instead of slice value.
            - `text_loc` : list of float, default=`va_config.text_loc`
                Relative axes coordinates for overlay text placement.
            - `text_color` : str, default=`va_config.text_color`
                Color of overlay text.
            - `colorbar` : bool, default=`va_config.cbar`
                Whether to add a colorbar.
            - `cbar_width` : float, default=`va_config.cbar_width`
                Width of the colorbar.
            - `cbar_pad` : float, default=`va_config.cbar_pad`
                Padding between axes and colorbar.
            - `clabel` : str, bool, or None, default=`va_config.clabel`
                Label for colorbar. If True, automatically generate from cube unit.
            - `xlabel` : str, default=`va_config.right_ascension`
                X axis label.
            - `ylabel` : str, default=`va_config.declination`
                Y axis label.
            - `spectral_label` : bool, optional, default=True
                Whether to draw spectral slice value as a label.
            - `highlight` : bool, optional, default=`va_config.highlight`
                Whether to highlight interactive ellipse if plotted.
            - `ellipses` : list or None, default=None
                Ellipse objects to overlay on the image.
            - `plot_ellipse` : bool, default=False
                If True, plot a default or interactive ellipse.
            - `center` : list of two ints, default=[Nx//2, Ny//2]
                Center of default ellipse.
            - `w`, `h` : float, default=X//5, Y//5
                Width and height of default ellipse.
            - `angle` : float or None, default=None
                Angle of ellipse in degrees.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        Notes
        –––––
        - If multiple cubes are provided, they are overplotted in sequence.
        '''
        # –––– KWARGS ––––
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        cubes = cubes if isinstance(cubes, (list, np.ndarray, tuple)) else [cubes]

        # define wcs figure axes
        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            wcs2d = get_data(cubes[0]).wcs.celestial
            ax = fig.add_subplot(111, projection=wcs2d)
            if style.split('/')[-1] == 'minimal.mplstyle':
                ax.coords['ra'].set_ticks_position('bl')
                ax.coords['dec'].set_ticks_position('bl')

            _ = plot_spectral_cube(cubes, idx, ax, vmin, vmax, norm,
                                   percentile, radial_vel, unit, cmap,
                                   mask_non_pos, **kwargs)
            if savefig:
                save_figure_2_disk(dpi)

            plt.show()


    @staticmethod
    def plot_spectrum(extracted_spectrums=None, plot_norm_continuum=False,
                      plot_continuum_fit=False, emission_line=None, wavelength=None,
                      flux=None, continuum_fit=None, colors=None, **kwargs):
        '''
        Convenience wrapper for `plot_spectrum`, which visualizes extracted
        spectra with optional continuum fits and emission-line overlays.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_spectrum` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        extracted_spectrums : ExtractedSpectrum or list of ExtractedSpectrum, optional
            Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
            and `flux` must be given.
        plot_norm_continuum : bool, optional, default=False
            If True, plot normalized flux instead of raw flux.
        plot_continuum_fit : bool, optional, default=False
            If True, overplot continuum fit.
        emission_line : str, optional, default=None
            Label for an emission line to annotate on the plot.
        wavelength : array-like, optional, default=None
            Wavelength array (required if `extracted_spectrums` is None).
        flux : array-like, optional, default=None
            Flux array (required if `extracted_spectrums` is None).
        continuum_fit : array-like, optional, default=None
            Fitted continuum array.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color palette from
            `va_config.default_palette`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color` or `c` : list of colors or None, optional, default=None
                Aliases for `colors`.
            - `linestyles`, `linestyle`, `ls` : str or list of str, default=`va_config.linestyle`
                Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`va_config.linewidth`
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=`va_config.alpha`
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - `zorders`, `zorder` : float, default=None
                Order of line placement. If None, will increment by 1 for
                each additional line plotted.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple, optional, default=None
                Wavelength range to display.
            - `ylim` : tuple, optional
                Flux range to display.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default=`va_config.loc`
                Location of legend.
            - `xlabel` : str, optional
                Label for the x-axis.
            - `ylabel` : str, optional
                Label for the y-axis.
            - `text_loc` : list of float, optional, default=`va_config.text_loc`
                Location for emission line annotation text in axes coordinates.
            - `use_brackets` : bool, optional, default=`va_config.use_brackets`
                If True, plot units in square brackets; otherwise, parentheses.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # –––– KWARGS ––––
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        # set plot style
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot_spectrum(extracted_spectrums, ax, plot_norm_continuum,
                              plot_continuum_fit, emission_line, wavelength,
                              flux, continuum_fit, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
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
        ––––––––––
        extracted_spectra : list of `ExtractedSpectrum`/`Spectrum1D`, or list of list of `ExtractedSpectrum`/`Spectrum1D`
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
            If True, return the concatenated `ExtractedSpectrum` object instead of only plotting.
            If True, `concatenate` is set to True.
        plot_normalize : bool, optional, default=False
            If True, plot the normalized flux instead of the raw flux.
        use_samecolor : bool, optional, default=True
            If True, use the same color for all spectra. If `concatenate` is True,
            `use_samecolor` is also set to True.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color palette from
            `va_config.default_palette`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - ylim : tuple, optional, default=None
                y-axis limits as (ymin, ymax).
            - `color` or `c` : list of colors or None, optional, default=None
                Aliases for `colors`.
            - `linestyles`, `linestyle`, `ls` : str or list of str, default=`va_config.linestyle`
                Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`va_config.linewidth`
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=`va_config.alpha`
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - cmap : str, optional, default=`va_config.cmap`
                Colormap name for generating colors.
            - label : str, optional, default=None.
                Label for the plotted spectrum.
            - loc : str, optional, default=`va_config.loc`
                Legend location (e.g., 'best', 'upper right').
            - xlabel, ylabel : str, optional, default=None
                Axis labels.
            - use_brackets : bool, optional, default=`va_config.use_brackets`
                If True, format axis labels with units in brackets instead of parentheses.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.

        Returns
        –––––––
        ExtractedSpectrum or None
            If `return_spectra` is True, returns the concatenated spectrum.
            Otherwise, returns None.

        Notes
        -----
        - If `concatenate` is True, all spectra are merged and plotted as one line.
        - If `wave_cuttofs` is provided, each spectrum is masked to its corresponding
        wavelength interval before plotting.
        '''
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        # set plot style
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
            if return_spectra:
                combined_spectra = plot_combine_spectrum(extracted_spectra, ax, idx,
                                                         wave_cuttofs, concatenate,
                                                         return_spectra, plot_normalize,
                                                         use_samecolor, colors, **kwargs)
            else:
                plot_combine_spectrum(extracted_spectra, ax, idx,
                                      wave_cuttofs, concatenate,
                                      return_spectra, plot_normalize,
                                      use_samecolor, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
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
        ––––––––––
        X : array-like or list of arrays
            The x-axis data or list of data arrays.
        Y : array-like or list of arrays
            The y-axis data or list of data arrays.
        bins : int, sequence, str, or None, optional, default=None
            Histogram bin specification. Passed directly to
            `matplotlib.pyplot.hist`. If None, uses the default
            value from `va_config.bins`. If `bins` is a str, use
            one of the supported binning strategies 'auto', 'fd',
            'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        xlog : bool or None, optional, default=None
            Whether to use a logarithmic x-axis scale for the scatter plot.
            If None, uses the default value from `va_config.xlog`.
        ylog : bool or None, optional, default=None
            Whether to use a logarithmic y-axis scale for the scatter plot.
            If None, uses the default value from `va_config.ylog`.
        xlog_hist : bool or None, optional, default=None
            Whether to use a logarithmic x-axis scale for the top histogram.
            If None, uses the default value from `va_config.xlog_hist`.
        ylog_hist : bool or None, optional, default=None
            Whether to use a logarithmic y-axis scale for the right histogram.
            If None, uses the default value from `va_config.ylog_hist`.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
            Type of histogram to draw. If None, uses the default value from `va_config.histtype`.
        normalize : bool, optional, default=None
            If True, normalize histograms to a probability density.
            If None, uses the default value from `va_config.normalize_hist`.
        colors : list of colors, str, or None, optional, default=None
            Colors for each dataset. If None, uses the
            default color palette from `va_config.default_palette`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keyword arguments include:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : list of colors, str, or None, optional, default=None
                aliases for `colors`.
            - `sizes`, `size`, `s` : float or list, optional, default=`va_config.scatter_size`
                Marker size(s) for scatter points.
            - `markers`, `marker`, `m` : str or list, optional, default=`va_config.marker`
                Marker style(s) for scatter points.
            - `alphas`, `alpha`, `a` : float or list, optional, default=`va_config.alpha`
                Transparency level(s).
            - `edgecolors`, `edgecolor`, `ec` : str or list, optional, default=`va_config.edgecolor`
                Edge colors for scatter points.
            - `linestyles`, `linestyle`, `ls` : str or list, optional, default=`va_config.linestyle`
                Line style(s) for histogram edges.
            - `linewidth`, `lw` : float or list, optional, default=`va_config.linewidth`
                Line width(s) for histogram edges.
            - `zorders`, `zorder` : int or list, optional, default=None
                Z-order(s) for drawing priority.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap name for automatic color assignment.
            - `xlim`, `ylim` : tuple, optional, default=None
                Axis limits for the scatter plot.
            - `labels`, `label`, `l` : list or str, optional, default=None
                Labels for legend entries.
            - `loc` : str, optional, default=`va_config.loc`
                Legend location.
            - `xlabel`, `ylabel` : str, optional, default=None
                Axis labels for the scatter plot.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        style = return_stylename(style)
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

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot_histogram(datas, bins=None, xlog=None,
                       ylog=None, histtype=None,
                       normalize=None, colors=None,
                       **kwargs):
        '''
        Convenience wrapper for `plot_histogram`, to plot one or
        more histograms.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_histogram` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        datas : array-like or list of array-like
            Input data to histogram. Can be a single 1D array or a
            list of 1D/2D arrays. 2D arrays are automatically flattened.
        ax : matplotlib.axes.Axes
            The Axes object on which to plot the histogram.
        bins : int, sequence, str, or None, optional, default=None
            Histogram bin specification. Passed directly to
            `matplotlib.pyplot.hist`. If None, uses the default
            value from `va_config.bins`. If `bins` is a str, use
            one of the supported binning strategies 'auto', 'fd',
            'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        xlog : bool or None, optional, default=None
            If True, set x-axis to logarithmic scale.
            If None, uses the default value from `va_config.xlog`.
        ylog : bool or None, optional, default=None
            If True, set y-axis to logarithmic scale.
            If None, uses the default value from `va_config.ylog`.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
            Matplotlib histogram type. If None, uses the default value from `va_config.histtype`.
        normalize : bool or None, optional, default=None
            If True, normalize histograms to a probability density.
            If None, uses the default value from `va_config.normalize_hist`.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each dataset. If None,
            uses the default color palette from `va_config.default_palette`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : list of colors, str, or None, optional, default=None
                aliases for `colors`.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple, optional
                X data range to display.
            - `ylim` : tuple, optional
                Y data range to display.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default=`va_config.loc`
                Location of legend.
            - `xlabel` : str or None, optional
                Label for the x-axis.
            - `ylabel` : str or None, optional
                Label for the y-axis.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # –––– KWARGS ––––
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot_histogram(datas, ax, bins, xlog,
                               ylog, histtype, normalize,
                               colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot(X, Y,
             normalize=False,
             xlog=None,
             ylog=None,
             colors=None,
             linestyle=None,
             linewidth=None,
             alpha=None,
             zorder=None,
             **kwargs):
        '''
        Convenience wrapper for `plot_lines`, to plot one or more lines.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_lines` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        X : array-like or list of array-like
            x-axis data for the lines. Can be a single array or a list of arrays.
        Y : array-like or list of array-like
            y-axis data for the lines. Must match the length of X if lists are provided.
        ax : matplotlib.axes.Axes
            The Axes object to plot on.
        normalize : bool or None, optional, default=None
            If True, normalize each line by its maximum value.
            If None, uses the default value from `va_config.normalize_data`.
        xlog : bool or None, optional, default=None
            If True, set the x-axis to logarithmic scale.
            If None, uses the default value from `va_config.xlog`.
        ylog : bool or None, optional, default=None
            If True, set the y-axis to logarithmic scale.
            If None, uses the default value from `va_config.ylog`.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each line. If None, uses the
            default color palette from `va_config.default_palette`.
        linestyle : str, list of str, or None, optional, default=None
            Line style(s) to use for plotting. Can be a single string or a list of
            styles for multiple lines. Accepted values are:
            {'-', '--', '-.', ':', ''}. If None, uses the default
            value set in `va_config.linestyle`.
        linewidth : float, list of float, or None, optional, default=None
            Line width for the plotted lines. If None, uses the
            default value set in `va_config.linewidth`.
        alpha : float, list of float or None, optional, default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            If None, uses the default value set in `va_config.alpha`.
        zorder : float or list of float, optional, default=None
            Order in which to plot lines in. Lines are drawn in order
            of greatest to lowest zorder. If None, starts at 0 and increments
            the zorder by 1 for each subsequent line drawn.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
                Aliases for `colors`.
            - `linestyles`, `ls` : str or list of str, default=`va_config.linestyle`
                Aliases for `linestyle`.
            - `linewidths`, `lw` : float or list of float, optional, default=`va_config.linewidth`
                Aliases for `linewidth`.
            - `alphas`, `a` : float or list of float, default=`va_config.alpha`
                Aliases for `alpha`.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple of two floats or None
                Limits for the x-axis.
            - `ylim` : tuple of two floats or None
                Limits for the y-axis.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default=`va_config.loc`
                Location of legend.
            - `xlabel` : str or None
                Label for the x-axis.
            - `ylabel` : str or None
                Label for the y-axis.
            - `xpad`/`ypad` : float
                padding along x and y axis used when computing
                axis limits. Defined as:
                    xmax/min ±= xpad * (xmax - xmin)
                    ymax/min ±= ypad * (ymax - ymin)
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot_lines(X, Y, ax, normalize=normalize,
                           xlog=xlog, ylog=ylog, colors=colors,
                           linestyle=linestyle, linewidth=linewidth,
                           alpha=alpha, zorder=zorder, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def scatter(X, Y, xerr=None, yerr=None, normalize=False,
                xlog=None, ylog=None, colors=None, size=None,
                marker=None, alpha=None, edgecolors=_default_flag,
                facecolors=_default_flag, **kwargs):
        '''
        Convenience wrapper for `plot_scatter`, to scatter plot one or more distributions.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_scatter` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        X : array-like or list of array-like
            x-axis data for the lines. Can be a single array or a list of arrays.
        Y : array-like or list of array-like
            y-axis data for the lines. Must match the length of X if lists are provided.
        ax : matplotlib.axes.Axes
            The Axes object to plot on.
        xerr : array-like or list of array-like, optional, default=None
            x-axis errors on `X`. Should be same shape as `X`.
        yerr : array-like or list of array-like, optional, default=None
            x-axis errors on `Y`. Should be same shape as `Y`.
        normalize : bool or None, optional, default=None
            If True, normalize each line by its maximum value.
            If None, uses the default value from `va_config.normalize_data`.
        xlog : bool or None, optional, default=None
            If True, set the x-axis to logarithmic scale. If
            None, uses the default value in `va_config.xlog`.
        ylog : bool or None, optional, default=None
            If True, set the y-axis to logarithmic scale. If
            None, uses the default value in `va_config.ylog`.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color palette from
            `va_config.default_palette`.
        size : float, list of float, or None, optional, default=None
            Size of scatter dots. If None, uses the default
            value in `va_config.scatter_size`.
        marker : str, list of str, or None, optional, default=None
            Marker style for scatter dots. If None, uses the
            default value in `va_config.marker`.
        alpha : float, list of float, or None, default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            If None, uses the default value from `va_config.alpha`.
        edgecolors : {'face', 'none', None}, color, list of color, or None, default=`_default_flag`
            The edge color of the marker. Possible values:
            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.
            If `_default_flag`, uses the default value in `va_config.edgecolor`.
        facecolors : {'none'}, color, list of colors, or None, default=`_default_flag`
            The face color of the marker. Possible values:
            - 'none': Sets the face color to transparent
            - A color or sequence of colors
            - None: No face color is set (facecolor is set to marker color).
            If `_default_flag`, uses the default value in `va_config.facecolor`.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
                Aliases for `colors`.
            - `sizes`, `s` : float or list of float, optional, default=`va_config.scatter_size`
                Aliases for `size`.
            - `markers`, `m` : str or list of str, optional, default=`va_config.marker`
                Aliases for `marker`.
            - `alphas`, `a` : float or list of float default=`va_config.alpha`
                Aliases for `alpha`.
            - `edgecolor`, `ec` : {'face', 'none', None}, color, list of color, or None, default=`va_config.edgecolor`
                Aliases for `edgecolors`.
            - `facecolor`, `fc` : {'none'}, color, list of colors, or None, default=`_default_flag`
                Aliases for `facecolors`.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple of two floats or None
                Limits for the x-axis.
            - `ylim` : tuple of two floats or None
                Limits for the y-axis.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default=`va_config.loc`
                Location of legend.
            - `xlabel` : str or None
                Label for the x-axis.
            - `ylabel` : str or None
                Label for the y-axis.
            - `ecolors`, `ecolor` : color or list of color, optional, default=`va_config.ecolors`
                Color(s) of the error bars.
            - `elinewidth` : float, default=`va_config.elinewidth`
                Line width of the error bars.
            - `capsize` : float, default=`va_config.capsize`
                Length of the error bar caps in points.
            - `capthick` : float, default=`va_config.capthick`
                Thickness of the error bar caps in points.
            - `barsabove` : bool, default=`va_config.barsabove`
                If True, draw error bars above the plot symbols; otherwise, below.
            - `figsize` : tuple of float, default=`va_config.figsize`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize)
        style = kwargs.get('style', va_config.style)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            _ = plot_scatter(X, Y, ax, xerr=xerr, yerr=yerr, normalize=normalize,
                             xlog=xlog, ylog=ylog, colors=colors, size=size,
                             marker=marker, alpha=alpha, edgecolors=edgecolors,
                             facecolors=facecolors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def scatter3D(X, Y, Z, elev=90, azim=-90, roll=0,
                  scale=None, axes_off=False, grid_lines=False,
                  colors=None, size=None, marker=None, alpha=None,
                  edgecolors=_default_flag, plot_contours=None, **kwargs):
        '''
        Convenience wrapper for `scatter3D`, to scatter plot one or more
        distributions in 3-Dimensional space.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `plot_scatter` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        X, Y, Z : array-like or list of array-like
            Coordinates of the data points. Each of `X`, `Y`, and `Z`
            may be a single array or a list of arrays for plotting
            multiple groups. All three must have the same number of arrays.
        elev : float, default=30
            Elevation angle in degrees (rotation around camera x-axis).
        azim : float, default=45
            Azimuth angle in degrees (rotation around the z-axis).
        roll : float, default=0
            Roll angle in degrees (rotation around the view direction).
        scale : float or None, default=None
            If given, sets symmetric limits for all axes as `[-scale, scale]`.
        axes_off : bool, default=False
            If True, hides all axes spines, ticks, and labels.
        grid_lines : bool, default=False
            If False, disables gridlines on the 3D plot.
        colors : list of colors, str, or None, optional, default=None
            Colors to use for each scatter group or dataset.
            If None, uses the default color palette from
            `va_config.default_palette`.
        size : float, list of float, or None, optional, default=None
            Size of scatter dots. If None, uses the default
            value in `va_config.scatter_size`.
        marker : str, list of str, or None, optional, default=None
            Marker style for scatter dots. If None, uses the
            default value in `va_config.marker`.
        alpha : float, list of float, or None, default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            If None, uses the default value from `va_config.alpha`.
        edgecolors : {'face', 'none', None}, color, list of color, or None, default=`_default_flag`
            The edge color of the marker. Possible values:
            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.
            If `_default_flag`, uses the default value in `va_config.edgecolor`.
        plot_contours : {'x', 'y', 'z', 'all'}, sequence of {'x', 'y', 'z'}, or None, optional, default=None
            Specifies which contour projections to draw onto the side planes of the 3D axes.
            Each entry indicates the axis *normal* to the projection plane:
            - 'x' : Project onto the **YZ** plane at a fixed X offset.
            - 'y' : Project onto the **XZ** plane at a fixed Y offset.
            - 'z' : Project onto the **XY** plane at a fixed Z offset.
            - 'all' : Equivalent to `['x', 'y', 'z']`.
            If None, no contour projections are drawn.

        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `rasterized` : bool, default=`va_config.rasterized`
                Whether to rasterize plot artists. Rasterization
                converts the artist to a bitmap when saving to
                vector formats (e.g., PDF, SVG), which can
                significantly reduce file size for complex plots.
            - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
                Aliases for `colors`.
            - `sizes`, `s` : float or list of float, optional, default=`va_config.scatter_size`
                Aliases for `size`.
            - `markers`, `m` : str or list of str, optional, default=`va_config.marker`
                Aliases for `marker`.
            - `alphas`, `a` : float or list of float default=`va_config.alpha`
                Aliases for `alpha`.
            - `edgecolor`, `ec` : {'face', 'none', None}, color, list of color, or None, default=`va_config.edgecolor`
                Aliases for `edgecolors`.
            - `cmap` : str, optional, default=`va_config.cmap`
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple of two floats or None
                Limits for the x-axis.
            - `ylim` : tuple of two floats or None
                Limits for the y-axis.
            - `zlim` : tuple of two floats or None
                Limits for the z-axis.
            - `plot_contour_offset` : float or sequence of float, optional, default=None
                Manual positional offsets for the contour projection planes.
                If a single float is given, the same offset is used for all projections.
                If a sequence is given (e.g., array-like), its length must match
                the number of entries in `plot_contours`, providing one offset per projection
                in the same order. If None, offsets are automatically chosen based
                on current axis limits (i.e., `ax.get_xlim()[0]`, `ax.get_ylim()[0]`,
                `ax.get_zlim()[0]`).
            - `xlabel` : str or None
                Label for the x-axis.
            - `ylabel` : str or None
                Label for the y-axis.
            - `zlabel` : str or None
                Label for the z-axis.
            - `minor_ticks` : bool, default=False
                If True, sets minor ticks for all axes.
            - `figsize` : tuple of float, default=`va_config.figsize3d`
                Figure size in inches.
            - `style` : str, default=`va_config.style`
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `tight_layout` : bool, optional, default=True
                If True, uses `plt.tight_layout()`.
            - `savefig` : bool, default=`va_config.savefig`
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=`va_config.dpi`
                Resolution (dots per inch) for saved figure.

        Returns
        –––––––
        scatter : `matplotlib.collections.Path3DCollection` or list of them
            The created scatter artist(s). Returns a single object
            if only one dataset is plotted.

        Raises
        ––––––
        ValueError
            If `X`, `Y`, and `Z` do not have the same number of arrays
            after unit consistency checks.

        Notes
        –––––
        - The function cycles through `colors`, `sizes`, `markers`,
          `alphas`, and `edgecolors` if fewer values are given than
          datasets.
        - Pane backgrounds are set to white (`(1, 1, 1, 1)`).
        - Axis limits are applied in the order of `xlim`, `ylim`, `zlim`,
          and finally `scale` if provided.
        '''
        # figure params
        figsize = kwargs.get('figsize', va_config.figsize3d)
        style = kwargs.get('style', va_config.style)
        tight_layout = kwargs.get('tight_layout', True)
        # savefig
        savefig = kwargs.get('savefig', va_config.savefig)
        dpi = kwargs.get('dpi', va_config.dpi)

        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            _ = scatter3D(X, Y, Z, ax, elev, azim, roll,
                          scale, axes_off, grid_lines,
                          colors, size, marker, alpha,
                          edgecolors, plot_contours, **kwargs)

            if tight_layout:
                plt.tight_layout()

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    # –––– VisualAstro Help ––––
    class help:
        @staticmethod
        def colors(user_color=None):
            '''
            Display VisualAstro color palettes.

            Displays predefined VisualAstro color schemes or, if specified, a custom
            user-provided palette. Each palette is shown as a horizontal row of color
            tiles, labeled by palette name. Two sets of colors are displayed for each
            scheme: 'plot colors' and 'model colors'.

            Parameters
            ––––––––––
            user_color : str or None, optional, default=None
                Name of a specific color scheme to display. If `None`,
                all default VisualAstro palettes are shown.
            Examples
            ––––––––
            Display all default VisualAstro color palettes:
            >>> va.help.colors()
            Display only the 'astro' palette, including plot and model colors:
            >>> va.help.colors('astro')
            '''
            style = return_stylename('astro')
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
                                plt.Rectangle((j, -i), 1, 1, color=c, ec="black")
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
                                plt.Rectangle((j, -i), 1, 1, color=c, ec="black")
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
                ax.axis("off")
                for i in range(2):
                    for j in range(len(color_palettes[i])):
                        ax.add_patch(
                            plt.Rectangle((j, -i), 1, 1, color=color_palettes[i][j], ec="black")
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
            '''
            Display example plots for one or more available matplotlib style sheets.

            This method is primarily intended for previewing and comparing the
            visual appearance of built-in style sheets such as 'astro',
            'latex', and 'minimal'.
            Parameters
            ––––––––––
            style_name : str or None, optional
                Name of a specific style to preview. If ``None`` (default),
                all predefined styles ``['astro', 'latex', 'minimal']`` are shown
                sequentially.
            Examples
            ––––––––
            Display all visualastro plotting styles:
            >>> va.help.styles()
            Display a matplotlib or visualastro plotting style:
            >>> va.help.styles('classic')
            '''
            style_names = ['astro', 'smplot', 'latex', 'minimal', 'default'] if style_name is None else [style_name]
            colors = ['k', 'darkslateblue', 'slateblue', 'plum', 'palevioletred', '#D81B60']
            print(
                'Here are sample plot made with the available visualastro plot styles. '
                '\nEach style sets the axes, fonts and font sizes, but leaves the color up to the user.\n'
            )
            for i, style_name in enumerate(style_names):
                style = return_stylename(style_name)
                with plt.style.context(style):
                    print(fr"Style : '{style_name}'")
                    fig, ax = plt.subplots(figsize=(7,2))
                    ax.set_xscale('log')

                    x = np.logspace(1, 9, 100)
                    y = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(x)**2
                    ax.scatter(x, y, color=colors[i%len(colors)], s=8, label=r'${\lambda}$')

                    ax.set_xlabel(r'Wavelength [$\mu m$]')
                    ax.set_ylabel('Counts')

                    ax.legend(loc='upper left')

                    plt.show()
