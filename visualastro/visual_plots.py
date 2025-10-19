from contextlib import contextmanager
from astropy.wcs import WCS
from astropy.io.fits import Header
import matplotlib.pyplot as plt
import numpy as np
from .data_cube import plot_spectral_cube
from .io import save_figure_2_disk
from .numerical_utils import get_data
from .plotting import imshow, panel_axes, plot_density_histogram, plot_histogram, plot_lines, scatter_plot
from .plot_utils import return_stylename, set_plot_colors
from .spectra import plot_combine_spectrum, plot_spectrum
from .visual_classes import DataCube, FitsFile


class va:
    @contextmanager
    def style(name):
        '''
        Context manager to temporarily apply a Matplotlib style.

        Parameters
        ––––––––––
        name : str
            Name of the Matplotlib or visualastro style to apply.
            The style name is passed to `return_stylename`, which
            returns the path to a visualastro mpl stylesheet.
            Matplotlib styles are also allowed (ex: 'classic').
        Yields
        ––––––
        None
            This context manager does not return a value. Code executed within
            the context will use the specified style, which is restored upon exit.
        Examples
        ––––––––
        >>> with style('astro'):
        ...     plt.plot(x, y)
        ...     plt.show()
        '''
        style_name = return_stylename(name)
        with plt.style.context(style_name):
            yield


    @staticmethod
    def imshow(datas, idx=None, vmin=None, vmax=None, norm='asinh',
               percentile=[3,99.5], origin='lower', wcs_input=None,
               invert_wcs=False, cmap='turbo', aspect=None, **kwargs):

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
        vmin, vmax : float, optional, default=None
            Lower and upper limits for colormap scaling. If not provided,
            values are determined from 'percentile'.
        norm : str, optional, default=None
            Normalization algorithm for colormap scaling.
            - 'asinh' -> AsinhStretch using 'ImageNormalize'
            - 'log' -> logarithmic scaling using 'LogNorm'
            - 'none' or None -> no normalization applied
        percentile : list of float, default=[3, 99.5]
            Default percentile range used to determine 'vmin' and 'vmax'.
        origin : str, {'upper', 'lower'}, default='lower'
            Pixel origin convention for imshow.
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
        cmap : str or list of str, default='turbo'
            Matplotlib colormap name or list of colormaps, cycled across images.
            ex: ['turbo', 'RdPu_r']
        aspect : str, {'auto', 'equal'} or float, optional, default=None
            Aspect ratio passed to imshow.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - `invert_xaxis` : bool, optional, default=False
                Invert the x-axis if True.
            - `invert_yaxis` : bool, optional, default=False
                Invert the y-axis if True.
            - `text_loc` : list of float, optional, default=[0.03, 0.03]
                Relative axes coordinates for text placement when plotting interactive ellipses.
            - `text_color` : str, optional, default='k'
                Color of the ellipse annotation text.
            - `xlabel` : str, optional, default=None
                X-axis label.
            - `ylabel` : str, optional, default=None
                Y-axis label.
            - `colorbar` : bool, optional, default=True
                Add colorbar if True.
            - `clabel` : str or bool, optional, default=True
                Colorbar label. If True, use default label; if None or False, no label.
            - `cbar_width` : float, optional, default=0.03
                Width of the colorbar.
            - `cbar_pad` : float, optional, default=0.015
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
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)
        # by default plot WCS if available
        wcs = None
        if wcs_input is not False:
            if wcs_input is None:
                # if provided data is a DataCube or FitsFile, use the header
                if isinstance(datas, (DataCube, FitsFile)):
                    wcs_input = datas.header[0] if isinstance(datas.header, list) else datas.header
                else:
                    # fall back to default axes
                    wcs_input = None
            # create wcs object if provided
            if isinstance(wcs_input, Header):
                try:
                    wcs = WCS(wcs_input)
                except:
                    wcs_input = None
            elif isinstance(wcs_input, (list, np.ndarray, tuple)):
                try:
                    wcs = WCS(wcs_input[0])
                except:
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

            imshow(datas, ax, idx, vmin, vmax, norm, percentile, origin,
                   cmap, aspect, wcs_input=wcs_input, **kwargs)

            if savefig:
                    save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot_spectral_cube(cubes, idx, vmin=None, vmax=None, percentile=[3,99.5],
                           norm='asinh', radial_vel=None, unit=None, **kwargs):
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
        vmin, vmax : float, optional, default=None
            Minimum and maximum values for image scaling. Overrides percentile if provided.
        percentile : list of two floats, default=[3, 99.5]
            Percentile values for automatic scaling if vmin/vmax are not specified.
        norm : str or None, default='asinh'
            Normalization type for `imshow`. Use None for linear scaling.
        radial_vel : float or astropy.units.Quantity, optional, default=None
            Radial velocity to shift spectral axis to the rest frame.
        unit : astropy.units.Unit or str, optional, default=None
            Desired spectral axis unit for labeling.
        cmap : str, list, or tuple, default='turbo'
            Colormap(s) to use for plotting.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            title : bool, default=False
                If True, display spectral slice label as plot title.
            emission_line : str or None, default=None
                Optional emission line label to display instead of slice value.
            text_loc : list of float, default=[0.03, 0.03]
                Relative axes coordinates for overlay text placement.
            text_color : str, default='k'
                Color of overlay text.
            colorbar : bool, default=True
                Whether to add a colorbar.
            cbar_width : float, default=0.03
                Width of the colorbar.
            cbar_pad : float, default=0.015
                Padding between axes and colorbar.
            clabel : str, bool, or None, default=True
                Label for colorbar. If True, automatically generate from cube unit.
            xlabel, ylabel : str, default='Right Ascension', 'Declination'
                Axes labels.
            spectral_label : bool, default=True
                Whether to draw spectral slice value as a label.
            highlight : bool, default=True
                Whether to highlight interactive ellipse if plotted.
            ellipses : list or None, default=None
                Ellipse objects to overlay on the image.
            plot_ellipse : bool, default=False
                If True, plot a default or interactive ellipse.
            center : list of two ints, default=[Nx//2, Ny//2]
                Center of default ellipse.
            w, h : float, default=X//5, Y//5
                Width and height of default ellipse.
            angle : float or None, default=None
                Angle of ellipse in degrees.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        Notes
        –––––
        - If multiple cubes are provided, they are overplotted in sequence.
        '''
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

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

            plot_spectral_cube(cubes, idx, ax, vmin, vmax, percentile,
                                norm, radial_vel, unit, **kwargs)
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
        colors : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - `colors`, `color` or `c` : list of colors or None, optional, default=None
                Colors to use for each dataset. If None, default
                color cycle is used.
            - `linestyles`, `linestyle`, `ls` : str or list of str, {'-', '--', '-.', ':', ''}, default='-'
                Line style of plotted lines.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=0.8
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=None
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - `zorders`, `zorder` : float, default=None
                Order of line placement. If None, will increment by 1 for
                each additional line plotted.
            - `cmap` : str, optional, default='turbo'
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple, optional
                Wavelength range to display.
            - `ylim` : tuple, optional
                Flux range to display.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default='best'
                Location of legend.
            - `xlabel` : str, optional
                Label for the x-axis.
            - `ylabel` : str, optional
                Label for the y-axis.
            - `text_loc` : list of float, optional, default=[0.025, 0.95]
                Location for emission line annotation text in axes coordinates.
            - `use_brackets` : bool, optional, default=False
                If True, plot units in square brackets; otherwise, parentheses.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''

        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        # set plot style
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_spectrum(extracted_spectrums, ax, plot_norm_continuum,
                          plot_continuum_fit, emission_line, wavelength,
                          flux, continuum_fit, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot_combine_spectrum(extracted_spectra, idx=0, wave_cuttofs=None,
                              concatenate=False, return_spectra=False,
                              plot_normalize=False, use_samecolor=True, **kwargs):
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
        colors : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - ylim : tuple, optional, default=None
                y-axis limits as (ymin, ymax).
            - linestyles : str, optional, default='-'
                Line style (e.g., '-', '--', ':').
            - linewidths : float, optional, default=0.8
                Line width in points.
            - alphas : float, optional, default=1
                Line transparency (0–1).
            - cmap : str, optional, default='turbo'
                Colormap name for generating colors.
            - label : str, optional, default=None.
                Label for the plotted spectrum.
            - loc : str, optional, default='best'
                Legend location (e.g., 'best', 'upper right').
            - xlabel, ylabel : str, optional, default=None
                Axis labels.
            - use_brackets : bool, optional, default=False
                If True, format axis labels with units in brackets instead of parentheses.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
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
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        # set plot style
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
            if return_spectra:
                combined_spectra = plot_combine_spectrum(extracted_spectra, ax, idx,
                                                         wave_cuttofs, concatenate,
                                                         return_spectra, plot_normalize,
                                                         use_samecolor, **kwargs)
            else:
                plot_combine_spectrum(extracted_spectra, ax, idx,
                                      wave_cuttofs, concatenate,
                                      return_spectra, plot_normalize,
                                      use_samecolor, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

        if return_spectra:
            return combined_spectra


    @staticmethod
    def plot_density_histogram(X, Y, bins='auto', xlog=False, ylog=False,
                             xlog_hist=True, ylog_hist=True, sharex=False,
                             sharey=False, histtype='step', normalize=True,
                             colors=None, **kwargs):
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
        bins : int, str, or sequence, optional, default='auto'
            Histogram bin specification. Passed directly to `matplotlib.pyplot.hist`.
        xlog : bool, optional, default=False
            Whether to use a logarithmic x-axis scale for the scatter plot.
        ylog : bool, optional, default=False
            Whether to use a logarithmic y-axis scale for the scatter plot.
        xlog_hist : bool, optional, default=False
            Whether to use a logarithmic x-axis scale for the top histogram.
        ylog_hist : bool, optional, default=False
            Whether to use a logarithmic y-axis scale for the right histogram.
        sharex : bool, default=False
            If True, share the x-axis among all subplots.
        sharey : bool, default=False
            If True, share the y-axis among all subplots.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional, default='step'
            Type of histogram to draw.
        normalize : bool, optional, default=True
            If True, normalize histograms.
        colors : list, str, or None, optional, default=None
            Colors for each dataset. If `None`, a colormap will be used.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keyword arguments include:

            - `sizes`, `size`, `s` : float or list, optional, default=10
                Marker size(s) for scatter points.
            - `markers`, `marker`, `m` : str or list, optional, default='o'
                Marker style(s) for scatter points.
            - `alphas`, `alpha`, `a` : float or list, optional, default=1
                Transparency level(s).
            - `edgecolors`, `edgecolor`, `ec` : str or list, optional, default=None
                Edge colors for scatter points.
            - `linestyles`, `linestyle`, `ls` : str or list, optional, default='-'
                Line style(s) for histogram edges.
            - `linewidth`, `lw` : float or list, optional, default=0.8
                Line width(s) for histogram edges.
            - `zorders`, `zorder` : int or list, optional, default=None
                Z-order(s) for drawing priority.
            - `cmap` : str, optional, default='turbo'
                Colormap name for automatic color assignment.
            - `xlim`, `ylim` : tuple, optional, default=None
                Axis limits for the scatter plot.
            - `labels`, `label`, `l` : list or str, optional, default=None
                Labels for legend entries.
            - `loc` : str, optional, default='best'
                Legend location.
            - `xlabel`, `ylabel` : str, optional, default=None
                Axis labels for the scatter plot.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # adjust grid layout to prevent overlap
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2),
                                    height_ratios=(1.2, 4),
                                    left=0.15, right=0.9, bottom=0.15,
                                    top=0.9, wspace=0.09, hspace=0.09)
            # create subplots
            ax = fig.add_subplot(gs[1, 0])
            sharex = ax if sharex is True else None
            sharey = ax if sharey is True else None
            ax_histx = fig.add_subplot(gs[0, 0], sharex=sharex)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=sharey)

            plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins,
                                   xlog, ylog, xlog_hist, ylog_hist,
                                   histtype, normalize, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot_histogram(datas, bins='auto', xlog=False, ylog=False,
                      histtype='step', colors=None, **kwargs):
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
        bins : int, sequence, or str, optional, default='auto'
            Number of bins or binning method. Passed to 'ax.hist'.
        xlog : bool, optional, default=False
            If True, set x-axis to logarithmic scale.
        ylog : bool, optional, Default=False
            If True, set y-axis to logarithmic scale.
        histtype : str, {'bar', 'barstacked', 'step', 'stepfilled'}, optional, default='step'
            Matplotlib histogram type.
        colors : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - `colors`, `color`, `c` : str, list of str or None, optional, default=None
                Colors to use for each line. If None, default color cycle is used.
            - `cmap` : str, optional, default='turbo'
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple, optional
                X data range to display.
            - `ylim` : tuple, optional
                Y data range to display.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default='best'
                Location of legend.
            - `xlabel` : str or None, optional
                Label for the x-axis.
            - `ylabel` : str or None, optional
                Label for the y-axis.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''
        # –––– KWARGS ––––
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_histogram(datas, ax, bins, xlog, ylog,
                           histtype, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot(X, Y, normalize=False,
             xlog=False, ylog=False,
             colors=None, linestyle='-',
             linewidth=0.8, alpha=1,
             zorder=None, **kwargs):
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
        normalize : bool, optional, default=False
            If True, normalize each line to its maximum value.
        xlog : bool, optional, default=False
            If True, set the x-axis to logarithmic scale.
        ylog : bool, optional, default=False
            If True, set the y-axis to logarithmic scale.
        colors : str, list of str or None, optional, default=None
            Colors to use for each line. If None, default color cycle is used.
        linestyle : str or list of str, {'-', '--', '-.', ':', ''}, default='-'
            Line style of plotted lines.
        linewidth : float or list of float, optional, default=0.8
            Line width for the plotted lines.
        alpha : float or list of float default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        zorder : float or list of float, optional, default=None
            Order in which to plot lines in. Lines are drawn in order
            of greatest to lowest zorder. If None, starts at 0 and increments
            the zorder by 1 for each subsequent line drawn.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - `colors`, `color`, `c` : str, list of str or None, optional, default=None
                Colors to use for each line. If None, default color cycle is used.
            - `linestyles`, `linestyle`, `ls` : str or list of str, {'-', '--', '-.', ':', ''}, default='-'
                Line style of plotted lines.
            - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=0.8
                Line width for the plotted lines.
            - `alphas`, `alpha`, `a` : float or list of float default=None
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - `cmap` : str, optional, default='turbo'
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple of two floats or None
                Limits for the x-axis.
            - `ylim` : tuple of two floats or None
                Limits for the y-axis.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default='best'
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
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_lines(X, Y, ax, normalize=normalize,
                       xlog=xlog, ylog=ylog, colors=colors,
                       linestyle=linestyle, linewidth=linewidth,
                       alpha=alpha, zorder=zorder, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def scatter(X, Y, xerr=None, yerr=None, normalize=False,
                xlog=False, ylog=False, colors=None, size=10,
                marker='o', alpha=1, edgecolors='face',
                **kwargs):

        '''
        Convenience wrapper for `scatter_plot`, to scatter plot one or more distributions.

        Initializes a Matplotlib figure and axis using the specified plotting
        style, then calls the core `scatter_plot` routine with the provided
        parameters. This method is intended for rapid visualization and consistent
        figure formatting, while preserving full configurability through **kwargs.
        Parameters
        ––––––––––
        X : array-like or list of array-like
            x-axis data for the lines. Can be a single array or a list of arrays.
        Y : array-like or list of array-like
            y-axis data for the lines. Must match the length of X if lists are provided.
        normalize : bool, optional, default=False
            If True, normalize each line to its maximum value.
        xlog : bool, optional, default=False
            If True, set the x-axis to logarithmic scale.
        ylog : bool, optional, default=False
            If True, set the y-axis to logarithmic scale.
        colors : list of str or None, optional, default=None
            Colors to use for each line. If None, default color cycle is used.
        size : float or list of float, optional, default=10
            Size of scatter dots.
        marker : str or list of str, optional, default='o'
            Marker style for scatter dots.
        alpha : float or list of float default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        edgecolors : {'face', 'none', None} or color or list of color, default='face'
            The edge color of the marker. Possible values:
            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.

        **kwargs : dict, optional
            Additional plotting parameters.

            Supported keywords:

            - `colors`, `color`, `c` : str, list of str or None, optional, default=None
                Colors to use for each line. If None, default color cycle is used.
            - `sizes`, `size`, `s` : float or list of float, optional, default=10
                Size of scatter dots.
            - `markers`, `marker`, `m` : str or list of str, optional, default='o'
                Marker style for scatter dots.
            - `alphas`, `alpha`, `a` : float or list of float default=None
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            - `edgecolors`, `edgecolor`, `ec` : {'face', 'none', None} or color or list of color, default='face'
                The edge color of the marker.
            - `cmap` : str, optional, default='turbo'
                Colormap to use if `colors` is not provided.
            - `xlim` : tuple of two floats or None
                Limits for the x-axis.
            - `ylim` : tuple of two floats or None
                Limits for the y-axis.
            - `labels`, `label`, `l` : str or list of str, default=None
                Legend labels.
            - `loc` : str, default='best'
                Location of legend.
            - `xlabel` : str or None
                Label for the x-axis.
            - `ylabel` : str or None
                Label for the y-axis.
            - `figsize` : tuple of float, default=(6, 6)
                Figure size in inches.
            - `style` : str, default='astro'
                Matplotlib or visualastro style name to apply during plotting.
                Ex: 'astro', 'classic', etc...
            - `savefig` : bool, default=False
                If True, saves the figure to disk using `save_figure_2_disk`.
            - `dpi` : int, default=600
                Resolution (dots per inch) for saved figure.
        '''
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            scatter_plot(X, Y, ax, xerr=xerr, yerr=yerr, normalize=normalize,
                         xlog=xlog, ylog=ylog, colors=colors, size=size,
                         marker=marker, alpha=alpha, edgecolors=edgecolors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

    # –––– VISUALASTRO HELP ––––

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
            color_map = ['visualastro', 'ibm_contrast', 'astro', 'MSG', 'ibm', 'ibm_r']
            if user_color is None:
                with plt.style.context(style):
                    fig, ax = plt.subplots(figsize=(8, len(color_map)))
                    ax.axis("off")
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
            style_names = ['astro', 'latex', 'minimal'] if style_name is None else [style_name]
            for style_name in style_names:
                style = return_stylename(style_name)
                with plt.style.context(style):
                    print(fr"Style : '{style_name}'")
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.set_xscale('log')

                    x = np.logspace(1, 9, 100)
                    y = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(x)**2
                    ax.scatter(x, y, color='darkslateblue')

                    ax.set_xlabel('Frequency [Hz]')
                    ax.set_ylabel('Counts')

                    plt.show()
