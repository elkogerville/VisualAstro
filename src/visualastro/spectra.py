'''
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2025-10-22
Description:
    Spectra science functions.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - scipy
    - specutils
Module Structure:
    - Spectra Extraction Functions
        Functions for extracting spectra from data.
    - Spectra Plotting Functions
        Functions for plotting extracted spectra.
    - Spectra Fitting Functions
        Fitting routines for spectra.
'''

from collections import namedtuple
import astropy.units as u
from astropy.units import Quantity
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from specutils.spectra import Spectrum
from tqdm import tqdm
from .io import get_kwargs, save_figure_2_disk
from .numerical_utils import (
    get_value,
    mask_within_range,
    shift_by_radial_vel,
    to_array,
    to_list
)
from .numerical_utils import interpolate as _interpolate
from .plot_utils import (
    return_stylename, sample_cmap, set_axis_labels,
    set_axis_limits, set_plot_colors
)
from .spectra_utils import (
    GaussianFitResult,
    deredden_flux,
    fit_continuum,
    get_spectral_axis,
)
from .spectra_utils import gaussian as _gaussian
from .spectra_utils import gaussian_continuum as _gaussian_continuum
from .spectra_utils import gaussian_line as _gaussian_line
from .SpectrumPlus import SpectrumPlus
from .units import ensure_common_unit, convert_quantity, get_unit
from .utils import _unwrap_if_single
from .config import get_config_value, config, _default_flag


# Spectra Extraction Functions
# ----------------------------
def extract_cube_spectra(cubes, flux_extract_method=None, extract_mode=None, fit_method=None,
                         region=None, radial_vel=_default_flag, rest_freq=_default_flag,
                         deredden=None, unit=_default_flag, emission_line=None,
                         plot_continuum=None, plot_norm_continuum=None, **kwargs):
    '''
    Extract 1D spectra from one or more data cubes, with optional continuum normalization,
    dereddening, and plotting.

    Parameters
    ----------
    cubes : DataCube, SpectralCube, or list of cubes
        Input cube(s) from which to extract spectra. The data must either be
        a SpectralCube, or a DataCube containing a SpectralCube.
    flux_extract_method : {'mean', 'median', 'sum'} or None, default=None
        Method for extracting the flux. If None, uses the default
        value set by `config.flux_extract_method`.
    extract_mode : {'cube', 'slice', 'ray'} or None, default=None
        Specifies how the spectral cube should be traversed during flux
        extraction. This controls memory usage and performance for large cubes.
            - 'cube' :
                Load and operate on the entire cube in memory. This is the
                simplest mode but may be slow or disabled for very large datasets
                unless `cube.allow_huge_operations = True` is set.
            - 'slice' :
                Process the cube slice-by-slice along the spectral axis. This
                avoids loading the full cube into memory and is recommended for
                moderately large datasets.
            - 'ray' :
                Traverse the cube voxel-by-voxel ('ray-wise'), minimizing memory
                load at the cost of speed. Recommended for extremely large cubes
                or low-memory environments.
        If None, uses the default value set by `config.spectral_cube_extraction_mode`.
    fit_method : {'fit_continuum', 'generic'} or None, optional, default=None
        Method used to fit the continuum. If None, uses the default
        value set by `config.spectrum_continuum_fit_method`.
    region : array-like or None, optional, default=None
        Wavelength or pixel region(s) to use when `fit_method='fit_continuum'`.
        Ignored for other methods. This allows the user to specify which
        regions to include in the fit. Removing strong peaks is preferable to
        avoid skewing the fit up or down.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
    radial_vel : float or None, optional, default=`_default_flag`
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional. If None, ignores the radial velocity.
        If `_default_flag`, uses the default value set by `config.radial_velocity`.
    rest_freq : float or None, optional, default=`_default_flag`
        Rest-frame frequency or wavelength of the spectrum. If None,
        ignores the rest frequency for unit conversions. If `_default_flag`,
        uses the default value set by `config.spectra_rest_frequency`.
    deredden : bool or None, optional, default=None
        Whether to apply dereddening to the flux using deredden_flux().
        If None, uses the default value set by `config.deredden_spectrum`.
    unit : str, astropy.units.Unit, or None, optional, default=`_default_flag`
        Desired units for the wavelength axis. Converts the default
        units if possible. If None, does not try and convert. If `_default_flag`,
        uses the default value set by `config.wavelength_unit`.
    emission_line : str, optional, default=None
        Name of an emission line to annotate on the plot.
    plot_continuum : bool or None, optional, default=None
        Whether to overplot the continuum fit. If None, uses the
        default value set by `config.plot_continuum_fit`.
    plot_norm_continuum : bool or None, optional, default=None
        Whether to plot the normalized extracted spectra. If None,
        uses the default value set by `config.plot_normalized_continuum`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `how` : str, optional, default=`config.spectral_cube_extraction_mode`
            Alias for `extract_mode`.
        - `convention` : str, optional
            Doppler convention.
        - `Rv` : float, optional, default=`config.Rv`
            Dereddening parameter.
        - `Ebv` : float, optional, default=`config.Ebv`
            Dereddening parameter.
        - `deredden_method` : str, optional, default=`config.deredden_method`
            Extinction law to use.
        - `deredden_region` : str, optional, default=`config.deredden_region`
            Region/environment for WD01 extinction law.
        - `figsize` : tuple, optional, default=`config.figsize`
            Figure size for plotting.
        - `style` : str, optional, default=`config.style`
            Plotting style.
        - `savefig` : bool, optional, default=`config.savefig`
            Whether to save the figure to disk.
        - `dpi` : int, optional, default=`config.dpi`
            Figure resolution for saving.
        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `colors`, `color` or `c` : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.
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
        - `loc` : str, default=`config.loc`
            Location of legend.
        - `xlabel` : str, optional
            Label for the x-axis.
        - `ylabel` : str, optional
            Label for the y-axis.
        - `text_loc` : list of float, optional, default=`config.text_loc`
            Location for emission line annotation text in axes coordinates.
        - `use_brackets` : bool, optional, default=`config.use_brackets`
            If True, plot units in square brackets; otherwise, parentheses.

    Returns
    -------
    SpectrumPlus or list of SpectrumPlus
        Single object if one cube is provided, list if multiple cubes are provided.
    '''
    # ---- KWARGS ----
    # spectra extraction memory mode
    extract_mode = get_kwargs(kwargs, 'how', default=extract_mode)
    # doppler convention
    convention = kwargs.get('convention', None)
    # dereddening parameters
    Rv = kwargs.get('Rv', config.Rv)
    Ebv = kwargs.get('Ebv', config.Ebv)
    deredden_method = kwargs.get('deredden_method', config.deredden_method)
    deredden_region = kwargs.get('deredden_region', config.deredden_region)
    # figure params
    figsize = kwargs.get('figsize', config.figsize)
    style = kwargs.get('style', config.style)
    # savefig
    savefig = kwargs.get('savefig', config.savefig)
    dpi = kwargs.get('dpi', config.dpi)

    # get default config values
    extract_mode = get_config_value(extract_mode, 'spectral_cube_extraction_mode')
    methods = {
        'mean': lambda cube: cube.mean(axis=(1, 2), how=extract_mode),
        'median': lambda cube: cube.median(axis=(1, 2), how=extract_mode),
        'sum': lambda cube: cube.sum(axis=(1, 2), how=extract_mode)
    }

    flux_extract_method = str(get_config_value(flux_extract_method, 'flux_extract_method')).lower()
    extract_method = methods.get(flux_extract_method)
    if extract_method is None:
        raise ValueError(
            f"Invalid flux_extract_method '{flux_extract_method}'. "
            f'Choose from {list(methods.keys())}.'
        )
    fit_method = get_config_value(fit_method, 'spectrum_continuum_fit_method')
    radial_vel = config.radial_velocity if radial_vel is _default_flag else radial_vel
    rest_freq = config.spectra_rest_frequency if rest_freq is _default_flag else rest_freq
    deredden = get_config_value(deredden, 'deredden_spectrum')
    unit = config.wavelength_unit if unit is _default_flag else unit
    plot_continuum = get_config_value(plot_continuum, 'plot_continuum_fit')
    plot_norm_continuum = get_config_value(plot_norm_continuum, 'plot_normalized_continuum')

    # ensure cubes are iterable
    cubes = to_list(cubes)
    cubes = ensure_common_unit(cubes)

    # set plot style and colors
    style = return_stylename(style)

    extracted_spectra = []

    for cube in cubes:

        # shift by radial velocity
        spectral_axis = shift_by_radial_vel(cube.spectral_axis, radial_vel)
        spectral_axis = convert_quantity(spectral_axis, unit, equivalencies=u.spectral())

        # extract spectrum flux
        flux = extract_method(cube)
        # convert to Quantity
        flux = flux.value * flux.unit

        if deredden:
            flux = deredden_flux(
                spectral_axis, flux, Rv, Ebv,
                deredden_method, deredden_region
            )

        # initialize Spectrum object
        spectrum = Spectrum(
            spectral_axis=spectral_axis,
            flux=flux,
            rest_value=rest_freq,
            velocity_convention=convention
        )

        # compute continuum fit
        continuum = fit_continuum(spectrum, fit_method, region)

        # compute normalized flux
        flux_normalized = spectrum / continuum

        # save computed spectrum
        extracted_spectra.append(SpectrumPlus(
            spectrum=spectrum,
            normalized=flux_normalized.flux,
            continuum=continuum,
            fit_method=fit_method,
            region=region
        ))

    # plot spectrum
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        _ = plot_spectrum(extracted_spectra, ax, plot_norm_continuum,
                          plot_continuum, emission_line, **kwargs)
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()

    # ensure a list is only returned if returning more than 1 spectrum
    spectra_out = _unwrap_if_single(extracted_spectra)

    return spectra_out


def extract_cube_pixel_spectra(
    cube, *, idx=None, idx_range=None,
    plot_combined=False, combine_method=None,
    vline=None, cmap=None, style=None, **kwargs,
):
    """
    Extract per-pixel spectra from a spectral cube, keeping only spatial
    pixels that contain at least one non-NaN value along the spectral axis.

    Optionally compute and plot a combined spectrum derived from the
    extracted pixel spectra.

    Parameters
    ----------
    cube : DataCube, SpectralCube, Quantity or array-like
        Spectral cube with shape (T, N, M). If `cube`
        has no units, it is assined `u.dimensionless_unscaled`.
    idx : int, sequence of int, or None, optional, default=None
        Index or indices of the extracted per-pixel spectra to plot.
        If None, all extracted pixel spectra are plotted.
        If an int, only the spectrum with that index is plotted.
        If a sequence of ints (e.g., list, tuple, or NumPy array),
        only the spectra with those indices are plotted.
        The indices correspond to the ordering shown in the legend,
        where each entry is labeled as:
            <index>: (x=<x>, y=<y>)
    idx_range : sequence of two int or None, optional, default=None
        Selects a range of indices of the extracted per-pixel spectra
        to plot. Internally equivalent to `idx=np.arange(range[0],range[1]+1,1)`.
        Overrides idx.
    plot_combined : bool, optional, default=False
        If True, compute and plot a combined spectrum from all extracted
        pixel spectra using `combine_method`.
    combine_method : {'sum', 'mean', 'median'} or None, optional, default=None
        Method used to combine per-pixel spectra when `plot_combined=True`.
        If None, uses the default value from `config.flux_extract_method`.
    vline : Quantity or float or None, optional
        If provided, draw a vertical dotted reference line at this wavelength.
        If unitless, the value is assumed to be in the same units as the
        spectral axis.
    cmap : str or None, optional, default=None
        Colormap used to sample per-spectrum colors.
        If None, uses the default value from `config.cmap`.
    style : str or None, optional, default=None
        Matplotlib style to use for plotting.
        If None, uses the default value from `config.style`.
    figsize : tuple, optional, default=(12,6)
        Plot figsize.
    fontsize : float, optional, default=8
        Font size of the legend.
    ncols : int, optional, default=8
        Number of columns for the legend.
    savefig : bool, optional, default=False
        If True, saves figure to disk.

    Returns
    -------
    spectra : list of SpectrumPlus
        List of length A, where each element is a per-pixel spectrum with
        shape (T,). A is the number of spatial pixels containing valid data.
    combined_spec : SpectrumPlus
        Combined spectrum, only returned if `plot_combined` is True.
    """
    figsize = kwargs.get('figsize', (12,6))
    fontsize = kwargs.get('fontsize', 8)
    ncols = kwargs.get('ncols', 8)
    savefig = kwargs.get('savefig', False)

    combine_method = get_config_value(combine_method, 'flux_extract_method')
    style = get_config_value(style, 'style')
    cmap = get_config_value(cmap, 'cmap')

    if plot_combined and combine_method not in {'sum', 'mean', 'median'}:
        raise ValueError(
            "`combine_method` must be one of {'sum', 'mean', 'median'} when "
            "`plot_combined=True`."
        )

    spectral_axis = get_spectral_axis(cube)
    if spectral_axis is None:
        raise ValueError('Could not determine spectral_axis from cube')

    data_unit = get_unit(cube)
    if data_unit is None:
        data_unit = u.dimensionless_unscaled

    data = to_array(cube, keep_units=False)
    if data.ndim != 3:
        raise ValueError('cube must have shape (T, N, M)')

    if idx_range is not None:
        if (not isinstance(idx_range, (list, tuple, np.ndarray))
            or len(idx_range) != 2):
            raise TypeError(
                'range must be a sequence of two integers (start, stop)'
            )

        start, stop = map(int, idx_range)
        if stop < start:
            raise ValueError('range[1] must be >= range[0]')

        idx_set = set(range(start, stop + 1))

    elif idx is None:
        idx_set = None
    elif isinstance(idx, (int, np.integer)):
        idx_set = {int(idx)}
    elif isinstance(idx, (list, tuple, np.ndarray)):
        idx_set = {int(i) for i in idx}
    else:
        raise TypeError(
            'idx must be None, an int, or a sequence of ints'
        )

    # spatial pixels (spectrums) that are not entirely NaN
    valid_mask = ~np.all(np.isnan(data), axis=0)
    coords = np.column_stack(np.where(valid_mask))

    if idx_set is None:
        extract_idx = np.arange(len(coords))
    else:
        extract_idx = np.array(
            [i for i in sorted(idx_set) if i < len(coords)],
            dtype=int,
        )
    if extract_idx.size == 0:
        raise ValueError('idx does not select any valid spectra')

    coords = coords[extract_idx]
    ys = coords[:, 0]
    xs = coords[:, 1]

    flux_matrix = data[:, ys, xs].T * data_unit
    spectra = [
        SpectrumPlus(
            Spectrum(spectral_axis=spectral_axis, flux=flux)
        )
        for flux in flux_matrix
    ]

    labels = [
        f"{i}: (x={x}, y={y})"
        for i, (y, x) in zip(extract_idx, coords)
    ]

    n_plot = len(spectra)

    combined_spec = None
    style = return_stylename(style)

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_autoscale_on(False)

        if isinstance(cmap, (list, tuple, np.ndarray)) and len(cmap) >= n_plot:
            colors = list(cmap[:n_plot])
        else:
            colors = sample_cmap(n_plot, cmap)

        for spec, label, color in zip(spectra, labels, colors):
            plot_spectrum(
                spec,
                ax,
                color=[color],
                label=label,
                plot_continuum=False,
            )

        fluxes = [spec.flux for spec in spectra]

        if plot_combined:
            if combine_method == 'sum':
                combined_flux = np.nansum(flux_matrix, axis=0)
            elif combine_method == 'mean':
                combined_flux = np.nanmean(flux_matrix, axis=0)
            elif combine_method == 'median':
                combined_flux = np.nanmedian(flux_matrix, axis=0)
            else:
                raise ValueError(f'Unknown combine_method: {combine_method}')

            combined_spec = SpectrumPlus(
                Spectrum(spectral_axis=spectral_axis, flux=combined_flux)
            )
            plot_spectrum(
                combined_spec,
                ax,
                color='k',
                ls='--',
                label=f'combined ({combine_method})',
                plot_continuum=False,
            )

            fluxes.append(combined_flux)

        if vline is not None:
            if isinstance(vline, Quantity):
                vline = vline.to(spectral_axis.unit).value
            ax.axvline(
                vline,
                ls=':',
                lw=1.0,
                color='k',
                alpha=0.7,
                zorder=0,
            )

        set_axis_limits(spectral_axis, fluxes, ax, **kwargs)

        ax.legend(
            fontsize=fontsize,
            ncols=ncols,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
        )

        if savefig:
            save_figure_2_disk(**kwargs)

        plt.show()

    spectra = _unwrap_if_single(spectra)
    if plot_combined:
        return spectra, combined_spec
    return spectra


# Spectra Plotting Functions
# --------------------------
def plot_spectrum(extracted_spectra=None, ax=None, plot_norm_continuum=None,
                  plot_continuum=None, emission_line=None, wavelength=None,
                  flux=None, continuum=None, colors=None, vline=None, **kwargs):
    '''
    Plot one or more extracted spectra on a matplotlib Axes.

    Parameters
    ----------
    extracted_spectrums : SpectrumPlus or list of SpectrumPlus, optional
        Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
        and `flux` must be given.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    plot_norm_continuum : bool, optional, default=None
        If True, plot normalized flux instead of raw flux.
        If None, uses the default value set by `plot_normalized_continuum`.
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
        If None, uses the default color palette from
        `config.default_palette`.
    vline : Quantity or float or None, optional
        If provided, draw a vertical dotted reference line at this wavelength.
        If unitless, the value is assumed to be in the same units as the
        spectral axis.

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
        - `loc` : str, default=`config.loc`
            Location of legend.
        - `xlabel` : str, optional
            Label for the x-axis.
        - `ylabel` : str, optional
            Label for the y-axis.
        - `text_loc` : list of float, optional, default=`config.text_loc`
            Location for emission line annotation text in axes coordinates.
        - `use_brackets` : bool, optional, default=`config.use_brackets`
            If True, plot units in square brackets; otherwise, parentheses.

    Returns
    -------
    lines : Line2D or list of Line2D, or PlotSpectrum
        The plotted line object(s) created by `Axes.plot`.

        - If `plot_continuum` is False, returns a single `Line2D` object
          or a list of `Line2D` objects corresponding to the main spectrum.
        - If `plot_continuum` is True, returns a `PlotSpectrum` named tuple
          with the following fields:
            * `lines` : Line2D or list of Line2D
              The plotted spectrum line(s).
            * `continuum_lines` : Line2D or list of Line2D
              The plotted continuum fit line(s), if available.
    '''
    # ---- KWARGS ----
    # fig params
    rasterized = kwargs.get('rasterized', config.rasterized)
    # line params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = get_kwargs(kwargs, 'linewidths', 'linewidth', 'lw', default=None)
    alphas = get_kwargs(kwargs, 'alphas', 'alpha', 'a', default=None)
    zorder = get_kwargs(kwargs, 'zorders', 'zorder', default=None)
    cmap = kwargs.get('cmap', config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    text_loc = kwargs.get('text_loc', config.plot_spectrum_text_loc)
    use_brackets = kwargs.get('use_brackets', config.use_brackets)

    # get default config values
    plot_norm_continuum = get_config_value(plot_norm_continuum, 'plot_normalized_continuum')
    plot_continuum = get_config_value(plot_continuum, 'plot_continuum_fit')
    colors = get_config_value(colors, 'colors')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    # ensure an axis is passed
    if ax is None:
        raise ValueError('ax must be a matplotlib axes object!')

    # construct SpectrumPlus if user passes in wavelength and flux
    if extracted_spectra is None:

        # disable normalization because the user provided raw arrays
        plot_norm_continuum = False

        # normalize continuum_fit into a list
        if isinstance(continuum, (list, tuple)):
            continuum_list = list(continuum)
        else:
            continuum_list = [continuum]

        # case 1: single wavelength/flux array
        if (
            isinstance(wavelength, (np.ndarray, Quantity)) and
            isinstance(flux, (np.ndarray, Quantity))
        ):
            extracted_spectra = SpectrumPlus(
                wavelength=wavelength,
                flux=flux,
                continuum=continuum_list[0]
            )
        # case 2: multiple arrays
        elif (
            isinstance(wavelength, (list, tuple)) and
            isinstance(flux, (list, tuple)) and
            len(wavelength) == len(flux)
        ):
            extracted_spectra = [
                SpectrumPlus(
                    wavelength=w,
                    flux=f,
                    continuum=continuum_list[i % len(continuum_list)]
                )
                for i, (w, f) in enumerate(zip(wavelength, flux))
            ]
        else:
            raise ValueError(
                'Either pass `extracted_spectra`, or provide matching '
                '`wavelength` and `flux` arguments. \nFor multiple spectra, '
                'use lists of wavelength and flux arrays with equal length.'
            )

    # ensure extracted_spectra is iterable
    extracted_spectra = to_list(extracted_spectra)
    extracted_spectra = ensure_common_unit(extracted_spectra)
    linestyles = linestyles if isinstance(linestyles, (list, tuple)) else [linestyles]
    linewidths = linewidths if isinstance(linewidths, (list, tuple)) else [linewidths]
    alphas = alphas if isinstance(alphas, (list, tuple)) else [alphas]
    zorders = zorder if isinstance(zorder, (list, tuple)) else [zorder]
    labels = labels if isinstance(labels, (list, tuple)) else [labels]

    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors, cmap=cmap)
    # add emission line text
    if emission_line is not None:
        ax.text(text_loc[0], text_loc[1], f'{emission_line}', transform=ax.transAxes)

    lines = []
    fit_lines = []
    wavelength_list = []

    # loop through each spectrum
    for i, extracted_spectrum in enumerate(extracted_spectra):

        # extract wavelength and flux
        wavelength = extracted_spectrum.spectral_axis
        if plot_norm_continuum:
            flux = extracted_spectrum.normalized
        else:
            flux = extracted_spectrum.flux

        # mask wavelength within data range
        mask = mask_within_range(wavelength, xlim=xlim)
        wavelength_list.append(wavelength[mask])

        # define plot params
        color = colors[i%len(colors)]
        fit_color = fit_colors[i%len(fit_colors)]
        linestyle = linestyles[i%len(linestyles)]
        linewidth = linewidths[i%len(linewidths)]
        alpha = alphas[i%len(alphas)]
        zorder = zorders[i%len(zorders)] if zorders[i%len(zorders)] is not None else i
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        # plot spectrum
        l = ax.plot(wavelength[mask], flux[mask], c=color,
                    ls=linestyle, lw=linewidth, alpha=alpha,
                    zorder=zorder, label=label, rasterized=rasterized)
        # plot continuum fit
        if plot_continuum and extracted_spectrum.continuum is not None:
            if plot_norm_continuum:
                # normalize continuum fit
                continuum = extracted_spectrum.continuum/extracted_spectrum.continuum
            else:
                continuum = extracted_spectrum.continuum
            fl = ax.plot(wavelength[mask], continuum[mask], c=fit_color,
                         ls=linestyle, lw=linewidth, alpha=alpha, rasterized=rasterized)

            fit_lines.append(fl)

        lines.append(l)

    # set plot axis limits and labels
    set_axis_limits(wavelength_list, None, ax, xlim, ylim)
    set_axis_labels(wavelength, extracted_spectrum.flux, ax,
                    xlabel, ylabel, use_brackets=use_brackets)

    if vline is not None:
        if isinstance(vline, Quantity):
            vline = vline.to(extracted_spectrum.unit).value
        ax.axvline(
            vline,
            ls=':',
            lw=1.0,
            color='k',
            alpha=0.7,
            zorder=0,
        )

    if labels[0] is not None:
        ax.legend(loc=loc)

    lines = _unwrap_if_single(lines)
    if plot_continuum:
        PlotHandles = namedtuple('PlotSpectrum', ['lines', 'continuum_lines'])
        fit_lines = _unwrap_if_single(fit_lines)

        return PlotHandles(lines, fit_lines)

    return lines


def plot_combine_spectrum(extracted_spectra, ax, idx=0, wave_cuttofs=None,
                          concatenate=False, return_spectra=False,
                          plot_normalize=False, use_samecolor=True,
                          colors=None, **kwargs):
    '''
    Allows for easily plotting multiple spectra and stiching them together into
    one `SpectrumPlus` object.

    Parameters
    ----------
    extracted_spectra : list of `SpectrumPlus`/`Spectrum`, or list of list of `SpectrumPlus`/`Spectrum`
        List of spectra to plot. Each element should contain wavelength and flux attributes,
        and optionally the normalize attribute.
    ax : matplotlib.axes.Axes
        Axis on which to plot the spectra.
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
        If None, uses the default color palette from
        `config.default_palette`.

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
        - loc : str, optional, default=`config.loc`
            Legend location (e.g., 'best', 'upper right').
        - xlabel, ylabel : str, optional, default=None
            Axis labels.
        - use_brackets : bool, optional, default=`config.use_brackets`
            If True, format axis labels with units in brackets instead of parentheses.

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
    # ---- KWARGS ----
    # figure params
    rasterized = kwargs.get('rasterized', config.rasterized)
    ylim = kwargs.get('ylim', None)
    # line params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = get_kwargs(kwargs, 'linewidths', 'linewidth', 'lw', default=None)
    alphas = get_kwargs(kwargs, 'alphas', 'alpha', 'a', default=None)
    cmap = kwargs.get('cmap', config.cmap)
    # labels
    label = kwargs.get('label', None)
    loc = kwargs.get('loc', config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    use_brackets = kwargs.get('use_brackets', config.use_brackets)

    # get default config values
    colors = get_config_value(colors, 'colors')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    # ensure units match and that extracted_spectra is a list
    extracted_spectra = to_list(extracted_spectra)
    extracted_spectra = ensure_common_unit(extracted_spectra)
    # hardcode behavior to avoid breaking
    if return_spectra:
        concatenate = True
    if concatenate:
        use_samecolor = True

    # set plot style and colors
    colors, _ = set_plot_colors(colors, cmap=cmap)

    wave_list = []
    flux_list = []
    wavelength_lims = []
    for i, spectrum in enumerate(extracted_spectra):
        # index spectrum if list
        spectrum = spectrum[idx] if isinstance(spectrum, list) else spectrum
        # extract wavelength and flux
        wavelength = spectrum.spectral_axis
        flux = spectrum.normalize if plot_normalize else spectrum.flux
        # compute minimum and maximum wavelength values
        wmin = np.nanmin(get_value(wavelength))
        wmax = np.nanmax(get_value(wavelength))
        wavelength_lims.append( [wmin, wmax] )
        # mask wavelength and flux if user passes in limits
        if wave_cuttofs is not None:
            wave_min = wave_cuttofs[i]
            wave_max = wave_cuttofs[i+1]
            mask = mask_within_range(get_value(wavelength), [wave_min, wave_max])
            wavelength = wavelength[mask]
            flux = flux[mask]

        c = colors[0] if use_samecolor else colors[i%len(colors)]
        # only plot a label for combined spectrum, not each sub spectra
        l = label if label is not None and i == len(extracted_spectra)-1 else None
        # append to lists if concatenate
        if concatenate:
            wave_list.append(wavelength)
            flux_list.append(flux)
        # plot spectrum if not concatenating
        else:
            ax.plot(wavelength, flux, color=c,
                    label=l, ls=linestyles,
                    lw=linewidths, alpha=alphas,
                    rasterized=rasterized)
    # plot entire spectrum if concatenate
    if concatenate:
        wavelength = np.concatenate(wave_list)
        flux = np.concatenate(flux_list)

        ax.plot(get_value(wavelength),
                get_value(flux),
                color=c, label=l, ls=linestyles,
                lw=linewidths, alpha=alphas,
                rasterized=rasterized)

    set_axis_labels(wavelength, flux, ax, xlabel, ylabel, use_brackets)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if wave_cuttofs is None:
        xmin = min(l[0] for l in wavelength_lims)
        xmax = max(l[1] for l in wavelength_lims)
    else:
        xmin = wave_cuttofs[0]
        xmax = wave_cuttofs[-1]
    ax.set_xlim(xmin, xmax)

    if label is not None:
        ax.legend(loc=loc)

    if return_spectra:
        extracted_spectrum = SpectrumPlus(wavelength, flux)

        return extracted_spectrum


# Spectra Fitting Functions
# -------------------------
def fit_gaussian_2_spec(
    extracted_spectrum, p0, *, model=None, spectral_range=None,
    fit_method=None, absolute_sigma=None, yerror=None,
    interpolate=None, samples=None, interp_method=None,
    error_interp_method=None, return_fit_params=None,
    plot_interp=False, print_vals=None, **kwargs
):
    '''
    Fit a Gaussian-like model to a Spectrum, optionally including the continuum.

    Parameters
    ----------
    extracted_spectrum : SpectrumPlus or Spectrum
        Spectrum object to be gaussian fitted.
    p0 : list
        Initial guess for the Gaussian fit parameters.
        This should match the input arguments of the
        gaussian model (excluding the first argument
        which is wavelength).
    model : {'gaussian', 'gaussian_line', 'gaussian_continuum'} or None, default=None
        Type of Gaussian model to fit:
        - 'gaussian' : standard Gaussian
        - 'gaussian_line' : Gaussian with linear continuum
        - 'gaussian_continuum' : Gaussian with computed continuum array
        The continuum can be computed with fit_continuum().
        If None, uses the default value set by `config.gaussian_model`.
    spectral_range : array-like or None, optional, default=None
        (min, max) wavelength range to restrict the fit.
        If None, computes the min and max from the wavelength.
    fit_method : {'lm', 'trf', 'dogbox'} or None, optional, default=None
        Curve fitting algorithm used by `scipy.optimize.curve_fit`.
        If None, uses the default value set by `config.curve_fit_method`.
    absolute_sigma : boolean, optional, default=None
        If True, the values provided in `yerror` are interpreted as absolute
        1σ uncertainties on the flux measurements. In this case, the returned
        covariance matrix reflects these absolute uncertainties, and parameter
        errors are reported in physical units.
        If False, the values in `yerror` are treated as relative weights only.
        The covariance matrix is rescaled such that the reduced χ² of the fit
        is unity, and the reported parameter uncertainties reflect relative
        errors rather than absolute measurement uncertainties.
        Set this to True when `yerror` represents well-calibrated observational
        uncertainties (e.g., photon-counting or pipeline-provided errors).
        Set this to False when `yerror` is used only for weighting the fit.
        If None, uses the default value set by `config.curve_fit_absolute_sigma`.
    yerror : array-like or None, optional, default=None
        Flux uncertainties to be used in the fit. If None,
        uncertainties are ignored when computing the fit.
        This is passed to `curve_fit` as the `sigma` parameter.
    interpolate : bool or None, default=None
        Whether to interpolate the spectrum over a regular wavelength grid.
        The number of samples is controlled by `samples`. If None, uses the
        default value set by `config.curve_fit_interpolate`.
    samples : int or None, default=None
        Number of points in interpolated wavelength grid. If
        None, uses the default value set by `config.interpolation_samples`.
    interp_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Interpolation method used. If None, uses the default
        value set by `config.interpolation_method`.
    error_interp_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Method to interpolate yerror if provided. If None, uses
        the default value set by `config.error_interpolation_method`.
    return_fit_params : bool or None, default=None
        If True, return full computed best-fit parameters for all parameters,
        including popt, pcov, and perr. If False, return only Flux, FWHM, and mu.
        If None, uses the default value set by `config.return_gaussian_fit_parameters`.
    plot_interp : bool, default=False
        If True, plot the interpolated spectrum. This is
        provided for debugging purposes.
    print_vals : bool or None, default=None
        If True, print a table of best-fit parameters,
        errors, and computed quantities. If None, uses the
        default value set by `config.print_gaussian_values`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `figsize` : list or tuple, optional, default=`config.figsize`
            Figure size.
        - `style` : str or {'astro', 'latex', 'minimal', 'default'}, optional, default=`config.style`
            Plot style used. Can either be a matplotlib mplstyle
            or an included visualastro style.
        - `xlim` : tuple, optional, default=None
            Wavelength range for plotting. If None, uses `wave_range`.
        - `plot_type` : {'plot', 'scatter'}, optional, default='plot'
            Matplotlib plotting style to use.
        - `label` : str, optional, default=None
            Spectrum legend label.
        - `xlabel` : str, optional, default=None
            Plot x-axis label.
        - `ylabel` : str, optional, default=None
            Plot y-axis label.
        - `colors` : str or list, optional, default=`config.colors`
            Plot colors. If None, will use default visualastro color palette.
        - `use_brackets` : bool, optional, default=`config.use_brackets`
            If True, use square brackets for plot units. If False, use parentheses.
        - `savefig` : bool, optional, default=`config.savefig`
            If True, save current figure to disk.
        - `dpi` : float or int, optional, default=`config.dpi`
            Resolution in dots per inch.

    Returns
    -------
    GaussianFitResult or GaussianHandles

        GaussianFitResult : dataclass (when `return_fit_params=True`):
            Contains all fitted parameters (amplitude, mu, sigma, and optionally
            slope/intercept for line models), derived quantities (flux, FWHM),
            and their 1σ uncertainties. Also includes raw fit outputs (popt, pcov, perr).
            See dataclass definition for complete attribute list.

        GaussianHandles : namedtuple (when `return_fit_params=False`):
            Simplified namedtuple with key results:

            - flux, FWHM, mu : Quantity
                Integrated flux, full width at half maximum, and center position.
            - flux_error, FWHM_error, mu_error : Quantity
                1σ uncertainties on the above quantities.
    '''
    # ---- KWARGS ----
    # figure params
    figsize = kwargs.get('figsize', config.figsize)
    style = kwargs.get('style', config.style)
    xlim = kwargs.get('xlim', None)
    plot_type = kwargs.get('plot_type', 'plot')
    # labels
    label = kwargs.get('label', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=None)
    use_brackets = kwargs.get('use_brackets', config.use_brackets)
    # savefig
    savefig = kwargs.get('savefig', config.savefig)
    dpi = kwargs.get('dpi', config.dpi)

    # get default config values
    colors = get_config_value(colors, 'colors')
    model = get_config_value(model, 'gaussian_model')
    fit_method = get_config_value(fit_method, 'curve_fit_method')
    absolute_sigma = get_config_value(absolute_sigma, 'curve_fit_absolute_sigma')
    interpolate = get_config_value(interpolate, 'curve_fit_interpolate')
    interp_method = get_config_value(interp_method, 'interpolation_method')
    error_interp_method = get_config_value(error_interp_method, 'error_interpolation_method')
    samples = get_config_value(samples, 'interpolation_samples')
    return_fit_params = get_config_value(return_fit_params, 'return_gaussian_fit_parameters')
    print_vals = get_config_value(print_vals, 'print_gaussian_values')

    # ensure arrays are not quantity objects
    spectral_axis = extracted_spectrum.spectral_axis
    flux = extracted_spectrum.flux

    spectral_unit = spectral_axis.unit
    flux_unit = flux.unit

    x0 = spectral_axis.to_value()
    y0 = flux.to_value()

    if yerror is not None:
        yerror = yerror.to_value()

    p0 = list(p0)

    # compute default wavelength range from wavelength
    if spectral_range is None:
        spectral_range = [np.nanmin(x0), np.nanmax(x0)]

    if model == 'gaussian_continuum':
        # remove continuum values to ensure it is not
        # included as a free parameter during minimization
        continuum = np.asarray(p0.pop(-1))
    else:
        continuum = None

    # interpolate arrays
    if interpolate:
        # interpolate wavelength and flux arrays
        x, y = _interpolate(
            x0, y0, spectral_range, samples, method=interp_method
        )
        # interpolate y error values
        if yerror is not None:
            _, yerror = _interpolate(
                x0, yerror, spectral_range, samples, method=error_interp_method
            )
        # interpolate continuum array
        if model == 'gaussian_continuum':
            _, continuum = _interpolate(
                x0, continuum, spectral_range, samples, method=interp_method
            )
    else:
        x, y = x0, y0

    # clip values outside wavelength range
    spectral_mask = mask_within_range(x, spectral_range)

    x_sub = x[spectral_mask]
    y_sub = y[spectral_mask]
    yerr_sub = yerror[spectral_mask] if yerror is not None else None

    if len(x_sub) == 0:
        raise ValueError(
            f'No data points within spectral_range {spectral_range}'
        )

    if model == 'gaussian_continuum' and continuum is not None:
        continuum_sub = continuum[spectral_mask]

        def function(x, A, mu, sigma):
            return _gaussian_continuum(x, A, mu, sigma, continuum_sub)

    elif model == 'gaussian_line':
        function = _gaussian_line

    else:
        function = _gaussian

    # fit gaussian model to data
    popt, pcov = curve_fit(
        function,
        x_sub,
        y_sub,
        p0,
        sigma=yerr_sub,
        absolute_sigma=absolute_sigma,
        method=fit_method
    )

    # estimate errors
    perr = np.sqrt(np.diag(pcov))

    # extract physical quantities from model fitting
    amplitude = popt[0] * flux_unit
    amplitude_error = perr[0] * flux_unit

    mu = popt[1] * spectral_unit
    mu_error = perr[1] * spectral_unit

    sigma = popt[2] * spectral_unit
    sigma_error = perr[2] * spectral_unit

    # compute integrated flux, FWHM, and their errors
    integrated_flux = amplitude * sigma * np.sqrt(2*np.pi)
    flux_error = integrated_flux * np.sqrt(
        (amplitude_error / amplitude)**2 +
        (sigma_error / sigma)**2
    )

    FWHM = 2*sigma * np.sqrt(2*np.log(2))
    FWHM_error = 2*sigma_error * np.sqrt(2*np.log(2))

    if model == 'gaussian_line' and len(popt) > 3:
        m = popt[3] * (flux_unit / spectral_unit)
        m_error = perr[3] * (flux_unit / spectral_unit)
        b = popt[4] * flux_unit
        b_error = perr[4] * flux_unit
    else:
        m = None
        m_error = None
        b = None
        b_error = None

    fit_config={
        'gaussian_model': model,
        'curve_fit_method': fit_method,
        'absolute_sigma': absolute_sigma,
        'interpolate': interpolate,
    }
    if interpolate:
        fit_config.update(
            interpolate_method=interp_method,
            error_interp_method=error_interp_method,
            interpolate_samples=samples
        )

    result = GaussianFitResult(
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        mu=mu,
        mu_error=mu_error,
        sigma=sigma,
        sigma_error=sigma_error,
        flux=integrated_flux,
        flux_error=flux_error,
        FWHM=FWHM,
        FWHM_error=FWHM_error,
        slope=m,
        slope_error=m_error,
        intercept=b,
        intercept_error=b_error,
        popt=popt,
        pcov=pcov,
        perr=perr,
        p0=p0,
        fit_config=fit_config

    )

    # set plot style and colors
    colors, _ = set_plot_colors(colors)
    style = return_stylename(style)

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        # determine plot type
        plt_plot = {
            'plot': ax.plot,
            'scatter': ax.scatter
        }.get(plot_type, ax.plot)
        # plot interpolated data
        if plot_interp and interpolate:
            plt_plot(
                x, y, c=colors[2%len(colors)], label='Interpolated'
            )
        # plot original data
        # clip values outisde of plotting range
        xlim = spectral_range if xlim is None else xlim
        plot_mask = mask_within_range(x0, xlim)
        gauss_mask = mask_within_range(x_sub, xlim)
        label = label if label is not None else 'Spectrum'

        plt_plot(x0[plot_mask], y0[plot_mask],
                 c=colors[0%len(colors)], label=label)
        # plot gaussian model
        gaussian = function(x_sub, *popt)
        ax.plot(
            x_sub,
            gaussian,
            c=colors[1%len(colors)],
            label='Gaussian Model'
        )
        # set axis labels and limits
        set_axis_labels(
            spectral_axis, flux, ax, xlabel, ylabel, use_brackets
        )
        set_axis_limits(x0[plot_mask], [y0[plot_mask], gaussian[gauss_mask]], ax, xlim=xlim)

        plt.legend()
        if savefig:
            save_figure_2_disk(dpi=dpi)
        plt.show()

    if print_vals:
        result.pretty_print(**kwargs)

    if return_fit_params:
        return result
    else:
        default_return = ['flux', 'FWHM', 'mu', 'flux_error', 'FWHM_error', 'mu_error']
        GaussianHandles = namedtuple('GaussianFit', default_return)

        return GaussianHandles(
            integrated_flux, FWHM, mu, flux_error, FWHM_error, mu_error
        )
