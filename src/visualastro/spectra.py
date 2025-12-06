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
from astropy.units import Quantity
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from specutils.spectra import Spectrum1D
from .io import get_kwargs, save_figure_2_disk
from .numerical_utils import (
    check_units_consistency, convert_units, interpolate_arrays,
    mask_within_range, return_array_values, shift_by_radial_vel
)
from .plot_utils import (
    return_stylename, set_axis_labels,
    set_axis_limits, set_plot_colors
)
from .spectra_utils import (
    compute_continuum_fit,
    deredden_flux, gaussian,
    gaussian_continuum, gaussian_line,
    get_config_value
)
from .va_config import get_config_value, va_config, _default_flag
from .ExtractedSpectrum import ExtractedSpectrum


# Spectra Extraction Functions
# ––––––––––––––––––––––––––––
def extract_cube_spectra(cubes, flux_extract_method=None, extract_mode=None, fit_method=None,
                         region=None, radial_vel=_default_flag, rest_freq=_default_flag,
                         deredden=None, unit=_default_flag, emission_line=None,
                         plot_continuum_fit=None, plot_norm_continuum=None, **kwargs):
    '''
    Extract 1D spectra from one or more data cubes, with optional continuum normalization,
    dereddening, and plotting.
    Parameters
    ––––––––––
    cubes : DataCube, SpectralCube, or list of cubes
        Input cube(s) from which to extract spectra. The data must either be
        a SpectralCube, or a DataCube containing a SpectralCube.
    flux_extract_method : {'mean', 'median', 'sum'} or None, default=None
        Method for extracting the flux. If None, uses the default
        value set by `va_config.flux_extract_method`.
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
        If None, uses the default value set by `va_config.spectral_cube_extraction_mode`.
    fit_method : {'fit_continuum', 'generic'} or None, optional, default=None
        Method used to fit the continuum. If None, uses the default
        value set by `va_config.spectrum_continuum_fit_method`.
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
        If `_default_flag`, uses the default value set by `va_config.radial_velocity`.
    rest_freq : float or None, optional, default=`_default_flag`
        Rest-frame frequency or wavelength of the spectrum. If None,
        ignores the rest frequency for unit conversions. If `_default_flag`,
        uses the default value set by `va_config.spectra_rest_frequency`.
    deredden : bool or None, optional, default=None
        Whether to apply dereddening to the flux using deredden_flux().
        If None, uses the default value set by `va_config.deredden_spectrum`.
    unit : str, astropy.units.Unit, or None, optional, default=`_default_flag`
        Desired units for the wavelength axis. Converts the default
        units if possible. If None, does not try and convert. If `_default_flag`,
        uses the default value set by `va_config.wavelength_unit`.
    emission_line : str, optional, default=None
        Name of an emission line to annotate on the plot.
    plot_continuum_fit : bool or None, optional, default=None
        Whether to overplot the continuum fit. If None, uses the
        default value set by `va_config.plot_continuum_fit`.
    plot_norm_continuum : bool or None, optional, default=None
        Whether to plot the normalized extracted spectra. If None,
        uses the default value set by `va_config.plot_normalized_continuum`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `how` : str, optional, default=`va_config.spectral_cube_extraction_mode`
            Alias for `extract_mode`.
        - `convention` : str, optional
            Doppler convention.
        - `Rv` : float, optional, default=`va_config.Rv`
            Dereddening parameter.
        - `Ebv` : float, optional, default=`va_config.Ebv`
            Dereddening parameter.
        - `deredden_method` : str, optional, default=`va_config.deredden_method`
            Extinction law to use.
        - `deredden_region` : str, optional, default=`va_config.deredden_region`
            Region/environment for WD01 extinction law.
        - `figsize` : tuple, optional, default=`va_config.figsize`
            Figure size for plotting.
        - `style` : str, optional, default=`va_config.style`
            Plotting style.
        - `savefig` : bool, optional, default=`va_config.savefig`
            Whether to save the figure to disk.
        - `dpi` : int, optional, default=`va_config.dpi`
            Figure resolution for saving.
        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `colors`, `color` or `c` : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.
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

    Returns
    –––––––
    ExtractedSpectrum or list of ExtractedSpectrum
        Single object if one cube is provided, list if multiple cubes are provided.
    '''
    # –––– KWARGS ––––
    # spectra extraction memory mode
    extract_mode = get_kwargs(kwargs, 'how', default=extract_mode)
    # doppler convention
    convention = kwargs.get('convention', None)
    # dereddening parameters
    Rv = kwargs.get('Rv', va_config.Rv)
    Ebv = kwargs.get('Ebv', va_config.Ebv)
    deredden_method = kwargs.get('deredden_method', va_config.deredden_method)
    deredden_region = kwargs.get('deredden_region', va_config.deredden_region)
    # figure params
    figsize = kwargs.get('figsize', va_config.figsize)
    style = kwargs.get('style', va_config.style)
    # savefig
    savefig = kwargs.get('savefig', va_config.savefig)
    dpi = kwargs.get('dpi', va_config.dpi)

    # get default va_config values
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
    radial_vel = va_config.radial_velocity if radial_vel is _default_flag else radial_vel
    rest_freq = va_config.spectra_rest_frequency if rest_freq is _default_flag else rest_freq
    deredden = get_config_value(deredden, 'deredden_spectrum')
    unit = va_config.wavelength_unit if unit is _default_flag else unit
    plot_continuum_fit = get_config_value(plot_continuum_fit, 'plot_continuum_fit')
    plot_norm_continuum = get_config_value(plot_norm_continuum, 'plot_normalized_continuum')

    # ensure cubes are iterable
    cubes = check_units_consistency(cubes)
    # set plot style and colors
    style = return_stylename(style)

    extracted_spectra = []
    for cube in cubes:

        # shift by radial velocity
        spectral_axis = shift_by_radial_vel(cube.spectral_axis, radial_vel)

        # extract spectrum flux
        flux = extract_method(cube)
        # convert to Quantity
        flux = flux.value * flux.unit

        # derreden
        if deredden:
            flux = deredden_flux(spectral_axis, flux, Rv, Ebv,
                                 deredden_method, deredden_region)

        # initialize Spectrum1D object
        spectrum1d = Spectrum1D(
            spectral_axis=spectral_axis,
            flux=flux,
            rest_value=rest_freq,
            velocity_convention=convention
        )

        # compute continuum fit
        continuum_fit = compute_continuum_fit(spectrum1d, fit_method, region)

        # compute normalized flux
        flux_normalized = spectrum1d / continuum_fit

        # variable for plotting wavelength
        wavelength = spectrum1d.spectral_axis
        # convert wavelength to desired units
        wavelength = convert_units(wavelength, unit)
        # rebuild spectrum1d after unit change
        spectrum1d = Spectrum1D(
            spectral_axis=wavelength,
            flux=flux,
            rest_value=rest_freq,
            velocity_convention=convention
        )

        # save computed spectrum
        extracted_spectra.append(ExtractedSpectrum(
            wavelength=wavelength,
            flux=flux,
            spectrum1d=spectrum1d,
            normalized=flux_normalized.flux,
            continuum_fit=continuum_fit
        ))

    # plot spectrum
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        _ = plot_spectrum(extracted_spectra, ax, plot_norm_continuum,
                          plot_continuum_fit, emission_line, **kwargs)
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()

    # ensure a list is only returned if returning more than 1 spectrum
    if len(extracted_spectra) == 1:
        return extracted_spectra[0]

    return extracted_spectra


# Spectra Plotting Functions
# ––––––––––––––––––––––––––
def plot_spectrum(extracted_spectra=None, ax=None, plot_norm_continuum=False,
                  plot_continuum_fit=False, emission_line=None, wavelength=None,
                  flux=None, continuum_fit=None, colors=None, **kwargs):
    '''
    Plot one or more extracted spectra on a matplotlib Axes.
    Parameters
    ––––––––––
    extracted_spectrums : ExtractedSpectrum or list of ExtractedSpectrum, optional
        Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
        and `flux` must be given.
    ax : matplotlib.axes.Axes
        Axis to plot on.
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

    Returns
    –––––––
    lines : Line2D or list of Line2D, or PlotSpectrum
        The plotted line object(s) created by `Axes.plot`.

        - If `plot_continuum_fit` is False, returns a single `Line2D` object
          or a list of `Line2D` objects corresponding to the main spectrum.
        - If `plot_continuum_fit` is True, returns a `PlotSpectrum` named tuple
          with the following fields:

            * `lines` : Line2D or list of Line2D
              The plotted spectrum line(s).

            * `continuum_lines` : Line2D or list of Line2D
              The plotted continuum fit line(s), if available.
    '''
    # –––– KWARGS ––––
    # fig params
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    # line params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = get_kwargs(kwargs, 'linewidths', 'linewidth', 'lw', default=None)
    alphas = get_kwargs(kwargs, 'alphas', 'alpha', 'a', default=None)
    zorder = get_kwargs(kwargs, 'zorders', 'zorder', default=None)
    cmap = kwargs.get('cmap', va_config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    text_loc = kwargs.get('text_loc', va_config.plot_spectrum_text_loc)
    use_brackets = kwargs.get('use_brackets', va_config.use_brackets)

    # get default va_config values
    colors = get_config_value(colors, 'colors')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    # ensure an axis is passed
    if ax is None:
        raise ValueError('ax must be a matplotlib axes object!')

    # construct ExtractedSpectrum if user passes in wavelenght and flux
    if extracted_spectra is None:

        # disable normalization because the user provided raw arrays
        plot_norm_continuum = False

        # normalize continuum_fit into a list
        if isinstance(continuum_fit, (list, tuple)):
            continuum_fit_list = list(continuum_fit)
        else:
            continuum_fit_list = [continuum_fit]

        # case 1: single wavelength/flux array
        if (
            isinstance(wavelength, (np.ndarray, Quantity)) and
            isinstance(flux, (np.ndarray, Quantity))
        ):
            extracted_spectra = ExtractedSpectrum(
                wavelength=wavelength,
                flux=flux,
                continuum_fit=continuum_fit_list[0]
            )
        # case 2: multiple arrays
        elif (
            isinstance(wavelength, (list, tuple)) and
            isinstance(flux, (list, tuple)) and
            len(wavelength) == len(flux)
        ):
            extracted_spectra = [
                ExtractedSpectrum(
                    wavelength=w,
                    flux=f,
                    continuum_fit=continuum_fit_list[i % len(continuum_fit_list)]
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
    extracted_spectra = check_units_consistency(extracted_spectra)
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
        wavelength = extracted_spectrum.wavelength
        if plot_norm_continuum:
            flux = extracted_spectrum.normalized
        else:
            flux = extracted_spectrum.flux

        # mask wavelength within data range
        mask = mask_within_range(wavelength, xlim=xlim)
        wavelength_list.append(wavelength[mask]) # type: ignore

        # define plot params
        color = colors[i%len(colors)]
        fit_color = fit_colors[i%len(fit_colors)]
        linestyle = linestyles[i%len(linestyles)]
        linewidth = linewidths[i%len(linewidths)]
        alpha = alphas[i%len(alphas)]
        zorder = zorders[i%len(zorders)] if zorders[i%len(zorders)] is not None else i
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        # plot spectrum
        l = ax.plot(wavelength[mask], flux[mask], c=color, # type: ignore
                    ls=linestyle, lw=linewidth, alpha=alpha,
                    zorder=zorder, label=label, rasterized=rasterized)
        # plot continuum fit
        if plot_continuum_fit and extracted_spectrum.continuum_fit is not None:
            if plot_norm_continuum:
                # normalize continuum fit
                continuum_fit = extracted_spectrum.continuum_fit/extracted_spectrum.continuum_fit
            else:
                continuum_fit = extracted_spectrum.continuum_fit
            fl = ax.plot(wavelength[mask], continuum_fit[mask], c=fit_color, # type: ignore
                         ls=linestyle, lw=linewidth, alpha=alpha, rasterized=rasterized)

            fit_lines.append(fl)

        lines.append(l)

    # set plot axis limits and labels
    set_axis_limits(wavelength_list, None, ax, xlim, ylim)
    set_axis_labels(wavelength, extracted_spectrum.flux, ax,
                    xlabel, ylabel, use_brackets=use_brackets)
    if labels[0] is not None:
        ax.legend(loc=loc)

    lines = lines[0] if len(lines) == 1 else lines
    if plot_continuum_fit:
        PlotHandles = namedtuple('PlotSpectrum', ['lines', 'continuum_lines'])
        fit_lines = fit_lines[0] if len(fit_lines) == 1 else fit_lines

        return PlotHandles(lines, fit_lines)

    return lines


def plot_combine_spectrum(extracted_spectra, ax, idx=0, wave_cuttofs=None,
                          concatenate=False, return_spectra=False,
                          plot_normalize=False, use_samecolor=True,
                          colors=None, **kwargs):
    '''
    Allows for easily plotting multiple spectra and stiching them together into
    one `ExtractedSpectrum` object.
    Parameters
    ––––––––––
    extracted_spectra : list of `ExtractedSpectrum`/`Spectrum1D`, or list of list of `ExtractedSpectrum`/`Spectrum1D`
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
    # –––– KWARGS ––––
    # figure params
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    ylim = kwargs.get('ylim', None)
    # line params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = get_kwargs(kwargs, 'linewidths', 'linewidth', 'lw', default=None)
    alphas = get_kwargs(kwargs, 'alphas', 'alpha', 'a', default=None)
    cmap = kwargs.get('cmap', va_config.cmap)
    # labels
    label = kwargs.get('label', None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    use_brackets = kwargs.get('use_brackets', va_config.use_brackets)

    # get default va_config values
    colors = get_config_value(colors, 'colors')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    # ensure units match and that extracted_spectra is a list
    extracted_spectra = check_units_consistency(extracted_spectra)
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
        wavelength = spectrum.wavelength
        flux = spectrum.normalize if plot_normalize else spectrum.flux
        # compute minimum and maximum wavelength values
        wmin = np.nanmin(return_array_values(wavelength))
        wmax = np.nanmax(return_array_values(wavelength))
        wavelength_lims.append( [wmin, wmax] )
        # mask wavelength and flux if user passes in limits
        if wave_cuttofs is not None:
            wave_min = wave_cuttofs[i]
            wave_max = wave_cuttofs[i+1]
            mask = mask_within_range(return_array_values(wavelength), [wave_min, wave_max])
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

        ax.plot(return_array_values(wavelength),
                return_array_values(flux),
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
        extracted_spectrum = ExtractedSpectrum(wavelength, flux)

        return extracted_spectrum


# Spectra Fitting Functions
# –––––––––––––––––––––––––
def fit_gaussian_2_spec(extracted_spectrum, p0, model=None, wave_range=None,
                        interpolate=None, interp_method=None, yerror=None,
                        error_method=None, samples=None, return_fit_params=None,
                        plot_interp=False, print_vals=None, **kwargs):
    '''
    Fit a Gaussian or Gaussian variant to a 1D spectrum, optionally including a continuum.
    Parameters
    ––––––––––
    extracted_spectrum : ExtractedSpectrum
        Spectrum object containing 'wavelength' and 'flux' arrays.
    p0 : list
        Initial guess for the Gaussian fit parameters.
        This should match the input arguments of the
        gaussian model (excluding the first argument
        which is wavelength).
    model : str or None, default=None
        Type of Gaussian model to fit:
        - 'gaussian' : standard Gaussian
        - 'gaussian_line' : Gaussian with linear continuum
        - 'gaussian_continuum' : Gaussian with computed continuum array
        The continuum can be computed with compute_continuum_fit().
        If None, uses the default value set by `va_config.gaussian_model`.
    wave_range : tuple or list, optional, default=None
        (min, max) wavelength range to restrict the fit.
        If None, computes the min and max from the wavelength.
    interpolate : bool or None, default=None
        Whether to interpolate the spectrum over
        a regular wavelength grid. The number of
        samples is controlled by `samples`. If None,
        uses the default value set by `va_config.interpolate`.
    interp_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Interpolation method used. If None, uses the default
        value set by `va_config.interpolation_method`.
    yerror : array-like or None, optional, default=None
        Flux uncertainties to be used in the fit. If None,
        uncertainties are ignored when computing the fit.
    error_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Method to interpolate yerror if provided. If None, uses
        the default value set by `va_config.error_interpolation_method`.
    samples : int or None, default=None
        Number of points in interpolated wavelength grid. If
        None, uses the default value set by `va_config.interpolation_samples`.
    return_fit_params : bool or None, default=None
        If True, return full computed best-fit parameters
        including derived flux and FWHM. If False, return
        only Flux, FWHM, and mu. If None, uses the default
        value set by `va_config.return_gaussian_fit_parameters`.
    plot_interp : bool, default=False
        If True, plot the interpolated spectrum. This is
        provided for debugging purposes.
    print_vals : bool or None, default=None
        If True, print a table of best-fit parameters,
        errors, and computed quantities. If None, uses the
        default value set by `va_config.print_gaussian_values`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `figsize` : list or tuple, optional, default=`va_config.figsize`
            Figure size.
        - `style` : str or {'astro', 'latex', 'minimal', 'default'}, optional, default=`va_config.style`
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
        - `colors` : str or list, optional, default=`va_config.colors`
            Plot colors. If None, will use default visualastro color palette.
        - `use_brackets` : bool, optional, default=`va_config.use_brackets`
            If True, use square brackets for plot units. If False, use parentheses.
        - `savefig` : bool, optional, default=`va_config.savefig`
            If True, save current figure to disk.
        - `dpi` : float or int, optional, default=`va_config.dpi`
            Resolution in dots per inch.

    Returns
    –––––––
    If return_fit_params:
        popt : np.ndarray
            Best-fit parameters including integrated flux and FWHM.
        perr : np.ndarray
            Uncertainties of fit parameters including flux and FWHM errors.
    Else:
        PlotHandles : namedtuple
            A `namedtuple` with the following fields:

            - `flux` : float
              Integrated flux of the fitted Gaussian.
            - `FWHM` : float
              Full width at half maximum of the fitted Gaussian.
            - `mu` : float
              Mean (central wavelength or position) of the fitted Gaussian.
            - `flux_error` : float
              1σ uncertainty on the integrated flux.
            - `FWHM_error` : float
              1σ uncertainty on the FWHM.
            - `mu_error` : float
              1σ uncertainty on the mean position.
    '''
    # –––– KWARGS ––––
    # figure params
    figsize = kwargs.get('figsize', va_config.figsize)
    style = kwargs.get('style', va_config.style)
    xlim = kwargs.get('xlim', None)
    plot_type = kwargs.get('plot_type', 'plot')
    # labels
    label = kwargs.get('label', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=None)
    use_brackets = kwargs.get('use_brackets', va_config.use_brackets)
    # savefig
    savefig = kwargs.get('savefig', va_config.savefig)
    dpi = kwargs.get('dpi', va_config.dpi)

    # get default va_config values
    colors = get_config_value(colors, 'colors')
    model = get_config_value(model, 'gaussian_model')
    interpolate = get_config_value(interpolate, 'interpolate')
    interp_method = get_config_value(interp_method, 'interpolation_method')
    error_method = get_config_value(error_method, 'error_interpolation_method')
    samples = get_config_value(samples, 'interpolation_samples')
    return_fit_params = get_config_value(return_fit_params, 'return_gaussian_fit_parameters')
    print_vals = get_config_value(print_vals, 'print_gaussian_values')

    # ensure arrays are not quantity objects
    wave_unit = extracted_spectrum.wavelength.unit
    flux_unit = extracted_spectrum.flux.unit
    wavelength = return_array_values(extracted_spectrum.wavelength)
    flux = return_array_values(extracted_spectrum.flux)
    # compute default wavelength range from wavelength
    wave_range = [np.nanmin(wavelength), np.nanmax(wavelength)] if wave_range is None else wave_range
    # guassian fitting function map
    function_map = {
        'gaussian': gaussian,
        'gaussian_line': gaussian_line,
        'gaussian_continuum': gaussian_continuum
    }
    if model == 'gaussian_continuum':
        continuum = p0[-1]
    # interpolate arrays
    if interpolate:
        # interpolate wavelength and flux arrays
        wavelength, flux = interpolate_arrays(wavelength, flux, wave_range,
                                              samples, method=interp_method)
        # interpolate y error values
        if yerror is not None:
            _, yerror = interpolate_arrays(extracted_spectrum.wavelength,
                                           yerror, wave_range, samples,
                                           method=error_method)
        # interpolate continuum array
        if model == 'gaussian_continuum':
            _, continuum = interpolate_arrays(extracted_spectrum.wavelength,
                                              continuum, wave_range, samples,
                                              method=interp_method)
            # remove continuum values to ensure it is not
            # included as a free parameter during minimization
            p0.pop(-1)

    # clip values outisde wavelength range
    wave_mask = mask_within_range(wavelength, wave_range)
    wave_sub = wavelength[wave_mask]
    flux_sub = flux[wave_mask]
    if yerror is not None:
        yerror = yerror[wave_mask]
    if model == 'gaussian_continuum':
        continuum = continuum[wave_mask]

    # extract fitting function from map
    function = function_map.get(model, gaussian)
    # fit gaussian model to data
    if model == 'gaussian_continuum':
        # define lambda function
        fitted_model = lambda x, A, mu, sigma: gaussian_continuum(x, A, mu, sigma, continuum)
        # fit gaussian to data
        popt, pcov = curve_fit(fitted_model, wave_sub, flux_sub, p0,
                               sigma=yerror, absolute_sigma=True, method='trf')
        # overwrite for plotting
        function = fitted_model
    else:
        # fit gaussian to data
        popt, pcov = curve_fit(function, wave_sub, flux_sub, p0, sigma=yerror,
                               absolute_sigma=True, method='trf')
    # estimate errors
    perr = np.sqrt(np.diag(pcov))
    # extract physical quantities from model fitting
    amplitude = popt[0] * flux_unit
    amplitude_error = perr[0] * flux_unit
    mu = popt[1] * wave_unit
    mu_error = perr[1] * wave_unit
    sigma = popt[2] * wave_unit
    sigma_error = perr[2] * wave_unit
    # compute integrated flux, FWHM, and their errors
    integrated_flux = amplitude * sigma * np.sqrt(2*np.pi)
    flux_error = np.sqrt(2*np.pi) * (
        np.sqrt((amplitude_error/amplitude)**2 + (sigma_error/sigma)**2) )
    FWHM = 2*sigma * np.sqrt(2*np.log(2))
    FWHM_error = 2*sigma_error * np.sqrt(2*np.log(2))

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
        if plot_interp:
            plt_plot(wavelength, flux,
                     c=colors[2%len(colors)],
                     label='Interpolated')
        # plot original data
        # re-extract values of original data
        wavelength = return_array_values(extracted_spectrum.wavelength)
        flux = return_array_values(extracted_spectrum.flux)
        # clip values outisde of plotting range
        xlim = wave_range if xlim is None else xlim
        plot_mask = mask_within_range(wavelength, xlim)
        label = label if label is not None else 'Spectrum'
        plt_plot(wavelength[plot_mask], flux[plot_mask],
                 c=colors[0%len(colors)], label=label)
        # plot gaussian model
        ax.plot(wave_sub, function(wave_sub, *popt),
                c=colors[1%len(colors)], label='Gaussian Model')
        # set axis labels and limits
        set_axis_labels(extracted_spectrum.wavelength, extracted_spectrum.flux,
                        ax, xlabel, ylabel, use_brackets)
        ax.set_xlim(xlim[0], xlim[1])
        plt.legend()
        if savefig:
            save_figure_2_disk(dpi=dpi)
        plt.show()

    if print_vals:
        # format list for printed table
        computed_vals = [return_array_values(integrated_flux), return_array_values(FWHM), '', '', '']
        computed_errors = [return_array_values(flux_error), return_array_values(FWHM_error), '', '', '']
        # table headers
        print('Best Fit Values:   | Best Fit Errors:   | Computed Values:   | Computed Errors:   \n'+'–'*81)
        params = ['A', 'μ', 'σ', 'm', 'b']
        computed_labels = ['Flux', 'FWHM', '', '', '']
        for i in range(len(popt)):
            # format best fit values
            fit_str = f'{params[i]+":":<2} {popt[i]:>15.6f}'
            # format best fit errors
            fit_err = f'{params[i]+"δ":<2}: {perr[i]:>14.8f}'
            # format computed values if value exists
            if computed_vals[i]:
                comp_str = f'{computed_labels[i]+":":<6} {computed_vals[i]:>10.9f}'
                comp_err = f'{computed_labels[i]+"δ:":<6} {computed_errors[i]:>11.8f}'
            else:
                comp_str = f"{computed_labels[i]:<6} {'':>11}"
                comp_err = f"{computed_labels[i]:<6} {'':>11}"

            print(f'{fit_str} | {fit_err} | {comp_str} | {comp_err}')

    GAUSSIAN_FIELDS = ['amplitude', 'mu', 'sigma', 'flux', 'FWHM',
                       'amplitude_error', 'mu_error', 'sigma_error', 'flux_error', 'FWHM_error']
    GAUSSIAN_LINE_FIELDS = ['amplitude', 'mu', 'sigma', 'm', 'b', 'flux', 'FWHM',
                            'amplitude_error', 'mu_error', 'sigma_error', 'm_error', 'b_error', 'flux_error', 'FWHM_error']
    GAUSSIAN_CONT_FIELDS = GAUSSIAN_FIELDS[:]
    DEFAULT_FIELDS = ['flux', 'FWHM', 'mu', 'flux_error', 'FWHM_error', 'mu_error']
    return_handle = {
        'gaussian': namedtuple('Gaussian', GAUSSIAN_FIELDS),
        'gaussian_line': namedtuple('GaussianLine', GAUSSIAN_LINE_FIELDS),
        'gaussian_continuum': namedtuple('GaussianContinuum', GAUSSIAN_CONT_FIELDS),
        'default': namedtuple('DefaultGaussian', DEFAULT_FIELDS)
    }
    model = model.lower() if return_fit_params else 'default'
    PlotHandles = return_handle[model.lower()]

    if return_fit_params:
        # concatenate computed values and errors
        fitted_params = [amplitude, mu, sigma]
        fitted_errors = [amplitude_error, mu_error, sigma_error]
        # add any extra fitting params
        if len(popt) > 3:
            fitted_params.extend(popt[3:])
            fitted_errors.extend(perr[3:])

        # added computed values and errors
        fitted_params += [integrated_flux, FWHM]
        fitted_errors += [flux_error, FWHM_error]

        return PlotHandles(*(fitted_params + fitted_errors))

    else:
        return PlotHandles(integrated_flux, FWHM, mu, flux_error, FWHM_error, mu_error)
