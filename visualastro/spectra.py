from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import numpy as np
from specutils.spectra import Spectrum1D
from scipy.optimize import curve_fit
from .io import save_figure_2_disk
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
)
from .visual_classes import ExtractedSpectrum


def extract_cube_spectra(cubes, normalize_continuum=False, plot_continuum_fit=False,
                         fit_method='fit_continuum', region=None, radial_vel=None,
                         rest_freq=None, deredden=False, unit=None, emission_line=None, **kwargs):
    '''
    Extract 1D spectra from one or more data cubes, with optional continuum normalization,
    dereddening, and plotting.
    Parameters
    ––––––––––
    cubes : DataCube, SpectralCube, or list of cubes
        Input cube(s) from which to extract spectra. The data must either be
        a SpectralCube, or a DataCube containing a SpectralCube.
    normalize_continuum : bool, optional, default=False
        Whether to normalize extracted spectra by a computed continuum fit
    plot_continuum_fit : bool, optional, default=False
        Whether to overplot the continuum fit.
    fit_method : str, {'fit_continuum', 'generic'}, optional, default='fit_continuum'
        Method used to fit the continuum.
    region : array-like, optional, default=None
        Wavelength or pixel region(s) to use when `fit_method='fit_continuum'`.
        Ignored for other methods. This allows the user to specify which
        regions to include in the fit. Removing strong peaks is preferable to
        avoid skewing the fit up or down.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
    radial_vel : float, optional, default=None
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional.
    rest_freq : float, optional, default=None
        Rest-frame frequency or wavelength of the spectrum.
    deredden : bool, optional, default=False
        Whether to apply dereddening to the flux using deredden_flux().
    unit : str or astropy.units.Unit, optional, default=None
        Desired units for the wavelength axis. Converts the default
        units if possible.
    emission_line : str, optional, default=None
        Name of an emission line to annotate on the plot.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `convention` : str, optional
            Doppler convention.
        - `Rv` : float, optional, default=3.1
            Dereddening parameter.
        - `Ebv` : float, optional, default=0.19
            Dereddening parameter.
        - `deredden_method` : str, optional, default='WD01'
            Extinction law to use.
        - `deredden_region` : str, optional, default='LMCAvg'
            Region/environment for WD01 extinction law.
        - `figsize` : tuple, optional, default=(6, 6)
            Figure size for plotting.
        - `style` : str, optional, default='astro'
            Plotting style.
        - `text_loc` : list, optional, default=[0.025, 0.95]
            Location of text annotations in axes coordinates.
        - `savefig` : bool, optional, default=False
            Whether to save the figure to disk.
        - `dpi` : int, optional, default=600
            Figure resolution for saving.
    Returns
    –––––––
    ExtractedSpectrum or list of ExtractedSpectrum
        Single object if one cube is provided, list if multiple cubes are provided.
    '''
    # –––– KWARGS ––––
    # doppler convention
    convention = kwargs.get('convention', None)
    # dereddening parameters
    Rv = kwargs.get('Rv', 3.1)
    Ebv = kwargs.get('Ebv', 0.19)
    deredden_method = kwargs.get('deredden_method', 'WD01')
    deredden_region = kwargs.get('deredden_region', 'LMCAvg')
    # figure params
    figsize = kwargs.get('figsize', (6,6))
    style = kwargs.get('style', 'astro')
    # labels
    text_loc = kwargs.get('text_loc', [0.025, 0.95])
    # savefig
    savefig = kwargs.get('savefig', False)
    dpi = kwargs.get('dpi', 600)

    # ensure cubes are iterable
    cubes = check_units_consistency(cubes)
    # set plot style and colors
    style = return_stylename(style)

    extracted_spectra = []
    for cube in cubes:

        # extract spectral axis converted to user specified units
        spectral_axis = shift_by_radial_vel(cube.spectral_axis, radial_vel)

        # extract spectrum flux
        flux = cube.mean(axis=(1,2))

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

        if emission_line is not None:
            ax.text(text_loc[0], text_loc[1], f'{emission_line}',
                    transform=plt.gca().transAxes)

        plot_spectrum(extracted_spectra, ax, normalize_continuum,
                        plot_continuum_fit, emission_line, **kwargs)
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()

    # ensure a list is only returned if returning more than 1 spectrum
    if len(extracted_spectra) == 1:
        return extracted_spectra[0]

    return extracted_spectra

def plot_spectrum(extracted_spectra=None, ax=None, plot_norm_continuum=False,
                  plot_continuum_fit=False, emission_line=None, wavelength=None,
                  flux=None, continuum_fit=None, **kwargs):
    '''
    Plot one or more extracted spectra on a matplotlib Axes.
    Parameters
    ----------
    extracted_spectrums : ExtractedSpectrum or list of ExtractedSpectrum, optional
        Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
        and `flux` must be given.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    normalize_continuum : bool, optional, default=False
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

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `xlim` : tuple, optional
            Wavelength range to display.
        - `ylim` : tuple, optional
            Flux range to display.
        - `labels` : list of str, optional
            Labels for each spectrum to use in the legend.
        - `xlabel` : str, optional
            Label for the x-axis.
        - `ylabel` : str, optional
            Label for the y-axis.
        - `colors` : list or str, optional
            Colors for each spectrum. If None, default palette is used.
        - `cmap` : str, optional, default='turbo'
            Colormap to use if `colors` is not provided.
        - `text_loc` : list of float, optional, default=[0.025, 0.95]
            Location for emission line annotation text in axes coordinates.
        - `use_brackets` : bool, optional, default=False
            If True, plot units in square brackets; otherwise, parentheses.

    Returns
    -------
    None
        The function plots directly on the provided matplotlib Axes.
    '''
    # –––– Kwargs ––––
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = kwargs.get('labels', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = kwargs.get('colors', None)
    cmap = kwargs.get('cmap', 'turbo')
    text_loc = kwargs.get('text_loc', [0.025, 0.95])
    use_brackets = kwargs.get('use_brackets', False)
    # ensure an axis is passed
    if ax is None:
        raise ValueError('ax must be a matplotlib axes object!')

    # construct ExtractedSpectrum if user passes in wavelenght and flux
    if extracted_spectra is None:
        if None not in (wavelength, flux):
            extracted_spectra = ExtractedSpectrum(
                wavelength=wavelength,
                flux=flux,
                continuum_fit=continuum_fit
            )
            # to avoid missing normalize attribute
            plot_norm_continuum = False
        else:
            raise ValueError(
                "Either `extracted_spectrums` must be provided, "
                "or both `wavelength` and `flux` must be given."
            )

    # ensure extracted_spectra is iterable
    extracted_spectra = check_units_consistency(extracted_spectra)

    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors, cmap=cmap)
    # add emission line text
    if emission_line is not None:
        ax.text(text_loc[0], text_loc[1], f'{emission_line}', transform=ax.transAxes)

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
        wavelength_list.append(wavelength[mask])

        # define spectrum label, color, and fit color
        label = labels[i] if (labels is not None and i < len(labels)) else None
        color = colors[i%len(colors)]
        fit_color = fit_colors[i%len(fit_colors)]

        # plot spectrum
        ax.plot(wavelength[mask], flux[mask], c=color, label=label)
        # plot continuum fit
        if plot_continuum_fit and extracted_spectrum.continuum_fit is not None:
            if plot_norm_continuum:
                # normalize continuum fit
                continuum_fit = extracted_spectrum.continuum_fit/extracted_spectrum.continuum_fit
            else:
                continuum_fit = extracted_spectrum.continuum_fit
            ax.plot(wavelength[mask], continuum_fit[mask], c=fit_color)

    # set plot axis limits and labels
    set_axis_limits(wavelength_list, None, ax, xlim, ylim)
    set_axis_labels(wavelength, extracted_spectrum.flux, ax,
                    xlabel, ylabel, use_brackets=use_brackets)
    if labels is not None:
        ax.legend()

def plot_combine_spectrum(spectra_dict_list, ax, idx=0, spec_lims=None,
                          concatenate=False, return_spectra=False,
                          use_samecolor=True, **kwargs):

    # figure params
    ylim = kwargs.get('ylim', None)
    # labels
    label = kwargs.get('label', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = kwargs.get('colors', None)
    cmap = kwargs.get('cmap', 'turbo')
    loc = kwargs.get('loc', 'best')
    use_brackets = kwargs.get('use_brackets', False)

    concatenate = True if return_spectra else concatenate
    # set plot style and colors
    colors, _ = set_plot_colors(colors, cmap=cmap)

    lims = []
    wave_list = []
    flux_list = []
    for i, spectra in enumerate(spectra_dict_list):
        spectra = spectra[idx] if isinstance(spectra, list) else spectra
        wavelength = spectra.wavelength
        flux = spectra.flux
        lims.append( [wavelength.value.min(), wavelength.value.max()] )
        if spec_lims is not None:
            spec_min = spec_lims[i]
            spec_max = spec_lims[i+1]
            mask = (wavelength.value > spec_min) & (wavelength.value < spec_max)
            wavelength = wavelength[mask]
            flux = flux[mask]

        c = colors[0] if use_samecolor else colors[i%len(colors)]
        l = label if label is not None and i == len(spectra_dict_list)-1 else None
        if concatenate:
            wave_list.append(wavelength)
            flux_list.append(flux)
        else:
            ax.plot(wavelength, flux, color=c, label=l, lw=0.5)

    if concatenate:
        wavelength = np.concatenate(wave_list)
        flux = np.concatenate(flux_list)
        ax.plot(wavelength.value, flux.value, color=c, label=l, lw=0.5)

    set_axis_labels(wavelength, flux, ax, xlabel, ylabel, use_brackets)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if spec_lims is None:
        xmin = min(l[0] for l in lims)
        xmax = max(l[1] for l in lims)
    else:
        xmin = spec_lims[0]
        xmax = spec_lims[-1]
    ax.set_xlim(xmin, xmax)

    if label is not None:
        plt.legend(loc=loc)

    if return_spectra:
        spectra_dict = return_spectra_dict(wavelength, flux)

        return spectra_dict

def return_spectra_dict(wavelength=None, flux=None, spectrum1d=None,
                        normalized=None, continuum_fit=None):
    spectra_dict = {}
    spectra_dict['wavelength'] = wavelength
    spectra_dict['flux'] = flux
    spectra_dict['spectrum1d'] = spectrum1d
    spectra_dict['normalized'] = normalized
    spectra_dict['continuum_fit'] = continuum_fit

    return spectra_dict

def fit_gaussian_2_spec(extracted_spectrum, p0, model='gaussian', wave_range=None,
                        interpolate=True, interp_method='cubic_spline', yerror=None,
                        error_method='cubic_spline', samples=1000, return_fit_params=False,
                        plot_interp=False, print_vals=True, **kwargs):
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
    model : str, default='gaussian'
        Type of Gaussian model to fit:
        - 'gaussian' : standard Gaussian
        - 'gaussian_line' : Gaussian with linear continuum
        - 'gaussian_continuum' : Gaussian with computed continuum array
        The continuum can be computed with compute_continuum_fit().
    wave_range : tuple or list, optional, default=None
        (min, max) wavelength range to restrict the fit.
        If None, computes the min and max from the wavelength.
    interpolate : bool, default=True
        Whether to interpolate the spectrum over
        a regular wavelength grid. The number of
        samples is controlled by `samples`.
    interp_method : str, {'cubic', 'cubic_spline', 'linear'} default='cubic_spline'
        Interpolation method used.
    yerror : array-like, optional, default=None
        Flux uncertainties.
    error_method : str, {'cubic', 'cubic_spline', 'linear'}, default='cubic_spline'
        Method to interpolate yerror if provided.
    samples : int, default=1000
        Number of points in interpolated wavelength grid.
    return_fit_params : bool, default=False
        If True, return full computed best-fit parameters
        including derived flux and FWHM. If False, return
        only Flux, FWHM, and mu.
    plot_interp : bool, default=False
        If True, plot the interpolated spectrum. This is
        provided for debugging purposes.
    print_vals : bool, default=True
        If True, print a table of best-fit parameters,
        errors, and computed quantities.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `figsize` : list or tuple, optional, default=(6, 6)
            Figure size.
        - `style` : str or {'astro', 'latex', 'minimal'}, optional, default='astro'
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
        - `colors` : str or list, optional, default=None
            Plot colors. If None, will use default visualastro color palette.
        - `use_brackets` : bool, optional, default=False
            If True, use square brackets for plot units. If False, use parentheses.
        - `savefig` : bool, optional, default=False
            If True, save current figure to disk.
        - `dpi` : float or int, optional, default=600
            Resolution in dots per inch.
    Returns
    –––––––
    If return_fit_params:
        popt : np.ndarray
            Best-fit parameters including integrated flux and FWHM.
        perr : np.ndarray
            Uncertainties of fit parameters including flux and FWHM errors.
    Else:
        list
            [integrated_flux, FWHM, mu]
        list
            [flux_error, FWHM_error, mu_error]
    '''
    # –––– Kwargs ––––
    # figure params
    figsize = kwargs.get('figsize', (6,6))
    style = kwargs.get('style', 'astro')
    xlim = kwargs.get('xlim', None)
    plot_type = kwargs.get('plot_type', 'plot')
    # labels
    label = kwargs.get('label', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = kwargs.get('colors', None)
    use_brackets = kwargs.get('use_brackets', False)
    # savefig
    savefig = kwargs.get('savefig', False)
    dpi = kwargs.get('dpi', 600)

    # ensure arrays are not quantity objects
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
    amplitude = popt[0]
    amplitude_error = perr[0]
    sigma = popt[2]
    sigma_error = perr[2]
    # compute integrated flux, FWHM, and their errors
    integrated_flux = amplitude * sigma * np.sqrt(2*np.pi)
    flux_error = np.sqrt(2*np.pi) * integrated_flux * (
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
        computed_vals = [integrated_flux, FWHM, '', '', '']
        computed_errors = [flux_error, FWHM_error, '', '', '']
        # table headers
        print('Best Fit Values:   | Best Fit Errors:   | Computed Values:   | Computed Errors:   \n'+'–'*81)
        params = ['A', 'μ', 'σ', 'm', 'b']
        computed_labels = ['Flux', 'FWHM', '', '', '']
        for i in range(len(popt)):
            # format best fit values
            fit_str = f'{params[i]+':':<2} {popt[i]:>15.6f}'
            # format best fit errors
            fit_err = f'{params[i]+'δ':<2}: {perr[i]:>14.8f}'
            # format computed values if value exists
            if computed_vals[i]:
                comp_str = f'{computed_labels[i]+':':<6} {computed_vals[i]:>10.9f}'
                comp_err = f'{computed_labels[i]+'δ:':<6} {computed_errors[i]:>11.8f}'
            else:
                comp_str = f'{computed_labels[i]:<6} {'':>11}'
                comp_err = f'{computed_labels[i]:<6} {'':>11}'

            print(f'{fit_str} | {fit_err} | {comp_str} | {comp_err}')
    # concatenate computed values and errors
    popt = np.concatenate([popt, [integrated_flux, FWHM]])
    perr = np.concatenate([perr, [flux_error, FWHM_error]])

    if return_fit_params:
        return popt, perr
    else:
        return [integrated_flux, FWHM, popt[1]], [flux_error, FWHM_error, perr[1]]

def gaussian_levmarLSQ(spectra_dict, p0, wave_range, N_samples=1000, subtract_continuum=False,
                       interp_method='cubic_spline', colors=None, style='astro', figsize=(6,6), xlim=None):

    wavelength = spectra_dict.wavelength
    flux = spectra_dict.flux.copy()
    wave_unit = spectra_dict.wavelength.unit
    flux_unit = spectra_dict.flux.unit

    if subtract_continuum:
        continuum = spectra_dict.continuum_fit
        flux -= continuum

    wave_range = [flux.min(), flux.max()] if wave_range is None else wave_range

    wave_sub, flux_sub = interpolate_arrays(wavelength, flux, wave_range, N_samples, method=interp_method)
    mask = ( (wave_sub*wave_unit > wave_range[0]*wave_unit) &
            (wave_sub*wave_unit < wave_range[1]*wave_unit) )

    wave_sub = wave_sub[mask]
    flux_sub = flux_sub[mask]

    amplitude, mean, stddev = p0

    g_init = models.Gaussian1D(amplitude=amplitude*flux_unit,
                               mean=mean*wave_unit,
                               stddev=stddev*wave_unit)
    fitter = fitting.LevMarLSQFitter()
    g_fit = fitter(g_init, wave_sub*wave_unit, flux_sub*flux_unit)
    y_fit = g_fit(wave_sub*wave_unit)

    # set plot style and colors
    colors, _ = set_plot_colors(colors)
    style = return_stylename(style)

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(wavelength, flux, c=colors[0])
        ax.plot(wave_sub, y_fit, c=colors[1])
        xlim = wave_range if xlim is None else xlim
        ax.set_xlim(xlim[0], xlim[1])
        set_axis_labels(spectra_dict.wavelength, spectra_dict.flux, ax, None, None)
        plt.show()

    integrated_flux = g_fit.amplitude.value * g_fit.stddev.value * np.sqrt(2*np.pi)
    print(f'μ: {g_fit.mean.value:.5f} {g_fit.mean.unit}, FWHM: {g_fit.fwhm:.5f}, Flux: {integrated_flux:.5}')

    return g_fit
