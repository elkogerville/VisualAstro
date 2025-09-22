#from astropy.coordinates import SpectralCoord
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import numpy as np
from specutils.spectra import Spectrum1D
from scipy.optimize import curve_fit, least_squares
from .numerical_utils import (
    interpolate_arrays, mask_within_range,
    return_array_values, shift_by_radial_vel
)
from .plot_utils import (
    return_stylename, save_figure_2_disk,
    set_axis_labels, set_axis_limits, set_plot_colors
)
from .spectra_utils import (
    convert_region_units, deredden_spectrum, gaussian,
    gaussian_continuum, gaussian_line, residuals,
    return_continuum_fit
)
from .visual_classes import ExtractedSpectrum

def extract_cube_spectra(cubes, normalize_continuum=False, plot_continuum_fit=False,
                         fit_method='fit_generic_continuum', region=None, radial_vel=None,
                         rest_freq=None, deredden=False, unit=None, emission_line=None, **kwargs):
    # –––– KWARGS ––––
    # doppler convention
    convention = kwargs.get('convention', None)
    # figure params
    figsize = kwargs.get('figsize', (6,6))
    style = kwargs.get('style', 'astro')
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
    # savefig
    savefig = kwargs.get('savefig', False)
    dpi = kwargs.get('dpi', 600)

    # ensure cubes are iterable
    cubes = [cubes] if not isinstance(cubes, list) else cubes
    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors, cmap=cmap)
    style = return_stylename(style)

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        if emission_line is not None:
            plt.text(text_loc[0], text_loc[1], f'{emission_line}',
                     transform=plt.gca().transAxes)

        wavelength_list = []
        extracted_spectrum_list = []
        for i, cube in enumerate(cubes):

            # extract spectral axis converted to user specified units
            spectral_axis = shift_by_radial_vel(cube.spectral_axis, radial_vel)

            # extract spectrum flux
            flux = cube.mean(axis=(1,2))

            # derreden
            if deredden:
                flux = deredden_spectrum(spectral_axis, flux, **kwargs)

            # initialize Spectrum1D object
            spectrum1d = Spectrum1D(
                spectral_axis=spectral_axis,
                flux=flux,
                rest_value=rest_freq,
                velocity_convention=convention
            )

            # convert region to default units
            region = convert_region_units(region, spectral_axis)

            # compute continuum fit
            continuum_fit = return_continuum_fit(spectrum1d, fit_method, region)

            # compute normalized flux
            spec_normalized = spectrum1d / continuum_fit

            # variable for plotting wavelength
            wavelength = spectrum1d.spectral_axis
            if unit is not None:
                try:
                    wavelength = wavelength.to(unit)
                except Exception:
                    print(
                        f'Could not convert to unit: {unit}. \n'
                        f'Defaulting to unit: {spectral_axis.unit}.'
                    )
            # mask data within its range
            mask = mask_within_range(wavelength, xlim=xlim)
            wavelength_list.append(wavelength[mask])

            # set spectrum label and color
            label = labels[i] if (labels is not None and i < len(labels)) else None
            color = colors[i%len(colors)]

            # plot normalized spectrum
            if normalize_continuum:
                ax.plot(wavelength[mask], spec_normalized.flux[mask],
                        color=color, label=label)
            # plot default spectrum
            else:
                ax.plot(wavelength[mask], flux[mask], color=color, label=label)
            # plot continuum fit
            if plot_continuum_fit:
                # normalize fit if spectrum is normalized
                continuum_plot = continuum_fit/continuum_fit if normalize_continuum else continuum_fit
                ax.plot(wavelength[mask], continuum_plot[mask], color=fit_colors[i%len(fit_colors)])
            # save computed spectrum
            extracted_spectrum_list.append(ExtractedSpectrum(
                wavelength=wavelength,
                flux=flux,
                spectrum1d=spectrum1d,
                normalized=spec_normalized,
                continuum_fit=continuum_fit
            ))
        # set plot axis limits and labels
        set_axis_limits(wavelength_list, None, ax, xlim, ylim)
        set_axis_labels(wavelength, flux, ax,
                        xlabel, ylabel, use_brackets)

        if labels is not None:
            plt.legend()
        plt.tight_layout()
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()
        # return computed ExtractedSpectrums
        if len(extracted_spectrum_list) == 1:
            extracted_spectrum_list = extracted_spectrum_list[0]
        return extracted_spectrum_list

def plot_spectrum(spectra_dicts, ax, normalize=False, plot_continuum=False,
                  emission_line=None, **kwargs):

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

    spectra_dicts = spectra_dicts if isinstance(spectra_dicts, list) else [spectra_dicts]

    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors, cmap=cmap)

    if emission_line is not None:
        ax.text(text_loc[0], text_loc[1], f'{emission_line}', transform=ax.transAxes)

    wavelength_list = []
    for i, spectra_dict in enumerate(spectra_dicts):
        if spectra_dict is not None:
            wavelength = spectra_dict['wavelength']
            flux = spectra_dict['normalized'] if normalize else spectra_dict['flux']

            mask = mask_within_range(wavelength, xlim=xlim)

            label = labels[i] if (labels is not None and i < len(labels)) else None

            ax.plot(wavelength[mask], flux[mask], c=colors[i%len(colors)], label=label)
            if plot_continuum:
                ax.plot(wavelength[mask], spectra_dict['continuum_fit'][mask], c=fit_colors[i%len(fit_colors)])

            wavelength_list.append(wavelength[mask])

    set_axis_limits(wavelength_list, None, ax, xlim, ylim)
    set_axis_labels(wavelength, spectra_dict['flux'], ax, xlabel, ylabel, use_brackets=use_brackets)
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
        wavelength = spectra['wavelength']
        flux = spectra['flux']
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





# def return_spectral_coord(cube, unit, radial_vel, rest_freq, convention='optical'):
#     '''
#     Return cube spectral axis shifted by radial velocity and converted to user specified units
#     '''
#     if convention is None:
#         axis_type = getattr(getattr(cube.spectral_axis, 'unit', None), 'physical_type', None)
#         convention = {
#             'frequency': 'radio',
#             'length': 'optical',
#             'speed': 'relativistic'
#         }.get(axis_type or '', 'optical')
#         print(convention)

#     spectral_axis = SpectralCoord(cube.spectral_axis, doppler_rest=rest_freq,
#                                   doppler_convention=convention)
#     spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)

#     if unit is not None:
#         try:
#             spectral_axis = spectral_axis.to(unit)
#         except Exception:
#             print(
#                 f'Could not convert to unit: {unit}. \n'
#                 f'Defaulting to unit: {spectral_axis.unit}.'
#             )

#     return spectral_axis

def fit_gaussian_2_spec(spectrum, p0, model='gaussian', wave_range=None, interpolate=True,
                        interp_method='cubic_spline', yerror=None, error_method='cubic_spline',
                        samples=1000, return_fit_params=False, plot_interp=False, print_vals=True,
                        **kwargs):

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

    wavelength = return_array_values(spectrum['wavelength'])
    flux = return_array_values(spectrum['flux'])

    wave_range = [wavelength.min(), wavelength.max()] if wave_range is None else wave_range

    function_map = {
        'gaussian': gaussian,
        'gaussian_line': gaussian_line,
        'gaussian_continuum': gaussian_continuum,
        'residuals': residuals,
    }

    if interpolate:
        wavelength, flux = interpolate_arrays(wavelength, flux, wave_range,
                                              samples, method=interp_method)
        if yerror is not None:
            _, yerror = interpolate_arrays(spectrum['wavelength'].value, yerror,
                                           wave_range, samples, method=error_method)
        if model == 'gaussian_continuum':
            _, continuum = interpolate_arrays(spectrum['wavelength'].value, p0[-1],
                                              wave_range, samples, method=interp_method)
            p0.pop(-1)

    wave_mask = (wavelength > wave_range[0]) & (wavelength < wave_range[1])
    wave_sub = wavelength[wave_mask]
    flux_sub = flux[wave_mask]
    if yerror is not None:
        yerror = yerror[wave_mask]
    if model == 'gaussian_continuum':
        continuum = continuum[wave_mask]

    function = function_map.get(model, gaussian)
    if model == 'residuals':
        res = least_squares(function, p0, args=(wave_sub, flux_sub), loss='soft_l1')
        popt = res.x
        function = function_map.get('gaussian_line', gaussian_line)
    if model == 'gaussian_continuum':
        fitted_model = lambda x, A, mu, sigma: gaussian_continuum(x, A, mu, sigma, continuum)
        popt, pcov = curve_fit(fitted_model, wave_sub, flux_sub, p0,
                               sigma=yerror, absolute_sigma=True, method='trf')
        function = fitted_model
    else:
        popt, pcov = curve_fit(function, wave_sub, flux_sub, p0, sigma=yerror,
                               absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))

    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors)
    style = return_stylename(style)

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        plt_plot = {
            'plot': ax.plot,
            'scatter': ax.scatter
        }.get(plot_type, ax.plot)

        if plot_interp:
            plt_plot(wavelength, flux, c=colors[2%len(colors)])

        wavelength = return_array_values(spectrum['wavelength'])
        flux = return_array_values(spectrum['flux'])
        xlim = wave_range if xlim is None else xlim
        mask = (wavelength > xlim[0]) & (wavelength < xlim[1])
        label = label if label is not None else 'Spectrum'
        plt_plot(wavelength[mask], flux[mask],
                 c=colors[0%len(colors)], label=label)

        ax.plot(wave_sub, function(wave_sub, *popt),
                c=colors[1%len(colors)], label='Gaussian Model')

        set_axis_labels(spectrum['wavelength'], spectrum['flux'],
                        ax, xlabel, ylabel, use_brackets)
        ax.set_xlim(xlim[0], xlim[1])
        plt.legend()
        if savefig:
            save_figure_2_disk(dpi=dpi)
        plt.show()

    amplitude = popt[0]
    amplitude_error = perr[0]
    sigma = popt[2]
    sigma_error = perr[2]

    integrated_flux = amplitude * sigma * np.sqrt(2*np.pi)
    flux_error = np.sqrt(2*np.pi) * integrated_flux * (
        np.sqrt((amplitude_error/amplitude)**2 + (sigma_error/sigma)**2) )
    FWHM = 2*sigma * np.sqrt(2*np.log(2))
    FWHM_error = 2*sigma_error * np.sqrt(2*np.log(2))
    computed_vals = [integrated_flux, FWHM, '', '', '']
    computed_errors = [flux_error, FWHM_error, '', '', '']

    if print_vals:
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

    popt = np.concatenate([popt, [integrated_flux, FWHM]])
    perr = np.concatenate([perr, [flux_error, FWHM_error]])
    if return_fit_params:
        return popt, perr
    else:
        return [integrated_flux, FWHM, popt[1]], [flux_error, FWHM_error, perr[1]]

def gaussian_levmarLSQ(spectra_dict, p0, wave_range, N_samples=1000, subtract_continuum=False,
                       interp_method='cubic_spline', colors=None, style='astro', figsize=(6,6), xlim=None):

    wavelength = spectra_dict['wavelength']
    flux = spectra_dict['flux'].copy()
    wave_unit = spectra_dict['wavelength'].unit
    flux_unit = spectra_dict['flux'].unit

    if subtract_continuum:
        continuum = spectra_dict['continuum_fit']
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
        set_axis_labels(spectra_dict['wavelength'], spectra_dict['flux'], ax, None, None)
        plt.show()

    integrated_flux = g_fit.amplitude.value * g_fit.stddev.value * np.sqrt(2*np.pi)
    print(f'μ: {g_fit.mean.value:.5f} {g_fit.mean.unit}, FWHM: {g_fit.fwhm:.5f}, Flux: {integrated_flux:.5}')

    return g_fit
