import warnings
import numpy as np
import astropy.units as u
from astropy.coordinates import SpectralCoord
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum, fit_continuum
import matplotlib.pyplot as plt
from .plot_utils import return_stylename, save_figure_2_disk, set_axis_labels, set_plot_colors

def plot_cube_spectra(cubes, normalize_continuum=False, plot_continuum_fit=False,
                      fit_method='fit_generic_continuum', region=None, radial_vel=None,
                      rest_freq=None, unit=None, emission_line=None, labels=None,
                      xlim=None, ylim=None, x_units=None, y_units=None, colors=None, return_spectra=False,
                      style='astro', use_brackets=False, savefig=False, dpi=600, figsize=(6,6)):
    # ensure cubes are iterable
    cubes = [cubes] if not isinstance(cubes, list) else cubes
    # set plot style and colors
    colors, fit_colors = set_plot_colors(colors)
    style = return_stylename(style)

    with plt.style.context(style):
        plt.figure(figsize=figsize)

        if emission_line is not None:
            plt.text(0.025, 0.95, f'{emission_line}', transform=plt.gca().transAxes)

        spectra_dict_list = []
        for i, cube in enumerate(cubes):

            # extract spectral axis converted to user specified units
            default_axis, spectral_axis = return_spectral_coord(cube, unit, radial_vel, rest_freq)
            # extract spectrum flux
            spectrum = cube.mean(axis=(1,2))

            # set plot limits
            xmin = xlim[0] if xlim is not None else spectral_axis.value.min()
            xmax = xlim[1] if xlim is not None else spectral_axis.value.max()
            mask = (spectral_axis.value > xmin) & (spectral_axis.value < xmax)
            # set plot labels
            label = labels[i] if (labels is not None and i < len(labels)) else None

            spectra_dict = return_spectra_dict(spectral_axis, spectrum)
            #spectra_dict['wavelength'] = spectral_axis
            #spectra_dict['flux'] = spectrum

            # compute continuum
            if normalize_continuum or plot_continuum_fit:

                spectrum1d = Spectrum1D(flux=spectrum, spectral_axis=default_axis)
                # convert region to default units
                region = convert_region_units(region, default_axis)
                # compute continuum fit
                continuum_fit = return_continuum_fit(default_axis, spectrum1d, fit_method, region)
                # compute normalized flux
                spec_normalized = spectrum1d / continuum_fit

                spectra_dict['normalized'] = spec_normalized.flux
                spectra_dict['continuum_fit'] = continuum_fit

                if normalize_continuum:
                    plt.plot(spectral_axis[mask], spec_normalized.flux[mask],
                             color=colors[i%len(colors)], label=label)
                if plot_continuum_fit:
                    # normalize fit if spectrum is normalized
                    continuum_fit = continuum_fit/continuum_fit if normalize_continuum else continuum_fit
                    plt.plot(spectral_axis[mask], continuum_fit[mask], color=fit_colors[i%len(fit_colors)])

            if not normalize_continuum:
                plt.plot(spectral_axis[mask], spectrum[mask], color=colors[i%len(colors)], label=label)

            spectra_dict_list.append(spectra_dict)

        plt.xlim(xmin, xmax)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        set_axis_labels(spectral_axis, spectrum, x_units, y_units, use_brackets=use_brackets)

        if labels is not None:
            plt.legend()
        plt.tight_layout()
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()

        if return_spectra:
            if len(spectra_dict_list) == 1:
                spectra_dict_list = spectra_dict_list[0]
            return spectra_dict_list

def plot_combine_spectrum(spectra_dict_list, idx=0, label=None, ylim=None, spec_lims=None,
                          concatenate=False, return_spectra=False, style='latex', colors=None,
                          use_samecolor=True, use_brackets=False, figsize=(12,6), loc='best',
                          savefig=False, dpi=600):

    concatenate = True if return_spectra else concatenate
    # set plot style and colors
    colors, _ = set_plot_colors(colors)
    style = return_stylename(style)

    with plt.style.context(style):
        plt.figure(figsize=figsize)
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
                plt.plot(wavelength, flux, color=c, label=l, lw=0.5)

        if concatenate:
            wavelength = np.concatenate(wave_list)
            flux = np.concatenate(flux_list)
            plt.plot(wavelength.value, flux.value, color=c, label=l, lw=0.5)

        set_axis_labels(wavelength, flux, None, None, use_brackets=use_brackets)

        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        if spec_lims is None:
            xmin = min(l[0] for l in lims)
            xmax = max(l[1] for l in lims)
        else:
            xmin = spec_lims[0]
            xmax = spec_lims[-1]
        plt.xlim(xmin, xmax)

        if label is not None:
            plt.legend(loc=loc)
        if savefig:
            save_figure_2_disk(dpi)
        plt.show()

        if return_spectra:
            spectra_dict = return_spectra_dict(wavelength, flux.value)
            #spectra_dict['wavelength'] = wavelength
            #spectra_dict['flux'] = flux.value

            return spectra_dict

def return_spectra_dict(wavelength=None, flux=None, normalized=None, continuum_fit=None):
    spectra_dict = {}
    spectra_dict['wavelength'] = wavelength
    spectra_dict['flux'] = flux
    spectra_dict['normalized'] = normalized
    spectra_dict['continuum_fit'] = continuum_fit

    return spectra_dict

def shift_by_radial_vel(spectral_axis, radial_vel):
    c = 299792.458 # [m/s]
    if radial_vel is not None:
        if spectral_axis.unit.is_equivalent(u.Hz):
            spectral_axis *= (1 + radial_vel / c)
        else:
            spectral_axis /= (1 + radial_vel / c)
        return spectral_axis
    return spectral_axis

def return_continuum_fit(spectral_axis, spectrum1d, fit_method, region):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if fit_method=='fit_continuum':
            fit = fit_continuum(spectrum1d, window=region)
        else:
            fit = fit_generic_continuum(spectrum1d)
    continuum_fit = fit(spectral_axis)

    return continuum_fit

def convert_region_units(region, default_axis):
    if region is not None:
        unit = default_axis.unit
        region_converted = []
        for rmin, rmax in region:
            rmin = rmin.to(unit)
            rmax = rmax.to(unit)
            region_converted.append((rmin, rmax))

        return region_converted

    return region

def return_spectral_coord(cube, unit, radial_vel, rest_freq):
    spectral_axis = SpectralCoord(cube.spectral_axis, doppler_rest=rest_freq, doppler_convention='radio')
    spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)

    if unit is not None:
        try:
            shifted_axis = spectral_axis.to(unit)
        except Exception:
            shifted_axis = spectral_axis
    else:
        shifted_axis = spectral_axis

    return spectral_axis, shifted_axis
