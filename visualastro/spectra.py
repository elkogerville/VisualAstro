import warnings
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
    spectra_dict_list = []
    cubes = [cubes] if not isinstance(cubes, list) else cubes
    colors, fit_colors = set_plot_colors(colors)
    style = return_stylename(style)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        if emission_line is not None:
            plt.text(0.025, 0.95, f'{emission_line}', transform=plt.gca().transAxes)
        for i, cube in enumerate(cubes):
            spectra_dict = return_spectra_dict()
            default_axis, spectral_axis = return_spectral_coord(cube, unit, radial_vel, rest_freq)

            spectrum = cube.mean(axis=(1,2))

            xmin = xlim[0] if xlim is not None else spectral_axis.value.min()
            xmax = xlim[1] if xlim is not None else spectral_axis.value.max()
            mask = (spectral_axis.value > xmin) & (spectral_axis.value < xmax)

            spectra_dict['wavelength'] = spectral_axis
            spectra_dict['flux'] = spectrum
            label = labels[i] if (labels is not None and i < len(labels)) else None

            if normalize_continuum or plot_continuum_fit:
                spectrum1d = Spectrum1D(flux=spectrum, spectral_axis=default_axis)
                region = convert_region_units(region, default_axis)
                continuum_fit = return_continuum_fit(default_axis, spectrum1d, fit_method, region)
                spec_normalized = spectrum1d / continuum_fit
                spectra_dict['normalized'] = spec_normalized.flux
                spectra_dict['continuum_fit'] = continuum_fit
                if normalize_continuum:
                    plt.plot(spectral_axis[mask], spec_normalized.flux[mask],
                             color=colors[i%len(colors)], label=label)
                if plot_continuum_fit:
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

def return_spectra_dict():
    spectra_dict = {}
    spectra_dict['wavelength'] = None
    spectra_dict['flux'] = None
    spectra_dict['normalized'] = None
    spectra_dict['continuum_fit'] = None

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
            shifted_axis = spectral_axis  # fallback if conversion fails
    else:
        shifted_axis = spectral_axis  # if no unit requested, keep original

    return spectral_axis, shifted_axis
