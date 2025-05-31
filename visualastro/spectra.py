import warnings
import astropy.units as u
from spectral_cube import SpectralCube
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum, fit_continuum
import matplotlib.pyplot as plt
from .plot_utils import return_stylename, save_figure_2_disk, set_axis_labels, set_plot_colors, set_spectral_axis

def plot_cube_spectra(cubes, normalize_continuum=False, plot_continuum_fit=False,
                      fit_method='fit_generic_continuum', region=None, radial_vel=None,
                      rest_freq=None, unit=None, emission_line=None, labels=None,
                      xlim=None, ylim=None, x_units=None, y_units=None, colors=None, return_spectra=False,
                      style='astro', use_brackets=False, savefig=False, dpi=600, figsize=(6,6)):
    c = 299792.458 # m/s
    spectra_dict_list = []
    colors, fit_colors = set_plot_colors(colors)
    cubes = [cubes] if not isinstance(cubes, list) else cubes
    style = return_stylename(style)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        if emission_line is not None:
            plt.text(0.025, 0.95, f'{emission_line}', transform=plt.gca().transAxes)
        for i, cube in enumerate(cubes):
            spectra_dict = {}
            spec_normalized, continuum_fit = None, None
            spectral_axis = set_spectral_axis(cube, unit)
            spectrum = cube.mean(axis=(1,2))

            if radial_vel is not None:
                if spectral_axis.unit.is_equivalent(u.Hz):
                    spectral_axis *= (1 + radial_vel / c)
                else:
                    spectral_axis /= (1 + radial_vel / c)
            if rest_freq is not None:
                spectral_axis = c*u.km/u.s * (rest_freq - spectral_axis) / rest_freq

            xmin = xlim[0] if xlim is not None else spectral_axis.value.min()
            xmax = xlim[1] if xlim is not None else spectral_axis.value.max()
            mask = (spectral_axis.value > xmin) & (spectral_axis.value < xmax)
            spectra_dict['wavelength'] = spectral_axis
            spectra_dict['flux'] = spectrum
            label = labels[i] if (labels is not None and i < len(labels)) else None
            if normalize_continuum != plot_continuum_fit:
                spectrum1d = Spectrum1D(flux=spectrum, spectral_axis=spectral_axis)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if fit_method=='fit_continuum':
                        fit = fit_continuum(spectrum1d, window=region)
                    else:
                        fit = fit_generic_continuum(spectrum1d)
                continuum_fit = fit(spectral_axis)
                spec_normalized = spectrum1d / continuum_fit
                if normalize_continuum:
                    plt.plot(spec_normalized.spectral_axis[mask], spec_normalized.flux[mask],
                             color=colors[i%len(colors)], label=label)
            if not normalize_continuum:
                plt.plot(spectral_axis[mask], spectrum[mask], color=colors[i%len(colors)], label=label)
            if plot_continuum_fit:
                plt.plot(spectral_axis[mask], continuum_fit[mask], color=fit_colors[i%len(fit_colors)])
            spectra_dict['spec_norm'] = spec_normalized
            spectra_dict['continuum_fit'] = continuum_fit
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
