import warnings
import astropy.units as u
from spectral_cube import SpectralCube
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum, fit_continuum
import matplotlib.pyplot as plt
from .plot_utils import return_stylename, save_figure_2_disk, set_axis_labels, set_plot_colors

def plot_cube_spectra(cubes, normalize_continuum=False, plot_continuum_fit=False,
                  fit_method='fit_generic_continuum', region=None, radial_vel=None,
                  emission_line=None, labels=None, x_limits=None, y_limits=None,
                  x_units=None, y_units=None, colors=None, return_spectra=False,
                  style='astro', savefig=False, dpi=600, figsize=(6,6)):
    c = 299792.458
    spec_normalized, continuum_fit = [], []
    colors, fit_colors = set_plot_colors(colors)
    cubes = [cubes] if isinstance(cubes, SpectralCube) else cubes
    style = return_stylename(style)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        if emission_line is not None:
            plt.text(0.025, 0.95, f'{emission_line}', transform=plt.gca().transAxes)
        for i, cube in enumerate(cubes):
            wavelengths = cube.spectral_axis.to(u.micron)
            if radial_vel is not None:
                wavelengths /= (1 + radial_vel/c)
            xmin = x_limits[0] if x_limits else wavelengths.value.min()
            xmax = x_limits[1] if x_limits else wavelengths.value.max()
            mask = (wavelengths.value > xmin) & (wavelengths.value < xmax)
            spectrum = cube.mean(axis=(1,2))
            label = labels[i] if (labels is not None and i < len(labels)) else None
            if normalize_continuum != plot_continuum_fit:
                spectrum1d = Spectrum1D(flux=spectrum, spectral_axis=wavelengths)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if fit_method=='fit_continuum':
                        fit = fit_continuum(spectrum1d, window=region)
                    else:
                        fit = fit_generic_continuum(spectrum1d)
                continuum_fit = fit(wavelengths)
                spec_normalized = spectrum1d / continuum_fit
                if normalize_continuum:
                    plt.plot(spec_normalized.spectral_axis[mask], spec_normalized.flux[mask],
                             color=colors[i%len(colors)], label=label)
            if not normalize_continuum:
                plt.plot(wavelengths[mask], spectrum[mask], color=colors[i%len(colors)], label=label)
            if plot_continuum_fit:
                plt.plot(wavelengths[mask], continuum_fit[mask], color=fit_colors[i%len(fit_colors)])

        x_min = x_limits[0] if x_limits is not None else wavelengths.value.min()
        x_max = x_limits[1] if x_limits is not None else wavelengths.value.max()
        plt.xlim(x_min, x_max)
        if y_limits is not None:
            plt.ylim(y_limits[0], y_limits[1])

        set_axis_labels(wavelengths, cubes[0], x_units, y_units)

        if labels is not None:
            plt.legend()
        plt.tight_layout()
        if savefig:
            save_figure_2_disk(dpi)

        plt.show()

        if return_spectra:
            return wavelengths, spectrum, spec_normalized, continuum_fit
