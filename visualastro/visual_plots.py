import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from .data_cube import plot_spectral_cube
from .plot_utils import return_stylename, save_figure_2_disk, set_axis_labels, set_plot_colors
from .spectra import compute_limits_mask, set_axis_limits

class va:
    @staticmethod
    def plotSpectralCube(cubes, idx, vmin=None, vmax=None, percentile=[3,99.5],
                        norm='asinh', radial_vel=None, unit=None, **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        # define wcs figure axes
        cubes = [cubes] if isinstance(cubes, SpectralCube) else cubes
        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            wcs2d = cubes[0].wcs.celestial
            ax = fig.add_subplot(111, projection=wcs2d)
            if style.split('/')[-1] == 'minimal.mplstyle':
                ax.coords['ra'].set_ticks_position('bl')
                ax.coords['dec'].set_ticks_position('bl')

            for cube in cubes:
                plot_spectral_cube(cube, idx, ax, vmin, vmax, percentile,
                                   norm, radial_vel, unit, **kwargs)
            if savefig:
                save_figure_2_disk(dpi)

            plt.show()

    @staticmethod
    def plotSpectrum(spectra_dicts, normalize=False, plot_continuum=False, emission_line=None, **kwargs):

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
        text_loc = kwargs.get('text_loc', [0.025, 0.95])
        use_brackets = kwargs.get('use_brackets', False)
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        spectra_dicts = spectra_dicts if isinstance(spectra_dicts, list) else [spectra_dicts]

        # set plot style and colors
        colors, fit_colors = set_plot_colors(colors)
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            if emission_line is not None:
                ax.text(text_loc[0], text_loc[1], f'{emission_line}', transform=ax.transAxes)

            wavelength_list = []
            for i, spectra_dict in enumerate(spectra_dicts):
                if spectra_dict is not None:

                    wavelength = spectra_dict['wavelength']
                    flux = spectra_dict['normalized'] if normalize else spectra_dict['flux']

                    mask = compute_limits_mask(wavelength, xlim=xlim)

                    label = labels[i] if (labels is not None and i < len(labels)) else None

                    ax.plot(wavelength[mask], flux[mask], c=colors[i%len(colors)], label=label)
                    if plot_continuum:
                        ax.plot(wavelength[mask], spectra_dict['continuum_fit'][mask], c=fit_colors[i%len(fit_colors)])

                    wavelength_list.append(wavelength[mask])

            set_axis_limits(wavelength_list, ax, xlim, ylim)
            set_axis_labels(wavelength, spectra_dict['flux'], ax, xlabel, ylabel, use_brackets=use_brackets)
            if labels is not None:
                ax.legend()
            if savefig:
                save_figure_2_disk(dpi)
            plt.show()
