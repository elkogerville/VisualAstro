import numpy as np
from astropy.wcs import WCS
from astropy.io.fits import Header
import matplotlib.pyplot as plt
from .data_cube import plot_spectral_cube
from .io import save_figure_2_disk
from .numerical_utils import get_data
from .plotting import imshow, plot_histogram
from .plot_utils import (
    return_stylename, set_axis_labels, set_plot_colors
)
from .spectra import plot_spectrum, return_spectra_dict
from .visual_classes import DataCube, FitsFile


class va:
    @staticmethod
    def imshow(datas, idx=None, vmin=None, vmax=None, norm='asinh',
               percentile=[3,99.5], origin='lower', wcs_input=None,
               invert_wcs=False, cmap='turbo', aspect=None, **kwargs):
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
                wcs = WCS(wcs_input)
            elif isinstance(wcs_input, WCS):
                wcs = wcs_input
            elif wcs_input is not None:
                raise TypeError(f'Unsupported wcs_input type: {type(wcs_input)}')
            if invert_wcs and isinstance(wcs, WCS):
                wcs = wcs.swapaxes(0, 1)

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
    def plotSpectralCube(cubes, idx, vmin=None, vmax=None, percentile=[3,99.5],
                        norm='asinh', radial_vel=None, unit=None, **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        # define wcs figure axes
        cubes = [cubes] if not isinstance(cubes, list) else cubes
        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            wcs2d = get_data(cubes[0]).wcs.celestial
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
    def plotSpectrum(extracted_spectrums=None, normalize_continuum=False, plot_continuum_fit=False,
                     emission_line=None, wavelength=None, flux=None, norm_flux=None, **kwargs):

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

            plot_spectrum(extracted_spectrums, ax, normalize_continuum,
                          plot_continuum_fit, emission_line, wavelength,
                          flux, norm_flux,**kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

    @staticmethod
    def plotCombineSpectrum(spectra_dict_list, idx=0, spec_lims=None,
                            concatenate=False, return_spectra=False,
                            use_samecolor=True, **kwargs):

        # figure params
        figsize = kwargs.get('figsize', (12,6))
        style = kwargs.get('style', 'astro')
        ylim = kwargs.get('ylim', None)
        # labels
        label = kwargs.get('label', None)
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        colors = kwargs.get('colors', None)
        cmap = kwargs.get('cmap', 'turbo')
        loc = kwargs.get('loc', 'best')
        use_brackets = kwargs.get('use_brackets', False)
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        concatenate = True if return_spectra else concatenate
        # set plot style and colors
        colors, _ = set_plot_colors(colors, cmap=cmap)
        style = return_stylename(style)

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
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
            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

            if return_spectra:
                spectra_dict = return_spectra_dict(wavelength, flux)

                return spectra_dict

    @staticmethod
    def plotHistogram(datas, bins='auto', xlog=False,
                      ylog=False, colors=None, **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_histogram(datas, ax, bins, xlog, ylog, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

    # –––– VISUALASTRO HELP ––––

    class help:
        @staticmethod
        def colors(user_color=None):
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
