from astropy.wcs import WCS
from astropy.io.fits import Header
import matplotlib.pyplot as plt
from .data_cube import plot_spectral_cube
from .io import save_figure_2_disk
from .numerical_utils import get_data
from .plotting import imshow, plot_density_histogram, plot_histogram, plot_lines, scatter_plot
from .plot_utils import return_stylename, set_plot_colors
from .spectra import plot_combine_spectrum, plot_spectrum
from .visual_classes import DataCube, FitsFile


class va:
    @staticmethod
    def imshow(datas, idx=None, vmin=None, vmax=None, norm='asinh',
               percentile=[3,99.5], origin='lower', wcs_input=None,
               invert_wcs=False, cmap='turbo', aspect=None, ax=None,
               show=True, **kwargs):
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
                try:
                    wcs = WCS(wcs_input)
                except:
                    wcs_input = None
            elif isinstance(wcs_input, (list, tuple)):
                try:
                    wcs = WCS(wcs_input[0])
                except:
                    wcs_input = None
            elif isinstance(wcs_input, WCS):
                wcs = wcs_input
            elif wcs_input is not None:
                raise TypeError(f'Unsupported wcs_input type: {type(wcs_input)}')
            if invert_wcs and isinstance(wcs, WCS):
                wcs = wcs.swapaxes(0, 1)

        style = return_stylename(style)
        with plt.style.context(style):
            if ax is None:
                plt.figure(figsize=figsize)
                ax = plt.subplot(111) if wcs_input is None else plt.subplot(111, projection=wcs)

            imshow(datas, ax, idx, vmin, vmax, norm, percentile, origin,
                   cmap, aspect, wcs_input=wcs_input, **kwargs)

            if savefig:
                    save_figure_2_disk(dpi)
            if show:
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

        cubes = cubes if isinstance(cubes, (list, tuple)) else [cubes]

        # define wcs figure axes
        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            wcs2d = get_data(cubes[0]).wcs.celestial
            ax = fig.add_subplot(111, projection=wcs2d)
            if style.split('/')[-1] == 'minimal.mplstyle':
                ax.coords['ra'].set_ticks_position('bl')
                ax.coords['dec'].set_ticks_position('bl')

            plot_spectral_cube(cubes, idx, ax, vmin, vmax, percentile,
                                norm, radial_vel, unit, **kwargs)
            if savefig:
                save_figure_2_disk(dpi)

            plt.show()


    @staticmethod
    def plotSpectrum(extracted_spectrums=None, plot_norm_continuum=False,
                     plot_continuum_fit=False, emission_line=None, wavelength=None,
                     flux=None, continuum_fit=None, colors=None, **kwargs):

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

            plot_spectrum(extracted_spectrums, ax, plot_norm_continuum,
                          plot_continuum_fit, emission_line, wavelength,
                          flux, continuum_fit, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plotCombineSpectrum(extracted_spectra, idx=0, wave_cuttofs=None,
                            concatenate=False, return_spectra=False,
                            plot_normalize=False, use_samecolor=True, **kwargs):

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
            if return_spectra:
                combined_spectra = plot_combine_spectrum(extracted_spectra, ax, idx,
                                                         wave_cuttofs, concatenate,
                                                         return_spectra, plot_normalize,
                                                         use_samecolor, **kwargs)
            else:
                plot_combine_spectrum(extracted_spectra, ax, idx,
                                      wave_cuttofs, concatenate,
                                      return_spectra, plot_normalize,
                                      use_samecolor, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()

        if return_spectra:
            return combined_spectra


    @staticmethod
    def plotDensityHistogram(X, Y, bins='auto', xlog=False, ylog=False,
                             xlog_hist=True, ylog_hist=True, sharex=False,
                             sharey=False, histtype='step', colors=None, **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # adjust grid layout to prevent overlap
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2),
                                    height_ratios=(1.2, 4),
                                    left=0.15, right=0.9, bottom=0.15,
                                    top=0.9, wspace=0.09, hspace=0.09)
            # create subplots
            ax = fig.add_subplot(gs[1, 0])
            sharex = ax if sharex is True else None
            sharey = ax if sharey is True else None
            ax_histx = fig.add_subplot(gs[0, 0], sharex=sharex)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=sharey)

            plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins,
                                   xlog, ylog, xlog_hist, ylog_hist,
                                   histtype, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plotHistogram(datas, bins='auto', xlog=False, ylog=False,
                      histtype='step', colors=None, **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_histogram(datas, ax, bins, xlog, ylog,
                           histtype, colors, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def plot(X, Y, normalize=False, xlog=False,
             ylog=False, colors=None, linestyle='-',
             linewidth=0.8, alpha=1, zorder=None, **kwargs):

        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            plot_lines(X, Y, ax, normalize=normalize,
                       xlog=xlog, ylog=ylog, colors=colors,
                       linestyle=linestyle, linewidth=linewidth,
                       alpha=alpha, zorder=zorder, **kwargs)

            if savefig:
                save_figure_2_disk(dpi)
            plt.show()


    @staticmethod
    def scatter(X, Y, xerr=None, yerr=None, normalize=False,
                xlog=False, ylog=False, colors=None, size=10,
                marker='o', alpha=1, edgecolors='face',
                **kwargs):
        # figure params
        figsize = kwargs.get('figsize', (6,6))
        style = kwargs.get('style', 'astro')
        # savefig
        savefig = kwargs.get('savefig', False)
        dpi = kwargs.get('dpi', 600)

        style = return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            scatter_plot(X, Y, ax, xerr=xerr, yerr=yerr, normalize=normalize,
                         xlog=xlog, ylog=ylog, colors=colors, size=size,
                         marker=marker, alpha=alpha, edgecolors=edgecolors, **kwargs)

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
