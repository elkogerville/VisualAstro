import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from .data_cube import plot_spectral_cube
from .plot_utils import return_stylename, save_figure_2_disk

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
