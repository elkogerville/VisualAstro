'''
Author: Elko Gerville-Reache
Date Created: 2025-10-20
Date Modified: 2025-10-27
Description:
    Visualastro configuration interface to update function defaults.
Dependencies:
    - numpy
Module Structure:
    - Plotting Params
        Confguration parameters related to plotting functions.
    - Science Params
        Configuration parameters related to science functions.
'''

import numpy as np


class VAConfig:
    '''
    Global configuration object for controlling default behavior
    across the visualastro package.

    Users can modify attributes to update default values for
    plotting functions globally.
    '''
    def __init__(self):
        # Plotting Params
        # –––––––––––––––

        # I/O params
        self.default_unit = np.float64
        self.hdu_idx = 0
        self.print_info = False
        self.transpose = False
        self.mask_non_positive = False
        self.mask_out_value = np.nan
        self.invert_wcs_if_transpose = True
        self.target_wcs = None

        # figure params
        self.style = 'astro' # default style
        self.style_fallback = 'default.mplstyle' # style if default style fails
        self.figsize = (6, 6)
        self.grid_figsize = (12, 6)
        self.figsize3d = (10, 10)
        self.colors = None # if None, defaults to `self.default_palette`. To define a custom default palette,
                           # define it in `set_plot_colors` and change the `default_palette`.
        self.default_palette = 'ibm_contrast' # see `set_plot_colors` in plot_utils.py
        self.alpha = 1
        self.nrows = 1 # make_grid_plot() nrows
        self.ncols = 2 # make_grid_plot() ncols
        self.rasterized = False # rasterize plot artists wherever possible
        self.wcs_grid = False

        # data params
        self.normalize_data = False

        # histogram params
        self.histtype = 'step'
        self.bins = 'auto'
        self.normalize_hist = True

        # line2D params
        self.linestyle = '-'
        self.linewidth = 0.8

        # scatter params
        self.scatter_size = 10
        self.marker = 'o'
        self.edgecolor = None
        self.facecolor = None

        # errorbar params
        self.eb_fmt = 'none' # use 'none' (case-insensitive) to plot errorbars without any data markers.
        self.ecolors = None
        self.elinewidth = 1
        self.capsize = 1
        self.capthick = 1
        self.barsabove = False

        # imshow params
        self.cmap = 'turbo'
        self.origin = 'lower'
        self.norm = 'asinh'
        self.linear_width = 1 # AsinhNorm linear width
        self.gamma = 0.5 # PowerNorm exponent
        self.vmin = None
        self.vmax = None
        self.percentile = [3.0, 99.5]
        self.aspect = None

        # axes params
        self.xpad = 0.0  # set_axis_limits() xpad
        self.ypad = 0.05 # set_axis_limits() ypad
        self.xlog = False
        self.ylog = False
        self.xlog_hist = True
        self.ylog_hist = True
        self.sharex = False
        self.sharey = False
        self.hspace = None
        self.wspace = None
        self.Nticks = None
        self.aspect = None

        # cbar params
        self.cbar = True
        self.cbar_width = 0.03
        self.cbar_pad = 0.015
        self.cbar_tick_which = 'both'
        self.cbar_tick_dir = 'out'
        self.clabel = True

        # text params
        self.fontsize = 10
        self.text_color = 'k'
        self.text_loc = [0.03, 0.03]

        # label params
        self.use_brackets = False # display units as [unit] instead of (unit)
        self.right_ascension = 'Right Ascension'
        self.declination = 'Declination'
        self.highlight = True
        self.loc = 'best'
        self.use_type_label = True
        self.use_unit_label = True
        self.unit_label_format = 'latex_inline'

        # savefig params
        self.savefig = False
        self.dpi = 600
        self.pdf_compression = 6
        self.bbox_inches = 'tight'
        self.allowed_formats = {'eps', 'pdf', 'png', 'svg'}

        # circles params
        self.circle_linewidth = 2
        self.circle_fill = False
        self.ellipse_label_loc = [0.03, 0.03]

        # Science Params
        # ––––––––––––––
        # data params
        self.wavelength_unit = None
        self.radial_velocity = None

        # extract_cube_spectrum params
        self.spectra_rest_frequency = None
        self.flux_extract_method = 'mean'
        self.spectral_cube_extraction_mode = 'cube'
        self.spectrum_continuum_fit_method = 'fit_continuum'
        self.deredden_spectrum = False
        self.plot_normalized_continuum = False
        self.plot_continuum_fit = False

        # plot_spectrum params
        self.plot_spectrum_text_loc = [0.025, 0.95]

        # deredden spectra params
        self.Rv = 3.1 # Milky Way average
        self.Ebv = 0.19
        self.deredden_method = None
        self.deredden_region = None

        # gaussian fitting params
        self.gaussian_model = 'gaussian'
        self.interpolate = False
        self.return_gaussian_fit_parameters = False
        self.print_gaussian_values = True

        # numerical parameters
        self.interpolation_samples = 10000
        self.interpolation_method = 'cubic_spline'
        self.error_interpolation_method = 'cubic_spline'

        # reprojection parameters
        self.reproject_method = 'interp'
        self.return_footprint = False
        self.reproject_block_size = None
        self.reproject_parallel = False

    def reset_defaults(self):
        self.__init__()

# instantiate va_config class
va_config = VAConfig()
# placeholder flag for default values by
# default, the placeholder flag is `None`,
# but when an argument can also take in
# `None`, `_default_flag` should be used.
_default_flag = object()

def get_config_value(var, attribute):
    '''
    Retrieve a configuration value, falling back to the
    default from `va_config` if `var` is None.
    Parameters
    ––––––––––
    var : any
        User-specified value. If not None, this value is returned.
    attribute : str
        Name of the attribute to retrieve from `va_config` when `var` is None.
    Returns
    –––––––
    value : any
        The user-specified `var` if provided, otherwise the
        corresponding default value from `va_config`.
    '''
    if var is None:
        return getattr(va_config, attribute)
    return var
