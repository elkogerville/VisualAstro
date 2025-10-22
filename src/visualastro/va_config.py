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
        self.print_info = True
        self.transpose = False

        # figure params
        self.style = 'astro' # default style
        self.style_fallback = 'default.mplstyle' # style if default style fails
        self.figsize = (6, 6)
        self.grid_figsize = (12, 6)
        self.colors = None # if None, defaults to `self.default_palette`. To define a custom default palette,
                           # define it in `set_plot_colors` and change the `default_palette`.
        self.default_palette = 'ibm_contrast' # see `set_plot_colors` in plot_utils.py
        self.alpha = 1
        self.nrows = 1 # make_grid_plot() nrows
        self.ncols = 2 # make_grid_plot() ncols

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
        self.edgecolor = 'face'

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
        self.text_color = 'k'
        self.text_loc = [0.03, 0.03]

        # label params
        self.use_brackets = False # display units as [unit] instead of (unit)
        self.right_ascension = 'Right Ascension'
        self.declination = 'Declination'
        self.highlight = True
        self.loc = 'best'

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

        # plot_spectrum params
        self.plot_spectrum_text_loc = [0.025, 0.95]

        # deredden spectra params
        self.Rv = 3.1
        self.Ebv = 0.19
        self.deredden_method = 'WD01'
        self.deredden_region = 'LMCAvg'

    def reset_defaults(self):
        self.__init__()


va_config = VAConfig()
_default_flag = object()

def get_config_value(var, attribute):
    if var is None:
        return getattr(va_config, attribute)
    return var
