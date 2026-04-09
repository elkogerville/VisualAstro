"""
Author: Elko Gerville-Reache
Date Created: 2025-10-20
Date Modified: 2026-04-08
Description:
    Visualastro configuration interface to update function defaults.
Dependencies:
    - astropy
    - numpy
Module Structure:
    - visualastro config class
    - function specific configuration dataclasses
    - _Unset value and related functions
"""

from dataclasses import field
from enum import Enum
from typing import Literal, TypeVar
from astropy.wcs import WCS
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import physical
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import DTypeLike
from regions.io.fits.write import dataclass


T = TypeVar('T')

class VisualAstroConfig:
    """
    Global configuration object for controlling default behavior
    across the visualastro package.

    visualastro function parameters are often set to ``_UNSET``,
    which at runtime gets resolved to the default hardcoded value
    set in ``VisualAstroConfig``. Modifying this file will update
    the default values.

    Users can also modify attributes at runtime by modifying the config
    object attributes. This avoids permanently changing the default values.
    ie:
        >>> import visualastro as va
        >>> va.config.style = 'latex'
        >>> va.config.figsize = (6, 6)
    """

    def __init__(self):
        # Plotting Params
        # ---------------

        # I/O params
        self.unit_mismatch: str = 'warn'
        self.default_dtype: DTypeLike = np.float64
        self.hdu_idx: int = 0
        self.print_info: bool = False
        self.transpose: bool = False
        self.mask_non_positive: bool = False
        self.mask_out_value: float = np.nan
        self.invert_wcs_if_transpose: bool = True
        self.target_wcs: Header | WCS | None = None
        self.hdu = HDUConfig()

        # figure params
        self.style: str = 'astro' # default style
        self.style_fallback: str = 'default.mplstyle' # style if default style fails
        self.figsize: tuple = (6, 6)
        self.grid_figsize: tuple = (12, 6)
        self.figsize3d: tuple = (10, 10)
        self.colors: str | None = None # if None, defaults to `self.default_palette`. To define a custom default palette,
                           # define it in `set_plot_colors` and change the `default_palette`.
        self.default_palette: str = 'ibm_contrast' # see `set_plot_colors` in plot_utils.py
        self.alpha: int = 1
        self.nrows: int = 1 # make_grid_plot() nrows
        self.ncols: int = 2 # make_grid_plot() ncols
        self.rasterized: bool = False # rasterize plot artists wherever possible

        # figure grid params
        self.grid_alpha: float = 1
        self.grid_color: 'str' = 'k'
        self.grid_linestyle: 'str' = 'solid'

        self.wcs_grid: bool = False
        self.wcs_grid_color: 'str' = 'w'
        self.wcs_grid_linestyle: 'str' = 'dotted'

        # data params
        self.normalize_data = False

        # histogram params
        self.histtype = 'step'
        self.bins = 'auto'
        self.normalize_hist = True

        # line2D params
        self.linestyle = '-'
        self.linewidth = 0.8

        self.axline = AXLineConfig()

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
        self.norm: Literal['asinh', 'asinhnorm', 'log', 'power'] | None = 'asinh'
        self.linear_width: float = 1 # AsinhNorm linear width
        self.gamma: float = 0.5 # PowerNorm exponent
        self.vmin = None
        self.vmax = None
        self.percentile: tuple[float, float] | None = (3.0, 99.5)
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
        self.fontsize: float = 10
        self.text_color: str = 'k'
        self.text_loc: tuple[float, float] = (0.03, 0.03)

        # label params
        self.use_brackets = True # display units as [unit] instead of (unit)
        self.right_ascension = 'Right Ascension'
        self.declination = 'Declination'
        self.highlight: bool = True
        self.loc = 'best'
        self.use_type_label = True
        self.use_unit_label = True
        self.unit_label_format = 'latex_inline'
        self._PHYSICAL_TYPE_LABELS = {
            u.adu.physical_type: 'ADU',          # type: ignore
            u.count.physical_type: 'Counts',     # type: ignore
            u.electron.physical_type: 'Counts',  # type: ignore
            u.mag.physical_type: 'Magnitude',    # type: ignore
            physical.length: 'Distance',
            physical.power_density: 'Flux',
            physical.spectral_flux_density: 'Flux Density',
            physical.surface_brightness: 'Surface Brightness',
        }

        self._SPECTRAL_TYPE_LABELS = {
            physical.length: 'Wavelength',
            physical.frequency: 'Frequency',
            physical.energy: 'Energy',
            physical.speed: 'Velocity',
        }

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
        # --------------
        # data params
        self.wavelength_unit = None
        self.radial_velocity = None

        # data cube params
        self.stack_cube_method = 'sum'

        # extract_cube_spectrum params
        self.spectra_rest_frequency = None
        self.flux_extract_method = 'mean'
        self.spectral_cube_extraction_mode = 'cube'
        self.spectrum_continuum_fit_method: str = 'fit_continuum'
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

        # curve_fit params
        #
        self.curve_fit = CurveFitConfig()
        self.curve_fit_interpolate = False
        self.curve_fit_method = 'trf'
        self.curve_fit_absolute_sigma = False
        self.interpolation_samples = 10000
        self.interpolation_method = 'cubic_spline'
        self.error_interpolation_method = 'cubic_spline'

        # gaussian fitting params
        self.gaussian_model = 'gaussian'
        self.return_gaussian_fit_parameters = True
        self.print_gaussian_values = True

        # reprojection parameters
        self.reproject_method = 'interp'
        self.return_footprint: bool = False
        self.reproject_block_size = None
        self.reproject_parallel = False

        # error propagation
        self.propagate_flux_error_method = None

        # Utils Params
        # ------------

        # pretty table params
        self.table_precision = 7
        self.table_sci_notation = True
        self.table_column_pad = 3

        # numpy save
        self.save_format = '.npy'

        self.spectral_line_marker = SpectralLineConfig()


    def reset_defaults(self):
        """
        Reset all configuration values to default.
        """
        self.__init__()


@dataclass
class CurveFitConfig:
    """scipy.curve_fit config"""
    method: Literal['lm', 'trf', 'dogbox'] = 'trf'
    absolute_sigma: bool = False
    sample: int = 10000
    interpolate: bool = False
    interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'
    error_interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'

@dataclass
class AXLineConfig:
    """ax.vline / ax.hline config"""
    linestyle: Literal['-', '--', '-.', ':', ''] = ':'
    linewidth: float = 1.0
    color: ColorType = 'k'
    alpha: float | None = 0.7
    zorder: float = 0

@dataclass
class SpectralLineConfig:
    """spectral_line_marker config"""
    marker_direction: Literal['up', 'down'] = 'down'
    label_offset_points: tuple[float, float] = (0.0, 4.0)
    label_position: Literal['center', 'left', 'right'] = 'center'
    label_anchor: Literal['center', 'left', 'right', 'auto'] = 'auto'
    label_reference: Literal['marker', 'hline', 'auto'] = 'auto'
    label_rotation: float = 0


config = VisualAstroConfig()


class _Unset(Enum):
    """
    Default placeholder sentinel value for
    visualastro functions.
    """
    UNSET = 'UNSET'

_UNSET = _Unset.UNSET


def resolve_default(value: T | _Unset, fallback: T) -> T:
    """
    Fallback to a default configuration value if
    value is ``_UNSET``.

    Parameters
    ----------
    value : T | _Unset
        Variable to be resolved.

    fallback : T
        Default value if ``value`` is ``_UNSET``.
        Should be a ``config`` attribute.
        ie. ``config.figsize``.

    Returns
    -------
    resolved_value : T
        Either ``value`` if value is not ``_UNSET``
        or ``fallback`` if ``value`` is ``_UNSET``.
    """
    if value is _UNSET:
        return fallback
    return value


def get_config_value(var, attribute):
    """
    Retrieve a configuration value, falling back to the
    default from `config` if `var` is None.

    Parameters
    ----------
    var : any
        User-specified value. If not None, this value is returned.
    attribute : str
        Name of the attribute to retrieve from `config` when `var` is None.

    Returns
    -------
    value : any
        The user-specified `var` if provided, otherwise the
        corresponding default value from `config`.
    """
    if var is None:
        return getattr(config, attribute)
    return var
