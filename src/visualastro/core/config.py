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

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Literal, TypeVar
from astropy.wcs import WCS
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import physical
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import DTypeLike


T = TypeVar('T')

class _Unset(Enum):
    """
    Default placeholder sentinel value for
    visualastro functions.
    """
    UNSET = 'UNSET'

_UNSET = _Unset.UNSET


@dataclass(slots=True)
class AxesConfig:
    """matplotlib.axes config"""
    xpad: float = 0.05  # set_axis_limits() xpad
    ypad: float = 0.05 # set_axis_limits() ypad
    xlog: bool = False
    ylog: bool = False
    xlog_hist: bool = True
    ylog_hist: bool = True
    sharex: bool = False
    sharey: bool = False
    hspace = None
    wspace = None
    Nticks = None
    aspect = None

@dataclass(slots=True)
class AXLineConfig:
    """ax.vline / ax.hline config"""
    linestyle: Literal['-', '--', '-.', ':', ''] = ':'
    linewidth: float = 1.0
    color: ColorType = 'k'
    alpha: float | None = 0.7
    zorder: float = 0

@dataclass(slots=True)
class CurveFitConfig:
    """scipy.curve_fit config"""
    method: Literal['lm', 'trf', 'dogbox'] = 'trf'
    absolute_sigma: bool = False
    sample: int = 10000
    interpolate: bool = False
    interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'
    error_interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'

@dataclass(slots=True)
class ErrorBarConfig:
    """ax.errorbar config"""
    fmt: str = 'none' # use 'none' to plot errorbars without any data markers.
    colors: ColorType | None = None
    linewidth: float | None = 1
    capsize: float = 3 # cap length in points
    capthick: float | None = 1 # cap thickness in points
    barsabove: bool = False
    markeredgecolor: ColorType | None = None

@dataclass(slots=True)
class SpectralLineConfig:
    """spectral_line_marker config"""
    marker_direction: Literal['up', 'down'] = 'down'
    label_offset_points: tuple[float, float] = (0.0, 4.0)
    label_position: Literal['center', 'left', 'right'] = 'center'
    label_anchor: Literal['center', 'left', 'right', 'auto'] = 'auto'
    label_reference: Literal['marker', 'hline', 'auto'] = 'auto'
    label_rotation: float = 0

@dataclass(slots=True)
class HDUConfig:
    """HDU config"""
    index: int = 0
    error_extensions: list[str] = field(
        default_factory=lambda: [
        'ERR', 'ERROR', 'UNCERT'
    ])
    variance_extensions: list[str] = field(
        default_factory=lambda: [
        'VAR', 'VARIANCE', 'VAR_POISSON', 'VAR_RNOISE', 'STAT'
    ])


@dataclass(slots=True)
class VisualAstroConfig:
    """
    Global configuration object for controlling default behavior
    across the visualastro package.

    visualastro function parameters are often set to `_UNSET`,
    which at runtime gets resolved to the default hardcoded value
    set in `VisualAstroConfig`. Modifying this file will update
    the default values.

    Users can also modify attributes at runtime by modifying the config
    object attributes. This avoids permanently changing the default values.
    ie:
        >>> import visualastro as va
        >>> va.config.style = 'latex'
        >>> va.config.figsize = (6, 6)
    """
    # I/O params
    unit_mismatch: Literal['warn', 'ignore', 'raise'] | None = 'warn'
    default_dtype: DTypeLike = np.float64
    hdu_idx: int = 0
    print_info: bool = False
    transpose: bool = False
    mask_non_positive: bool = False
    mask_out_value: float = np.nan
    invert_wcs_if_transpose: bool = True
    target_wcs: Header | WCS | None = None
    hdu: HDUConfig = field(default_factory=HDUConfig)
    array_order: Literal['C', 'c', 'F', 'fortran'] = 'c'

    # figure params
    style: str = 'astro' # default style
    style_fallback: str = 'default.mplstyle' # style if default style fails
    figsize: tuple = (6, 6)
    reference_idx: int = 0 # which index is considered the reference for plot labels, cbars, etc..
    grid_figsize: tuple = (12, 6)
    figsize3d: tuple = (10, 10)
    # if _UNSET, defaults to `self.default_colorset`.
    # To define a custom default colorset,
    # define it in `get_colors` and change the `default_colorset`.
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET
    default_colorset: str = 'ibm_contrast' # see `get_colors` in plot_utils.py
    alpha: int = 1
    nrows: int = 1 # make_grid_plot() nrows
    ncols: int = 2 # make_grid_plot() ncols
    rasterized: bool = False # rasterize plot artists wherever possible

    # figure grid params
    grid_alpha: float = 1
    grid_color: ColorType = 'k'
    grid_linestyle: 'str' = 'solid'

    wcs_grid: bool = False
    wcs_grid_color: ColorType = 'w'
    wcs_grid_linestyle: 'str' = 'dotted'

    # data params
    normalize_data: bool = False

    # histogram params
    histtype = 'step'
    bins = 'auto'
    normalize_hist = True

    # line2D params
    linestyle: Literal['solid', '-', 'dotted', ':', 'dashed', '--', 'dashdot', '-.'] = '-'
    linewidth = 0.8

    axline: AXLineConfig = field(default_factory=AXLineConfig)

    # scatter params
    scatter_size = 10
    marker = 'o'
    edgecolor = None
    facecolor = None

    # errorbar params
    errorbar: ErrorBarConfig = field(default_factory=ErrorBarConfig)
    eb_fmt = 'none' # use 'none' (case-insensitive) to plot errorbars without any data markers.
    ecolors = None

    # imshow params
    cmap = 'turbo'
    origin = 'lower'
    norm: Literal['asinh', 'asinhnorm', 'log', 'power'] | None = 'asinh'
    linear_width: float = 1 # AsinhNorm linear width
    gamma: float = 0.5 # PowerNorm exponent
    vmin = None
    vmax = None
    percentile: tuple[float, float] | None = (3.0, 99.5)
    aspect = None

    # axes params
    xpad: float = 0.05  # set_axis_limits() xpad
    ypad: float = 0.05 # set_axis_limits() ypad
    xlog = False
    ylog = False
    xlog_hist = True
    ylog_hist = True
    sharex = False
    sharey = False
    hspace = None
    wspace = None
    Nticks = None
    aspect = None

    # cbar params
    cbar = True
    cbar_width = 0.03
    cbar_pad = 0.015
    cbar_tick_which = 'both'
    cbar_tick_dir = 'out'
    clabel = True

    # text params
    fontsize: float = 10
    text_color: str = 'k'
    text_loc: tuple[float, float] = (0.03, 0.03)

    # label params
    use_brackets = True # display units as [unit] instead of (unit)
    right_ascension = 'Right Ascension'
    declination = 'Declination'
    highlight: bool = True
    loc = 'best'
    show_type_label = True
    show_unit_label = True
    unit_label_format = 'latex_inline'
    _PHYSICAL_TYPE_LABELS = {
        u.adu.physical_type: 'ADU',          # type: ignore
        u.count.physical_type: 'Counts',     # type: ignore
        u.electron.physical_type: 'Counts',  # type: ignore
        u.mag.physical_type: 'Magnitude',    # type: ignore
        physical.length: 'Distance',
        physical.power_density: 'Flux',
        physical.spectral_flux_density: 'Flux Density',
        physical.surface_brightness: 'Surface Brightness',
    }

    _SPECTRAL_TYPE_LABELS = {
        physical.length: 'Wavelength',
        physical.frequency: 'Frequency',
        physical.energy: 'Energy',
        physical.speed: 'Velocity',
    }

    # savefig params
    savefig = False
    dpi = 600
    pdf_compression = 6
    bbox_inches = 'tight'
    allowed_formats = {'eps', 'pdf', 'png', 'svg'}

    # circles params
    circle_linewidth = 2
    circle_fill = False
    ellipse_label_loc = [0.03, 0.03]

    # Science Params
    # --------------
    # data params
    wavelength_unit = None
    radial_velocity = None

    # data cube params
    stack_cube_method = 'sum'

    # extract_cube_spectrum params
    spectra_rest_frequency = None
    flux_extract_method = 'mean'
    spectral_cube_extraction_mode = 'cube'
    spectrum_continuum_fit_method: str = 'fit_continuum'
    deredden_spectrum = False
    plot_normalized_continuum = False
    plot_continuum_fit = False

    # plot_spectrum params
    plot_spectrum_text_loc = [0.025, 0.95]

    # deredden spectra params
    Rv = 3.1 # Milky Way average
    Ebv = 0.19
    deredden_method = None
    deredden_region = None

    # curve_fit params
    curve_fit: CurveFitConfig = field(default_factory=CurveFitConfig)
    curve_fit_interpolate = False
    curve_fit_method = 'trf'
    curve_fit_absolute_sigma = False
    interpolation_samples = 10000
    interpolation_method = 'cubic_spline'
    error_interpolation_method = 'cubic_spline'

    # gaussian fitting params
    gaussian_model = 'gaussian'
    return_gaussian_fit_parameters = True
    print_gaussian_values = True

    # reprojection parameters
    reproject_method = 'interp'
    return_footprint: bool = False
    reproject_block_size = None
    reproject_parallel = False

    # error propagation
    propagate_flux_error_method = None

    # Utils Params
    # ------------

    # pretty table params
    table_precision = 7
    table_sci_notation = True
    table_column_pad = 3

    # numpy save
    save_format = '.npy'

    spectral_line_marker: SpectralLineConfig = field(default_factory=SpectralLineConfig)

    @property
    def elinewidth(self) -> float | None:
        return self.errorbar.linewidth

    @elinewidth.setter
    def elinewidth(self, value: float | None) -> None:
        self.errorbar.linewidth = value

    def reset(self):
        """
        Reset all configuration values to defaults.
        """
        default = type(self)()

        for f in fields(self):
            setattr(self, f.name, getattr(default, f.name))


config = VisualAstroConfig()


def _resolve_default(value: T | _Unset, fallback: T) -> T:
    """
    Fallback to a default configuration value if
    value is `_UNSET`.

    Parameters
    ----------
    value : T | _Unset
        Variable to be resolved.

    fallback : T
        Default value if `value` is `_UNSET`.
        Should be a `config` attribute.
        ie. `config.figsize`.

    Returns
    -------
    resolved_value : T
        Either `value` if value is not `_UNSET`
        or `fallback` if `value` is `_UNSET`.
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
