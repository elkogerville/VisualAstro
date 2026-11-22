"""
Author: Elko Gerville-Reache, Qiushi Chris Tian
Date Created: 2025-05-23
Date Modified: 2026-11-18
"""

from importlib.metadata import version as _version

__version__ = _version('visualastro')


from visualastro.analysis.ic import blob
from visualastro.analysis.image_utils import (
    compute_line,
    compute_sobel_filter,
    detect_edges,
    get_filter_wavelength,
    get_hdu,
    get_hdu_from_hdu_list,
    image_2_grayscale,
    load_data_cube,
    load_fits,
    load_spectral_cube,
    mask_image,
    stack_cube,
)
from visualastro.analysis.spectra import (
    ExtractedPixelSpectra,
    GaussianFitResult,
    compute_continuum,
    continuum_spectrum,
    deredden_flux,
    estimate_spectrum_line_flux,
    extract_region,
    gaussian,
    gaussian_continuum,
    gaussian_line,
    gaussian_p0,
    propagate_flux_errors,
    shift_to_rest_frame,
    sort_spectra_by_line_strength,
    spectral_idx_2_world,
    spectral_region,
    spectral_region_mask,
    spectral_world_2_idx,
    splice_spectra,
)
from visualastro.core.configuration import config
from visualastro.core.io import (
    get_errors,
    load_quantity,
    save_array,
    save_quantity,
    savefig,
    write_arrays_2_file,
    write_cube_2_fits,
)
from visualastro.core.numerical import (
    flatten,
    get_finite_mask,
    interpolate,
    kde1d,
    kde2d,
    keep_finite,
    mask_within_range,
    number_density,
)
from visualastro.core.sequences import (
    match_length,
)
from visualastro.core.stats import normalize, percent_difference, relative_error
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_physical_type,
    get_spectral_unit,
    get_unit,
    get_unit_label,
    get_units,
    stack_quantities,
    to_fits_unit,
    to_latex_unit,
    to_unit,
    unit_2_string,
)
from visualastro.core.validation import allclose
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.datamodels.spectrumplus import SpectrumPlus
from visualastro.help.help import getsource, help
from visualastro.plotting.ax import ax
from visualastro.plotting.base.plots import (
    hist,
    plot,
    plot_density_histogram,
    scatter,
    scatter3D,
    scatter_polyfit,
    scatter_project,
    step,
)
from visualastro.plotting.core.axes import (
    ax3d_axis_style,
    ax3d_pane_color,
    figure,
    gridspec,
    inset_axes,
    set_limits,
    set_title,
    subplots,
    subplots_3d,
    subplots_wcs,
    tripanel_figure,
)
from visualastro.plotting.core.backend import (
    close,
    inline,
    interactive,
)
from visualastro.plotting.core.colorbar import add_colorbar
from visualastro.plotting.core.colormaps import (
    create_cmap,
    get_cmap,
    plot_cmap_lightness,
)
from visualastro.plotting.core.colors import (
    Colors,
    Colorsets,
    as_color,
    darken_colors,
    desaturate_colors,
    get_color,
    get_colors,
    get_colorset,
    get_namedcolor,
    lighten_colors,
    plot_color_deltaE,
    plot_colors,
    plot_colortable,
    preview_colorset,
    random_colors,
    sample_cmap,
    saturate_colors,
    simulate_colorblindness,
)
from visualastro.plotting.core.contour import (
    contour_kde,
    contourf_kde,
)
from visualastro.plotting.core.image_utils import (
    get_imshow_norm,
    get_vmin_vmax,
    nanpercentile_limits,
    thorlabs_logo,
)
from visualastro.plotting.core.legend import legend
from visualastro.plotting.core.lines import axhlines, axvlines, grid, wcsgrid
from visualastro.plotting.core.style import (
    print_rcParams,
    reset_rcParams,
    set_plot_cycle,
)
from visualastro.plotting.core.utils import (
    ellipse_patch,
    plot_circles,
    plot_ellipses,
    plot_interactive_ellipse,
    plot_points,
)
from visualastro.plotting.science.spectra_plots import (
    extract_cube_pixel_spectra,
    extract_cube_spectrum,
    fit_gaussian_2_spec,
    plot_combine_spectrum,
    plot_extracted_pixel_map,
    plot_spectrum,
)
from visualastro.plotting.science.spectral_axis import spectral_axis_label
from visualastro.plotting.science.spectral_line_markers import (
    spectral_elbow_markers,
    spectral_markers,
    spectral_vlines,
)
from visualastro.plotting.science.wcs_plots import (
    color_image,
    imshow,
    plot_spectral_cube,
)
from visualastro.utils.text_utils import (
    check_font,
    find_font,
    pprint,
    pretty_table,
    print_font_info,
    print_pretty_table,
)
from visualastro.utils.wcs_utils import (
    crop2D,
    get_header_wcs,
    get_wcs,
    get_wcs_celestial,
    reproject_wcs,
)


# REGISTER FONTS
# --------------
def _register_fonts() -> None:
    """
    Register additional fonts into Matplotlib.
    To add more fonts, simply add a folder to
    `VisualAstro/src/visualastro/stylelib/fontlib`
    with `.ttf` or `.otf` files.
    """
    import warnings
    from pathlib import Path

    import matplotlib.font_manager as fm

    src = Path(__file__).parent
    fonts_dir = src / 'stylelib' / 'fontlib'

    if not fonts_dir.exists():
        warnings.warn(
            '[visualastro] Font directory not found. '
            'Falling back to Matplotlib default fonts.',
            stacklevel=2
        )
        return

    font_files = list(fonts_dir.rglob('*.ttf')) + list(fonts_dir.rglob('*.otf'))
    registered = {font.fname for font in fm.fontManager.ttflist}

    for font_file in font_files:
        if str(font_file) not in registered:
            try:
                fm.fontManager.addfont(str(font_file))
            except Exception as e:
                warnings.warn(
                    f'[visualastro] Could not register font {font_file.name}: {e}',
                    stacklevel=2
                )

_register_fonts()

# REGISTER STYLES
# ---------------
def _register_styles():
    """
    Register additional styles with Matplotlib.

    Available under `plt.style.use('stylename')`
    after importing VisualAstro.
    """
    import warnings
    from importlib.resources import files
    from pathlib import Path

    import matplotlib.pyplot as plt

    from visualastro.plotting.core.style import STYLE_ALIASES, VisualAstroStyles

    stylelib = files('visualastro') / 'stylelib'

    # matplotlib >= 3.11
    try:
        _plt_read_style_dir = plt.style.read_style_directory
    except AttributeError:
        _plt_read_style_dir = plt.style.core.read_style_directory

    styledict = {}
    style_root = Path(stylelib)

    available_styles = sorted(list({p.stem for p in style_root.rglob('*.mplstyle')}))

    # create set of each valid parent directory to a .mplstyle in stylelib
    dirs = {p.parent for p in style_root.rglob('*.mplstyle')}
    dirs.add(style_root)
    for directory in dirs:
        styledict.update(_plt_read_style_dir(directory))

    for key, style in styledict.items():
        aliases = STYLE_ALIASES.get(key, ())

        for name in (key, *aliases):
            if name in plt.style.library:
                warnings.warn(
                    f"Found custom VisualAstro style '{name}' that conflicts "
                    "with an existing Matplotlib style. Skipping registration.",
                    stacklevel=2,
                )
                continue

            plt.style.library[name] = style.copy()

    plt.style.available[:] = sorted(plt.style.library.keys())

    return VisualAstroStyles(available_styles)

from visualastro.optional_dependencies import _scienceplots

styles = _register_styles()


# REGISTER COLORSETS
# ------------------
def _register_colorsets() -> Colorsets:
    """Register VisualAstro colorsets with matplotlib."""
    import matplotlib as mpl

    from visualastro.plotting.core.colors import COLORSET_ALIASES, COLORSETS, Colorsets

    registry = mpl.color_sequences
    for name, colors in COLORSETS.items():
        if name not in registry:
            registry.register(name, colors)

    for alias, name in COLORSET_ALIASES.items():
        if alias not in registry and name in registry:
            registry.register(alias, registry[name])

    return Colorsets()

colorsets = _register_colorsets()


# REGISTER NAMED COLORS
# ---------------------
def _register_colors() -> Colors:
    """
    Register additional named colors with Matplotlib.

    Notes
    -----
    - Existing colors are never overwritten.
    - Each name is also registered in lowercase.
    - Warns on any name collision.
    """
    import warnings

    import matplotlib as mpl

    from visualastro.plotting.core.colors import (
        NAMED_COLOR_ALIASES,
        VISUALASTRO_NAMED_COLORS,
    )

    cmap = mpl.colors.get_named_colors_mapping()

    def _register(key, color):
        for k in {key, key.lower()}:
            if k in cmap:
                warnings.warn(
                    f"Color name '{k}' (from '{key}') conflicts with an "
                    'existing named color. Skipping registration.',
                    stacklevel=3,
                )
            else:
                cmap[k] = color

    for name, color in VISUALASTRO_NAMED_COLORS.items():
        _register(name, color)

    for alias, name in NAMED_COLOR_ALIASES.items():
        if name not in cmap:
            warnings.warn(
                f"Alias '{alias}' targets unknown color '{name}'. Skipping.",
                stacklevel=3,
            )
            continue
        _register(alias, cmap[name])

    return Colors()

colors = _register_colors()


# REGISTER COLORMAPS
# ------------------
def _register_cmaps() -> None:
    """Register additional colormaps with Matplotlib."""
    import matplotlib as mpl

    from visualastro.optional_dependencies._cmasher import _HAS_CMASHER, cmasher

    # register tol-colors cmaps upon import
    from visualastro.optional_dependencies._tol_colors import tol_colors
    from visualastro.plotting.core.colormaps import VISUALASTRO_CMAPS

    mpl_cmaps = set(mpl.colormaps.keys())

    if _HAS_CMASHER:
        cmasher_cmaps_names = cmasher.get_cmap_list()
        for name in cmasher_cmaps_names:
            cmap = getattr(cmasher, name, None)
            if cmap is not None and name not in mpl_cmaps:
                mpl.colormaps.register(cmap, name=name)

    for name, cmap in VISUALASTRO_CMAPS.items():
        mpl.colormaps.register(cmap, name=name)
        mpl.colormaps.register(cmap.reversed(), name=name+'_r')

_register_cmaps()


__all__ = [name for name in dir() if not name.startswith('_')]
