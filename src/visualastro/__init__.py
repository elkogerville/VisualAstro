 # core classes
# ------------
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.datamodels.spectrumplus import SpectrumPlus

# submodules
# ----------
from visualastro.analysis.ic import (
    blob,
)
from visualastro.analysis.image_utils import (
    compute_sobel_filter,
    detect_edges,
    image_2_grayscale,
    load_data_cube,
    load_fits,
    load_spectral_cube,
    compute_line,
    mask_image,
    stack_cube
)
from visualastro.analysis.spectra_utils import (
    ExtractedPixelSpectra,
    GaussianFitResult,
    construct_gaussian_p0,
    deredden_flux,
    estimate_spectrum_line_flux,
    fit_continuum,
    gaussian,
    gaussian_continuum,
    gaussian_line,
    mask_spectral_region,
    propagate_flux_errors,
    shift_by_radial_vel,
    sort_spectra_by_line_strength,
    spectral_idx_2_world,
    spectral_world_2_idx
)
from visualastro.core.config import config
from visualastro.core.io import (
    get_errors,
    load_quantity,
    save_array,
    savefig,
    save_quantity,
    write_arrays_2_file,
    write_cube_2_fits
)
from visualastro.core.numerical import (
    interpolate,
    kde1d,
    kde2d,
    number_density,
)
from visualastro.core.numerical_utils import (
    finite,
    flatten,
    mask_finite,
    mask_within_range,
)
from visualastro.core.stats import (
    normalize,
    percent_difference,
    relative_error
)
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_physical_type,
    get_spectral_unit,
    get_unit,
    get_units,
    get_unit_label,
    stack_quantities,
    to_unit,
    to_fits_unit,
    to_latex_unit,
    to_spectral_region,
    unit_2_string
)
from visualastro.core.validation import allclose
from visualastro.help.help import help, getsource
from visualastro.plotting.ax import ax
from visualastro.plotting.base.plots import (
    hist,
    plot_density_histogram,
    plot,
    scatter,
    scatter_fit,
    scatter_project,
    scatter3D
)
from visualastro.plotting.core.axes import (
    add_subplot,
    add_ax3d,
    ax3d,
    ax3d_axis_style,
    ax3d_pane_color,
    gridspec,
    set_axis_labels,
    set_axis_limits,
    subplot,
    tripanel_figure,
)
from visualastro.plotting.core.colors import (
    as_color,
    _color as color,
    create_cmap,
    desaturate_colors,
    get_cmap,
    get_colors,
    get_complimentary_colors,
    lighten_colors,
    plot_colortable,
    random_colors,
    sample_cmap,
    saturate_colors,
    simulate_colorblindness,
)
from visualastro.plotting.core.image_utils import (
    get_imshow_norm,
    get_vmin_vmax,
    nanpercentile_limits,
    thorlabs_logo,
)
from visualastro.plotting.core.utils import (
    add_colorbar,
    close,
    contour,
    contourf,
    ellipse_patch,
    inline,
    interactive,
    legend,
    plot_circles,
    plot_ellipses,
    plot_hlines,
    plot_interactive_ellipse,
    plot_points,
    plot_vlines,
    style
)
from visualastro.plotting.science.spectra_plots import (
    extract_cube_pixel_spectra,
    extract_cube_spectra,
    fit_gaussian_2_spec,
    plot_combine_spectrum,
    plot_extracted_pixel_map,
    plot_spectra
)
from visualastro.plotting.science.spectra_plot_utils import (
    mark_spectral_lines,
    spectral_axis_label,
    spectral_line_marker
)
from visualastro.plotting.science.wcs_plots import (
    imshow,
    plot_spectral_cube
)
from visualastro.utils.text_utils import (
    pretty_table,
    print_pretty_table
)
from visualastro.utils.wcs_utils import (
    crop2D,
    get_header_wcs,
    get_wcs,
    get_wcs_celestial,
    reproject_wcs
)


__all__ = [name for name in dir() if not name.startswith('_')]


# REGISTER FONTS
# --------------
def _register_fonts():
    """
    Register additional fonts into matplotlib.
    To add more fonts, simply add a folder to
    `VisualAstro/src/visualastro/stylelib/fontlib`
    with `.ttf` or `.otf` files.
    """
    from pathlib import Path
    import warnings
    import matplotlib.font_manager as fm

    src = Path(__file__).parent
    fonts_dir = src / 'stylelib' / 'fontlib'

    if not fonts_dir.exists():
        warnings.warn(
            '[visualastro] Font directory not found. '
            'Falling back to matplotlib default fonts.',
            stacklevel=2
        )
        return

    font_files = list(fonts_dir.rglob('*.ttf')) + list(fonts_dir.rglob('*.otf'))

    for font_file in font_files:
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
    Register additional styles with matplotlib.

    Available under `plt.style.use('stylename')`
    after importing visualastro.
    """
    from importlib.resources import files
    from pathlib import Path
    import warnings
    import matplotlib.pyplot as plt

    stylelib = files('visualastro') / 'stylelib'

    # matplotlib >= 3.11
    try:
        _plt_read_style_dir = plt.style.read_style_directory
    except AttributeError:
        _plt_read_style_dir = plt.style.core.read_style_directory

    styledict = {}
    style_root = Path(stylelib)
    # create set of each valid parent directory to a .mplstyle in stylelib
    dirs = {p.parent for p in style_root.rglob("*.mplstyle")}
    dirs.add(style_root)
    for directory in dirs:
        styledict.update(_plt_read_style_dir(directory))

    for key in styledict.keys():
        if key not in plt.style.library:
            plt.style.library[key] = styledict[key]
        else:
            warnings.warn(
                f"Found custom visualastro style of name: '{key}', which "
                'conflicts with a pre-existing matplotlib style! '
                'Skipping registration, please change name collision.',
                stacklevel=2
            )

    plt.style.available[:] = sorted(plt.style.library.keys())

import scienceplots
_register_styles()
