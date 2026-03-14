# core classes
# ------------
from visualastro.dataclasses.datacube import DataCube
from visualastro.dataclasses.fitsfile import FitsFile
from visualastro.dataclasses.spectrumplus import SpectrumPlus

# submodules
# ----------
from visualastro.analysis.image_utils import (
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
    get_continuum,
    get_flux,
    get_spectral_axis,
    mask_spectral_region,
    propagate_flux_errors,
    shift_by_radial_vel,
    sort_spectra_by_line_strength,
    spectral_idx_2_world,
    spectral_world_2_idx
)
from visualastro.core.config import config
from visualastro.core.io import (
    get_dtype,
    get_errors,
    get_kwargs,
    load_quantity,
    save_array,
    save_figure_2_disk,
    save_quantity,
    write_cube_2_fits
)
from visualastro.core.numerical_utils import (
    compute_density_kde,
    finite,
    flatten,
    get_data,
    get_value,
    interpolate,
    mask_finite,
    mask_within_range,
    percent_difference,
    to_array,
    to_list
)
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_physical_type,
    get_spectral_unit,
    get_unit,
    require_spectral_region,
    to_unit,
    to_fits_unit,
    to_latex_unit,
    to_spectral_region,
    unit_2_string
)
from visualastro.core.validation import allclose
from visualastro.help.help import help
from visualastro.plotting.image_plots import (
    imshow,
    plot_spectral_cube
)
from visualastro.plotting.plot_utils import (
    add_colorbar,
    add_contours,
    add_subplot,
    close,
    ellipse_patch,
    inline,
    interactive,
    lighten_color,
    make_plot_grid,
    nanpercentile_limits,
    plot_circles,
    plot_ellipses,
    plot_hlines,
    plot_interactive_ellipse,
    plot_points,
    plot_vlines,
    return_imshow_norm,
    get_stylepath,
    sample_cmap,
    set_axis_labels,
    set_axis_limits,
    set_plot_colors,
    set_vmin_vmax,
    spectral_axis_label,
    style
)
from visualastro.plotting.plots import (
    plot_density_histogram,
    plot_histogram,
    plot_lines,
    plot_scatter,
    scatter3D
)
from visualastro.plotting.spectra_plots import (
    extract_cube_pixel_spectra,
    extract_cube_spectra,
    fit_gaussian_2_spec,
    plot_combine_spectrum,
    plot_extracted_pixel_map,
    plot_spectrum
)
from visualastro.plotting.visual_plots import ax
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

def _register_fonts():
    """
    Register additional fonts into matplotlib.
    To add more fonts, simply add a folder to
    VisualAstro/src/visualastro/stylelib/Fonts
    with .ttf or .otf files.
    """
    from pathlib import Path
    import warnings
    import matplotlib.font_manager as fm

    package_dir = Path(__file__).parent
    fonts_dir = package_dir / 'stylelib' / 'Fonts'

    if not fonts_dir.exists():
        return

    font_files = list(fonts_dir.rglob('*.ttf')) + list(fonts_dir.rglob('*.otf'))

    for font_file in font_files:
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception as e:
            warnings.warn(
                f'[visualastro] Could not register font {font_file.name}: {e}'
            )

_register_fonts()
