'''
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2025-10-20
Description:
    Datacube related visualization and masking functions.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - spectral_cube
    - tqdm
Module Structure:
    - Datacube I/O Functions
        Functions for loading datacubes into visualastro.
    - Cube Plotting Functions
        Functions for plotting datacubes
'''

import glob
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm
from .config import get_config_value, config, _default_flag
from .datacube import DataCube
from .data_cube_utils import stack_cube
from .io import get_dtype, get_errors
from .numerical_utils import (
    get_data, get_value, to_list
)
from .plot_utils import (
    add_colorbar, plot_ellipses,
    plot_interactive_ellipse,
    return_imshow_norm, set_vmin_vmax,
    spectral_axis_label,
)
from .units import ensure_common_unit, to_latex_unit
from .utils import _unwrap_if_single

warnings.filterwarnings('ignore', category=AstropyWarning)


# Datacube I/O Functions
# ----------------------
def load_data_cube(filepath, error=True, hdu=None,
                   dtype=None, print_info=None,
                   transpose=None, invert_wcs=None):
    '''
    Load a sequence of FITS files into a 3D data cube.

    This function searches for all FITS files matching a
    given path pattern, loads them into a NumPy array of shape
    (T, M, N), and bundles the data, headers, errors, and WCS
    into a `DataCube` object.

    Parameters
    ----------
    filepath : str
        Path pattern to FITS files. Wildcards are supported.
        Example: 'Spectro-Module/raw/HARPS*.fits'
    hdu : int or None, default=None
        Hdu extension to use. If None, uses the
        default value set by `config.hdu_idx`.
    dtype : numpy.dtype, optional, default=None
        Data type for the loaded FITS data. If None, will use
        the dtype of the provided data, promoting integer or
        unsigned to `np.float64`.
    print_info : bool or None, optional, default=None
        If True, print summary information about the loaded cube.
        If None, uses the default value set by `config.print_info`.
    transpose : bool or None, optional, default=None
        If True, transpose each 2D image before stacking into the cube.
        This will also transpose each error array if available and
        swap the WCS axes for consistency. The swapping of the WCS
        can be disabled by `config.invert_wcs_if_transpose`.
        If None, uses the default value set by `config.transpose`.
    invert_wcs : bool or None, optional, default=None
        If True, will perform a swapaxes(0,1) on the wcs if `transpose=True`.
        If None, uses the default value set by `config.invert_wcs_if_transpose`.

    Returns
    -------
    cube : DataCube
        A DataCube object containing:
        - `cube.data` : np.ndarray of shape (T, M, N)
        - `cube.header` : list of astropy.io.fits.Header objects
        - `cube.error` : np.ndarray of shape (T, M, N)
        - `cube.wcs` : list of `astropy.wcs.wcs.WCS`

    Examples
    --------
    Search for all fits files starting with 'HARPS' with .fits extention and load them:
        >>> filepath = 'Spectro-Module/raw/HARPS.*.fits'
    '''
    # get default config values
    hdu = get_config_value(hdu, 'hdu_idx')
    print_info = get_config_value(print_info, 'print_info')
    transpose = get_config_value(transpose, 'transpose')
    invert_wcs = get_config_value(invert_wcs, 'invert_wcs_if_transpose')

    # searches for all files within a directory
    fits_files = sorted(glob.glob(filepath))
    if not fits_files:
        raise FileNotFoundError(f'No FITS files found for pattern: {filepath}')
    # allocate ixMxN data cube array and header array
    n_files = len(fits_files)

    # load first file to determine shape, dtype, and check for errors
    with fits.open(fits_files[0]) as hdul:
        if print_info:
            hdul.info()

        data = hdul[hdu].data
        header = hdul[hdu].header
        err = get_errors(hdul, dtype)

    dt = get_dtype(data, dtype)

    try:
        wcs = WCS(header)
    except ValueError:
        wcs = None

    if transpose:
        data = data.T
        if wcs is not None and invert_wcs:
            wcs = wcs.swapaxes(0,1)
        if err is not None:
            err = err.T

    # Preallocate data cube and headers
    datacube = np.zeros((n_files, data.shape[0], data.shape[1]), dtype=dt)
    datacube[0] = data.astype(dt)
    headers = [None] * n_files
    headers[0] = header
    wcs_list = [None] * n_files
    wcs_list[0] = wcs
    # preallocate error array if needed and error exists
    error_array = None
    if error and err is not None:
        error_array = np.zeros_like(datacube, dtype=dt)
        error_array[0] = err.astype(dt)

    # loop through remaining files
    for i, file in enumerate(tqdm(fits_files[1:], desc='Loading FITS')):
        with fits.open(file) as hdul:
            data = hdul[hdu].data
            headers[i+1] = hdul[hdu].header
            err = get_errors(hdul, dt)
            try:
                wcs = WCS(headers[i+1])
            except ValueError:
                wcs = None

        if transpose:
            data = data.T
            if wcs is not None and invert_wcs:
                wcs = wcs.swapaxes(0,1)
            if err is not None:
                err = err.T
        datacube[i+1] = data.astype(dt)
        if error_array is not None and err is not None:
            error_array[i+1] = err.astype(dt)
        wcs_list[i+1] = wcs

    if all(w is None for w in wcs_list):
        wcs_list = None
    elif any(w is None for w in wcs_list):
        missing_indices = [i for i, w in enumerate(wcs_list) if w is None]
        raise ValueError(
            f'Inconsistent WCS: files at indices {missing_indices} have no WCS, '
            f'but other files do. Either all files must have WCS or none should.'
        )

    return DataCube(datacube, headers, error_array, wcs_list)


def load_spectral_cube(filepath, hdu, error=True,
                       header=True, dtype=None,
                       print_info=None):
    '''
    Load a spectral cube from a FITS file,
    optionally including errors and header.

    Parameters
    ----------
    filepath : str
        Path to the FITS file to read.
    hdu : int or str
        HDU index or name to read from the FITS file.
    error : bool, optional, default=True
        If True, load the associated error array using `get_errors`.
    header : bool, optional, default=True
        If True, load the HDU header.
    dtype : data-type, optional, default=None
        Desired NumPy dtype for the error array. If None, inferred
        from FITS data, promoting integer and unsigned to `np.float64`.
    print_info : bool or None, optional, default=None
        If True, print FITS file info to the console.
        If None, uses default value set by `config.print_info`.

    Returns
    -------
    DataCube
        A `DataCube` object containing:
        - data : SpectralCube
            Fits file data loaded as SpectralCube object.
        - header : astropy.io.fits.Header
            Fits file header.
        - error : np.ndarray
            Fits file error array.
        - value : np.ndarray
            Fits file data as np.ndarray.
        Ex:
        data = cube.data
    '''
    print_info = get_config_value(print_info, 'print_info')

    # load SpectralCube from filepath
    spectral_cube = SpectralCube.read(filepath, hdu=hdu)
    # initialize error and header objects
    error_array = None
    hdr = None
    # open fits file
    with fits.open(filepath) as hdul:
        # print fits info
        if print_info:
            hdul.info()
        # load error array
        if error:
            error_array = get_errors(hdul, dtype)
        # load header
        if header:
            hdr = hdul[hdu].header

    return DataCube(spectral_cube, header=hdr, error=error_array)


# Cube Plotting Functions
# -----------------------
def plot_spectral_cube(cubes, idx=None, ax=None, vmin=_default_flag,
                       vmax=_default_flag, norm=_default_flag,
                       percentile=_default_flag, stack_method=None,
                       radial_vel=None, spectral_unit=None, cmap=None,
                       mask_non_pos=None, wcs_grid=None, **kwargs):
    """
    Plot a single spectral slice from one or more spectral cubes.

    Parameters
    ----------
    cubes : DataCube, SpectralCube, or list of such
        One or more spectral cubes to plot. All cubes should have consistent units.
    idx : int or None, optional, default=None
        Index along the spectral axis corresponding to the slice to plot.
        If None, collapses the entire cube into a 2D map according
        to ``stack_method``.
    ax : matplotlib.axes.Axes or WCSAxes
        The axes on which to draw the slice.
    vmin : float or None, optional, default=`_default_flag`
        Lower limit for colormap scaling; overides `percentile[0]`.
        If None, values are determined from `percentile[0]`.
        If `_default_flag`, uses the default value in `config.vmin`.
    vmax : float or None, optional, default=`_default_flag`
        Upper limit for colormap scaling; overides `percentile[1]`.
        If None, values are determined from `percentile[1]`.
        If `_default_flag`, uses the default value in `config.vmax`.
    norm : str or None, optional, default=`_default_flag`
        Normalization algorithm for colormap scaling.
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'linear', 'none', or None -> no normalization applied
        If `_default_flag`, uses the default value in `config.norm`.
    percentile : list or tuple of two floats, or None, default=`_default_flag`
        Default percentile range used to determine `vmin` and `vmax`.
        If None, use no percentile stretch (as long as vmin/vmax are None).
        If `_default_flag`, uses default value from `config.percentile`.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'}, default=None
        Stacking method. If None, uses the default value set
        by ``config.stack_cube_method``.
    radial_vel : float or None, optional, default=None
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional. If None, uses the default
        value set by `config.radial_velocity`.
    spectral_unit : astropy.units.Unit or str, optional, default=None
        Desired spectral axis unit for labeling.
    cmap : str, list or tuple of str, or None, default=None
        Colormap(s) to use for plotting. If None,
        uses the default value set by `config.cmap`.
    mask_non_pos : bool or None, optional, default=None
        If True, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If None, uses the
        default value set by `config.mask_non_positive`.
    wcs_grid : bool or None, optional, default=None
        If True, display WCS grid ontop of plot. Requires
        using WCSAxes for `ax`. If None, uses the default
        value set by `config.wcs_grid`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `title` : bool, default=False
            If True, display spectral slice label as plot title.
        - `emission_line` : str or None, default=None
            Optional emission line label to display instead of slice value.
        - `text_loc` : list of float, default=`config.text_loc`
            Relative axes coordinates for overlay text placement.
        - `text_color` : str, default=`config.text_color`
            Color of overlay text.
        - `colorbar` : bool, default=`config.cbar`
            Whether to add a colorbar.
        - `cbar_width` : float, default=`config.cbar_width`
            Width of the colorbar.
        - `cbar_pad` : float, default=`config.cbar_pad`
            Padding between axes and colorbar.
        - `clabel` : str, bool, or None, default=`config.clabel`
            Label for colorbar. If True, automatically generate from cube unit.
        - `xlabel` : str, default=`config.right_ascension`
            X axis label.
        - `ylabel` : str, default=`config.declination`
            Y axis label.
        - `spectral_label` : bool, default=True
            Whether to draw spectral slice value as a label.
        - `highlight` : bool, optional, default=`config.highlight`
            Whether to highlight interactive ellipse or wavelength label if plotted.
        - `mask_out_val` : float, optional, default=`config.mask_out_value`
            Value to use when masking out non-positive values.
            Ex: np.nan, 1e-6, np.inf
        - `ellipses` : list or None, default=None
            Ellipse objects to overlay on the image.
        - `plot_ellipse` : bool, default=False
            If True, plot a default or interactive ellipse.
        - `center` : list of two ints, default=[Nx//2, Ny//2]
            Center of default ellipse.
        - `w`, `h` : float, default=X//5, Y//5
            Width and height of default ellipse.
        - `angle` : float or None, default=None
            Angle of ellipse in degrees.

    Returns
    -------
    images : matplotlib.image.AxesImage or list of matplotlib.image.AxesImage
            Image object if a single array is provided, otherwise a list of image
            objects created by `ax.imshow`.

    Notes
    -----
    - If multiple cubes are provided, they are overplotted in sequence.
    """
    # check cube units match and ensure cubes is iterable
    cubes = to_list(cubes)
    ref_unit = ensure_common_unit(cubes)
    # ---- Kwargs ----
    # fig params
    rasterized = kwargs.get('rasterized', config.rasterized)
    as_title = kwargs.get('as_title', False)
    # labels
    emission_line = kwargs.pop('emission_line', None)
    text_loc = kwargs.get('text_loc', config.text_loc)
    text_color = kwargs.get('text_color', config.text_color)
    colorbar = kwargs.get('colorbar', config.cbar)
    cbar_width = kwargs.get('cbar_width', config.cbar_width)
    cbar_pad = kwargs.get('cbar_pad', config.cbar_pad)
    clabel = kwargs.get('clabel', config.clabel)
    xlabel = kwargs.get('xlabel', config.right_ascension)
    ylabel = kwargs.get('ylabel', config.declination)
    draw_spectral_label = kwargs.get('spectral_label', True)
    highlight = kwargs.get('highlight', config.highlight)
    # mask out value
    mask_out_val = kwargs.get('mask_out_val', config.mask_out_value)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = kwargs.get('plot_ellipse', False)
    _, X, Y = get_data(cubes[0]).shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    angle = kwargs.get('angle', None)

    # get default config values
    vmin = config.vmin if vmin is _default_flag else vmin
    vmax = config.vmax if vmax is _default_flag else vmax
    norm = config.norm if norm is _default_flag else norm
    percentile = config.percentile if percentile is _default_flag else percentile
    stack_method = get_config_value(stack_method, 'stack_cube_method')
    radial_vel = get_config_value(radial_vel, 'radial_velocity')
    cmap = get_config_value(cmap, 'cmap')
    mask_non_pos = get_config_value(mask_non_pos, 'mask_non_positive')
    wcs_grid = get_config_value(wcs_grid, 'wcs_grid')

    if not isinstance(ax, WCSAxes):
        raise ValueError(
            'ax must be a WCSAxes instance!'
        )

    images = []
    cmap = cmap if isinstance(cmap, (list, np.ndarray, tuple)) else [cmap]

    for i, cube in enumerate(cubes):
        cube = get_data(cube)
        if not isinstance(cube, SpectralCube):
            raise ValueError(
                'Input cubes must contain a SpectralCube! '
                'For non SpectralCube data, use imshow.'
            )

        # return data cube slices
        cube_slice = stack_cube(
            cube, idx=idx, method=stack_method, axis=0
        )
        data = get_value(cube_slice)

        if mask_non_pos:
            data = np.where(data > 0.0, data, mask_out_val)

        # compute imshow stretch
        vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
        cube_norm = return_imshow_norm(vmin, vmax, norm)

        # imshow data
        if norm is None:
            im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax,
                           cmap=cmap[i%len(cmap)], rasterized=rasterized)
        else:
            im = ax.imshow(data, origin='lower', cmap=cmap[i%len(cmap)],
                           norm=cube_norm, rasterized=rasterized)

        images.append(im)

    # determine unit of colorbar
    cbar_unit = to_latex_unit(ref_unit)
    # set colorbar label
    if clabel is True:
        clabel = cbar_unit if cbar_unit is not None else None
    # set colorbar
    if colorbar:
        add_colorbar(
            images[0], ax, cbar_width, cbar_pad, clabel, rasterized=rasterized
        )

    if ellipses is not None:
        plot_ellipses(ellipses, ax)

    if plot_ellipse:
        plot_interactive_ellipse(
            center, w, h, ax, text_loc,
            text_color, highlight,
            rotation_step=kwargs.get('rotation_step', 5)
        )
        draw_spectral_label = False

    # plot wavelength/frequency of current spectral slice, and emission line
    if draw_spectral_label:
        spectral_axis_label(
            cubes[0], idx, ax,
            ref_unit=spectral_unit,
            radial_vel=radial_vel,
            emission_line=emission_line,
            as_title=as_title,
            **kwargs
        )

    # set axes labels
    ax.coords['ra'].set_axislabel(xlabel)
    ax.coords['dec'].set_axislabel(ylabel)
    ax.coords['dec'].set_ticklabel(rotation=90)
    if wcs_grid:
        ax.coords.grid(True, color='white', ls='dotted')

    images = _unwrap_if_single(images)

    return images
