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
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from matplotlib.patches import Ellipse
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm
from .data_cube_utils import get_spectral_slice_value, slice_cube
from .io import get_dtype, get_errors
from .numerical_utils import (
    check_units_consistency, convert_units,
    get_data, shift_by_radial_vel
)
from .plot_utils import (
    add_colorbar, plot_ellipses, plot_interactive_ellipse,
    return_imshow_norm, set_unit_labels, set_vmin_vmax, va_config,
)
from .visual_classes import DataCube

warnings.filterwarnings('ignore', category=AstropyWarning)


# Datacube I/O Functions
# ––––––––––––––––––––––
def load_data_cube(filepath, error=True, hdu=va_config.hdu_idx,
                   dtype=None, print_info=va_config.print_info,
                   transpose=va_config.transpose):
    '''
    Load a sequence of FITS files into a 3D data cube.
    This function searches for all FITS files matching a given path pattern,
    loads them into a NumPy array of shape (T, M, N), and bundles the data
    and headers into a `DataCube` object.
    Parameters
    ––––––––––
    filepath : str
        Path pattern to FITS files. Wildcards are supported.
        Example: 'Spectro-Module/raw/HARPS*.fits'
    dtype : numpy.dtype, optional, default=None
        Data type for the loaded FITS data. If None, will use
        the dtype of the provided data, promoting integer or
        unsigned to `np.float64`.
    print_info : bool, optional, default=True
        If True, print summary information about the loaded cube.
    transpose : bool, optional, default=False
        If True, transpose each 2D image before stacking into the cube.
    Returns
    –––––––
    cube : DataCube
        A DataCube object containing:
        - 'cube.data' : np.ndarray of shape (T, M, N)
        - 'cube.headers' : list of astropy.io.fits.Header objects
    Example
    –––––––
    Search for all fits files starting with 'HARPS' with .fits extention and load them.
        filepath = 'Spectro-Module/raw/HARPS.*.fits'
    '''
    # searches for all files within a directory
    fits_files = sorted(glob.glob(filepath))
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

    if transpose:
        data = data.T
        if err is not None:
            err = err.T

    # Preallocate data cube and headers
    datacube = np.zeros((n_files, data.shape[0], data.shape[1]), dtype=dt)
    datacube[0] = data.astype(dt)
    headers = [None] * n_files
    headers[0] = header
    # preallocate error array if needed and error exists
    error_array = None
    if error and err is not None:
        error_array = np.zeros_like(datacube, dtype=dt)
        error_array[0] = err.astype(dt)

    # loop through remaining files
    for i, file in enumerate(tqdm(fits_files[1:], desc="Loading FITS")):
        with fits.open(file) as hdul:
            data = hdul[hdu].data
            headers[i+1] = hdul[hdu].header
            err = get_errors(hdul, dt)
        if transpose:
            data = data.T
            if err is not None:
                err = err.T
        datacube[i+1] = data.astype(dt)
        if error_array is not None and err is not None:
            error_array[i+1] = err.astype(dt)

    return DataCube(datacube, headers, error_array)


def load_spectral_cube(filepath, hdu, error=True, header=True, dtype=None, print_info=va_config.print_info):
    '''
    Load a spectral cube from a FITS file, optionally including errors and header.
    Parameters
    ––––––––––
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
    print_info : bool, optional, default=False
        If True, print FITS file info to the console.
    Returns
    –––––––
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

    return DataCube(spectral_cube, headers=hdr, errors=error_array)


# Cube Plotting Functions
# –––––––––––––––––––––––
def plot_spectral_cube(cubes, idx, ax, vmin=va_config.vmin, vmax=va_config.vmax,
                       percentile=va_config.percentile, norm=va_config.norm,
                       radial_vel=None, unit=None, cmap=va_config.cmap, **kwargs):
    '''
    Plot a single spectral slice from one or more spectral cubes.
    Parameters
    ––––––––––
    cubes : DataCube, SpectralCube, or list of such
        One or more spectral cubes to plot. All cubes should have consistent units.
    idx : int
        Index along the spectral axis corresponding to the slice to plot.
    ax : matplotlib.axes.Axes
        The axes on which to draw the slice.
    vmin, vmax : float, optional, default=None
        Minimum and maximum values for image scaling. Overrides percentile if provided.
    percentile : list of two floats, default=[3, 99.5]
        Percentile values for automatic scaling if vmin/vmax are not specified.
    norm : str or None, default='asinh'
        Normalization type for `imshow`. Use None for linear scaling.
    radial_vel : float or astropy.units.Quantity, optional, default=None
        Radial velocity to shift spectral axis to the rest frame.
    unit : astropy.units.Unit or str, optional, default=None
        Desired spectral axis unit for labeling.
    cmap : str, list, or tuple, default='turbo'
        Colormap(s) to use for plotting.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `title` : bool, default=False
            If True, display spectral slice label as plot title.
        - `emission_line` : str or None, default=None
            Optional emission line label to display instead of slice value.
        - `text_loc` : list of float, default=[0.03, 0.03]
            Relative axes coordinates for overlay text placement.
        - `text_color` : str, default='k'
            Color of overlay text.
        - `colorbar` : bool, default=True
            Whether to add a colorbar.
        - `cbar_width` : float, default=0.03
            Width of the colorbar.
        - `cbar_pad` : float, default=0.015
            Padding between axes and colorbar.
        - `clabel` : str, bool, or None, default=True
            Label for colorbar. If True, automatically generate from cube unit.
        - `xlabel`, `ylabel` : str, default='Right Ascension', 'Declination'
            Axes labels.
        - `spectral_label` : bool, default=True
            Whether to draw spectral slice value as a label.
        - `highlight` : bool, default=True
            Whether to highlight interactive ellipse if plotted.
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
    Notes
    –––––
    - If multiple cubes are provided, they are overplotted in sequence.
    '''
    # check cube units match and ensure cubes is iterable
    cubes = check_units_consistency(cubes)
    # –––– Kwargs ––––
    # labels
    title = kwargs.get('title', False)
    emission_line = kwargs.get('emission_line', None)
    text_loc = kwargs.get('text_loc', va_config.text_loc)
    text_color = kwargs.get('text_color', va_config.text_color)
    colorbar = kwargs.get('colorbar', va_config.cbar)
    cbar_width = kwargs.get('cbar_width', va_config.cbar_width)
    cbar_pad = kwargs.get('cbar_pad', va_config.cbar_pad)
    clabel = kwargs.get('clabel', va_config.clabel)
    xlabel = kwargs.get('xlabel', va_config.right_ascension)
    ylabel = kwargs.get('ylabel', va_config.declination)
    draw_spectral_label = kwargs.get('spectral_label', True)
    highlight = kwargs.get('highlight', True)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = (
        True if ellipses is not None else kwargs.get('plot_ellipse', False)
    )
    _, X, Y = get_data(cubes[0]).shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    angle = kwargs.get('angle', None)

    for cube in cubes:
        # extract data component
        cube = get_data(cube)

        # return data cube slices
        cube_slice = slice_cube(cube, idx)
        data = cube_slice.value

        # compute imshow stretch
        vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
        cube_norm = return_imshow_norm(vmin, vmax, norm)

        # imshow data
        if norm is None:
            im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            im = ax.imshow(data, origin='lower', cmap=cmap, norm=cube_norm)

    # determine unit of colorbar
    cbar_unit = set_unit_labels(cube.unit)
    # set colorbar label
    if clabel is True:
        clabel = f'${cbar_unit}$' if cbar_unit is not None else None
    # set colorbar
    if colorbar:
        add_colorbar(im, ax, cbar_width, cbar_pad, clabel)

    # plot ellipses
    if plot_ellipse:
        # plot Ellipse objects
        if ellipses is not None:
            plot_ellipses(ellipses, ax)
        # plot ellipse with angle
        elif angle is not None:
            e = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=False)
            ax.add_patch(e)
        # plot default/interactive ellipse
        else:
            plot_interactive_ellipse(center, w, h, ax, text_loc, text_color, highlight)
            draw_spectral_label = False

    # plot wavelength/frequency of current spectral slice, and emission line
    if draw_spectral_label:
        # compute spectral axis value of slice for label
        spectral_axis = convert_units(cube.spectral_axis, unit)
        spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)
        spectral_value = get_spectral_slice_value(spectral_axis, idx)
        unit_label = set_unit_labels(spectral_axis.unit)

        # lambda for wavelength, f for frequency
        spectral_type = r'\lambda = ' if spectral_axis.unit.physical_type == 'length' else r'f = '
        # replace spectral type with emission line if provided
        if emission_line is None:
            slice_label = fr'${spectral_type}{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
        else:
            # replace spaces with latex format
            emission_line = emission_line.replace(' ', r'\ ')
            slice_label = fr'$\mathrm{{{emission_line}}}\,{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
        # display label as either a title or text in figure
        if title:
            ax.set_title(slice_label, color=text_color, loc='center')
        else:
            ax.text(text_loc[0], text_loc[1], slice_label,
                    transform=ax.transAxes, color=text_color)

    # set axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.coords['dec'].set_ticklabel(rotation=90)
