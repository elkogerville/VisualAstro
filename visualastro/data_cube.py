import glob
import warnings
from astropy.io import fits
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from matplotlib.patches import Ellipse
import numpy as np
from regions import PixCoord, EllipsePixelRegion, EllipseAnnulusPixelRegion
from spectral_cube import SpectralCube
from tqdm import tqdm
from .data_cube_utils import compute_line, get_spectral_slice_value, slice_cube
from .io import get_dtype, get_errors
from .numerical_utils import (
    check_units_consistency, convert_units,
    get_data, get_units, shift_by_radial_vel
)
from .plot_utils import (
    add_colorbar, plot_ellipses, plot_interactive_ellipse,
    return_imshow_norm, set_unit_labels, set_vmin_vmax,
)
from .visual_classes import DataCube, FitsFile

warnings.filterwarnings('ignore', category=AstropyWarning)


def load_data_cube(filepath, error=True, hdu=0, dtype=None,
                   print_info=True, transpose=False):
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


def load_spectral_cube(filepath, hdu, error=True, header=True, dtype=None, print_info=False):
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

def plot_spectral_cube(cubes, idx, ax, vmin=None, vmax=None, percentile=[3,99.5],
                        norm='asinh', radial_vel=None, unit=None, cmap='turbo', **kwargs):
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

        title : bool, default=False
            If True, display spectral slice label as plot title.
        emission_line : str or None, default=None
            Optional emission line label to display instead of slice value.
        text_loc : list of float, default=[0.03, 0.03]
            Relative axes coordinates for overlay text placement.
        text_color : str, default='k'
            Color of overlay text.
        colorbar : bool, default=True
            Whether to add a colorbar.
        cbar_width : float, default=0.03
            Width of the colorbar.
        cbar_pad : float, default=0.015
            Padding between axes and colorbar.
        clabel : str, bool, or None, default=True
            Label for colorbar. If True, automatically generate from cube unit.
        xlabel, ylabel : str, default='Right Ascension', 'Declination'
            Axes labels.
        spectral_label : bool, default=True
            Whether to draw spectral slice value as a label.
        highlight : bool, default=True
            Whether to highlight interactive ellipse if plotted.
        ellipses : list or None, default=None
            Ellipse objects to overlay on the image.
        plot_ellipse : bool, default=False
            If True, plot a default or interactive ellipse.
        center : list of two ints, default=[Nx//2, Ny//2]
            Center of default ellipse.
        w, h : float, default=X//5, Y//5
            Width and height of default ellipse.
        angle : float or None, default=None
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
    text_loc = kwargs.get('text_loc', [0.03, 0.03])
    text_color = kwargs.get('text_color', 'k')
    colorbar = kwargs.get('colorbar', True)
    cbar_width = kwargs.get('cbar_width', 0.03)
    cbar_pad = kwargs.get('cbar_pad', 0.015)
    clabel = kwargs.get('clabel', True)
    xlabel = kwargs.get('xlabel', 'Right Ascension')
    ylabel = kwargs.get('ylabel', 'Declination')
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


def mask_image(image, ellipse_region=None, region=None,
               line_points=None, invert_region=False, above_line=True,
               preserve_shape=True, existing_mask=None, **kwargs):
    '''
    Mask an image with modular filters.
    Supports applying an elliptical or annular region mask, an optional
    line cut (upper or lower half-plane), and combining with an existing mask.
    Parameters
    ––––––––––
    image : array-like, DataCube, FitsFile, or SpectralCube
        Input image or cube. If higher-dimensional, the mask is applied
        to the last two axes.
    ellipse_region : `EllipsePixelRegion` or `EllipseAnnulusPixelRegion`, optional, default=None
        Region object specifying an ellipse or annulus.
    region : str {'annulus', 'ellipse'}, optional, default=None
        Type of region to apply. Ignored if `ellipse_region` is provided.
    line_points : array-like, shape (2, 2), optional, default=None
        Two (x, y) points defining a line for masking above/below.
        Ex: [[0,2], [20,10]]
    invert_region : bool, default=False
        If True, invert the region mask.
    above_line : bool, default=True
        If True, keep the region above the line. If False, keep below.
    preserve_shape : bool, default=True
        If True, return an array of the same shape with masked values set to NaN.
        If False, return only the unmasked pixels.
    existing_mask : ndarray of bool, optional, default=None
        An existing mask to combine (union) with the new mask.
    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - center : tuple of float, optional, default=None
            Center coordinates (x, y).
        - w : float, optional, default=None
            Width of ellipse.
        - h : float, optional, default=None
            Height of ellipse.
        - angle : float, optional, default=0
            Rotation angle in degrees.
        - tolerance : float, optional, default=2
            Tolerance for annulus inner/outer radii
    Returns
    –––––––
    masked_image : ndarray or SpectralCube
        Image with mask applied. Type matches input.
    masks : ndarray of bool or list
        If multiple masks are combined, returns a list containing the
        master mask followed by individual masks. Otherwise returns a single mask.
    '''
    # –––– Kwargs ––––
    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)
    tolerance = kwargs.get('tolerance', 2)

    # ensure working with array
    if isinstance(image, (DataCube, FitsFile)):
        image = image.data
    elif isinstance(image, (list, tuple)):
        image = np.asarray(image)

    # determine image shape
    N, M = image.shape[-2:]
    y, x = np.indices((N, M))
    # empty list to hold all masks
    masks = []

    # early return if just applying an existing mask
    if ellipse_region is None and region is None and line_points is None and existing_mask is not None:
        if existing_mask.shape != image.shape[-2:]:
            raise ValueError("existing_mask must have same shape as image")

        if isinstance(image, np.ndarray):
            if preserve_shape:
                masked_image = np.full_like(image, np.nan, dtype=float)
                masked_image[..., existing_mask] = image[..., existing_mask]
            else:
                masked_image = image[..., existing_mask]
        else:
            # if spectral cube or similar object
            masked_image = image.with_mask(existing_mask)

        return masked_image

    # –––– Region Mask ––––
    # if ellipse region is passed in use those values
    if ellipse_region is not None:
        center = ellipse_region.center
        a = ellipse_region.width / 2
        b = ellipse_region.height / 2
        angle = ellipse_region.angle if ellipse_region.angle is not None else 0
    # accept user defined center, w, and h values if used
    elif None not in (center, w, h):
        a = w / 2
        b = h / 2
    # stop program if attempting to plot a region without necessary data
    elif region is not None:
        raise ValueError("Either 'ellipse_region' or 'center', 'w', 'h' must be provided.")

    # construct region
    if region is not None:
        if region.lower() == 'annulus':
            region_obj = EllipseAnnulusPixelRegion(
                center=PixCoord(center[0], center[1]),
                inner_width=2*(a - tolerance),
                inner_height=2*(b - tolerance),
                outer_width=2*(a + tolerance),
                outer_height=2*(b + tolerance),
                angle=angle * u.deg
            )
        elif region.lower() == 'ellipse':
            region_obj = EllipsePixelRegion(
                center=PixCoord(center[0], center[1]),
                width=2*a,
                height=2*b,
                angle=angle * u.deg
            )
        else:
            raise ValueError("region must be 'annulus' or 'ellipse'")

        # filter by region mask
        region_mask = region_obj.to_mask(mode='center').to_image((N, M)).astype(bool)
        if invert_region:
            region_mask = ~region_mask
        masks.append(region_mask.copy())
    else:
        # empty mask if no region
        region_mask = np.ones((N, M), dtype=bool)

    # –––– Line Mask ––––
    if line_points is not None:
        # start from previous mask
        line_mask = region_mask.copy()
        # compute slope and intercept of line
        m, b_line = compute_line(line_points)
        # filter out points above/below line
        line_mask &= (y >= m*x + b_line) if above_line else (y <= m*x + b_line)
        # add line region to mask array
        masks.append(line_mask.copy())
    else:
        # empty mask if no region
        line_mask = region_mask.copy()

    # –––– Combine Masks ––––
    # start master mask with line_mask (or region if no line)
    mask = line_mask.copy()

    # union with existing mask if provided
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError("existing_mask must have the same shape as the image")
        mask |= existing_mask

    # –––– Apply Mask ––––
    # if numpy array:
    if isinstance(image, np.ndarray):
        if preserve_shape:
            masked_image = np.full_like(image, np.nan, dtype=float)
            masked_image[..., mask] = image[..., mask]
        else:
            masked_image = image[..., mask]
    # if spectral cube object
    else:
        masked_image = image.with_mask(mask)

    # ––––– Final Mask List –––––
    # Return master mask as first element
    masks = [mask] + masks if len(masks) > 1 else mask

    return masked_image, masks
