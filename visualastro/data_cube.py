import glob
import warnings
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from matplotlib.patches import Ellipse
import numpy as np
from regions import PixCoord, EllipsePixelRegion, EllipseAnnulusPixelRegion
from spectral_cube import SpectralCube
from tqdm import tqdm
from .data_cube_utils import (
    compute_line, extract_spectral_axis,
    get_spectral_slice_value, return_cube_slice
)
from .io import get_dtype, get_errors
from .numerical_utils import get_data, shift_by_radial_vel
from .plot_utils import (
    add_colorbar, plot_ellipses, plot_interactive_ellipse,
    return_imshow_norm, set_unit_labels, set_vmin_vmax,
)
from .visual_classes import DataCube

warnings.filterwarnings('ignore', category=AstropyWarning)

def load_data_cube(filepath, error=True, dtype=None,
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
    i = len(fits_files)
    headers = np.empty(i, dtype=object)
    data, headers[0] = fits.getdata(fits_files[0], header=True) # type: ignore
    dt = get_dtype(data, dtype)
    if transpose:
        data = data.T
    datacube = np.zeros((i, data.shape[0], data.shape[1]), dtype=dt)
    # save first file to data arrays
    datacube[0] = data.astype(dt)
    # loop through each array in data list and store in data cube
    for i in tqdm(range(1, len(fits_files))):
        data, headers[i] = fits.getdata(fits_files[i], header=True) # type: ignore
        if transpose:
            data = data.T
        datacube[i] = data.astype(dt)

    cube = DataCube(datacube, headers)

    if print_info:
        with fits.open(fits_files[0]) as hdul:
            hdul.info()

    return cube

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

def plot_spectral_cube(cube, idx, ax, vmin=None, vmax=None, percentile=[3,99.5],
                        norm='asinh', radial_vel=None, unit=None, cmap='turbo', **kwargs):

    cube = get_data(cube)
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
    _, X, Y = cube.shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    angle = kwargs.get('angle', None)

    # return data cube slices
    cube_slice = return_cube_slice(cube, idx)
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
    if clabel is True and cbar_unit is not None:
        clabel = f'${cbar_unit}$'
    # set colorbar
    if colorbar:
        add_colorbar(im, ax, cbar_width, cbar_pad, clabel)

    # plot ellipses
    if plot_ellipse:
        if ellipses is not None:
            plot_ellipses(ellipses, ax)
        elif angle is not None:
            e = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=False)
            ax.add_patch(e)
        else:
            plot_interactive_ellipse(center, w, h, ax, text_loc, text_color, highlight)
            draw_spectral_label = False

    # plot spectral label
    if draw_spectral_label:
        # compute spectral axis value of slice for label
        spectral_axis = extract_spectral_axis(cube, unit)
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











# def mask_image(image, ellipse_region=None, region='annulus', line_points=None,
#                invert_region=False, upper=True, preserve_shape=True, **kwargs):

#     center = kwargs.get('center', None)
#     w = kwargs.get('w', None)
#     h = kwargs.get('h', None)
#     angle = kwargs.get('angle', 0)
#     tolerance = kwargs.get('tolerance', 2)
#     existing_mask = kwargs.get('exsisting_mask', None)
#     if None in (center, w, h) and ellipse_region is None:
#         region = None

#     # determine image shape
#     N, M = image.shape
#     y, x = np.indices((N, M))
#     # construct empty boolean mask for image
#     mask = np.ones((N, M), dtype=bool)
#     # empty list to hold all masks
#     masks = []

#     # if ellipse region is passed in use those values
#     if ellipse_region is not None:
#         center = ellipse_region.center
#         a = ellipse_region.width/2
#         b = ellipse_region.height/2
#         angle = ellipse_region.angle if ellipse_region.angle is not None else 0
#     # accept user defined center, w, and h values if used
#     elif None not in (center, w, h):
#         a = w/2
#         b = h/2
#     # stop program if attempting to plot a region without necessary data
#     elif region is not None:
#         raise ValueError("Either 'ellipse_region' or 'center', 'w', 'h' must be provided.")

#     # filter by region mask
#     if region is not None:
#         if region == 'annulus':
#             region = EllipseAnnulusPixelRegion(
#                 center=PixCoord(center[0], center[1]),
#                 inner_width=2*(a - tolerance),
#                 inner_height=2*(b - tolerance),
#                 outer_width=2*(a + tolerance),
#                 outer_height=2*(b + tolerance),
#                 angle=angle * u.deg
#             )
#         elif region == 'ellipse':
#             region = EllipsePixelRegion(
#                 center=PixCoord(center[0], center[1]),
#                 width=2*a,
#                 height=2*b,
#                 angle=angle * u.deg
#             )
#         else:
#             raise ValueError("region must be 'annulus' or 'ellipse'")

#         # mask image
#         region_mask = region.to_mask(mode='center').to_image((N, M)).astype(bool)
#         if invert_region:
#             region_mask = ~region_mask
#         # add region mask to mask array
#         mask &= region_mask
#         masks.append(region_mask)

#     # filter by linear line
#     if line_points is not None:
#         # create copy of mask
#         region_line_mask = mask.copy()
#         # compute slope and intercept of line
#         m, b_line = compute_line(line_points)
#         # filter out points above/below line
#         line_mask = (y >= m*x + b_line) if upper else (y <= m*x + b_line)
#         # add line region to mask array
#         region_line_mask &= line_mask
#         masks.append(region_line_mask)
#         mask = region_line_mask

#     # if user passes an existing mask
#     if existing_mask is not None:
#         if existing_mask.shape != mask.shape:
#             raise ValueError('Existing_mask must have the same shape as the image')
#         mask |= existing_mask
#         masks.append(existing_mask.copy())

#     # apply mask to data
#     if isinstance(image, np.ndarray):
#         # if numpy array:
#         if preserve_shape:
#             masked_image = np.full_like(image, np.nan, dtype=float)
#             masked_image[..., mask] = image[..., mask]
#         else:
#             masked_image = image[..., mask]
#     # if spectral cube object
#     else:
#         masked_image = image.with_mask(mask)

#     masks = [mask] + masks if len(masks) > 1 else mask

#     return masked_image, masks

    # # filter by linear line
    # if line_points is not None:
    #     region_line_mask = mask.copy()
    #     m, b_line = compute_line(line_points)
    #     line_mask = (y >= m*x + b_line) if upper else (y <= m*x + b_line)
    #     region_line_mask &= line_mask
    #     if isinstance(image, np.ndarray):
    #         if preserve_shape:
    #             region_line_image = np.full_like(image, np.nan)
    #             region_line_image[..., region_line_mask] = image[..., region_line_mask]
    #         else:
    #             region_line_image = image[..., region_line_mask]

    #     else:
    #         region_line_image = image.with_mask(region_line_mask)
    #     return region_line_image, region_image

    # else:
    #     return region_image

def mask_image(image, ellipse_region=None, region='annulus', line_points=None,
               invert_region=False, upper=True, preserve_shape=True, **kwargs):
    '''
    Mask an image with modular filters:
    - Ellipse or annulus region
    - Line cut (upper/lower)
    - Existing mask (unioned)

    Returns:
        masked_image: image with mask applied
        masks: master mask + individual masks
    '''

    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)
    tolerance = kwargs.get('tolerance', 2)
    existing_mask = kwargs.get('existing_mask', None)

    # determine image shape
    N, M = image.shape
    y, x = np.indices((N, M))
    # empty list to hold all masks
    masks = []

    # ----- REGION MASK -----
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
        if region == 'annulus':
            region_obj = EllipseAnnulusPixelRegion(
                center=PixCoord(center[0], center[1]),
                inner_width=2*(a - tolerance),
                inner_height=2*(b - tolerance),
                outer_width=2*(a + tolerance),
                outer_height=2*(b + tolerance),
                angle=angle * u.deg
            )
        elif region == 'ellipse':
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
        region_mask = np.ones((N, M), dtype=bool)

    # ----- LINE MASK -----
    if line_points is not None:
        # start from region_mask only for this line module
        line_mask = region_mask.copy()
        # compute slope and intercept of line
        m, b_line = compute_line(line_points)
        # filter out points above/below line
        line_mask &= (y >= m*x + b_line) if upper else (y <= m*x + b_line)
        # add line region to mask array
        masks.append(line_mask.copy())
    else:
        line_mask = region_mask.copy()

    # ----- COMBINE MASKS -----
    # start master mask with line_mask (or region if no line)
    mask = line_mask.copy()

    # union with existing mask if provided
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError("existing_mask must have the same shape as the image")
        mask |= existing_mask
        masks.append(existing_mask.copy())

    # ----- APPLY MASK -----
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

    # ----- FINAL MASK LIST -----
    # Return master mask as first element
    masks = [mask] + masks if len(masks) > 1 else mask

    return masked_image, masks

def mask_cube(cube, ellipse_region=None, composite_mask=False, region='annulus',
              tolerance=2, line_points=None, invert=False, upper=True, preserve_shape=True, **kwargs):
    '''
    Function to mask a data cube using user defined regions
    Paramerters
    –––––––––––
    composite_mask: bool
        set to True if layering multiple masks ontop of cube


    '''
    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)

    cube = get_data(cube)

    _, N, M = cube.shape
    y, x = np.indices((N, M))

    mask = np.ones((N, M), dtype=bool) if not composite_mask else cube.mask.include().copy()

    # if ellipse region is passed in use those values
    if ellipse_region is not None:
        center = ellipse_region.center
        a = ellipse_region.width/2
        b = ellipse_region.height/2
        angle = ellipse_region.angle if ellipse_region.angle is not None else 0

    if region is not None:
        if region == 'annulus':
            region = EllipseAnnulusPixelRegion(
                center=PixCoord(center[0], center[1]),
                inner_width=2*(a - tolerance),
                inner_height=2*(b - tolerance),
                outer_width=2*(a + tolerance),
                outer_height=2*(b + tolerance),
                angle=angle * u.deg
            )
        elif region == 'ellipse':
            region = EllipsePixelRegion(
                center=PixCoord(center[0], center[1]),
                width=2*a,
                height=2*b,
                angle=angle * u.deg
            )
        else:
            raise ValueError()

        region_mask = region.to_mask(mode='center').to_image((N, M)).astype(bool)
        if invert:
            region_mask = ~region_mask
        mask &= region_mask

    if isinstance(cube, np.ndarray):
        if preserve_shape:
            region_cube = np.full_like(cube, np.nan)
            region_cube[:, mask] = cube[:, mask]
        else:
            region_cube = cube[:, mask]

    else:
        region_cube = cube.with_mask(mask)

    # apply line mask if provided
    if line_points is not None:
        region_line_mask = mask.copy()
        m, b_line = compute_line(line_points)
        line_mask = (y >= m*x + b_line) if upper else (y <= m*x + b_line)
        region_line_mask &= line_mask
        if isinstance(cube, np.ndarray):
            if preserve_shape:
                region_line_cube = np.full_like(cube, np.nan)
                region_line_cube[:, region_line_mask] = cube[:, region_line_mask]
            else:
                region_line_cube = cube[:, region_line_mask]

        else:
            region_line_cube = cube.with_mask(region_line_mask)
        return region_line_cube, region_cube

    else:
        return region_cube

# def mask_cube(cube, composite_mask=False, center=None, w=None, h=None, angle=0, region='annulus',
#               tolerance=2, ellipse_region=None, line_points=None, outer=False, upper=True, return_full=True):

#     _, N, M = cube.shape
#     y, x = np.indices((N, M))

#     mask = np.ones((N, M), dtype=bool) if not composite_mask else cube.mask.include().copy()

#     # if ellipse region is passed in use those values
#     if ellipse_region is not None:
#         center = ellipse_region.center
#         a = ellipse_region.width/2
#         b = ellipse_region.height/2
#         angle = ellipse_region.angle if ellipse_region.angle is not None else 0

#     if region is not None:
#         if region == 'annulus':
#             region = EllipseAnnulusPixelRegion(
#                 center=PixCoord(center[0], center[1]),
#                 inner_width=2*(a - tolerance),
#                 inner_height=2*(b - tolerance),
#                 outer_width=2*(a + tolerance),
#                 outer_height=2*(b + tolerance),
#                 angle=angle * u.deg
#             )
#         elif region == 'ellipse':
#             region = EllipsePixelRegion(
#                 center=PixCoord(center[0], center[1]),
#                 width=2*a,
#                 height=2*b,
#                 angle=angle * u.deg
#             )
#         else:
#             raise ValueError("region must be 'annulus' or 'ellipse'")

#         region_mask = region.to_mask(mode='center').to_image((N, M)).astype(bool)
#         if outer:
#             region_mask = ~region_mask
#         mask &= region_mask

#     if isinstance(cube, np.ndarray):
#         if return_full:
#             region_cube = np.full_like(cube, np.nan)
#             region_cube[:, mask] = cube[:, mask]
#         else:
#             region_cube = cube[:, mask]

#     else:
#         region_cube = cube.with_mask(mask)

#     # apply line mask if provided
#     if line_points is not None:
#         region_line_mask = mask.copy()
#         m, b_line = compute_line(line_points)
#         line_mask = (y >= m*x + b_line) if upper else (y <= m*x + b_line)
#         region_line_mask &= line_mask
#         if isinstance(cube, np.ndarray):
#             if return_full:
#                 region_line_cube = np.full_like(cube, np.nan)
#                 region_line_cube[:, region_line_mask] = cube[:, region_line_mask]
#             else:
#                 region_line_cube = cube[:, region_line_mask]

#         else:
#             region_line_cube = cube.with_mask(region_line_mask)
#         return region_line_cube, region_cube

#     else:
#         return region_cube
