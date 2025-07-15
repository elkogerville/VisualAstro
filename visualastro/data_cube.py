import glob
import warnings
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from regions import PixCoord, EllipsePixelRegion, EllipseAnnulusPixelRegion
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from .plot_utils import (
    return_cube_slice, return_imshow_norm, return_spectral_axis_idx,
    set_spectral_axis, set_unit_labels, set_vmin_vmax, shift_by_radial_vel
)

warnings.filterwarnings('ignore', category=AstropyWarning)


def load_fits(filepath, header=True, print_info=True):
    if print_info:
        with fits.open(filepath) as hdul:
            hdul.info()
    data, fits_header = fits.getdata(filepath, header=True)
    result = [data, fits_header] if header else data

    return result

def load_data_cube(filepath, header=True, dtype=np.float64, print_info=True):
    '''
    Searches for all data fits files in a directory and loads them into a numpy 3D data cube
    Parameters
    ----------
    path: string
        path to directory with fits files, will search for all files of specified extension
    Returns
    -------
    data_cube: np.ndarray[np.float64]
        ixMxN array where each i index corresponds to a different fits data file
        each data file is transposed into a MxN matrix
    header_list: list[astropy.io.fits.Header]
        list of each corresponding fits header file
    Example
    -------
    search for all fits files starting with 'HARPS' with .fits extention
        path = 'Spectro-Module/raw/HARPS.*.fits'
    '''
    # searches for all files within a directory
    fits_files = sorted(glob.glob(filepath))
    # allocate ixMxN data cube array and header array
    i = len(fits_files)
    headers = np.empty(i, dtype=object)
    data, headers[0] = fits.getdata(fits_files[0], header=True)
    data_cube = np.zeros((i, data.shape[0], data.shape[1]), dtype=dtype)
    # save first file to data arrays
    data_cube[0] = data.astype(dtype)
    # loop through each array in data list and store in data cube
    for i in tqdm(range(1, len(fits_files))):
        data, headers[i] = fits.getdata(fits_files[i], header=True)
        data_cube[i] = data.astype(dtype)

    data_dict = {}
    data_dict['data'] = data_cube
    data_dict['header'] = headers
    if print_info:
        with fits.open(fits_files[0]) as hdul:
            hdul.info()
    return data_dict

def load_spectral_cube(filepath, hdu, error=True, header=True, error_key='ERR', print_info=False):

    spectral_cube = SpectralCube.read(filepath, hdu=hdu)

    if error or header:
        result = [spectral_cube]
    else:
        result = spectral_cube

    with fits.open(filepath) as hdul:

        if print_info:
            print( hdul.info() )

        if error:
            if error_key in hdul:
                result.append( hdul[error_key].data )
            else:
                raise KeyError(f"HDU '{error_key}' not found in file")

        if header:
            result.append( hdul[hdu].header )

    return result

def plot_spectral_cube(cube, idx, ax, vmin=None, vmax=None, percentile=[3,99.5],
                        norm='asinh', radial_vel=None, unit=None, **kwargs):
    # plot params
    cmap = kwargs.get('cmap', 'turbo')
    # labels
    title = kwargs.get('title', False)
    emission_line = kwargs.get('emission_line', None)
    text_loc = kwargs.get('text_loc', [0.03, 0.03])
    text_color = kwargs.get('text_color', 'k')
    cbar_width = kwargs.get('cbar_width', 0.03)
    cbar_pad = kwargs.get('cbar_pad', 0.015)
    xlabel = kwargs.get('xlabel', 'Right Ascension')
    ylabel = kwargs.get('ylabel', 'Declination')
    # plot ellipse
    plot_ellipse = kwargs.get('plot_ellipse', False)
    ellipses = kwargs.get('ellipses', None)
    _, X, Y = cube.shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    angle = kwargs.get('angle', None)

    fig = ax.figure

    # return data cube slices
    slice_data = return_cube_slice(cube, idx)
    data = slice_data.value

    # compute imshow stretch
    vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
    cube_norm = return_imshow_norm(vmin, vmax, norm)

    if norm is None:
        im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.imshow(data, origin='lower', cmap=cmap, norm=cube_norm)

    cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                        cbar_width, ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax, pad=0.02)
    cbar.ax.tick_params(which='both', direction='out')
    cbar.set_label(fr'${set_unit_labels(cube.unit)}$')

    spectral_axis = set_spectral_axis(cube, unit)
    spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)
    spectral_value = return_spectral_axis_idx(spectral_axis, idx)
    unit_label = set_unit_labels(spectral_axis.unit)

    if (plot_ellipse and ellipses is None):
        text = ax.text(0.5, 0.5, '', size='small', color=text_color)
    else:
        # lambda for wavelength, f for frequency
        spectral_type = r'\lambda = ' if spectral_axis.unit.physical_type == 'length' else r'f = '

        if emission_line is None:
            slice_label = fr'${spectral_type}{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
        else:
            emission_line = emission_line.replace(' ', r'\ ')
            slice_label = fr'$\mathrm{{{emission_line}}}\,{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
        if title:
            plt.title(slice_label, color=text_color)
        else:
            plt.text(text_loc[0], text_loc[1], slice_label, transform=ax.transAxes, color=text_color)

    def update_region(region):
        x_center = region.center.x
        y_center = region.center.y
        width = region.width
        height = region.height
        major = max(width, height)
        minor = min(width, height)

        # Update text display (positioned in data coordinates)
        text.set_text(
            f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
            f'Major: {major:.1f}\n'
            f'Minor: {minor:.1f}\n'
        )

    if plot_ellipse:
        if ellipses is not None:
            ellipses = ellipses if isinstance(ellipses, list) else [ellipses]
            for ellipse in ellipses:
                ax.add_patch(copy_ellipse(ellipse))
        else:
            if angle is not None:
                e = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=False)
                ax.add_patch(e)
            else:
                ellipse_region = EllipsePixelRegion(center=PixCoord(x=center[0], y=center[1]), width=w, height=h)
                selector = ellipse_region.as_mpl_selector(ax, callback=update_region)
                ax._ellipse_selector = selector

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.coords['dec'].set_ticklabel(rotation=90)

def header_2_array(cube, key):
    headers = cube['header']
    array = []
    for i in range(len(headers)):
        array.append(headers[i][key])

    return np.asarray(array)

def write_cube_2_fits(cube, filename, overwrite=False):
    N_frames, N, M = cube.shape
    print(f'Writing {N_frames} fits files to {filename}_reduced_i.fits')
    for i in tqdm(range(N_frames)):
        output_name = filename + f'_reduced_{i}.fits'
        fits.writeto(output_name, cube[i], overwrite=overwrite)

def mask_cube(cube, composite_mask=False, center=None, w=None, h=None, angle=0, region='annulus',
              tolerance=2, ellipse_region=None, line_points=None, outer=False, upper=True, return_full=True):

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
        if outer:
            region_mask = ~region_mask
        mask &= region_mask

    if isinstance(cube, np.ndarray):
        if return_full:
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
            if return_full:
                region_line_cube = np.full_like(cube, np.nan)
                region_line_cube[:, region_line_mask] = cube[:, region_line_mask]
            else:
                region_line_cube = cube[:, region_line_mask]

        else:
            region_line_cube = cube.with_mask(region_line_mask)
        return region_line_cube, region_cube

    else:
        return region_cube

def return_ellipse_region(center, w, h, angle=0):
    ellipse = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=False)

    return ellipse

def copy_ellipse(ellipse):
    return Ellipse(
        xy=ellipse.center,
        width=ellipse.width,
        height=ellipse.height,
        angle=ellipse.angle,
        edgecolor=ellipse.get_edgecolor(),
        facecolor=ellipse.get_facecolor(),
        lw=ellipse.get_linewidth(),
        ls=ellipse.get_linestyle(),
        alpha=ellipse.get_alpha()
    )

def compute_line(points):
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    b = points[0][1] - m*points[0][0]

    return m, b

def compute_cube_percentile(cube, slice_idx, vmin, vmax):

    data = return_cube_slice(cube, slice_idx)
    vmin = np.nanpercentile(data.value, vmin)
    vmax = np.nanpercentile(data.value, vmax)

    return vmin, vmax
