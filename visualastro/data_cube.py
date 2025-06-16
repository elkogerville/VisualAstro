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
    return_cube_slice, return_imshow_norm, return_spectral_axis_idx, return_stylename,
    save_figure_2_disk, set_spectral_axis, set_unit_labels, set_vmin_vmax
)

warnings.filterwarnings('ignore', category=AstropyWarning)

# def load_fits_as_dict(filepath, data_idx=1, header_idx=0):
#     '''
#     loads fits data from HARPS spectrograph and outputs the header and data
#     Parameters
#     ----------
#     file_path: string
#         filename including path to a fits file
#     Returns
#     -------
#     data: np.ndarray[np.float64]
#         NxM array of intensities
#     header:
#         header of fits file
#     '''
#     with fits.open(filepath) as hdu:
#         header = hdu[header_idx].header
#         data = hdu[data_idx].data.astype(np.float64)
#     #data =

#     data, header = fits.getdata(filepath, header=True)

#     return data, header

def load_data_cube(filepath, header=True, dtype=np.float64, print_info=True):
    '''
    searches for all data fits files in a directory and loads them into a numpy 3D data cube
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
    result = [spectral_cube]

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

    return tuple(result)

def plot_spectral_cube(cubes, idx, vmin=None, vmax=None, percentile=[3,99.5], norm='asinh',
                       radial_vel=None, unit=None, plot_ellipse=False, center=[None,None],
                       w=None, h=None, angle=None, ellipses=None, emission_line=None,
                       cmap='turbo', style='astro', title=False, label_color='k', savefig=False,
                       dpi=600, figsize=(6,6)):
    c = 299792.458
    # define wcs figure axes
    cubes = [cubes] if isinstance(cubes, SpectralCube) else cubes
    style = return_stylename(style)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        wcs2d = cubes[0].wcs.celestial
        ax = fig.add_subplot(111, projection=wcs2d)
        if style.split('/')[-1] == 'minimal.mplstyle':
            ax.coords['ra'].set_ticks_position('bl')
            ax.coords['dec'].set_ticks_position('bl')

        for cube in cubes:
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
            spectral_axis = set_spectral_axis(cube, unit)
            unit_label = set_unit_labels(spectral_axis.unit)
            spectral_value = return_spectral_axis_idx(spectral_axis, idx)
            if radial_vel is not None:
                if spectral_axis.unit.is_equivalent(u.Hz):
                    spectral_axis *= (1 + radial_vel / c)
                else:
                    spectral_axis /= (1 + radial_vel / c)

        if (plot_ellipse and ellipses is None):
            text = ax.text(0.5, 0.5, '', size='small', color=label_color)
        else:
            spectral_type = r'\lambda = ' if spectral_axis.unit.physical_type == 'length' else r'f = '
            if emission_line is None:
                slice_label = fr'${spectral_type}{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
            else:
                emission_line = emission_line.replace(' ', r'\ ')
                slice_label = fr'$\mathrm{{{emission_line}}}\,{spectral_value:0.2f}\,\mathrm{{{unit_label}}}$'
            if title:
                plt.title(slice_label, color=label_color)
            else:
                plt.text(0.03, 0.03, slice_label, transform=ax.transAxes, color=label_color)

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
            if isinstance(ellipses, list):
                for ellipse in ellipses:
                    ax.add_patch(copy_ellipse(ellipse))
            else:
                _, X, Y = cube.shape
                x_center = center[0] if center[0] is not None else ellipses.center[0] if ellipses is not None else X//2
                y_center = center[1] if center[1] is not None else ellipses.center[1] if ellipses is not None else Y//2
                w = w if w is not None else ellipses.width if ellipses is not None else X//5
                h = h if h is not None else ellipses.height if ellipses is not None else Y//5
                angle = angle if angle is not None else ellipses.angle if ellipses is not None else None
                e = Ellipse(xy=(x_center, y_center), width=w, height=h, angle=angle, fill=False)

                if angle is not None:
                    ax.add_patch(e)
                else:
                    ellipse_region = EllipsePixelRegion(center=PixCoord(x=x_center, y=y_center), width=w, height=h)
                    selector = ellipse_region.as_mpl_selector(ax, callback=update_region)

        cax = fig.add_axes([ax.get_position().x1+0.02, ax.get_position().y0,
                            0.03, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax, pad=0.04)
        cbar.ax.tick_params(which='both', direction='out')
        cbar.set_label(fr'${set_unit_labels(cube.unit)}$')

        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        ax.coords['dec'].set_ticklabel(rotation=90)

        if savefig:
            save_figure_2_disk(dpi)

        plt.show()

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
