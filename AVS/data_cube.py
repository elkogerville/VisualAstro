import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

def load_fits(file_path):
    '''
    loads fits data from HARPS spectrograph and outputs the header and data
    Parameters
    ----------
    file_path: string
        filename including path to a fits file
    Returns
    -------
    data: np.ndarray[np.float64]
        NxM array of intensities
    header:
        header of fits file
    '''
    return fits.getdata(file_path)

def load_data_cube(path):
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
    fits_files = sorted(glob.glob(path))
    # allocate ixMxN data cube array
    i = len(fits_files)
    data = load_fits(fits_files[0])
    data_cube = np.zeros((i, data.shape[0], data.shape[1]))
    # save first file to data arrays
    data_cube[0,:,:] = data
    # loop through each array in data list and store in data cube
    for i in tqdm(range(1, len(fits_files))):
        data = load_fits(fits_files[i])
        data_cube[i,:,:] = data

    return data_cube

def imshow(data, cmap='turbo', style='astro', vmin=None, vmax=None,
           percentile=[3,99.5], circles=None, plot_boolean=False):
    if plot_boolean:
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanpercentile(data, percentile[0]) if vmin is None else vmin
        vmax = np.nanpercentile(data, percentile[1]) if vmax is None else vmax
    with plt.style.context(style + '.mplstyle'):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(data.T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        if circles is not None:
            circle_colors = ['r', 'mediumvioletred', 'magenta']
            for i, circle in enumerate(circles):
                x, y, r = circle
                circle = Circle((x, y), radius=r, fill=False, linewidth=2,
                                color=circle_colors[i%len(circle_colors)])
                ax.add_patch(circle)

        plt.show()
