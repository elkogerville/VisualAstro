import glob
import re
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def load_fits_as_dict(file_path):
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
    data, header = fits.getdata(file_path, header=True)

    return data, header

def load_data_cube(path, header=True):
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
    # allocate ixMxN data cube array and header array
    i = len(fits_files)
    headers = np.empty(i, dtype=object)
    data, headers[0] = fits.getdata(fits_files[0], header=True)
    data_cube = np.zeros((i, data.shape[0], data.shape[1]))
    # save first file to data arrays
    data_cube[0] = data
    # loop through each array in data list and store in data cube
    for i in tqdm(range(1, len(fits_files))):
            data_cube[i], headers[i] = fits.getdata(fits_files[0], header=True)
    data_dict = {}
    data_dict['data'] = data_cube
    data_dict['header'] = headers

    return data_dict
