from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .visual_classes import FitsFile

def load_fits(filepath, header=True, print_info=True,
              transpose=False, dtype=np.float64):
    '''
    Load a FITS file and return its data and optional header.
    Parameters
    ––––––––––
    filepath : str
        Path to the FITS file to load.
    header : bool, default=True
        If True, return the FITS header along with the data
        as a FitsFile object.
        If False, only the data is returned.
    print_info : bool, default=True
        If True, print HDU information using 'hdul.info()'.
    transpose : bool, default=False
        If True, transpose the data array before returning.
    dtype : data-type, default=np.float64
            Data type to convert the FITS data to.
    Returns
    –––––––
    FitsFile
        If header is True, returns an object containing:
        - data: 'np.ndarray' of the FITS data
        - header: 'astropy.io.fits.Header' if 'header=True', else 'None'
    data : np.ndarray
        If header is False, returns just the data component.
    '''
    # print fits file info
    if print_info:
        with fits.open(filepath) as hdul:
            hdul.info()
    # extract data and optionally the header from the file
    # if header is not requested, return None
    result = fits.getdata(filepath, header=header)
    data, fits_header = result if isinstance(result, tuple) else (result, None)

    data = data.astype(dtype, copy=False)

    if transpose:
        data = data.T
    if header:
        return FitsFile(data, fits_header)
    else:
        return data

def save_figure_2_disk(dpi=600):
    '''
    Saves current figure to disk as a pdf, png, or svg,
    and prompts user for a filename and format.
    Parameters
    ––––––––––
    dpi: float or int
        Resolution in dots per inch.
    '''
    allowed_formats = {'pdf', 'png', 'svg'}
    # prompt user for filename, and extract extension
    filename = input('Input filename for image (ex: myimage.pdf): ').strip()
    basename, *extension = filename.rsplit('.', 1)
    # if extension exists, and is allowed, extract extension from list
    if extension and extension[0].lower() in allowed_formats:
        extension = extension[0]
    # else prompt user to input a valid extension
    else:
        extension = ''
        while extension not in allowed_formats:
            extension = input(
                f'Please choose a format from ({", ".join(allowed_formats)}): '
            ).strip().lower()
    # construct complete filename
    filename = f'{basename}.{extension}'

    # save figure
    plt.savefig(filename, format=extension, bbox_inches='tight', dpi=dpi)

def write_cube_2_fits(cube, filename, overwrite=False):
    '''
    Write a 3D data cube to a series of FITS files.
    Parameters
    ––––––––––
    cube : ndarray (N_frames, N, M)
        Data cube containing N_frames images of shape (N, M).
    filename : str
        Base filename (without extension). Each
        output file will be saved as "{filename}_i.fits".
    overwrite : bool, optional, default=False
        If True, existing files with the same name
        will be overwritten.
    Notes
    –––––
    Prints a message indicating the number of
    frames and the base filename.
    '''
    N_frames, N, M = cube.shape
    print(f'Writing {N_frames} fits files to {filename}_i.fits')
    for i in tqdm(range(N_frames)):
        output_name = filename + f'_{i}.fits'
        fits.writeto(output_name, cube[i], overwrite=overwrite)
