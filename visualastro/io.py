from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    overwrite : bool, optional, default False
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
