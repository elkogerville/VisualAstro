from astropy.io import fits
from tqdm import tqdm

def write_cube_2_fits(cube, filename, overwrite=False):
    N_frames, N, M = cube.shape
    print(f'Writing {N_frames} fits files to {filename}_reduced_i.fits')
    for i in tqdm(range(N_frames)):
        output_name = filename + f'_reduced_{i}.fits'
        fits.writeto(output_name, cube[i], overwrite=overwrite)
