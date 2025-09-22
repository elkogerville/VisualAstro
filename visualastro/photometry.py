import numpy as np
from scipy.ndimage import center_of_mass
from tqdm import tqdm
from .plotting import imshow, check_is_array

def compute_flux(cube, target_pixel_loc, star_radius, sky_radii=None, window_half_width=100, plot=False):
    cube = check_is_array(cube)
    # initial target location guess
    x_pixel, y_pixel = target_pixel_loc
    # array to store target and sky flux
    star_flux = np.zeros((cube.shape[0]))
    sky_flux = np.zeros_like(star_flux)
    # loop through each image in cube
    for i in tqdm(range(len(cube))):
        # extract subimage centered around target
        xmin, xmax = x_pixel - window_half_width, x_pixel + window_half_width
        ymin, ymax = y_pixel - window_half_width, y_pixel + window_half_width
        sub_image = cube[i][xmin:xmax, ymin:ymax]
        # compute approximate center pixels of target
        cenx, ceny = center_of_mass(sub_image)
        # compute distance between target center each pixel in subimage
        x, y = np.indices(sub_image.shape)
        distance_from_center = np.sqrt( (x - cenx)**2 + (y - ceny)**2)
        # mask out pixels outside star aperture
        star_aperture = distance_from_center < star_radius
        star_flux[i] = np.nansum(sub_image[star_aperture])
        if sky_radii is not None:
            sky_inner_r, sky_outer_r = sky_radii
            sky_aperture = (distance_from_center < sky_outer_r) & (distance_from_center > sky_inner_r)
            sky_flux[i] = np.nanmedian(sub_image[sky_aperture])
            star_flux[i] -= sky_flux[i]
        if plot:
            circles = [[cenx, ceny, star_radius]]
            if sky_radii is not None:
                for r in sky_radii:
                    circles.append([cenx, ceny, r])
            imshow(sub_image, circles=circles)

    return star_flux, sky_flux
