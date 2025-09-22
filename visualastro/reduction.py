import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
from .numerical_utils import check_is_array
from .plotting import plot_histogram

def compute_master_bias(bias_cube):
    bias_cube = check_is_array(bias_cube)
    master_bias = np.median(bias_cube, axis=0)
    return master_bias

def compute_norm_flat(flats_cube, master_bias, remove_zeros=False, plot_hist=False,
                      xlog=False, ylog=True, colors=None, style='astro', bins='auto'):
    flats_cube = check_is_array(flats_cube)
    flats_bias_sub = flats_cube - master_bias
    median_flat = np.median(flats_bias_sub, axis=0)
    N, M = median_flat.shape
    x_midpoint = N//2
    y_midpoint = M//2
    x_min, x_max = x_midpoint-500, x_midpoint+500
    y_min, y_max = y_midpoint-500, y_midpoint+500
    flat_inner_region = median_flat[x_min:x_max, y_min:y_max]
    inner_region_mean = np.mean(flat_inner_region)
    normalized_flat = median_flat / inner_region_mean
    if remove_zeros:
        normalized_flat = np.where(normalized_flat == 0, np.nan, normalized_flat)
    if plot_hist:
        labels = ['Counts', 'Number of Pixels']
        plot_histogram(normalized_flat, bins, style, colors=colors,
                       xlog=xlog, ylog=ylog, labels=labels)

    return normalized_flat

def reduce_science_frames(data_cube, master_bias, master_flat, trim=None, vectorize=False):
    data_cube = check_is_array(data_cube)
    if vectorize:
        data_bias_sub = data_cube - master_bias
        norm_data_cube = data_bias_sub / master_flat
        if trim is not None:
            norm_data_cube = norm_data_cube[:, trim:-trim, trim:-trim]
    else:
        if trim is None:
            norm_data_cube = np.zeros_like(data_cube)
        else:
            norm_data_cube = np.zeros_like(data_cube[:, trim:-trim, trim:-trim])
        for i in tqdm(range(len(data_cube))):
            data_bias_sub = data_cube[i] - master_bias
            norm_data = data_bias_sub / master_flat
            if trim is not None:
                norm_data = norm_data[trim:-trim, trim:-trim]
            norm_data_cube[i] = norm_data

    return norm_data_cube

@njit(parallel=True)
def njit_reduce_science_frames(data_cube, master_bias, master_flat, trim=None, vectorize=False):
    if isinstance(data_cube, dict):
        data_cube = np.asarray(data_cube['data'])
    else:
        data_cube = np.asarray(data_cube)

    if trim is None:
        norm_data_cube = np.zeros_like(data_cube)
    else:
        norm_data_cube = np.zeros_like(data_cube[:, trim:-trim, trim:-trim])
    for i in prange(len(data_cube)):
        data_bias_sub = data_cube[i] - master_bias
        norm_data = data_bias_sub / master_flat
        if trim is not None:
            norm_data = norm_data[trim:-trim, trim:-trim]
        norm_data_cube[i] = norm_data

    return norm_data_cube

def subtract_sky_values(science_cube, N_bins=3000, xlog=False, ylog=True, N_legend=5):
    '''
    Subtract the sky value from a science image data cube
    Parameters
    ----------
    science_cube: np.ndarray[np.float64]
        ixNxM fits data cube where i is the number
        of images and NxM is the image dimensions
    N_bins: int
        number of bins to sample data, by default is 3000
    Returns
    -------
    reduced_science: np.ndarray[np.float64]
        ixNxM fits data cube where each image
        is sky subtracted
    '''
    # determine range of bins from data
    min_val = np.nanmin(science_cube)
    max_val = np.nanmax(science_cube)
    bins = np.linspace(min_val, max_val, N_bins)

    # compute number of images
    N = science_cube.shape[0]

    # compute the histogram of each image
    histograms = [np.histogram(science_cube[i, :, :].flatten(), bins=bins)
                  for i in tqdm(range(N), 'Computing histograms')]
    # plot histogram
    colors = plt.get_cmap('rainbow')(np.linspace(0.1, 1, len(histograms)))
    plt.figure(figsize=(10,4))
    plt.rcParams['axes.linewidth'] = 0.5
    plt.minorticks_on()
    plt.tick_params(axis = 'both', length = 3, direction = 'in',
                    which = 'both', right = True, top = True)

    for i in tqdm(range(N), 'Computing sky value'):
        # unpack the counts and edges of each image histogram
        counts, edges = histograms[i]
        # convert to arrays and compute the centers of each set of bins
        counts, edges = np.asarray(counts), np.asarray(edges)
        centers = (edges[1:] + edges[:-1])/2
        # find index of each image's histogram peak
        peak_loc = np.argmax(counts)
        # sky values are the values of the bins associated with each peak
        sky_value = centers[peak_loc]

        # subtract sky value from each science image
        science_cube[i] -= sky_value
        label = f'image {i}, sky val: {sky_value}' if i <= N_legend else None
        plt.step(centers, counts, c=colors[i], label=label)

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel('Bin Centers')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

    return science_cube
