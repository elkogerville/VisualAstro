import numpy as np
from tqdm import tqdm
from .plot_utils import plot_histogram, check_is_array

def compute_master_bias(bias_cube):
    bias_cube = check_is_array(bias_cube)
    master_bias = np.median(bias_cube, axis=0)
    return master_bias

def compute_norm_flat(flats_cube, master_bias, remove_zeros=False, plot_hist=False,
                      xlog=False, ylog=True, style='astro', bins='auto'):
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
        plot_histogram(normalized_flat, bins, style, xlog=xlog, ylog=ylog, labels=labels)

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
