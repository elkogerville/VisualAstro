import warnings
from dust_extinction.parameter_averages import M14, G23
from dust_extinction.grain_models import WD01
import numpy as np
from specutils.fitting import fit_generic_continuum, fit_continuum
from .numerical_utils import return_array_values

def deredden_spectrum(wavelength, flux, **kwargs):

    Rv = kwargs.get('Rv', 3.1)
    Ebv = kwargs.get('Ebv', 0.19)
    deredden_method = kwargs.get('deredden_method', 'WD01')
    region = kwargs.get('region', 'LMCAvg')

    # Rv=3.1 and Ebv=0.19 are LMC parameters
    deredden = {
        'G23': G23,
        'WD01': WD01,
        'M14': M14
    }.get(deredden_method, M14)

    if deredden_method == 'WD01':
        extinction = deredden(region)
    else:
        extinction = deredden(Rv=Rv)

    deredden_flux = flux / extinction.extinguish(wavelength, Ebv=Ebv)

    return deredden_flux

def convert_region_units(region, spectral_axis):
    if region is None:
        return region

    unit = spectral_axis.unit
    return [(rmin.to(unit), rmax.to(unit)) for rmin, rmax in region]

def return_continuum_fit(spectrum1d, fit_method, region):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if fit_method=='fit_continuum':
            fit = fit_continuum(spectrum1d, window=region)
        else:
            fit = fit_generic_continuum(spectrum1d)
    spectral_axis = spectrum1d.spectral_axis
    continuum_fit = fit(spectral_axis)

    return continuum_fit

def propagate_flux_errors(errors):
    N = np.sum(~np.isnan(errors), axis=1)
    flux_errors = np.sqrt( np.nansum(errors**2, axis=1) ) / N

    return flux_errors

def construct_p0(extracted_spectrum, args, xlim=None):
    wavelength = return_array_values(extracted_spectrum.wavelength)
    flux = return_array_values(extracted_spectrum.flux)

    if xlim is not None:
        mask = (wavelength > xlim[0]) & (wavelength < xlim[1])
        wavelength = wavelength[mask]
        flux = flux[mask]
    peak_idx = int(np.argmax(flux))
    p0 = [np.nanmax(flux), wavelength[peak_idx]]
    p0.extend(args)

    return p0

def gaussian(x, A, mu, sigma):
    '''
    compute a gaussian curve given x values, amplitude, mean, and standard deviation
    Parameters
    ––––––––––
    x: np.ndarray[np.int64]
        (N,) shaped range of x values (pixel indeces) to compute the gaussian function over
    A: float
        amplitude of gaussian function
    mu: int
        mean or center of gaussian function
    sigma: float
        standard deviation of gaussian function
    Returns
    –––––––
    y: np.ndarray[np.float64]
        (N,) shaped array of values of gaussian function evaluated at each x
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y

def gaussian_line(x, A, mu, sigma, m, b):
    '''
    compute a gaussian curve given x values, amplitude, mean, and standard deviation
    Parameters
    ----------
    x: np.ndarray[np.int64]
        (N,) shaped range of x values (pixel indeces) to compute the gaussian function over
    A: float
        amplitude of gaussian function
    mu: int
        mean or center of gaussian function
    sigma: float
        standard deviation of gaussian function
    Returns
    -------
    y: np.ndarray[np.float64]
        (N,) shaped array of values of gaussian function evaluated at each x
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m*x+b

    return y

def gaussian_continuum(x, A, mu, sigma, continuum):
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y + continuum

def residuals(params, x, y):
    A, mu, sigma, m, b = params
    model = A*np.exp(-0.5*((x - mu) / sigma)**2) + m*x + b
    return y - model
