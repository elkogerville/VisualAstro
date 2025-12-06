'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-17
Description:
    Spectra utility functions.
Dependencies:
    - dust_extinction
    - numpy
    - specutils
Module Structure:
    - Science Spectrum Functions
        Utility functions for scientific spectra work.
    - Science Helper Functions
        Utility functions for Science Spectrum Functions.
    - Axes Labels, Format, and Styling
        Axes related utility functions.
    - Model Fitting Functions
        Model fitting utility functions.
'''

import warnings
from dust_extinction.parameter_averages import M14, G23
from dust_extinction.grain_models import WD01
import numpy as np
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum, fit_continuum
from .numerical_utils import mask_within_range, return_array_values
from .ExtractedSpectrum import ExtractedSpectrum
from .va_config import get_config_value


# Science Spectrum Functions
# ––––––––––––––––––––––––––
def compute_continuum_fit(spectrum1d, fit_method='fit_continuum', region=None):
    '''
    Fit the continuum of a 1D spectrum using a specified method.
    Parameters
    ––––––––––
    spectrum1d : Spectrum1D or ExtractedSpectrum
        Input 1D spectrum object containing flux and spectral_axis.
        ExtractedSpectrum is supported only if it contains a
        spectrum1d object.
    fit_method : str, optional, default='generic'
        Method used for fitting the continuum.
        - 'fit_continuum': uses `fit_continuum` with a specified window
        - 'generic'      : uses `fit_generic_continuum`
    region : array-like, optional, default=None
        Wavelength or pixel region(s) to use when `fit_method='fit_continuum'`.
        Ignored for other methods. This allows the user to specify which
        regions to include in the fit. Removing strong peaks is preferable to
        avoid skewing the fit up or down.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
    Returns
    –––––––
    continuum_fit : np.ndarray
        Continuum flux values evaluated at `spectrum1d.spectral_axis`.
    Notes
    –––––
    - Warnings during the fitting process are suppressed.
    '''
    # if input spectrum is ExtractedSpectrum object
    # extract the spectrum1d attribute
    if not isinstance(spectrum1d, Spectrum1D):
        if hasattr(spectrum1d, 'spectrum1d'):
            spectrum1d = spectrum1d.spectrum1d
        else:
            raise ValueError (
                'Input object is not a Spectrum1d '
                "or has no `spectrum1d` attribute. "
                f'type: {type(spectrum1d)}'
            )
    # extract spectral axis
    spectral_axis = spectrum1d.spectral_axis

    # suppress warnings during continuum fitting
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # fit continuum with selected method
        if fit_method=='fit_continuum':
            # convert region to default units
            region = _convert_region_units(region, spectral_axis)
            fit = fit_continuum(spectrum1d, window=region)
        else:
            fit = fit_generic_continuum(spectrum1d)

    # fit the continuum of the provided spectral axis
    continuum_fit = fit(spectral_axis)

    return continuum_fit


def deredden_flux(wavelength, flux, Rv=None, Ebv=None,
                  deredden_method=None, region=None):
    '''
    Apply extinction correction (dereddening) to a spectrum.
    Default values are for LMC parameters.
    Parameters
    ––––––––––
    wavelength : array-like
        Wavelength array (in Angstroms, microns, or units expected by the
        extinction law being used).
    flux : array-like
        Observed flux values at the corresponding wavelengths. Must be in
        linear units (e.g., erg/s/cm^2/Å, Jy).
    Rv : float or None, optional, default=None
        Ratio of total-to-selective extinction (A_V / E(B-V)).
        If None, uses default value set by `va_config.Rv`.
    Ebv : float or None, optional, default=None
        Color excess E(B-V), representing the amount of reddening.
        If None, uses default value set by `va_config.Ebv`.
    deredden_method : {'G23', 'WD01', 'M14'} or None, optional, default=None
        Choice of extinction law:
        - 'G23' : Gordon et al. (2023)
        - 'WD01': Weingartner & Draine (2001)
        - 'M14' : Maíz Apellániz et al. (2014)
        If None, uses default value set by `va_config.deredden_method`.
    region : str or None, optional, default=None
        For WD01 extinction, the environment/region to use (e.g., 'MWAvg',
        'LMC', 'LMCAvg', 'SMCBar'). Ignored for other methods.
        If None, uses default value set by `va_config.deredden_region`.
    Returns
    –––––––
    deredden_flux : array-like
        Flux array corrected for extinction.
    '''
    # get default va_config values
    Rv = get_config_value(Rv, 'Rv')
    Ebv = get_config_value(Ebv, 'Ebv')
    deredden_method = get_config_value(deredden_method, 'deredden_method')
    region = get_config_value(region, 'deredden_region')

    # select appropriate dereddening method
    methods = {
        'G23': G23,
        'WD01': WD01,
        'M14': M14
    }
    if deredden_method not in methods:
        raise ValueError(
            f"Unknown deredden_method '{deredden_method}'. "
            "Choose from 'G23', 'WD01', or 'M14'."
        )
    deredden = methods[deredden_method]

    if deredden_method == 'WD01':
        extinction = deredden(region)
    else:
        extinction = deredden(Rv=Rv)
    # deredden flux
    dereddened_flux = flux / extinction.extinguish(wavelength, Ebv=Ebv)

    return dereddened_flux


def propagate_flux_errors(errors, method=None):
    '''
    Compute propagated flux errors from individual pixel errors in a spectrum.
    Parameters
    ––––––––––
    errors : np.ndarray
        Either:
        - 2D array with shape (N_spectra, N_pixels), or
        - 1D array with shape (N_pixels,) for a single spectrum.
    method : {'mean', 'sum', 'median'} or None, optional
        Flux extraction method.
        If None, falls back to va_config.flux_extract_method.

    Returns
    –––––––
    flux_errors : np.ndarray
        1D array of propagated flux errors (shape N_spectra).
    '''
    # get default va_config value
    method = get_config_value(method, 'flux_extract_method').lower()

    # ensure errors are 2-dimensional
    if errors.ndim == 1:
        errors = errors[np.newaxis, :]

    # number of valid (non-NaN) pixels per spectrum
    N = np.sum(~np.isnan(errors), axis=1)

    # quadratic sum per spectrum
    quad_sum = np.sqrt( np.nansum(errors**2, axis=1) )

    # propagation method based on flux extraction method
    if method == 'mean':
        flux_errors = quad_sum / N

    elif method == 'sum':
        flux_errors = quad_sum

    elif method == 'median':
        # statistically correct median error scaling
        flux_errors = 1.253 * (quad_sum / N)

    else:
        raise ValueError(f"Unknown flux extraction method '{method}'.")

    return flux_errors


# Science Helper Functions
# ––––––––––––––––––––––––
def _convert_region_units(region, spectral_axis):
    '''
    Convert the units of a list of spectral regions to match
    a given spectral axis. Helper function used when fitting
    a spectrum continuum.
    Parameters
    ––––––––––
    region : list of tuple of astropy.units.Quantity or None
        Each element is a tuple `(rmin, rmax)` defining a spectral region.
        Both `rmin` and `rmax` should be `Quantity` objects with units.
        If `None`, the function returns `None`.
    spectral_axis : astropy.units.Quantity
        The spectral axis whose unit is used for conversion.
    Returns
    –––––––
    list of tuple of astropy.units.Quantity or None
        The input regions converted to the same unit as 'spectral_axis'.
        Returns `None` if `region` is `None`.
    Examples
    ––––––––
    >>> regions = [(1*u.micron, 2*u.micron), (500*u.nm, 700*u.nm)]
    '''
    if region is None:
        return region
    # extract unit
    unit = spectral_axis.unit
    # convert each element to spectral axis units
    return [(rmin.to(unit), rmax.to(unit)) for rmin, rmax in region]


# Model Fitting Functions
# –––––––––––––––––––––––
def construct_gaussian_p0(extracted_spectrum, args, xlim=None):
    '''
    Construct an initial guess (`p0`) for Gaussian fitting of a spectrum.
    Parameters
    ––––––––––
    extracted_spectrum : `ExtractedSpectrum`
        `ExtractedSpectrum` object containing `wavelength` and `flux` attributes.
        These can be `numpy.ndarray` or `astropy.units.Quantity`.
    args : list or array-like
        Additional parameters to append to the initial guess after
        amplitude and center (e.g., sigma, linear continuum slope/intercept).
    xlim : tuple of float, optional, default=None
        Wavelength range `(xmin, xmax)` to restrict the fitting region.
        If None, the full spectrum is used.
    Returns
    –––––––
    p0 : list of float
        Initial guess for Gaussian fitting parameters:
        - First element: amplitude (`max(flux)` in the region)
        - Second element: center (`wavelength` at max flux)
        - Remaining elements: values from `args`
    Notes
    –––––
    - Useful for feeding into `scipy.optimize.curve_fit`
      or similar fitting routines.
    '''
    # extract wavelength and flux from ExtractedSpectrum object
    wavelength = return_array_values(extracted_spectrum.wavelength)
    flux = return_array_values(extracted_spectrum.flux)
    # clip arrays by xlim
    if xlim is not None:
        mask = mask_within_range(wavelength, xlim)
        wavelength = wavelength[mask]
        flux = flux[mask]
    # compute index of peak flux value
    peak_idx = int(np.argmax(flux))
    # compute max amplitude and corresponding wavelength value
    p0 = [np.nanmax(flux), wavelength[peak_idx]]
    # extend any arguments needed for gaussian fitting
    p0.extend(args)

    return p0

def gaussian(x, A, mu, sigma):
    '''
    Compute a gaussian curve.
    Parameters
    ––––––––––
    x : np.ndarray
        (N,) shaped range of x values (pixel indices) to
        compute the gaussian function over.
    A : float
        Amplitude of gaussian function.
    mu : float
        Mean or center of gaussian function.
    sigma : float
        Standard deviation of gaussian function.
    Returns
    –––––––
    y : np.ndarray
        (N,) shaped array of values of gaussian function
        evaluated at each `x`.
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y

def gaussian_line(x, A, mu, sigma, m, b):
    '''
    Compute a Gaussian curve with a linear continuum.
    Parameters
    ––––––––––
    x : np.ndarray
        (N,) shaped array of x values (e.g., pixel indices)
        to evaluate the Gaussian.
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean or center of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    m : float
        Slope of the linear continuum.
    b : float
        Y-intercept of the linear continuum.
    Returns
    –––––––
    y : np.ndarray
        (N,) shaped array of the Gaussian function evaluated
        at each `x`, including the linear continuum `m*x + b`.
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m*x+b

    return y

def gaussian_continuum(x, A, mu, sigma, continuum):
    '''
    Compute a Gaussian curve with a continuum offset.
    Parameters
    ––––––––––
    x : np.ndarray
        (N,) shaped array of x values (e.g., pixel indices)
        to evaluate the Gaussian.
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean or center of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    continuum : np.ndarray or array-like
        Continuum values to add to the Gaussian.
        Must be the same shape as `x`.
    Returns
    –––––––
    y : np.ndarray
        (N,) shaped array of the Gaussian function evaluated
        at `x`, including the continuum offset.
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y + continuum
