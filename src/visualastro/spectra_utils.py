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
from dataclasses import dataclass, fields
from typing import Any, Optional
import warnings
from astropy.units import Quantity
from dust_extinction.parameter_averages import M14, G23
from dust_extinction.grain_models import WD01
import numpy as np
from specutils import SpectralAxis, SpectralRegion, Spectrum
from specutils.fitting import fit_continuum as _fit_continuum
from specutils.fitting import fit_generic_continuum as _fit_generic
from .text_utils import print_pretty_table
from .numerical_utils import get_value, mask_within_range
from .config import get_config_value, config


# Science Spectrum Functions
# --------------------------
def get_spectral_axis(obj: Any) -> SpectralAxis | Quantity | None:
    """
    Get the `spectral_axis` associated with an object, if one exists.

    The function follows a recursive search strategy:

    1. If `obj` is a `SpectralAxis`, it is returned directly.
    2. If `obj` has a `.spectral_axis` attribute, that attribute is returned.
    3. If `obj` has a `.data` attribute, the function is applied recursively
        to `obj.data`.

    If no spectral axis can be identified, the function returns None.

    Parameters
    ----------
    obj : Any
        Object from which to extract a spectral axis. This may include:
        - a `SpectralAxis` instance
        - objects exposing a `.spectral_axis` attribute
        - container objects with a `.data` attribute holding one of the above

    Returns
    -------
    spectral_axis : astropy.coordinates.SpectralAxis or astropy.units.Quantity or None
        The spectral axis associated with the object, or None if unavailable.
    """
    if isinstance(obj, SpectralAxis):
        return obj

    if hasattr(obj, 'spectral_axis'):
        return obj.spectral_axis

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            spectral_axis = get_spectral_axis(data)
            if spectral_axis is not None:
                return spectral_axis

    return None

def fit_continuum(spectrum, fit_method='fit_continuum', region=None):
    '''
    Fit the continuum of a 1D spectrum using a specified method.

    Parameters
    ----------
    spectrum : Spectrum or SpectrumPlus
        Input 1D spectrum object containing flux and spectral_axis.
    fit_method : {'fit_continuum', 'generic'}, optional, default='fit_continuum'
        Method used for fitting the continuum.
        - 'fit_continuum': uses `fit_continuum` with a specified window
        - 'generic' : uses `fit_generic_continuum`
    region : array-like of tuple, optional
        Spectral region(s) to include in the continuum fit when
        `fit_method='fit_continuum'`. Each region is specified as a
        `(lower, upper)` bound in wavelength or pixel coordinates
        (typically `Quantity`). Multiple regions may be provided to
        restrict the fit to selected portions of the spectrum. This
        is commonly used to exclude strong emission or absorption features
        that would otherwise bias the continuum model.Ignored for all
        other fitting methods.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]

    Returns
    -------
    continuum : Quantity
        Continuum flux values evaluated at `spectrum.spectral_axis`.

    Notes
    -----
    - Warnings during the fitting process are suppressed.
    '''
    # if input spectrum is SpectrumPlus object
    # extract the spectrum attribute
    if not isinstance(spectrum, Spectrum):
        if hasattr(spectrum, 'spectrum'):
            spectrum = spectrum.spectrum
        else:
            raise ValueError (
                'Input object is not a Spectrum '
                "or has no `spectrum` attribute. "
                f'type: {type(spectrum)}'
            )
    # extract spectral axis
    spectral_axis = spectrum.spectral_axis

    # suppress warnings during continuum fitting
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # fit continuum with selected method
        if fit_method=='fit_continuum':
            # convert region to default units
            region = _convert_region_units(region, spectral_axis)
            fit = _fit_continuum(spectrum, window=region)
        else:
            fit = _fit_generic(spectrum)

    # fit the continuum of the provided spectral axis
    return fit(spectral_axis)


def deredden_flux(wavelength, flux, Rv=None, Ebv=None,
                  deredden_method=None, region=None):
    '''
    Apply extinction correction (dereddening) to a spectrum.
    Default values are for LMC parameters.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (in Angstroms, microns, or units expected by the
        extinction law being used).
    flux : array-like
        Observed flux values at the corresponding wavelengths. Must be in
        linear units (e.g., erg/s/cm^2/Å, Jy).
    Rv : float or None, optional, default=None
        Ratio of total-to-selective extinction (A_V / E(B-V)).
        If None, uses default value set by `config.Rv`.
    Ebv : float or None, optional, default=None
        Color excess E(B-V), representing the amount of reddening.
        If None, uses default value set by `config.Ebv`.
    deredden_method : {'G23', 'WD01', 'M14'} or None, optional, default=None
        Choice of extinction law:
        - 'G23' : Gordon et al. (2023)
        - 'WD01': Weingartner & Draine (2001)
        - 'M14' : Maíz Apellániz et al. (2014)
        If None, uses default value set by `config.deredden_method`.
    region : str or None, optional, default=None
        For WD01 extinction, the environment/region to use (e.g., 'MWAvg',
        'LMC', 'LMCAvg', 'SMCBar'). Ignored for other methods.
        If None, uses default value set by `config.deredden_region`.

    Returns
    -------
    deredden_flux : array-like
        Flux array corrected for extinction.
    '''
    # get default config values
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
    ----------
    errors : np.ndarray
        Either:
        - 2D array with shape (N_spectra, N_pixels), or
        - 1D array with shape (N_pixels,) for a single spectrum.
    method : {'mean', 'sum', 'median'} or None, optional
        Flux extraction method. If None, uses the default
        value set by `config.propagate_flux_error_method`.

    Returns
    -------
    flux_errors : np.ndarray
        1D array of propagated flux errors (shape N_spectra).
    '''
    # get default config value
    method = get_config_value(method, 'propagate_flux_error_method').lower()
    if method is None:
        raise ValueError(
            "method must be : {'mean', 'sum', 'median'}"
        )

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
# ------------------------
def _convert_region_units(region, spectral_axis):
    '''
    Convert the units of a list of spectral regions to match
    a given spectral axis. Helper function used when fitting

    a spectrum continuum.
    Parameters
    ----------
    region : list of tuple of astropy.units.Quantity or None
        Each element is a tuple `(rmin, rmax)` defining a spectral region.
        Both `rmin` and `rmax` should be `Quantity` objects with units.
        If `None`, the function returns `None`.
    spectral_axis : astropy.units.Quantity
        The spectral axis whose unit is used for conversion.

    Returns
    -------
    list of tuple of astropy.units.Quantity or None
        The input regions converted to the same unit as 'spectral_axis'.
        Returns `None` if `region` is `None`.

    Examples
    --------
    >>> regions = [(1*u.micron, 2*u.micron), (500*u.nm, 700*u.nm)]
    '''
    if region is None:
        return region

    # extract unit
    unit = spectral_axis.unit

    # convert each element to spectral axis units
    if isinstance(region, SpectralRegion):
        converted_bounds = []
        for subregion in region:
            rmin = subregion.lower.to(unit)
            rmax = subregion.upper.to(unit)
            converted_bounds.append((rmin, rmax))
        return converted_bounds

    elif isinstance(region, list):
        return [(rmin.to(unit), rmax.to(unit)) for rmin, rmax in region]

    else:
        raise TypeError(f"region must be SpectralRegion or list of tuples, got {type(region)}")


# Model Fitting Functions
# -----------------------
def construct_gaussian_p0(extracted_spectrum, args, xlim=None):
    '''
    Construct an initial guess (`p0`) for Gaussian fitting of a spectrum.

    Parameters
    ----------
    extracted_spectrum : `SpectrumPlus`
        `SpectrumPlus` object containing `wavelength` and `flux` attributes.
        These can be `numpy.ndarray` or `astropy.units.Quantity`.
    args : list or array-like
        Additional parameters to append to the initial guess after
        amplitude and center (e.g., sigma, linear continuum slope/intercept).
    xlim : tuple of float, optional, default=None
        Wavelength range `(xmin, xmax)` to restrict the fitting region.
        If None, the full spectrum is used.

    Returns
    -------
    p0 : list of float
        Initial guess for Gaussian fitting parameters:
        - First element: amplitude (`max(flux)` in the region)
        - Second element: center (`wavelength` at max flux)
        - Remaining elements: values from `args`

    Notes
    -----
    - Useful for feeding into `scipy.optimize.curve_fit`
      or similar fitting routines.
    '''
    # extract wavelength and flux from SpectrumPlus object
    wavelength = get_value(extracted_spectrum.spectral_axis)
    flux = get_value(extracted_spectrum.flux)
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
    ----------
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
    -------
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
    ----------
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
    -------
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
    ----------
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
    -------
    y : np.ndarray
        (N,) shaped array of the Gaussian function evaluated
        at `x`, including the continuum offset.
    '''
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y + continuum


@dataclass
class GaussianFitResult:
    '''
    Lightweight dataclass for gaussian fitting results.

    Attributes
    ----------
    amplitude : Any
        Amplitude of gaussian.
    amplitude_error : Any
        Error on `amplitude`.
    mu : Any
        Mu or center of gaussian.
    mu_error : Any
        Error on `mu`.
    sigma : Any
        Sigma or standard deviation of gaussian.
    sigma_error : Any
        Error on `sigma`.
    flux : Any
        Integrated flux of gaussian.
    flux_error : Any
        Error on `flux`.
    FWHM : Any
        Full width at half maxixum or the width of
        the gaussian at half of its maximum value.
    FWHM_error : Any
        Error on `FWHM`.
    slope : Any
        Slope of continuum, if modelled as a linear line
        using the `gaussian_line` model.
    slope_error : Any
        Error on `slope`.
    intercept : Any
        Y-intercept of continuum, if modelled as a linear
        line using the `gaussian_line` model.
    intercept_error : Any
        Error on `intercept`.
    popt : Any
        Optimal values for the parameters so that the sum of the
        squared residuals of `f(xdata, *popt) - ydata` is minimized.
        Returned by `scipy.optimize.curve_fit`.
    pcov : Any
        The estimated approximate covariance of popt. The diagonals
        provide the variance of the parameter estimate. Returned by
        `scipy.optimize.curve_fit`.
    perr : Any
        The one standard deviation errors on the parameters.
        Computed as `perr = np.sqrt(np.diag(pcov))`.

    Notes
    -----
    - If new gaussian models are added, make sure to update
      the `additional_parameters` list.
    '''
    # fitted parameters
    amplitude: Any
    amplitude_error: Any
    mu: Any
    mu_error: Any
    sigma: Any
    sigma_error: Any

    # derived quantities
    flux: Any
    flux_error: Any
    FWHM: Any
    FWHM_error: Any

    # additional parameters
    slope: Optional[Any] = None
    slope_error: Optional[Any] = None
    intercept: Optional[Any] = None
    intercept_error: Optional[Any] = None

    # NOTE: ensure these are updated!
    additional_parameters = ['slope', 'intercept']
    parameters_labels = ['Slope (m)', 'Intercept (b)']

    # raw curve fit results
    popt: Optional[Any] = None
    pcov: Optional[Any] = None
    perr: Optional[Any] = None
    p0: Optional[Any] = None
    fit_config: Optional[dict] = None

    def pretty_print(self, **kwargs):
        '''
        Pretty print the results in table format.
        '''

        precision = kwargs.get('precision', config.table_precision)
        sci_notation = kwargs.get('sci_notation', config.table_sci_notation)
        pad = kwargs.get('pad', config.table_column_pad)

        fitted_data = [
            ['Amplitude', self.amplitude, self.amplitude_error],
            ['Mu (μ)', self.mu, self.mu_error],
            ['Sigma (σ)', self.sigma, self.sigma_error],
        ]
        for param, name in zip(
            self.additional_parameters, self.parameters_labels
        ):
            if getattr(self, param, None) is not None:
                value = getattr(self, param)
                error = getattr(self, param+'_error', '')
                fitted_data.append([name, value, error])

        print('Fitted Parameters:')
        print_pretty_table(
            headers=['Parameter', 'Value', 'Error'],
            data=fitted_data,
            precision=precision,
            sci_notation=sci_notation,
            pad=pad
        )

        print('\nDerived Parameters: ')
        print_pretty_table(
            headers=None,
            data=[
                ['Integrated Flux', self.flux, self.flux_error],
                ['FWHM', self.FWHM, self.FWHM_error],
            ],
            precision=precision,
            sci_notation=sci_notation,
            pad=pad
        )

    def __repr__(self):
        '''
        Returns values ± errors.
        '''
        exclude = {'popt', 'pcov', 'perr', 'additional_parameters', 'parameters_labels'}

        lines = ['GaussianFitResult(']

        for field in fields(self):
            if field.name in exclude or field.name.endswith('_error'):
                continue

            value = getattr(self, field.name)
            if value is not None:
                error = getattr(self, f'{field.name}_error', None)
                if error is not None:
                    lines.append(f'  {field.name} : {value} ± {error}')
                else:
                    lines.append(f'  {field.name} : {value}')

        lines.append(')')
        return '\n'.join(lines)
