"""
Author: Elko Gerville-Reache
Date Created: 2025-11-16
Date Modified: 2026-11-16
Description:
    Spectra model fitting module.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray

from visualastro.core.config import _UNSET
from visualastro.core.data import get_value
from visualastro.core.numerical import mask_within_range
from visualastro.optional_dependencies._specutils import Spectrum
from visualastro.optional_dependencies.register import _require_dependency
from visualastro.utils.text_utils import pretty_table

__all__ = [
    'GaussianFitResult',
    'ExtractedPixelSpectra',
    'gaussian',
    'gaussian_continuum',
    'gaussian_line',
    'gaussian_p0'
]


def gaussian_p0(extracted_spectrum, args, xlim=None):
    """
    Construct an initial guess (`p0`) for Gaussian fitting of a spectrum.

    Parameters
    ----------
    extracted_spectrum : Spectrum
        `Spectrum` object containing `wavelength` and `flux` attributes.
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
    """
    _require_dependency('specutils')
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
    """
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
    """
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y


def gaussian_line(x, A, mu, sigma, m, b):
    """
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
    """
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m*x+b

    return y

def gaussian_continuum(x, A, mu, sigma, continuum):
    """
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
    """
    y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y + continuum


@dataclass
class GaussianFitResult:
    """
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
    peak_height : Any
        Height (y value) of gaussian peak.
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
      the `additional_parameters` and `parameters_labels` list.
    """
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
    slope: Any | None = None
    slope_error: Any | None = None
    intercept: Any | None = None
    intercept_error: Any | None = None
    peak_height: Any | None = None

    # NOTE: ensure these are updated!
    additional_parameters = ['slope', 'intercept', 'peak_height']
    parameters_labels = ['Slope (m)', 'Intercept (b)', 'Peak Height']

    # raw curve fit results
    popt: Any | None = None
    pcov: Any | None = None
    perr: Any | None = None
    p0: Any | None = None
    fit_config: dict | None = None

    def pretty_print(self, **kwargs):
        """
        Print fit results as aligned tables.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional parameters.

            Supported keywords:

            - `precision` : int | _Unset, optional, default=_UNSET
              Number of decimal places. If `_UNSET`, uses `config.table_precision`.
            - `sci_notation` : bool | _Unset, optional, default=_UNSET
              If True, use scientific notation. If `_UNSET`, uses `config.table_sci_notation`.
            - `pad` : int | _Unset, optional, default=_UNSET
              Spaces between columns. If `_UNSET`, uses `config.table_column_pad`.
        """
        def _row(name, value, error):
            if error is None or isinstance(error, str):
                return [name, value, None]
            err = error.to(value.unit)
            if not np.isfinite(err.value) or err.value <= 0:
                return [name, value, 'N/A']
            return [name, value, err]

        fitted = [
            _row('Amplitude', self.amplitude, self.amplitude_error),
            _row('Mu (μ)', self.mu, self.mu_error),
            _row('Sigma (σ)', self.sigma, self.sigma_error),
        ]
        for attr, label in zip(self.additional_parameters, self.parameters_labels):
            value = getattr(self, attr, None)
            if value is not None:
                fitted.append(_row(label, value, getattr(self, attr + '_error', None)))

        derived = [
            _row('Integrated Flux', self.flux, self.flux_error),
            _row('FWHM', self.FWHM, self.FWHM_error),
        ]

        lines = pretty_table(
            headers=['Parameter', 'Value', 'Error'],
            data=fitted + derived,
            precision=kwargs.get('precision', _UNSET),
            sci_notation=kwargs.get('sci_notation', _UNSET),
            pad=kwargs.get('pad', _UNSET),
        ).splitlines()

        header, rule, body = lines[0], lines[1], lines[2:]
        nfit = len(fitted)

        print('Fitted Parameters:')
        print(header)
        print(rule)
        print('\n'.join(body[:nfit]))

        print('\nDerived Parameters:')
        print(header)
        print(rule)
        print('\n'.join(body[nfit:]))

    def __repr__(self) -> str:
        """Returns values ± errors."""
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


@dataclass(frozen=True)
class ExtractedPixelSpectra:
    """
    spectra : Spectrum | list[Spectrum]
        Extracted per-pixel spectra. If a single index is selected,
        this is returned as a single `Spectrum` instance;
        otherwise, a list is returned.
    cube_array : np.ndarray | Quantity
        Cube of array values corresponding to the extracted pixels.
        All other pixels are set to NaN.
    error_array : np.ndarray | Quantity | None, optional
        Cube of error values corresponding to the extracted pixels.
        All other pixels are set to NaN. Set to None if no error
        extension is present.
    extract_idx : np.ndarray[int], shape (N,)
        Indices of the extracted spectra corresponding to the ordering
        of valid spatial pixels.
    coords : np.ndarray[int], shape (N, 2)
        Spatial coordinates of extracted pixels in `(y, x)` order.
    colors : list
        Colors assigned to each extracted pixel/spectrum.
    labels : list[str]
        Human-readable labels of the form `"<idx>: (x=<x>, y=<y>)"`.
    combined_spectrum : Spectrum | None
        Combined spectrum computed using `combine_method` if
        `combine_spectra=True`; otherwise None.
    """
    spectra: Spectrum | list[Spectrum]
    cube_array : NDArray | Quantity
    error_array : NDArray | Quantity | None
    extract_idx: NDArray
    coords: NDArray
    colors: Sequence | NDArray
    labels: Sequence
    combined_spectrum: object | None = None

    def __post_init__(self):
        _require_dependency('specutils')

    def __repr__(self) -> str:
        n = len(self.extract_idx)

        spectra_type = (
            'Spectrum'
            if isinstance(self.spectra, Spectrum)
            else 'list[Spectrum]'
        )

        combined = self.combined_spectrum is not None

        return (
            f'{self.__class__.__name__}('
            f'N={n}, '
            f'spectra={spectra_type}, '
            f'coords_shape={self.coords.shape}, '
            f'combined={combined})'
        )
