"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-11-14
Description:
    Spectra utility functions.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray

from visualastro.analysis.spectra.fitting import ExtractedPixelSpectra
from visualastro.analysis.spectra.manipulation import compute_continuum
from visualastro.core.config import (
    config, _resolve_default, _Unset, _UNSET
)
from visualastro.core.data import get_value, to_list
from visualastro.core.sequences import _unwrap_if_single
from visualastro.core.units import (
    ensure_common_unit,
    get_spectral_unit,
    get_unit,
    _is_spectral_axis,
)
from visualastro.optional_dependencies._spectralcube import SpectralCube
from visualastro.optional_dependencies._specutils import (
    SpectralAxis,
    SpectralRegion,
    Spectrum,
    _HAS_SPECUTILS,
)


def estimate_spectrum_line_flux(spectra, spec_range):
    """
    Estimate the integrated line flux over a spectral interval.

    This function supports both single-spectrum inputs and iterable
    containers of spectra. For each spectrum, the continuum-subtracted
    flux is numerically integrated over the specified spectral range
    using the trapezoidal rule.

    Parameters
    ----------
    spectrum : Any | Sequence[Any]
        Spectral object(s) from which a spectral axis, flux, and continuum
        can be extracted. This includes `ExtractedPixelSpectra` or
        lists of `Spectrum`.
    spec_range : Quantity | ArrayLike of length 2
        Lower and upper bounds of the spectral interval over which to
        integrate. Must be compatible with the spectral axis units.
        If no units, the units of `spectrum` are assumed.

    Returns
    -------
    line_flux : astropy.units.Quantity | list[Quantity]
        Integrated line flux(es) over the specified interval.
    """
    if isinstance(spectra, ExtractedPixelSpectra):
        spectra = spectra.spectra

    spectra_list = to_list(spectra)
    results = [
        _estimate_spectrum_line_flux(spec, spec_range)
        for spec in spectra_list
    ]

    return _unwrap_if_single(results)


def _estimate_spectrum_line_flux(
    spectrum: Any,
    spec_range: Quantity | tuple[float, float] | list[float] | NDArray
) -> Quantity | float:
    """
    Estimate the integrated line flux over a spectral interval.

    The line flux is computed by numerically integrating the
    continuum-subtracted spectrum over the specified spectral range
    using the trapezoidal rule. This provides a model-independent
    estimate of the line strength.

    Parameters
    ----------
    spectrum : Any
        Spectral object from which a spectral axis, flux, and continuum
        can be extracted. Typically a `Spectrum`.
        `ExtractedPixelSpectra` or compatible container.
    spec_range : Quantity of length 2
        Lower and upper bounds of the spectral interval over which to
        integrate. Values must be comparable to the spectral axis.
        If no units, the units of `spectrum` are assumed.

    Returns
    -------
    line_flux : astropy.units.Quantity | float
        Integrated line flux over the specified interval. If no spectral
        samples fall within the interval, a zero-valued result is
        returned (with units if available).

    Raises
    ------
    ValueError
        If the spectral axis, flux, or continuum cannot be determined,
        or if `spec_range` is not a length-2 array-like object.
    """
    spectral_axis = _get_spectral_axis(spectrum)
    flux = _get_flux(spectrum)
    continuum = _get_continuum(spectrum)

    if spectral_axis is None or flux is None or continuum is None:
        raise ValueError(
            'Could not determine spectral_axis, flux, and continuum '
            f'from the provided spectrum! Got: {type(spectral_axis).__name__}, '
            f'{type(flux).__name__} and {type(continuum).__name__}.'
        )

    if (not isinstance(spec_range, (list, tuple, np.ndarray, Quantity))
        or len(spec_range) != 2):
        raise ValueError(
            'spec_range must be array-like of length 2, containing '
            'the lower and upper bounds of the spectral axis'
        )

    spec_unit = get_unit(spectral_axis)
    flux_unit = get_unit(flux)

    if not isinstance(spec_range, Quantity) and spec_unit is not None:
        spec_range = Quantity(spec_range, unit=spec_unit)

    spec_min, spec_max = spec_range
    try:
        if spec_unit is not None:
            spec_min = Quantity(spec_min).to(spec_unit).value
            spec_max = Quantity(spec_max).to(spec_unit).value
        else:
            spec_min = float(spec_min)
            spec_max = float(spec_max)
    except Exception as exc:
        raise ValueError(
            'spec_range units are incompatible with the spectral axis.'
        ) from exc

    x = np.asarray(spectral_axis)
    y = np.asarray(flux - continuum)
    mask = (x >= spec_min) & (x <= spec_max)

    if not np.any(mask):
        if flux_unit is not None and spec_unit is not None:
            return Quantity(0, unit=flux_unit * spec_unit)
        return 0

    value = np.trapezoid(y[mask], x[mask])

    if flux_unit is not None and spec_unit is not None:
        return Quantity(value, unit=flux_unit * spec_unit)
    return float(value)


def sort_spectra_by_line_strength(
    spectra,
    spec_range,
    descending=True,
    emission_only=None
):
    """
    Sort spectra by their integrated line flux over a spectral range.

    Parameters
    ----------
    spectra : list, array-like, or ExtractedPixelSpectra
        Spectrum objects to sort. If `ExtractedPixelSpectra`, returns
        a new `ExtractedPixelSpectra` with all attributes reordered.
    spec_range : Quantity of length 2 or array-like
        Lower and upper bounds of the spectral interval for integration.
    descending : bool, default=True
        If True, sort from strongest to weakest emission.
    emission_only : bool or None, optional, default=None
        If True, only return spectra with positive line flux (emission).
        If False, only return spectra with negative line flux (absorption)
        If None, return all spectra.

    Returns
    -------
    If input is ExtractedPixelSpectra:
        ExtractedPixelSpectra
            New instance with all attributes sorted by line strength.
    Otherwise:
        sorted_indices : np.ndarray
            Indices that would sort the input spectra by line strength.
        line_strengths : Quantity
            Integrated line flux for each spectrum, in sorted order.
    """
    spec_list = spectra.spectra if isinstance(spectra, ExtractedPixelSpectra) else spectra

    fluxes = estimate_spectrum_line_flux(spec_list, spec_range)

    unit = ensure_common_unit(fluxes, on_mismatch='ignore')
    # flatten list[Quantity] -> Quantity
    line_strengths = Quantity(fluxes, unit=unit)

    if emission_only is True:
        valid_mask = line_strengths > 0
    elif emission_only is False:
        valid_mask = line_strengths < 0
    else:
        valid_mask = np.ones(len(line_strengths), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    valid_strengths = line_strengths[valid_mask]

    sort_order = np.argsort(valid_strengths)
    if descending:
        sort_order = sort_order[::-1]

    sorted_indices = valid_indices[sort_order]
    sorted_strengths = valid_strengths[sort_order]

    if isinstance(spectra, ExtractedPixelSpectra):
        return ExtractedPixelSpectra(
            spectra=[spec_list[i] for i in sorted_indices],
            cube_array=spectra.cube_array,
            error_array=None,
            extract_idx=spectra.extract_idx[sorted_indices],
            coords=spectra.coords[sorted_indices],
            colors=[spectra.colors[i] for i in sorted_indices],
            labels=[spectra.labels[i] for i in sorted_indices],
            combined_spectrum=spectra.combined_spectrum
        )
    else:
        return sorted_indices, sorted_strengths


def spectral_idx_2_world(
    spectral_axis: SpectralAxis | Quantity,
    idx: int | tuple[int, int] | list[int | tuple[int, int]] | None,
    keep_unit: bool = True
):
    """
    Return a representative value from a spectral axis
    given an index or index range.

    Parameters
    ----------
    spectral_axis : SpectralAxis | Quantity
        The spectral axis (e.g., wavelength, frequency, or
        velocity) as an `astropy.units.Quantity` or a
        `specutils.spectra.spectral_axis.SpectralAxis` array.
    idx : int | tuple[int, int] | list[int | tuple[int, int]] | None
        Indices specifying the slice along the first axis:

        * `i` -> returns `spectral_axis[i]`
        * `[i]` -> returns `spectral_axis[i]`
        * `[i, j]` -> returns `(spectral_axis[i] + spectral_axis[j+1])/2`
        * `None` -> returns `(spectral_axis[0] + spectral_axis[-1]) / 2`

    keep_unit : bool, optional, default=True
        If `True`, return the result as an `astropy.units.Quantity`.
        If `False`, return the raw float value without units.

    Returns
    -------
    spectral_value : float or Quantity
        The spectral value at the specified index or index
        range, in the units of 'spectral_axis'.
    """
    spec_unit = get_spectral_unit(spectral_axis)
    if idx is None:
        result = (spectral_axis[0] + spectral_axis[-1]) / 2

    elif isinstance(idx, int):
        result = spectral_axis[idx]

    elif isinstance(idx, list):
        if len(idx) == 1:
            result = spectral_axis[idx[0]]
        elif len(idx) == 2:
            result = (
                (spectral_axis[idx[0]] + spectral_axis[idx[1]]) / 2
            )
        else:
            raise ValueError("'idx' must be an int or a list of one or two integers")
    else:
        raise ValueError("'idx' must be an int or a list of one or two integers")

    if spec_unit is not None:
        result = result * spec_unit if get_spectral_unit(result) is None else result

    return result if keep_unit else get_value(result)


def spectral_world_2_idx(
    spectral_axis: SpectralAxis | Quantity,
    value: u.Quantity | float
) -> int:
    """
    Return the index of the nearest spectral channel to a given value.
    If `value` has no unit, the cube unit is assumed.

    Parameters
    ----------
    spectral_axis : Quantity | SpectralAxis
        The spectral axis (e.g., wavelength, frequency, or
        velocity) as an `astropy.units.Quantity` or a
        `specutils.spectra.spectral_axis.SpectralAxis` array.
    value : Quantity | float
        Spectral value in the same units as the cube's spectral axis.

    Returns
    -------
    int :
        Index of the nearest spectral channel.

    Raises
    ------
    ValueError :
        If `spectral_axis` is not a `SpectralAxis` or `Quantity`.
    UnitsError :
        If a unit mismatch is detected.
    """
    spectral_axis = _get_spectral_axis(spectral_axis)
    _valid_types = (Quantity, SpectralAxis) if _HAS_SPECUTILS else (Quantity,)
    if not isinstance(spectral_axis, _valid_types):
        raise ValueError(
            'spectral_axis must be a Quantity or SpectralAxis!'
        )
    if not isinstance(value, Quantity) and spectral_axis.unit is not None:
        value = value * spectral_axis.unit

    ensure_common_unit([spectral_axis, value], on_mismatch='raise')

    return int(np.argmin(np.abs(spectral_axis - value)))


def propagate_flux_errors(
    errors: NDArray,
    method: Literal['mean', 'sum', 'median'] | _Unset = _UNSET
):
    """
    Compute propagated flux errors from individual
    pixel errors in a spectrum.

    Parameters
    ----------
    errors : np.ndarray
        Either:

        * 2D array with shape (N_spectra, N_pixels), or
        * 1D array with shape (N_pixels,) for a single spectrum.

    method : {'mean', 'sum', 'median'} | _Unset, optional, default=_UNSET
        Flux extraction method. If `_UNSET`, uses
        `config.propagate_flux_error_method`.

    Returns
    -------
    flux_errors : np.ndarray
        1D array of propagated flux errors (shape N_spectra).
    """
    method = _resolve_default(method, config.propagate_flux_error_method)
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

    elif method == 'interquartile':
        q1 = np.nanpercentile(errors, 25, axis=1)
        q3 = np.nanpercentile(errors, 75, axis=1)
        flux_errors = q3 - q1

    else:
        raise ValueError(f"Unknown flux extraction method '{method}'.")

    return flux_errors


@overload
def _get_spectral_axis(
    obj: SpectralAxis | Quantity | SpectralCube
) -> SpectralAxis | Quantity: ...

@overload
def _get_spectral_axis(
    obj: None
) -> None: ...

@overload
def _get_spectral_axis(
    obj: Any
) -> SpectralAxis | Quantity | None: ...

def _get_spectral_axis(obj: Any) -> SpectralAxis | Quantity | None:
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
    if _HAS_SPECUTILS and isinstance(obj, SpectralAxis):
        return obj

    if isinstance(obj, Quantity) and _is_spectral_axis(obj):
        return obj

    spectral_axis = getattr(obj, 'spectral_axis', None)
    if spectral_axis is not None and _is_spectral_axis(spectral_axis):
        return spectral_axis

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            spectral_axis = _get_spectral_axis(data)
            if spectral_axis is not None:
                return spectral_axis

    return None


def _get_flux(obj: Any) -> Quantity | None:
    """
    Get the flux associated with an object, if one exists.

    The function follows a recursive search strategy:

    1. If `obj` exposes a `.flux` attribute, that attribute is returned.
    2. If `obj` is an `astropy.units.Quantity`, it is returned directly.
    3. If `obj` exposes a `.data` attribute, the function is applied recursively
       to `obj.data`.

    If no flux can be identified, the function returns None.

    Parameters
    ----------
    obj : Any
        Object from which to extract a flux array. This may include:
        - objects exposing a `.flux` attribute
        - an `astropy.units.Quantity`
        - container objects with a `.data` attribute holding one of the above

    Returns
    -------
    flux : astropy.units.Quantity or None
        Flux array associated with the object, or None if unavailable.
    """
    if hasattr(obj, 'flux'):
        return obj.flux

    if isinstance(obj, Quantity):
        return obj

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            return _get_flux(data)

    return None


def _get_continuum(
    obj: Any,
    fit_method: Literal['fit_continuum', 'generic'] = 'fit_continuum',
    region: SpectralRegion | list[tuple] | None = None
) -> Quantity | None:
    """
    Get or compute the continuum associated with an object, if possible.

    The function follows a recursive search-and-construction strategy:

    1. If `obj` exposes a `.continuum` attribute, that value is returned.
    2. If `obj` is a `specutils.Spectrum`, a continuum is computed using
       `compute_continuum`.
    3. If `obj` exposes a `.spectrum` attribute, the function is applied
       recursively to that attribute.
    4. If both a spectral axis and flux can be extracted from `obj`,
       a temporary `Spectrum` is constructed and a continuum is computed.

    If no continuum can be retrieved or constructed, the function returns None.

    Parameters
    ----------
    obj : Any
        Object from which to retrieve or compute a continuum. This may include:

        * a `Spectrum`-like object exposing `.continuum` or `.continuum_fit`
        * a `specutils.Spectrum`
        * container objects exposing `.spectrum`
        * objects from which both a spectral axis and flux can be inferred

    fit_method : str, optional, default='fit_continuum'
        Continuum fitting method passed to `fit_continuum`.
        Either `'fit_continuum'`, or `'generic'`.
    region : SpectralRegion | list[tuple] | None, optional, default=None
        Spectral region(s) used during continuum fitting.

    Returns
    -------
    continuum : astropy.units.Quantity | None
        Continuum flux array evaluated on the spectral axis, or None if
        no continuum can be determined.
    """
    if hasattr(obj, 'continuum'):
        return obj.continuum

    if _HAS_SPECUTILS and isinstance(obj, Spectrum):
        return compute_continuum(
            obj, fit_method=fit_method, region=region
        )

    if hasattr(obj, 'spectrum'):
        spectrum = obj.spectrum
        if spectrum is not obj:
            return _get_continuum(
                spectrum, fit_method=fit_method, region=region
            )

    spectral_axis = _get_spectral_axis(obj)
    flux = _get_flux(obj)

    if (
        _HAS_SPECUTILS and
        spectral_axis is not None and
        spectral_axis is not obj and
        flux is not None and
        flux is not obj
    ):
        spectrum = Spectrum(
            spectral_axis=spectral_axis, flux=flux
        )
        return _get_continuum(
            spectrum, fit_method=fit_method, region=region
        )

    return None
