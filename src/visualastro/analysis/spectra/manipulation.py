"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-11-14
Description:
    Spectra manipulation functions.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import Literal, overload

import astropy.constants as constants
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike

from visualastro.analysis.spectra.regions import (
    SpectralRegionInput,
    spectral_region,
)
from visualastro.core.array_types import _is_iterable
from visualastro.core.configuration import _UNSET, _resolve_default, _Unset, config
from visualastro.core.data import get_value, to_list
from visualastro.core.numerical import mask_within_range
from visualastro.core.units import ensure_common_unit
from visualastro.optional_dependencies._dust_extinction import G23, M14, WD01
from visualastro.optional_dependencies._specutils import (
    Spectrum,
    _extract_region,
    _fit_continuum,
    _fit_generic,
)
from visualastro.optional_dependencies.register import _require_dependency

__all__ = [
    'compute_continuum',
    'continuum_spectrum',
    'deredden_flux',
    'extract_region',
    'splice_spectra',
    'shift_to_rest_frame',
]


def compute_continuum(
    spectrum: Spectrum,
    fit_method: Literal['fit_continuum', 'generic'] = 'fit_continuum',
    region: SpectralRegionInput | None = None
) -> Quantity:
    """
    Compute the continuum of a 1D spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Input 1D spectrum containing `flux` and `spectral_axis`.
    fit_method : {'fit_continuum', 'generic'}, optional, default='fit_continuum'
        Continuum fitting method (case-insensitive).

        - 'fit_continuum' : `specutils.fitting.fit_continuum` restricted to `region`.
        - 'generic' : `specutils.fitting.fit_generic_continuum`.

    region : SpectralRegionInput | None, optional, default=None
        Spectral region(s) included in the fit. Used only when
        `fit_method='fit_continuum'`; ignored otherwise.

        - Each region is a `(lower, upper)` pair of `Quantity` bounds
        - Multiple regions restrict the fit to disjoint windows, e.g. to
          exclude emission/absorption features that bias the continuum.

        Example: exclude an emission peak at 7 um::

            region = [(6.5 * u.um, 6.9 * u.um), (7.1 * u.um, 7.9 * u.um)]

    Returns
    -------
    continuum : Quantity
        Continuum flux evaluated at `spectrum.spectral_axis`.

    Raises
    ------
    TypeError
        If `spectrum` is not a `Spectrum` instance.
    ValueError
        If `fit_method` is not one of {'fit_continuum', 'generic'}.

    Notes
    -----
    - All warnings raised during fitting are suppressed.
    - `region` is silently ignored when `fit_method='generic'`.
    """
    _require_dependency('specutils')

    if not isinstance(spectrum, Spectrum):
        raise TypeError(
            f'Input object is not a Spectrum: {type(spectrum).__name__}'
        )

    if spectrum.flux.ndim != 1:
        raise ValueError(f'Expected 1D spectrum, got ndim={spectrum.flux.ndim}')

    method = str(fit_method).lower()
    if method not in {'fit_continuum', 'generic'}:
        raise ValueError(
            f"Unknown fit_method: {fit_method!r}. "
            "Expected 'fit_continuum' or 'generic'."
        )

    spectral_axis = spectrum.spectral_axis

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'fit_continuum':
            window = spectral_region(region, spectral_axis.unit)
            fit = _fit_continuum(spectrum, window=window)
        else:
            fit = _fit_generic(spectrum)

    return fit(spectral_axis)


@overload
def continuum_spectrum(
    spectrum: Spectrum,
    fit_method: Literal['fit_continuum', 'generic'] = ...,
    region: SpectralRegionInput | None = ...,
) -> Spectrum: ...

@overload
def continuum_spectrum(
    spectrum: Sequence[Spectrum],
    fit_method: Literal['fit_continuum', 'generic'] = ...,
    region: SpectralRegionInput | None = ...,
) -> list[Spectrum]: ...

@overload
def continuum_spectrum(
    spectrum: Spectrum | Sequence[Spectrum],
    fit_method: Literal['fit_continuum', 'generic'] = ...,
    region: SpectralRegionInput | None = ...,
) -> Spectrum | list[Spectrum]: ...

def continuum_spectrum(
    spectrum: Spectrum | Sequence[Spectrum],
    fit_method: Literal['fit_continuum', 'generic'] = 'fit_continuum',
    region: SpectralRegionInput | None = None,
) -> Spectrum | list[Spectrum]:
    """
    Compute the continuum of one or more 1D spectra and return it as `Spectrum`.

    Parameters
    ----------
    spectrum : Spectrum | Sequence[Spectrum]
        Input 1D spectrum or sequence of 1D spectra. Sequence elements may
        differ in length and spectral axis.
    fit_method : {'fit_continuum', 'generic'}, optional, default='fit_continuum'
        Continuum fitting method. See `compute_continuum`.
    region : SpectralRegionInput | None, optional, default=None
        Spectral region(s) included in the fit, applied to every spectrum.
        See `compute_continuum`.

    Returns
    -------
    continuum : Spectrum | list[Spectrum]
        `Spectrum` for `Spectrum` input; `list[Spectrum]` (same order) for
        sequence input.

        - `meta`, `mask`, redshift, `rest_value`, `velocity_convention`,
          and `bin_specification` are inherited from the input.
        - `uncertainty` is not propagated (the fit provides none).
        - `meta['continuum_fit']` records `fit_method` and `region`.

    Raises
    ------
    TypeError
        If `spectrum` is not a `Spectrum` or a sequence of `Spectrum`.
    ValueError
        If `spectrum` is an empty sequence, any spectrum is not 1D, or
        `fit_method` is invalid.

    Notes
    -----
    - Errors in any single fit propagate; no partial results are returned.
    - Fits run serially.
    """
    _require_dependency('specutils')
    single = isinstance(spectrum, Spectrum)
    if single:
        return _continuum_spectrum_1d(spectrum, fit_method, region)

    if not isinstance(spectrum, Sequence) or not _is_iterable(spectrum):
        raise TypeError(
            'Expected Spectrum or Sequence[Spectrum], '
            f'got {type(spectrum).__name__}'
        )
    if len(spectrum) == 0:
        raise ValueError('Empty sequence of spectra.')

    bad = {type(s).__name__ for s in spectrum if not isinstance(s, Spectrum)}
    if bad:
        raise TypeError(f'Non-Spectrum elements in sequence: {sorted(bad)}')

    results = [_continuum_spectrum_1d(s, fit_method, region) for s in spectrum]

    return results[0] if single else results


def _continuum_spectrum_1d(
    spectrum: Spectrum,
    fit_method: Literal['fit_continuum', 'generic'],
    region: SpectralRegionInput | None,
) -> Spectrum:
    """Helper for `continuum_spectrum`."""
    continuum = compute_continuum(spectrum, fit_method=fit_method, region=region)

    meta = copy.deepcopy(spectrum.meta)

    return Spectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=continuum,
        mask=spectrum.mask,
        meta=meta,
        velocity_convention=spectrum.velocity_convention,
    )


def deredden_flux(
    wavelength: ArrayLike | u.Quantity,
    flux: ArrayLike | u.Quantity,
    Rv: float | _Unset =_UNSET,
    Ebv: float | _Unset =_UNSET,
    deredden_method: Literal['G23', 'WD01', 'M14'] | _Unset = _UNSET,
    region: str | _Unset = _UNSET
):
    """
    Apply extinction correction (dereddening) to a spectrum.
    Default values are for LMC parameters.

    Parameters
    ----------
    wavelength : ArrayLike | u.Quantity
        Wavelength array (in Angstroms, microns, or units expected by the
        extinction law being used).
    flux : ArrayLike | u.Quantity
        Observed flux values at the corresponding wavelengths. Must be in
        linear units (e.g., erg/s/cm^2/Å, Jy).
    Rv : float | _Unset, optional, default=_UNSET
        Ratio of total-to-selective extinction (A_V / E(B-V)).
        If `_UNSET`, uses `config.deredden.Rv`.
    Ebv : float | _Unset, optional, default=_UNSET
        Color excess E(B-V), representing the amount of reddening.
        If `_UNSET, uses `config.deredden.Ebv`.
    deredden_method : {'G23', 'WD01', 'M14'} | _Unset, optional, default=_UNSET
        Choice of extinction law:

        * 'G23' : Gordon et al. (2023)
        * 'WD01': Weingartner & Draine (2001)
        * `'M14'` : Maíz Apellániz et al. (2014)

        If `_UNSET`, uses `config.deredden.method`.
    region : str | _Unset, optional, default=_Unset
        For `deredden_method='WD01'` extinction: the environment/region
        to use (e.g., `'MWAvg'`, `'LMC'`, `'LMCAvg'`, `'SMCBar'`).
        Ignored for other methods. If `_UNSET`, uses `config.deredden.region`.

    Returns
    -------
    deredden_flux : array-like
        Flux array corrected for extinction.
    """
    _require_dependency('dust_extinction')
    Rv = _resolve_default(Rv, config.deredden.Rv)
    Ebv = _resolve_default(Ebv, config.deredden.Ebv)
    dered_method = str(_resolve_default(deredden_method, config.deredden.method))
    region = _resolve_default(region, config.deredden.region)

    methods = {
        'G23': G23,
        'WD01': WD01,
        'M14': M14
    }
    if dered_method not in methods:
        raise ValueError(
            f"Unknown deredden_method '{dered_method}'. "
            "Choose from 'G23', 'WD01', or 'M14'."
        )
    deredden = methods[dered_method]

    extinction = deredden(region) if dered_method == 'WD01' else deredden(Rv=Rv)

    dereddened_flux = flux / extinction.extinguish(wavelength, Ebv=Ebv)

    return dereddened_flux


def extract_region(
    spectrum: Spectrum | Sequence[Spectrum],
    region: SpectralRegionInput,
    return_single_spectrum: bool = False,
    preserve_wcs: bool = False,
) -> Spectrum | list[Spectrum] | list[Spectrum | list[Spectrum]]:
    """
    Extract one or more spectral regions from one or more spectra.

    Wrapper around `specutils.manipulation.extract_region` that also accepts
    a sequence of spectra.

    Parameters
    ----------
    spectrum : Spectrum | Sequence[Spectrum]
        Input spectrum or sequence of spectra. All spectra are processed
        with identical parameters.
    region : SpectralRegionInput
        Region(s) to extract, converted with `spectral_region`. Applied
        to every spectrum. A region may contain multiple sub-regions.
    return_single_spectrum : bool, optional, default=False
        If `region` has multiple sub-regions, concatenate them into a single
        `Spectrum` instead of returning a list. No effect for a
        single-sub-region `region`.

    Returns
    -------
    Spectrum | list[Spectrum] | list[Spectrum | list[Spectrum]]
        The return type depends on the input type and on `region`, never on
        the sequence length.

        - `Spectrum` input, single sub-region (or `return_single_spectrum=True`):
          `Spectrum`.
        - `Spectrum` input, multiple sub-regions, `return_single_spectrum=False`:
          `list[Spectrum]`, one per sub-region.
        - `Sequence[Spectrum]` input: a list, same order as the input, whose
          elements follow the two rules above.

    Raises
    ------
    TypeError
        If `spectrum` is not a `Spectrum` or a sequence of `Spectrum`
        (only if validation is added; see below).
    """
    single = isinstance(spectrum, Spectrum)
    spectra = [spectrum] if single else list(spectrum)
    region = spectral_region(region)

    results = [
        _extract_region(
            s,
            region=region,
            return_single_spectrum=return_single_spectrum,
        )
        for s in spectra
    ]

    return results[0] if single else results


def splice_spectra(
    spectra: Spectrum | Sequence[Spectrum],
    wave_cutoffs: Sequence[float] | None = None,
    normalized: bool = False,
) -> Spectrum:
    """
    Splice 1D spectra into a single `Spectrum` by concatenation.

    Parameters
    ----------
    spectra : Spectrum | Sequence[Spectrum]
        Spectra to splice, in spectral order.
    wave_cutoffs : Sequence[float] | None, optional, default=None
        Boundary wavelengths `[λ0, ..., λN]` for N spectra. Spectrum `i` is
        masked to `λi <= wavelength <= λi+1`. Must have N+1 entries.
    normalized : bool, optional, default=False
        If True, splice `spectrum.normalize` instead of `spectrum.flux`.

    Returns
    -------
    Spectrum
        Spliced spectrum. Only `spectral_axis` and `flux` are set.
    """
    _require_dependency('specutils')
    segments = _masked_segments(to_list(spectra), wave_cutoffs, normalized)
    return Spectrum(
        spectral_axis=np.concatenate([w for w, _ in segments]),
        flux=np.concatenate([f for _, f in segments]),
    )


def _masked_segments(spectra, wave_cutoffs, normalized):
    """Return `(spectral_axis, flux)` per spectrum, masked to its cutoff window."""
    ensure_common_unit(spectra)
    segments = []
    for i, s in enumerate(spectra):
        wave = s.spectral_axis
        flux = s.normalize if normalized else s.flux
        if wave_cutoffs is not None:
            mask = mask_within_range(get_value(wave), (wave_cutoffs[i], wave_cutoffs[i + 1]))
            wave, flux = wave[mask], flux[mask]
        segments.append((wave, flux))
    return segments


def shift_to_rest_frame(
    spectral_axis: u.Quantity,
    radial_vel: float | u.Quantity | None,
) -> u.Quantity:
    """
    Shift spectral axis to rest frame using a radial velocity.

    Uses the non-relativistic Doppler approximation. If ``radial_vel`` is
    None, ``spectral_axis`` is returned unchanged.

    Parameters
    ----------
    spectral_axis : astropy.units.Quantity
        The spectral axis to shift. Can be in frequency or wavelength units.
    radial_vel : float, astropy.units.Quantity or None
        Radial velocity. Floats are interpreted as km/s. Positive values
        correspond to a redshift (moving away). If ``None``, no shift is applied.

    Returns
    -------
    shifted_axis : astropy.units.Quantity
        The spectral axis shifted to the rest frame according to the given
        radial velocity. If the input is in frequency units, the classical
        Doppler formula for frequency is applied; otherwise, the classical
        formula for wavelength is applied.
    """
    if radial_vel is None:
        return spectral_axis

    beta = (u.Quantity(radial_vel, unit=u.km/u.s) / constants.c).to_value(
        u.dimensionless_unscaled
    )

    if spectral_axis.unit.is_equivalent(u.m):
        return spectral_axis / (1 + beta)
    if spectral_axis.unit.is_equivalent(u.Hz, u.spectral()):
        return spectral_axis / (1 - beta)
    raise u.UnitsError(
        f"Cannot apply a Doppler shift to unit '{spectral_axis.unit}'."
    )
