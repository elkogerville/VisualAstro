"""
Author: Elko Gerville-Reache
Date Created: 2025-11-16
Date Modified: 2026-11-16
Description:
    Spectral region module.
"""

from __future__ import annotations

from typing import Any, TypeAlias, overload

import numpy as np
from astropy.units import Quantity, StructuredUnit
from astropy.units.core import UnitBase

from visualastro.core.data import get_value
from visualastro.core.units import get_unit
from visualastro.optional_dependencies._specutils import SpectralRegion
from visualastro.optional_dependencies.register import _require_dependency

SpectralRegionInput: TypeAlias = (
    SpectralRegion |
    Quantity |
    tuple[Quantity, Quantity] |
    list[tuple[Quantity, Quantity]]
)


__all__ = [
    'spectral_region',
    'spectral_region_mask',
]


@overload
def spectral_region(
    obj: SpectralRegionInput,
    unit: UnitBase | StructuredUnit | None = None
) -> SpectralRegion: ...

@overload
def spectral_region(
    obj: None,
    unit: UnitBase | StructuredUnit | None = None
) -> None: ...

def spectral_region(
    obj: SpectralRegionInput | None,
    unit: UnitBase | StructuredUnit | None = None
) -> SpectralRegion | None:
    """
    Create a :class:`specutils.spectra.spectral_region.SpectralRegion`
    instance from a wide range of inputs.

    Call signatures::

        spectral_region((8.3, 9.8) * u.um)
        spectral_region([(8.3, 9.8), ...] * u.um)
        spectral_region((8.3*u.um, 9.8*u.um))
        spectral_region([(8.3*u.um, 9.8*u.um), ...])

    Parameters
    ----------
    obj : SpectralRegion | Quantity | tuple | list[tuple] | None
        Region specification. Accepted forms:

        - :class:`~specutils.SpectralRegion`: returned as-is
        - `(low, high) * unit`: single region with shared unit
        - `[(low, high), ...] * unit`: multiple regions with shared unit
        - `(Quantity, Quantity)`: single region with explicit units
        - `[(Quantity, Quantity), ...]`: multiple regions with explicit units
        - `None`: returned as-is

    Returns
    -------
    SpectralRegion :
        A SpectralRegion object containing one or more spectral subregions.

    Raises
    ------
    ValueError
        If a Quantity has incorrect shape (must be (2,) or (N, 2)).
    ValueError
        If the input `obj` cannot be coerced into a `SpectralRegion`.
    """
    if obj is None:
        return None

    spectralregion = _to_spectral_region(obj)
    if unit is None:
        return spectralregion

    unit = get_unit(unit)
    bounds = []
    for lower, upper in spectralregion.subregions:
        lo = lower.to(unit, equivalencies=u.spectral())
        hi = upper.to(unit, equivalencies=u.spectral())
        bounds.append((min(lo, hi), max(lo, hi)))
    return SpectralRegion(bounds)


def _to_spectral_region(obj: SpectralRegionInput) -> SpectralRegion:
    """Helper for `spectral_region`."""
    _require_dependency('specutils')

    if isinstance(obj, SpectralRegion):
        return obj

    region: Any = obj

    if isinstance(region, Quantity):
        arr = np.asarray(obj)

        # single region
        if arr.shape == (2,):
            lo = region[0]
            hi = region[1]
            region = [(lo, hi)]

        # multiple regions
        elif arr.ndim == 2 and arr.shape[1] == 2:
            region = [(row[0], row[1]) for row in region]

        else:
            raise ValueError(f'Quantity must have shape (2,) or (N,2), got {arr.shape}')

    # case: quantities inside list/tuple
    elif (isinstance(region, tuple) and
        len(region) == 2 and
        all(isinstance(x, Quantity) for x in region)):
        region = [region]

    elif not isinstance(obj, (list, tuple)):
        raise TypeError(
            f'Expected SpectralRegion, Quantity, tuple, or list, got {type(region).__name__}'
        )

    try:
        return SpectralRegion(region)
    except Exception as e:
        raise ValueError(
            f'Could not construct SpectralRegion from {type(region).__name__}: {e}'
        ) from e


def spectral_region_mask(x, spectral_region):
    """
    Return a boolean mask selecting values within a SpectralRegion.

    This function constructs the union of all subregions in the input
    SpectralRegion and returns a boolean array that is True for elements
    of `x` that fall within any subregion.

    Parameters
    ----------
    x : array-like or `~astropy.units.Quantity`
        1D array of spectral coordinates (e.g., wavelength or frequency).
        If a Quantity is provided, its values are converted to plain
        numeric values.

    spectral_region : `~specutils.SpectralRegion` or region-like
        Spectral region definition. May contain multiple disjoint
        subregions. The mask is the logical union of all subregions.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask with the same shape as `x`. True where the spectral
        coordinate lies within any subregion, False otherwise.

    Raises
    ------
    TypeError
        If `spectral_region.subregions` is None.

    Notes
    -----
    - Subregions are combined using logical OR (union semantics).
    - This function assumes that `x` and the subregion bounds are in
        compatible units. No explicit unit conversion is performed.
    - Boundary comparisons are inclusive (`>= xmin` and `<= xmax`).

    Examples
    --------
    >>> mask = spectral_region_mask(wavelength, region)
    >>> n_valid = np.count_nonzero(mask & np.isfinite(flux))
    """
    x = get_value(x)
    spectral_region = _require_spectral_region(spectral_region)
    subregions = spectral_region.subregions
    if subregions is None:
        raise TypeError(
            'spectral_region.subregions is None!'
        )

    mask = np.zeros(x.shape, dtype=bool)

    for region in subregions:
        xmin, xmax = region[0].value, region[1].value
        mask |= (x >= xmin) & (x <= xmax)

    return mask


def _require_spectral_region(
    obj: SpectralRegionInput
) -> SpectralRegion:
    """
    Enforce that the input `obj` is a `SpectralRegion`.

    This is a wrapper of `spectral_region`, which unlike
    this function allows for None inputs.

    Parameters
    ----------
    obj : SpectralRegion, Quantity, tuple, or list
        Region specification. Accepted forms:

        - `SpectralRegion`: returned as-is
        - `(low, high) * unit`: single region with shared unit
        - `[(low, high), ...] * unit`: multiple regions with shared unit
        - `(Quantity, Quantity)`: single region with explicit units
        - `[(Quantity, Quantity), ...]`: multiple regions with explicit units

    Returns
    -------
    SpectralRegion :
        Input region as a SpectralRegion

    Raises
    ------
    ValueError :
        If input `obj` results in `None` when converted to
        a SpectralRegion.
    """
    region = spectral_region(obj)

    if region is None or not isinstance(region, SpectralRegion):
        raise ValueError(f'A SpectralRegion is required; got {type(region).__name__}')
    return region
