'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-01-18
Description:
    Utility functions for astropy units.
Dependencies:
    - astropy
    - numpy
'''

import warnings
from typing import Any
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import (
    dimensionless_unscaled, physical,
    Quantity, spectral, Unit, UnitBase,
    UnitConversionError, UnitsError, StructuredUnit
)
from astropy.units.physical import PhysicalType
import numpy as np
from .config import get_config_value, config


def convert_units(quantity, unit):
    '''
    Convert an Astropy Quantity to a specified unit, with a fallback if conversion fails.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        The input quantity to convert.
    unit : str, astropy.units.Unit, or None
        The unit to convert to. If None, no conversion is performed.

    Returns
    -------
    astropy.units.Quantity
        The quantity converted to the requested unit if possible; otherwise,
        the original quantity with its existing unit.

    Notes
    -----
    - Uses 'spectral()' equivalencies to allow conversions between
        wavelength, frequency, and velocity units.
    - If conversion fails, prints a warning and returns the original quantity.
    '''
    if unit is None:
        return quantity
    try:
        # convert string unit to Unit if necessary
        target_unit = Unit(unit) if isinstance(unit, str) else unit
        return quantity.to(target_unit, equivalencies=spectral())
    except UnitConversionError:
        print(
            f'Could not convert to unit: {unit}.'
            f'Defaulting to unit: {quantity.unit}.'
            )
        return quantity


def get_units(obj: Any) -> UnitBase | StructuredUnit | None:
    """
    Extract the unit from an object, if it exists.

    This function checks if the object has a .unit attribute,
    or if 'BUNIT' exists in `obj`.

    Parameters
    ----------
    obj : Object
        The input object from which to extract a unit. This can be:
        - an astropy.units.Quantity
        - a fits.Header with a 'BUNIT' key
        - any object with a .data attribute
        - any object with a .header attribute
    Returns
    -------
    astropy.units.Unit or None
        The unit associated with the input object, if it exists.
        Returns None if the object has no unit or if the unit cannot be parsed.
    """
    if isinstance(obj, Quantity):
        return obj.unit

    if isinstance(obj, Header):
        bunit = obj.get('BUNIT')
        return to_unit(bunit)

    unit = getattr(obj, 'unit', None)
    if unit is not None:
        return to_unit(unit)

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            unit = get_units(data)
            if unit is not None:
                return unit

    if hasattr(obj, 'header'):
        header = obj.header
        if header is not obj:
            unit = get_units(header)
            if unit is not None:
                return unit

    return None


def get_physical_type(obj: Any) -> PhysicalType | None:
    """
    Extract the physical type associated with an object's unit.

    Returns None if no physical type is found, or if the unit
    is a `StructuredUnit`.

    Parameters
    ----------
    obj : any
        Object from which a unit can be extracted via `get_units`.

    Returns
    -------
    astropy.units.physical.PhysicalType or None
        Physical type of the unit, or None if unavailable.
    """
    unit = get_units(obj)

    if unit is None:
        return None
    elif isinstance(unit, StructuredUnit):
        return None

    try:
        return unit.physical_type
    except Exception:
        return None


def to_unit(obj: Any) -> UnitBase | StructuredUnit | None:
    """
    Normalize an input into an astropy Unit.

    Parameters
    ----------
    obj : Unit, Quantity, str, or None

    Returns
    -------
    unit : astropy.units.Unit or None
    """
    if isinstance(obj, UnitBase):
        return obj

    elif isinstance(obj, Quantity):
        return obj.unit

    elif isinstance(obj, str):
        try:
            return Unit(obj)
        except Exception:
            return None

    return None


def to_latex_unit(unit: Any, fmt=None) -> str | None:
    """
    Convert an astropy unit string into a LaTeX-formatted label
    for plotting. Returns None if no unit is found.

    Parameters
    ----------
    unit : str or astropy.Unit
        The astropy.Unit or unit string to convert.
    fmt : {'latex', 'latex_inline', 'inline'} or None, optional, default=None
        The format of the unit label. 'latex_inline' and 'inline' uses
        negative exponents while 'latex' uses fractions. If None, uses
        the default value set by `config.unit_label_format`.

    Returns
    -------
    str or None
        A LaTeX-formatted unit label if the input is recognized.
        Returns None if the unit is invalid.
    """
    fmt = get_config_value(fmt, 'unit_label_format')

    if fmt.lower() == 'inline':
        fmt = 'latex_inline'

    try:
        unit = to_unit(unit)
        return unit.to_string(fmt)
    except Exception:
        return None


def ensure_unit_consistency(
    objs,
    *,
    unit=None,
    on_mismatch=None,
    label=None,
):
    """
    Check unit consistency across one or more objects.

    This function verifies that all objects with defined units share the same
    unit. No unit conversion is performed; objects are returned unchanged.
    The function will either raise an UnitsError, issue a warning, or ignore.

    Parameters
    ----------
    objs : object or list or tuple of objects
        Input objects carrying units (e.g. Quantity, DataCube, SpectralCube).
        Objects without units are allowed.
    unit : astropy.units.Unit or str or None, optional, default=None
        Reference unit to compare against. If None, the first non-None unit
        encountered among the inputs is used.
    on_mismatch : {'warn', 'ignore', 'raise'}, optional, default=None
        Action to take when a unit mismatch is detected. If None, uses
        the default value from `config.unit_mismatch`.
    label : str or None, optional
        Optional context label prepended to warnings or errors.

    Returns
    -------
    objs : list
        List of input objects, returned unchanged.

    Raises
    ------
    UnitsError
        If a unit mismatch is detected and `on_mismatch='raise'`.
    """
    on_mismatch = get_config_value(on_mismatch, 'unit_mismatch')

    objs = to_list(objs)
    units = [get_units(obj) for obj in objs]

    if all(u is None for u in units):
        return objs

    ref_unit = Unit(unit) if unit is not None else next(
        u for u in units if u is not None
    )

    prefix = f"{label}: " if label else ""

    out = []
    for i, (obj, u) in enumerate(zip(objs, units)):
        if u is None or u == ref_unit:
            out.append(obj)
            continue

        if on_mismatch == 'raise':
            raise UnitsError(
                f"{prefix}Unit mismatch at index {i}: {u} != {ref_unit}"
            )

        if on_mismatch == 'warn':
            warnings.warn(
                f"{prefix}Unit mismatch at index {i}: {u} != {ref_unit}. "
                "Values may be interpreted incorrectly."
            )

        out.append(obj)

    return out


def _is_spectral_axis(obj):
    """
    Determine whether an object represents a spectral axis.

    An object is considered a spectral axis if it explicitly exposes
    spectral metadata (e.g., `spectral_axis` or `spectral_unit`), or if
    its unit corresponds to a spectral quantity such as wavelength,
    frequency, energy, or Doppler velocity. Length units are only treated
    as spectral when they are convertible via Astropy spectral
    equivalencies.

    Parameters
    ----------
    obj : any
        An object describing an axis. This may be a SpectralCube,
        Spectrum1D, Quantity, or any object exposing a unit.

    Returns
    -------
    bool
        True if the object represents a spectral axis, False otherwise.

    Notes
    -----
    - Length units (e.g., meters, microns) are only interpreted as
      wavelengths when spectral context can be inferred.
    - Plain length quantities without spectral context are not
      automatically treated as spectral axes.
    """

    if hasattr(obj, 'spectral_axis') or hasattr(obj, 'spectral_unit'):
        return True

    unit = get_units(obj)
    if unit is None:
        return False

    physical_type = unit.physical_type

    if physical_type in {
        physical.frequency,
        physical.energy,
        physical.speed,
    }:
        return True

    if physical_type is physical.length:
        try:
            unit.to(u.Hz, equivalencies=u.spectral())
            return True
        except Exception:
            pass

    return False


def _infer_physical_type_label(obj: Any) -> str | None:
    """
    Infer a human-readable physical-type label for an object.

    This function determines an appropriate scientific label
    based on the physical type of an object's associated unit.
    Spectral axes are treated with higher precedence than other
    physical quantities (i.e. 'um' is mapped to 'Wavelength'
    rather than 'Distance'). If no suitable label can be inferred,
    the function returns None.

    The inference logic follows this order:

    1. If the object represents a spectral axis, return a spectral-axis
        label (e.g., 'Wavelength', 'Frequency', 'Velocity').
    2. Otherwise, return a curated physical-type label (e.g., 'Flux',
        'Surface Brightness').
    3. If no mapping exists, return None.

    Parameters
    ----------
    obj : any
        An object describing an axis or quantity. This may be an Astropy
        `Quantity`, a Spectrum-like object exposing spectral metadata, or
        any object from which a unit can be extracted via `get_units`.

    Returns
    -------
    label : str or None
        A human-readable physical-type label, or None if no appropriate
        label can be inferred.

    Notes
    -----
    - Spectral axes take precedence over generic physical types.
    - Length units are only interpreted as wavelengths when spectral context
        can be inferred (via Astropy spectral equivalencies or explicit spectral
        metadata). For example, units such as 'um', 'm', etc.. are mapped to
        'Wavelength' but distance units like 'pc' would return 'Distance'.
    - Structured units and non-scalar physical types are ignored.
    - This function does not raise exceptions.
    """
    physical_type = get_physical_type(obj)
    if physical_type is None:
        return None

    if _is_spectral_axis(obj):
        return config._SPECTRAL_TYPE_LABELS.get(
            physical_type, 'Spectral Axis'
        )

    return config._PHYSICAL_TYPE_LABELS.get(
        physical_type, None
    )


def _check_unit_equality(unit1, unit2, name1='unit1', name2='unit2'):
    """
    Validate that two units are exactly equal.

    Parameters
    ----------
    unit1, unit2 : str or astropy.units.Unit or None
        Units to compare. None means 'unitless'.
    name1, name2 : str
        Labels used in error messages.

    Raises
    ------
    UnitsError
        If units differ (either convertible or incompatible).
    """
    # case 1: either of the units are None
    if unit1 is None or unit2 is None:
        return

    try:
        u1 = Unit(unit1)
        u2 = Unit(unit2)
    except Exception:
        raise UnitsError('Invalid unit(s) supplied')

    # case 1: units are exactly equal
    if u1 == u2:
        return

    # case 2: equivalent but not equal
    if u1.is_equivalent(u2):
        raise UnitsError(
            f'{name1} and {name2} units are equivalent but not equal '
            f'({u1} vs {u2}). Convert one to match.'
        )

    # case 3: mismatch
    raise UnitsError(
        f'{name1} and {name2} have incompatible units: '
        f'{u1} vs {u2}.'
    )


def _is_unitless(obj):
    """
    Validate that an object has no unit.

    u.dimensionless_unscaled is treated
    as no units.

    Parameters
    ----------
    obj : object
        Object to check if unitless.

    Returns
    -------
    bool :
        If object has a unit.
    """
    if isinstance(obj, Quantity):
        return obj.unit == dimensionless_unscaled

    unit = getattr(obj, 'unit', None)
    if isinstance(unit, UnitBase):
        return unit == dimensionless_unscaled

    return True


def _validate_common_unit(objs, *, label=None):
    '''
    Extract units of each object in objs
    and validate that units match. This is
    an internal function used for input validation.

    Parameters
    ----------
    obj : array-like
        A single object or list/array of objects with unit data.
        Can be Quantities, Headers with 'BUNIT', or a mix of both.

    Returns
    -------
    None
        If no units are present.
    astropy.units.Unit
        If units are present and are consistent.

    Raises
    ------
    UnitsError
        If units exist and do not match, or if BUNIT is invalid.
    '''
    if not isinstance(objs, (list, tuple)):
        objs = [objs]

    # create unique set of each unit
    units = set()
    for i, obj in enumerate(objs):
        if isinstance(obj, Quantity):
            units.add(obj.unit)
        elif hasattr(obj, 'unit'):
            units.add(obj.unit)
        elif isinstance(obj, Header) and 'BUNIT' in obj:
            try:
                units.add(Unit(obj['BUNIT'])) # type: ignore
            except Exception as e:
                raise UnitsError(
                    f'Invalid BUNIT in header at index {i}: '
                    f"'{obj['BUNIT']}' ({e})"
                )
    # raise error if more than one unit found
    if len(units) > 1:
        prefix = f'{label} ' if label is not None else ''
        raise UnitsError(
            f'Inconsistent {prefix}units found: {units}'
        )

    # return either single unit or None
    return next(iter(units), None)
