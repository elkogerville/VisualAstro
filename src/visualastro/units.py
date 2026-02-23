"""
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-02-23
Description:
    Utility functions for astropy units.
Dependencies:
    - astropy
    - numpy
    - specutils
"""

import warnings
from typing import Any, overload
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import (
    dimensionless_unscaled,
    Quantity, StructuredUnit, Unit, UnitBase,
    UnitConversionError, UnitsError
)
from astropy.units.physical import PhysicalType
import numpy as np
from specutils import SpectralAxis, SpectralRegion
from .config import get_config_value, config
from .numerical_utils import to_list
from .utils import _type_name


def convert_quantity(
    quantity,
    unit,
    *,
    equivalencies=None,
    on_failure: str = 'warn',
):
    """
    Convert an Astropy Quantity to a specified unit.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        Input quantity to convert.
    unit : str or astropy.units.Unit or None
        Target unit. If None, the input is returned unchanged.
    equivalencies : list or None, optional
        Unit equivalencies to use during conversion
        (e.g. `spectral()`). Default is None.
    on_failure : {'warn', 'ignore', 'raise'}, optional, default='warn'
        Behavior if unit conversion fails.

    Returns
    -------
    astropy.units.Quantity
        Converted quantity, or the original quantity if conversion
        fails and `on_failure` is not 'raise'.

    Raises
    ------
    UnitConversionError
        If conversion fails and `on_failure='raise'`.
    """
    if unit is None:
        return quantity

    target_unit = to_unit(unit)

    try:
        return quantity.to(target_unit, equivalencies=equivalencies)
    except UnitConversionError as exc:
        if on_failure == 'raise':
            raise UnitConversionError
        if on_failure == 'warn':
            warnings.warn(
                f'Could not convert from {quantity.unit} to {target_unit}; '
                f'leaving quantity unchanged.'
            )
        return quantity


def get_unit(obj: Any) -> UnitBase | StructuredUnit | None:
    """
    Extract the unit from an object, if it exists.

    This function checks if the object has a `.unit` attribute,
    or if 'BUNIT' exists in `obj`.

    Parameters
    ----------
    obj : Object
        The input object from which to extract a unit. This can be:
        - an astropy UnitBase
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
    if obj is None:
        return None

    if isinstance(obj, UnitBase):
        return obj

    if isinstance(obj, Quantity):
        return obj.unit

    if isinstance(obj, Header):
        bunit = obj.get('BUNIT', None)
        return to_unit(bunit)

    unit = getattr(obj, 'unit', None)
    if unit is not None:
        return to_unit(unit)

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            unit = get_unit(data)
            if unit is not None:
                return unit

    if hasattr(obj, 'header'):
        header = obj.header
        if header is not obj:
            unit = get_unit(header)
            if unit is not None:
                return unit

    return None


def get_spectral_unit(obj: Any) -> UnitBase | StructuredUnit | None:
    """
    Extract a spectral (wavelength/frequency/energy) unit
    from an object, if one can be identified.

    The function follows a recursive search strategy:

    1. If ``obj`` is a ``SpectralAxis``, its ``.unit`` attribute is returned.
    2. If ``obj`` is an ``astropy.units.Quantity``, its unit is
       returned if it is convertible to a spectral unit using
       ``u.spectral()`` or ``u.doppler_radio`` equivalencies.
    3. If ``obj`` has a ``.spectral_unit`` attribute, the function
       is applied recursively to ``obj.spectral_unit``.
    4. If ``obj`` has a ``.data`` attribute, the function is applied
       recursively to ``obj.data``.
    5. If ``obj`` has a ``.spectral_axis`` attribute, the function is applied
       recursively to ``obj.spectral_axis``.

    If no spectral unit can be identified, the function returns None.

    Parameters
    ----------
    obj : Any
        Input object from which to extract a spectral unit:
        - `astropy.coordinates.SpectralAxis`
        - `astropy.units.Quantity` with spectral-equivalent units
        - objects exposing a `.spectral_unit` attribute
        - objects exposing a `.spectral_axis` attribute
        - container objects with a `.data` attribute holding one of the above

    Returns
    -------
    astropy.units.UnitBase or astropy.units.StructuredUnit or None
        A spectral unit (e.g. micron, Angstrom, Hz, eV) if found and valid,
        otherwise None.
    """
    if isinstance(obj, SpectralAxis):
        return to_unit(obj.unit)

    if isinstance(obj, Quantity) and _is_spectral_axis(obj):
        return to_unit(obj.unit)

    unit = getattr(obj, 'spectral_unit', None)
    if unit is not None and _is_spectral_axis(unit):
        return to_unit(unit)

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            unit = get_spectral_unit(data)
            if unit is not None:
                return unit

    if hasattr(obj, 'spectral_axis'):
        spectral_axis = obj.spectral_axis
        if spectral_axis is not obj:
            unit = get_spectral_unit(spectral_axis)
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
        Object from which a unit can be extracted via `get_unit`.

    Returns
    -------
    astropy.units.physical.PhysicalType or None
        Physical type of the unit, or None if unavailable.
    """
    unit = get_unit(obj)

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
    if obj is None:
        return None

    if isinstance(obj, UnitBase):
        return obj

    elif isinstance(obj, Quantity):
        return obj.unit

    elif isinstance(obj, str):
        try:
            return Unit(obj)
        except UnitsError:
            return None

    return None

@overload
def unit_2_string(
    unit: None,
    fmt: str | None = None
) -> None: ...

@overload
def unit_2_string(
    unit: Quantity | UnitBase | StructuredUnit | str,
    fmt: str | None = None
) -> str: ...

def unit_2_string(
    unit: Quantity | UnitBase | StructuredUnit | str | None,
    fmt: str | None = None
) -> str | None:
    """
    Convert an astropy unit to a fits compliant string representation.

    Parameters
    ----------
    unit : Quantity, Unit, str, or None
        Astropy unit. If ``Quantity``, uses
        the unit of the ``Quantity``.
    fmt : {'latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip'} or None, optional, default=None
        String format for returned unit. ``'inline'`` works as an alias for
        ``'latex_inline'``. If ``None``, does not use a format.

    Returns
    -------
    str :
        Fits compliant string representation of the unit.
    """
    unit = to_unit(unit)
    if unit is None:
        return None

    if isinstance(fmt, str):
        fmt = fmt.lower()
        if fmt == 'inline':
            fmt = 'latex_inline'

        return unit.to_string(format=fmt)

    return unit.to_string()


@overload
def to_latex_unit(
    unit: Quantity | UnitBase | StructuredUnit | str
) -> str: ...

@overload
def to_latex_unit(
    unit: None
) -> None: ...

def to_latex_unit(
    unit: Quantity | UnitBase | StructuredUnit | str | None,
    fmt: str | None = None
) -> str | None:
    """
    Convert an astropy unit into a LaTeX-formatted label
    for plotting. Returns None if no unit is found.

    Parameters
    ----------
    unit : str or astropy.Unit
        The astropy.Unit or unit string to convert.
    fmt : {'latex', 'latex_inline', 'inline'} or None, optional, default=None
        The format of the unit label. ``'latex_inline'`` and ``'inline'``
        (alias for ``'latex_inline'``) uses negative exponents while
        ``'latex'`` uses fractions. If ``None``, uses
        the default value set by ``config.unit_label_format``.

    Returns
    -------
    str or None
        A LaTeX-formatted unit label if the input is recognized.
        Returns None if the unit is invalid.
    """
    fmt = get_config_value(fmt, 'unit_label_format')
    fmt = str(fmt).lower()
    if fmt not in {'latex', 'latex_inline', 'inline'}:
        raise ValueError(
            "format must be: {'latex', 'latex_inline', 'inline'}"
            f'got: {fmt}'
        )

    try:
        return unit_2_string(unit, fmt=fmt)
    except Exception:
        return None


@overload
def to_fits_unit(
    unit: Quantity | UnitBase | StructuredUnit | str
) -> str: ...

@overload
def to_fits_unit(
    unit: None
) -> None: ...

def to_fits_unit(
    unit: Quantity | UnitBase | StructuredUnit | str | None
) -> str | None:
    """
    Convert an astropy unit into a fits compliant label.
    Returns None if no unit is found.

    Parameters
    ----------
    unit : Quantity | UnitBase | StructuredUnit | str | None
        Unit to convert.

    Returns
    -------
    str or None :
        Unit converted as a fits formatted string or ``None``
        if no unit is found.
    """
    return unit_2_string(unit, fmt='fits')


def to_spectral_region(
    obj: (
        SpectralRegion
        | Quantity
        | tuple[Quantity, Quantity]
        | list[tuple[Quantity, Quantity]]
        | None
    )
) -> SpectralRegion | None:
    """
    Coerce input into a SpectralRegion with units.

    Parameters
    ----------
    obj : SpectralRegion, Quantity, tuple, list, or None
        Region specification. Accepted forms:

        - ``SpectralRegion``: returned as-is
        - ``(low, high) * unit``: single region with shared unit
        - ``[(low, high), ...] * unit``: multiple regions with shared unit
        - ``(Quantity, Quantity)``: single region with explicit units
        - ``[(Quantity, Quantity), ...]``: multiple regions with explicit units
        - ``None``: returned as-is

    Returns
    -------
    SpectralRegion :
        A SpectralRegion object containing one or more spectral subregions.

    Raises
    ------
    ValueError :
        If a Quantity has incorrect shape (must be (2,) or (N, 2)).
    ValueError :
        If the input ``obj`` cannot be coerced into a ``SpectralRegion``.

    Examples
    --------
    >>> import astropy.units as u
    >>> # Single region with shared unit
    >>> to_spectral_region((6.5, 6.6) * u.um)

    >>> # Multiple regions with shared unit
    >>> to_spectral_region([(6.5, 6.6), (7.0, 7.5)] * u.um)

    >>> # Single region with explicit units
    >>> to_spectral_region((6.5*u.um, 6.6*u.um))

    >>> # Multiple regions with explicit units
    >>> to_spectral_region([(6.5*u.um, 6.6*u.um), (7.0*u.um, 7.5*u.um)])
    """
    if obj is None:
        return None

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
            f'Expected SpectralRegion, Quantity, tuple, or list, got {_type_name(region)}'
        )

    try:
        return SpectralRegion(region)
    except Exception as e:
        raise ValueError(
            f'Could not construct SpectralRegion from {_type_name(region)}: {e}'
        ) from e


def require_spectral_region(
    obj: (
        SpectralRegion
        | Quantity
        | tuple[Quantity, Quantity]
        | list[tuple[Quantity, Quantity]]
    )
) -> SpectralRegion:
    """
    Enforce that the input ``obj`` is a ``SpectralRegion``.

    This is a wrapper of ``to_spectral_region``, which unlike
    this function allows for None inputs.

    Parameters
    ----------
    obj : SpectralRegion, Quantity, tuple, or list
        Region specification. Accepted forms:

        - ``SpectralRegion``: returned as-is
        - ``(low, high) * unit``: single region with shared unit
        - ``[(low, high), ...] * unit``: multiple regions with shared unit
        - ``(Quantity, Quantity)``: single region with explicit units
        - ``[(Quantity, Quantity), ...]``: multiple regions with explicit units

    Returns
    -------
    SpectralRegion :
        Input region as a SpectralRegion

    Raises
    ------
    ValueError :
        If input ``obj`` results in ``None`` when converted to
        a SpectralRegion.
    """
    region = to_spectral_region(obj)

    if region is None or not isinstance(region, SpectralRegion):
        raise ValueError(f'A SpectralRegion is required; got {_type_name(region)}')
    return region


def ensure_common_unit(
    objs: Any,
    *,
    unit: UnitBase | str | None = None,
    on_mismatch: str | None = None,
    label: str | None = None
) -> UnitBase | StructuredUnit | None:
    """
    Check unit consistency across one or more objects.

    This function verifies that all objects with defined units share the same
    unit. No unit conversion is performed. The function will either raise an
    UnitsError, issue a warning, or ignore.

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
        the default value from ``config.unit_mismatch``.
    label : str or None, optional, default=None
        Optional context label prepended to warnings or errors.
        Is ignored if None.

    Returns
    -------
    ref_unit : UnitBase
        Common unit between ``objs``.

    Raises
    ------
    UnitsError :
        If a unit mismatch is detected and `on_mismatch='raise'`.
    """
    on_mismatch = get_config_value(on_mismatch, 'unit_mismatch')

    objs = to_list(objs)
    units = [get_unit(obj) for obj in objs]

    if all(u is None for u in units):
        return None

    if unit is None:
        ref_unit = next(u for u in units if u is not None)
    elif isinstance(unit, (UnitBase, StructuredUnit)):
        ref_unit = unit
    else:
        ref_unit = Unit(unit)

    prefix = f'{label}: ' if isinstance(label, str) else ''

    for i, u in enumerate(units):
        if u is None or u == ref_unit:
            continue

        if on_mismatch == 'raise':
            raise UnitsError(
                f"{prefix}Unit mismatch at index {i}: {u} != {ref_unit}"
            )

        if on_mismatch == 'warn':
            warnings.warn(
                f'{prefix}Unit mismatch at index {i}: {u} != {ref_unit}. '
                'Values may be interpreted incorrectly.'
            )

    return ref_unit


def _is_spectral_axis(obj: Any) -> bool:
    """
    Determine whether an object represents a spectral axis or a spectral quantity.

    An object is considered spectral if any of the following is true:
    1. It is a ``SpectralAxis`` instance.
    2. It exposes a ``.spectral_axis`` or ``.spectral_unit`` attribute that itself
        represents a spectral axis.
    3. Its unit corresponds to a spectral quantity such as wavelength, frequency,
        energy, or Doppler velocity, as determined via Astropy's ``u.spectral()``
        or ``u.doppler_radio()`` equivalencies.

    Parameters
    ----------
    obj : Any
        The object to evaluate. This may be a ``SpectralAxis``, ``Quantity``,
        unit, or any object exposing spectral metadata.

    Returns
    -------
    bool
        True if the object represents a spectral axis or spectral quantity, False otherwise.

    Notes
    -----
    - Length units (e.g., meters, microns) are only treated as spectral when
        they can be converted to frequency using ``u.spectral()`` equivalencies.
    - Plain length quantities without spectral context are not automatically
        considered spectral axes.
    - Doppler velocity units are recognized only if they can be converted
        to a spectral representation using ``u.doppler_radio()`` equivalencies.
    """
    if isinstance(obj, SpectralAxis):
        return True

    for attr in ('spectral_axis', 'spectral_unit'):
        val = getattr(obj, attr, None)
        if val is not None and _is_spectral_axis(val):
            return True

    unit = get_unit(obj)
    physical_type = get_physical_type(obj)
    if unit is None:
        return False
    if physical_type is None:
        return False

    if physical_type == 'length':
        try:
            unit.to(u.Hz, equivalencies=u.spectral())
            return True
        except UnitConversionError:
            return False

    if physical_type == 'speed':
        try:
            unit.to(u.Hz, equivalencies=u.doppler_radio(u.Hz))
            return True
        except UnitConversionError:
            return False

    if str(physical_type) in {'frequency', 'energy'}:
        return True

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
        any object from which a unit can be extracted via `get_unit`.

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
    try:
        u1 = to_unit(unit1)
        u2 = to_unit(unit2)
    except Exception:
        raise UnitsError('Invalid unit(s) supplied')

    # case 1: either of the units are None
    if u1 is None or u2 is None:
        return None

    u1_str = unit_2_string(u1)
    u2_str = unit_2_string(u2)

    # case 1: units are exactly equal
    if u1 == u2:
        return None

    # case 2: equivalent but not equal
    if u1.is_equivalent(u2):
        raise UnitsError(
            f'{name1} and {name2} units are equivalent but not equal '
            f'({u1_str} vs {u2_str}). Convert one to match.'
        )

    # case 3: mismatch
    raise UnitsError(
        f'{name1} and {name2} have incompatible units: '
        f'{u1_str} vs {u2_str}.'
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
