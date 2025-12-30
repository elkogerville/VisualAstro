'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-25-05
Description:
    SpectrumPlus data structure for 1D spectrum objects.
    This is an extension of the specutils.Spectrum class
    that offers convenience methods to work with spectra.
Dependencies:
    - astropy
    - numpy
    - specutils
'''

import copy
import numpy as np
from specutils import SpectralRegion
from specutils.manipulation import extract_region as _extract_region
from specutils.spectra import Spectrum
from .units import _check_unit_equality, _validate_units_consistency
from .va_config import get_config_value


class SpectrumPlus:
    '''
    Lightweight extension of specutils.Spectrum, offering convenience methods.

    Parameters
    ----------
    spectrum : `~specutils.Spectrum`, optional
        Spectrum object containing wavelength, flux, and unit information.
        Takes precedent over `wavelength` and `flux`.
    wavelength : array-like or `~astropy.units.Quantity`, optional
        Wavelength array corresponding to the spectral axis.
    flux : array-like or `~astropy.units.Quantity`, optional
        Flux values of the spectrum. Units are inferred if possible.
    normalized : array-like or `~astropy.units.Quantity`, optional
        Normalized flux values of the spectrum, if available.
    continuum_fit : array-like or callable, optional
        Continuum fit to the spectrum or a callable used to generate it.

    Attributes
    ----------
    spectrum : specutils.Spectrum
        Underlying Spectrum object containing the spectrum
        spectral axis and flux arrays.
    continuum_fit : array-like or callable or None
        Continuum fit data or fitting function.
    normalized : array-like or None
        Normalized flux array, if available.
    fit_method : {'fit_continuum', 'generic'}
        Method used to compute the continuum fit.
    region : SpectralRegion or None
        Region used to compute the continuum fit.
        Can be used to remove strong absorption/emission
        lines that can skew the fit.
    log : Header
        Log file to track SpectrumPlus operations using
        `HISTORY` cards. Methods that return a new SpectrumPlus
        instance will transfer over any existing logs.

    Properties
    ----------
    spectral_axis : Quantity or SpectralAxis
        Spectral axis array of the spectrum.
    wavelength : Quantity or SpectralAxis
        Spectral axis array of the spectrum converted to Angstroms.
    frequency : Quantity or SpectralAxis
        Spectral axis array of the spectrum converted to GHz.
    flux : Quantity
        Flux array of the spectrum.
    spectral_unit : Astropy.Unit
        Unit of spectral axis.
    unit : Astropy.Unit
        Unit of flux array.

    Methods
    -------
    extract_region(region, return_single_spectrum=False)
        Extract a subregion of the spectrum.
    replace_flux_where(mask, values)
        Replace flux values at selected locations.

    Array Interface
    ---------------
    __get_item__
        Return a slice of the spectrum.
    __getattr__
        Delegate undefined attributes or methods
        to the underlying `Spectrum` object.
    __mul__
        Multiply the spectrum flux by a scalar factor.
    __rmul__
        Multiply a scalar value by the spectrum flux.
    __truediv__
        Divide the spectrum flux by a scalar factor.
    __rtruediv__
        Raise an error as dividing by the spectrum flux
        is undefined behavior.

    Helper Methods
    --------------
    _apply_region
        Helper method for `extract_region`.
    _fit_continuum
        Helper method to fit the spectrum continuum.
    _construct_spectrum
        Helper method to construct a Spectrum object.

    '''

    def __init__(
        self, spectrum=None, *, spectral_axis=None, flux=None,
        normalized=None, continuum_fit=None, log_file=None, **kwargs
    ):

        # kwargs and config
        fit_method = kwargs.get('fit_method', None)
        fit_method = get_config_value(
            fit_method, 'spectrum_continuum_fit_method'
        )
        region = kwargs.get('region', None)
        if region is not None and not isinstance(region, SpectralRegion):
            region = SpectralRegion(region)

        # validate that spectral axis and flux units are consistent
        spectral_candidates = (
            spectral_axis,
            getattr(spectrum, 'spectral_axis', None)
        )
        flux_candidates = (
            spectrum,
            getattr(spectrum, 'flux', None),
            flux,
            continuum_fit
        )

        _validate_units_consistency(spectral_candidates, label='spectral axis')
        _validate_units_consistency(flux_candidates, label='flux')

        # generate Spectrum
        spectrum = self._construct_spectrum(
            spectrum=spectrum,
            spectral_axis=spectral_axis,
            flux=flux,
            **kwargs
        )
        # fit continuum and normalize
        if continuum_fit is None:
            continuum_fit = self._fit_continuum(spectrum, fit_method, region)

        if normalized is None:
            normalized = spectrum / continuum_fit

        self.spectrum = spectrum
        self.continuum_fit = continuum_fit
        self.normalized = normalized
        self.fit_method = fit_method
        self.region = region

    # Properties
    # ----------
    @property
    def spectral_axis(self):
        '''
        Returns
        -------
        Quantity or SpectralAxis : Spectral axis array.
        '''
        return self.spectrum.spectral_axis
    @property
    def wavelength(self):
        '''
        Returns
        -------
        Quantity or SpectralAxis : Wavelength array in Å units.
        '''
        return self.spectrum.wavelength
    @property
    def frequency(self):
        '''
        Returns
        -------
        Quantity or SpectralAxis : wavelength array in GHz units.
        '''
        return self.spectrum.frequency
    @property
    def flux(self):
        '''
        Returns
        -------
        Quantity : Flux array.
        '''
        return self.spectrum.flux
    @property
    def spectral_unit(self):
        '''
        Returns
        -------
        Unit : Spectral axis unit.
        '''
        return self.spectrum.spectral_axis.unit
    @property
    def unit(self):
        '''
        Returns
        -------
        Unit : Flux unit.
        '''
        return self.spectrum.flux.unit

    # Methods
    # -------
    def extract_region(self, region, return_single_spectrum=False):
        '''
        Extract a spectral sub-region.

        Parameters
        ----------
        region : SpectralRegion or array-like of tuple
            Spectral region(s) to extract. If array-like,
            it must be a sequence of `(lower, upper)`
            bounds, typically as `Quantity`.
            Ex : [(6.5*u.um, 8*u.um), (8.5*u.um, 9*u.um)]
            Ex : [(6000, 6200), (7000, 8000)] * u.AA
        return_single_spectrum : boolean, optional, default=False
            If True, the resulting spectra will be concatenated
            together into a single Spectrum object instead.

        Returns
        -------
        SpectrumPlus or list of SpectrumPlus
            A new SpectrumPlus object (or list of objects) restricted
            to the specified region(s).
        '''
        fit_method = self.fit_method

        if not isinstance(region, SpectralRegion):
            region = SpectralRegion(region)

        new_spectrum = self._apply_region(
            self.spectrum,
            region,
            return_single_spectrum=return_single_spectrum
        )

        if return_single_spectrum:
            return SpectrumPlus(
                spectrum=new_spectrum, fit_method=fit_method
            )

        # convert each subregion to a SpectrumPlus
        return [
            SpectrumPlus(
                spectrum=spec, fit_method=fit_method
            ) for spec in new_spectrum
        ]

    def replace_flux_where(self, mask, values):
        '''
        Replace flux values at selected locations and return a new spectrum.

        This method returns a new `SpectrumPlus` in which the flux values
        corresponding to `mask` are replaced by `values`. The WCS, velocity
        convention, rest value and meta attributes are preserved. The operation
        is non-mutating; the original spectrum remains unchanged.

        Parameters
        ----------
        mask : array-like of bool
            Boolean mask with the same shape as the spectrum flux. Elements
            set to True indicate locations where the flux will be replaced.
        values : `Quantity`
            Replacement flux values. Must have the same shape and units as
            the spectrum flux.

        Returns
        -------
        `SpectrumPlus`
            A new spectrum with modified flux values. The spectral axis and
            coordinate metadata are preserved, while uncertainty and masking
            information are not propagated.

        Raises
        ------
        ValueError :
            If `values` does not have the same shape as the spectrum flux.
        TypeError :
            If `values` is not a `Quantity` with units compatible with the
            spectrum flux.
        UnitsError :
            If `flux` and `values` don't have the same units.
        '''
        spectrum = self.spectrum
        spectral_axis = spectrum.spectral_axis
        flux = spectrum.flux.copy()

        if values.shape != flux.shape:
            raise ValueError(
                'Replacement values must match flux shape. '
                f'Flux shape: {flux.shape}, values shape: {values.shape}.'
            )

        if not hasattr(values, 'unit'):
            raise TypeError("`values` must be an astropy Quantity.")

        _check_unit_equality(flux.unit, values.unit, 'flux', 'values')

        flux[mask] = values[mask]

        new_spectrum = Spectrum(
            spectral_axis=spectral_axis,
            flux=flux,
            wcs=spectrum.wcs,
            velocity_convention=spectrum.velocity_convention,
            rest_value=spectrum.rest_value,
            meta=dict(spectrum.meta) if spectrum.meta is not None else None,
        )

        fit_method = self.fit_method

        return SpectrumPlus(
            spectrum=new_spectrum, fit_method=fit_method
        )

    # helper functions
    # ----------------
    def __getitem__(self, key):
        '''
        Slice the underlying `Spectrum` object, and return a new
        SpectrumPlus. The continuum fit and normalized spectra
        are automatically recomputed for the spectra slice.

        Parameters
        ----------
        key : int, slice, or array-like
            Index or slice used to select specific elements from
            the wavelength, flux, and other stored arrays.

        Returns
        -------
        SpectrumPlus
            A new `SpectrumPlus` instance containing the sliced
            spectrum object.
        '''
        sub_spectrum = self.spectrum[key]
        fit_method = self.fit_method

        return SpectrumPlus(
            spectrum=sub_spectrum, fit_method=fit_method
        )

    def __getattr__(self, name):
        '''
        Delegate attribute/method access to underlying Spectrum object
        if it is not defined in SpectrumPlus.

        Parameters
        ----------
        name : str
            Name of attribute, property, method, etc...
        '''
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            return getattr(self.spectrum, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __mul__(self, factor):
        '''
        Multiply spectrum by a scalar.

        Parameters
        ----------
        factor : scalar
        '''
        scaled_spectrum = self.spectrum * factor
        fit_method = self.fit_method
        region = copy.copy(self.region)

        return SpectrumPlus(
            spectrum=scaled_spectrum,
            fit_method=fit_method,
            region=region
        )

    __rmul__ = __mul__

    def __truediv__(self, factor):
        '''
        Divide spectrum by a scalar.

        Parameters
        ----------
        factor : scalar
        '''
        scaled_spectrum = self.spectrum / factor
        fit_method = self.fit_method
        region = copy.copy(self.region)

        return SpectrumPlus(
            spectrum=scaled_spectrum,
            fit_method=fit_method,
            region=region
        )

    def __rtruediv__(self, factor):
        raise TypeError(
            'Division by an SpectrumPlus is not defined.'
        )

    def _apply_region(self, spectrum, region, return_single_spectrum=False):
        '''
        Apply a spectral region to a Spectrum object.

        This is a thin wrapper around `specutils.manipulation.extract_region`.
        If `region` is not already a `SpectralRegion`, it is coerced into one
        before extraction.

        Parameters
        ----------
        spectrum : Spectrum or None
            Input spectrum to which the region will be applied. If None,
            the function returns None.
        region : SpectralRegion or array-like
            Spectral region to extract. If array-like, it is interpreted as
            region bounds and converted to a `SpectralRegion`.

        Returns
        -------
        Spectrum or None
            A new `Spectrum` object containing only the portion defined
            by `region`, or None if the input spectrum is None.
        '''

        if not isinstance(region, SpectralRegion):
            region = SpectralRegion(region)

        return _extract_region(
            spectrum, region, return_single_spectrum=return_single_spectrum
        )

    def _fit_continuum(self, spectrum, fit_method, region):
        '''
        Fit spectrum continuum.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum object.
        fit_method : {'fit_continuum', 'generic'}
            Continuum fitting method.
        region : array-like of tuple, optional
            Spectral region(s) to include in the continuum fit.
            Ex:
            region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
        '''
        from .spectra_utils import fit_continuum

        continuum_fit = fit_continuum(spectrum, fit_method, region)

        return continuum_fit

    def _construct_spectrum(
        self, *, spectrum=None, spectral_axis=None, flux=None, **kwargs
    ):
        '''
        Construct and return a Spectrum instance from either an existing spectrum
        or from spectral axis and flux arrays.

        Exactly one of the following input combinations must be provided:
        - `spectrum`: a pre-existing `specutils.Spectrum` instance, or
        - `spectral_axis` and `flux`: array-like objects used to construct a new
            `Spectrum`.

        Parameters
        ----------
        spectrum : `specutils.Spectrum`, optional
            Existing spectrum object to return directly. If provided, `spectral_axis`
            and `flux` must be None.
        spectral_axis : array-like or `astropy.units.Quantity`, optional
            Spectral axis values. Must be provided together with `flux` if
            `spectrum` is not given.
        flux : array-like or `astropy.units.Quantity`, optional
            Flux values corresponding to `spectral_axis`. Must be provided together
            with `spectral_axis` if `spectrum` is not given.

        **kwargs
            Additional keyword arguments forwarded to the `Spectrum` constructor
            (e.g., velocity_convention, rest_value, wcs, radial_velocity. etc).

        Returns
        -------
        spectrum : `specutils.Spectrum`
            A validated spectrum instance.

        Raises
        ------
        ValueError
            If both `spectrum` and (`wavelength` or `flux`) are provided, if neither
            input path is fully specified, or if `spectrum` is not a `Spectrum`
            instance.
        '''
        if spectrum is not None:
            if spectral_axis is not None or flux is not None:
                raise ValueError(
                    'Provide either spectrum OR spectral_axis + flux, not both!'
                )
            elif not isinstance(spectrum, Spectrum):
                raise ValueError(
                    "'spectrum' must be a Spectrum instance!"
                )
            return spectrum

        if spectral_axis is None or flux is None:
            raise ValueError(
                'spectral_axis and flux must both be provided if spectrum is None.'
            )

        return Spectrum(
            spectral_axis=spectral_axis,
            flux=flux,
            **kwargs
        )

    def __dir__(self):
        '''
        Ensure both `SpectrumPlus` and `specutils.Spectrum`
        attributes are included in dir().
        '''
        return sorted(
            set(
                dir(type(self)) +
                list(self.__dict__.keys()) +
                dir(self.spectrum)
            )
        )

    def __repr__(self):
        '''
        Returns
        -------
        str : String representation of `SpectrumPlus`.
        '''
        flux = (
            f'flux=({self.flux.value[0]:.3e} ... {self.flux.value[-1]:.3e}) '
            f'[{self.unit}]; '
        )
        spectral_axis = (
            f'spectral_axis=({self.spectral_axis.value[0]:.3e} ... '
            f'{self.spectral_axis.value[-1]:.3e}) '
            f'[{self.spectral_unit}]; '
        )
        stats = (
            f'mean={np.nanmean(self.flux.value):.3e} [{self.unit}], '
            f'len={len(self.spectral_axis)}'
        )

        return f'<SpectrumPlus: {flux}{spectral_axis}{stats}>'
