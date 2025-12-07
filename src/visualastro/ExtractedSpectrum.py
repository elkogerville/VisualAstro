'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-12-05
Description:
    ExtractedSpectrum data structure for 1D spectrum objects.
Dependencies:
    - astropy
    - numpy
    - specutils
Module Structure:
    - ExtractedSpectrum
        Data class for extracted spectra.
    - Utility Functions
        Utility functions for ExtractedSpectrum methods.
'''

from astropy.units import Quantity, UnitsError
import numpy as np
from specutils.spectra import Spectrum1D
from .va_config import get_config_value


class ExtractedSpectrum:
    '''
    Lightweight container class for extracted 1D spectra with associated metadata.

    Parameters
    ––––––––––
    wavelength : array-like or `~astropy.units.Quantity`, optional
        Wavelength array corresponding to the spectral axis.
    flux : array-like or `~astropy.units.Quantity`, optional
        Flux values of the spectrum. Units are inferred if possible.
    spectrum1d : `~specutils.Spectrum1D`, optional
        Spectrum object containing wavelength, flux, and unit information.
        Used as an alternative input to `wavelength` and `flux`.
    normalized : array-like or `~astropy.units.Quantity`, optional
        Normalized flux values of the spectrum, if available.
    continuum_fit : array-like or callable, optional
        Continuum fit to the spectrum or a callable used to generate it.

    Attributes
    ––––––––––
    wavelength : array-like or `~astropy.units.Quantity`
        Wavelength values of the spectrum.
    flux : array-like or `~astropy.units.Quantity`
        Flux values of the spectrum.
    spectrum1d : `~specutils.Spectrum1D` or None
        Original Spectrum1D object, if provided.
    normalized : array-like or None
        Normalized flux array, if available.
    continuum_fit : array-like or callable or None
        Continuum fit data or fitting function.
    wave_unit : `~astropy.units.Unit` or None
        Wavelength unit inferred from `wavelength` or `spectrum1d`.
    unit : `~astropy.units.Unit` or None
        Flux unit inferred from `flux` or `spectrum1d`.
    '''

    def __init__(self, wavelength=None, flux=None, spectrum1d=None,
                 normalized=None, continuum_fit=None):
        self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

    def _initialize(self, wavelength, flux, spectrum1d, normalized, continuum_fit):

        # validate that wavelength and flux units are consistent
        wave_candidates = (
            wavelength,
            getattr(spectrum1d, 'spectral_axis', None)
        )
        flux_candidates = (
            flux,
            spectrum1d,
            getattr(spectrum1d, 'flux', None),
            continuum_fit
        )

        wave_unit = self._validate_units(wave_candidates, label='wavelength')
        unit = self._validate_units(flux_candidates, label='flux')

        self.wavelength = wavelength
        self.flux = flux
        self.spectrum1d = spectrum1d
        self.normalized = normalized
        self.continuum_fit = continuum_fit
        self.wave_unit = wave_unit
        self.unit = unit

    # support slicing
    def __getitem__(self, key):
        '''
        Return a sliced view of the `ExtractedSpectrum` object.

        Parameters
        ––––––––––
        key : int, slice, or array-like
            Index or slice used to select specific elements from
            the wavelength, flux, and other stored arrays.

        Returns
        –––––––
        ExtractedSpectrum
            A new `ExtractedSpectrum` instance containing the sliced
            wavelength, flux, normalized flux, continuum fit, and
            `Spectrum1D` object (if present).

        Notes
        –––––
        - Metadata such as `rest_value` and `velocity_convention` are
            preserved when slicing `spectrum1d`.
        - Attributes that are `None` remain `None` in the returned object.
        '''
        wavelength = None
        flux = None
        spectrum1d = None
        normalized = None
        continuum_fit = None

        if self.wavelength is not None:
            wavelength = self.wavelength[key]
        if self.flux is not None:
            flux = self.flux[key]
        if self.spectrum1d is not None:
            spectrum1d = Spectrum1D(
                spectral_axis=self.spectrum1d.spectral_axis[key],
                flux=self.spectrum1d.flux[key],
                rest_value=self.spectrum1d.rest_value,
                velocity_convention=self.spectrum1d.velocity_convention
            )
        if self.normalized is not None:
            normalized = self.normalized[key]
        if self.continuum_fit is not None:
            continuum_fit = self.continuum_fit[key]

        return ExtractedSpectrum(
            wavelength,
            flux,
            spectrum1d,
            normalized,
            continuum_fit
        )

    def update(self, wavelength=None, flux=None, spectrum1d=None, normalized=None, continuum_fit=None, **kwargs):
        '''
        Update one or more attributes of the ExtractedSpectrum object.
        Any argument provided to this method overrides the existing value.
        Arguments left as None will retain the current stored values.
        Dependent attributes are automatically recomputed using the newest available
        inputs, falling back to the previously stored values when needed.

        Parameters
        ––––––––––
        wavelength : `~astropy.units.Quantity`, optional
            New spectral axis array to assign to the spectrum. If provided,
            the stored `Spectrum1D` object will be rebuilt (if it exists).
        flux : `~astropy.units.Quantity`, optional
            New flux array to assign to the spectrum. If provided,
            the stored `Spectrum1D` object will be rebuilt (if it exists).
        spectrum1d : `~specutils.Spectrum1D`, optional
            A full Spectrum1D object to replace the internal representation.
            If passed, this overrides both `wavelength` and `flux`, and no
            further updates are applied.

        Returns
        –––––––
        None

        Notes
        –––––
        - If `spectrum1d` is passed, it takes precedence and replaces `wavelength`
          and `flux`.
        '''
        from .spectra_utils import compute_continuum_fit
        # –––– KWARGS ––––
        rest_value = kwargs.get('rest_value', None)
        velocity_convention = kwargs.get('velocity_convention', None)
        fit_method = kwargs.get('fit_method', None)
        region = kwargs.get('region', None)

        # get default va_config values
        fit_method = get_config_value(fit_method, 'spectrum_continuum_fit_method')

        # use spectrum1d to update ExtractedSpectrum if provided
        if spectrum1d is not None:
            wavelength = spectrum1d.spectral_axis
            flux = spectrum1d.flux
            continuum_fit = compute_continuum_fit(spectrum1d, fit_method, region)
            normalized = (spectrum1d / continuum_fit).flux

            self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

            return None

        wavelength = self.wavelength if wavelength is None else wavelength
        flux = self.flux if flux is None else flux

        # rebuild spectrum1d if wavelength or flux is passed in
        if not self._allclose(wavelength, self.wavelength) or not self._allclose(flux, self.flux):

            # rest value and velocity convention defaults are set
            # by the previous spectrum1d values if they existed
            if self.spectrum1d is not None:
                rest_value = self.spectrum1d.rest_value if rest_value is None else rest_value
                velocity_convention = (
                                self.spectrum1d.velocity_convention
                                if velocity_convention is None else velocity_convention
                            )

            spectrum1d = Spectrum1D(
                    spectral_axis=wavelength,
                    flux=flux,
                    rest_value=rest_value,
                    velocity_convention=velocity_convention
                )
            # recompute continuum fit and normalized flux if not passed in
            if continuum_fit is None:
                continuum_fit = compute_continuum_fit(spectrum1d, fit_method, region)
            if normalized is None:
                normalized = (spectrum1d / continuum_fit).flux

        # use previous values unless provided / recomputed
        spectrum1d = self.spectrum1d if spectrum1d is None else spectrum1d
        normalized = self.normalized if normalized is None else normalized
        continuum_fit = self.continuum_fit if continuum_fit is None else continuum_fit

        self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

        return None

    # helper functions
    # ––––––––––––––––
    @staticmethod
    def _allclose(a, b):
        '''
        Determine whether two array-like objects are equal within a tolerance,
        with additional handling for `astropy.units.Quantity` and None.
        This function behaves like `numpy.allclose`, but adds logic to safely
        compare Quantities (ensuring matching units) and to treat None as
        a valid sentinel value.
        Parameters
        ––––––––––
        a, b : array-like, `~astropy.units.Quantity`, scalar, or None
            The inputs to compare. Inputs may be numerical arrays, scalars, or
            `Quantity` objects with units. If one argument is None, the result is
            False unless both are None.

        Returns
        –––––––
        bool
            True if the inputs are considered equal, False otherwise.
            Equality rules:
            - Both None → True
            - One None → False
            - Quantities with mismatched units → False
            - Quantities with identical units → value arrays compared via
                `numpy.allclose`
            - Non-Quantity arrays/scalars → compared via `numpy.allclose`

        Notes
        –––––
        - This function does **not** attempt unit conversion.
          Quantities must already share identical units.
        - This function exists to support `.update()` logic where user-supplied
          wavelength/flux arrays should only trigger recomputation if they
          differ from stored values.
        '''
        # case 1: both are None → equal
        if a is None and b is None:
            return True

        # case 2: only one is None → different
        if a is None or b is None:
            return False

        # case 3: one is Quantity, one is not
        if isinstance(a, Quantity) != isinstance(b, Quantity):
            return False

        # case 4: both Quantities
        if isinstance(a, Quantity) and isinstance(b, Quantity):
            if a.unit != b.unit:
                return False
            return np.allclose(a.value, b.value)

        # case 5: both unitless arrays/scalars
        return np.allclose(a, b)

    def _validate_units(self, objs, label):
        '''
        Validate that the units match between a list of objects.

        Parameters
        ––––––––––
        objs : array-like of objects
            A list or array-like of objects with or without 'unit'
            attribute.

        Returns
        –––––––
        None : if no units are present.
        Astropy Unit : If units are present and are consistent.

        Raises
        ––––––
        ValueError : If units exist and do not match.
        '''
        # create set of each unit in objs, and remove any None values
        # this contains each unique unit across the set of objects
        units = {getattr(o, 'unit', None) for o in objs} - {None}

        # return None if no units found
        if not units:
            return None
        # raise an error if more than 1 unique unit
        elif len(units) > 1:
            raise UnitsError(
                f'Inconsistent {label} units: {units}'
            )
        # return the unit
        else:
            return units.pop()

    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of `ExtractedSpectrum`.
        '''
        return (
            f'<ExtractedSpectrum: wave_unit={self.wave_unit}, flux_unit={self.unit}, len={len(self.wavelength)}>'
        )
