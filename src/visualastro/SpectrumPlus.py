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
from astropy.io.fits import Header
from astropy.units import Quantity
import numpy as np
from specutils import SpectralRegion
from specutils.manipulation import extract_region as _extract_region
from specutils.spectra import Spectrum
from .config import get_config_value
from .fits_utils import _copy_headers, _get_history, _log_history
from .units import ensure_common_unit, to_spectral_region, _check_unit_equality


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
    continuum : array-like or `~astropy.units.Quantity`, optional
        Continuum flux values evaluated on the same spectral axis as the
        spectrum. Must have the same shape and physical units as `flux`.
    log_file : Header
        Log file to track SpectrumPlus operations using
        `HISTORY` cards. Methods that return a new SpectrumPlus
        instance will transfer over any existing logs.

    Attributes
    ----------
    spectrum : specutils.Spectrum
        Underlying Spectrum object containing the spectrum
        spectral axis and flux arrays.
    continuum : Quantity
        Quantity array of continuum values.
    normalized : array-like
        Normalized flux array.
    fit_method : {'fit_continuum', 'generic'}
        Method used to compute the continuum fit.
    region : SpectralRegion or None
        Region used to compute the continuum fit.
        Can be used to remove strong absorption/emission
        lines that can skew the fit.
    log : list of str
        List of each log output in primary_header['HISTORY']

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
    __getattr__
        Delegate undefined attributes or methods
        to the underlying `Spectrum` object.
    __getitem__
        Return a slice of the spectrum.
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
        normalized=None, continuum=None, log_file=None, **kwargs
    ):

        fit_method = kwargs.pop('fit_method', None)
        fit_method = get_config_value(
            fit_method, 'spectrum_continuum_fit_method'
        )
        region = kwargs.pop('region', None)
        if region is not None:
            region = to_spectral_region(region)

        if log_file is None:
            log_file = Header()
        elif not isinstance(log_file, Header):
            raise ValueError(
                'log_file should be a astropy.fits.Header! '
                f'Got type: {type(log_file)}'
            )

        # validate that spectral axis and flux units are consistent
        spectral_candidates = (
            spectral_axis,
            getattr(spectrum, 'spectral_axis', None)
        )
        flux_candidates = (
            spectrum,
            getattr(spectrum, 'flux', None),
            flux,
            continuum
        )

        ensure_common_unit(
            spectral_candidates, on_mismatch='raise', label='spectral axis'
        )
        ensure_common_unit(
            flux_candidates, on_mismatch='raise', label='flux'
        )

        # generate Spectrum
        spectrum = self._construct_spectrum(
            spectrum=spectrum,
            spectral_axis=spectral_axis,
            flux=flux,
            log_file=log_file,
            **kwargs
        )
        # fit continuum and normalize
        if continuum is None:
            continuum = self._fit_continuum(spectrum, fit_method, region)
            _log_history(log_file, f"Computing continuum fit with '{fit_method}'")

        if normalized is None:
            normalized = spectrum / continuum

        self.spectrum = spectrum
        self.continuum = continuum
        self.normalized = normalized
        self.fit_method = fit_method
        self.region = region
        self.log_file = log_file

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
    @property
    def log(self):
        '''
        Get the processing history from the FITS HISTORY cards.

        Returns
        -------
        list of str or None
            List of HISTORY entries, or None if no header exists.
        '''
        return _get_history(self.log_file)

    # Methods
    # -------
    def extract_region(
        self, region, return_single_spectrum=False, continuum_region=None
    ):
        """
        Extract a spectral sub-region.

        Parameters
        ----------
        region : SpectralRegion or array-like of tuple
            Spectral region(s) to extract. If array-like,
            it must be a sequence of ``(lower, upper)``
            bounds, typically as ``Quantity``.
            Ex : [(6.5*u.um, 8*u.um), (8.5*u.um, 9*u.um)]
            Ex : [(6000, 6200), (7000, 8000)] * u.AA
        return_single_spectrum : boolean, optional, default=False
            If True, the resulting spectra will be concatenated
            together into a single Spectrum object instead.
        continuum_region : SpectralRegion, array-like, or list thereof, optional
            Spectral region(s) to use when fitting the continuum.
            - If ``return_single_spectrum=True`` :
                single region specification
            - If ``return_single_spectrum=False`` and multiple regions extracted :
                must be a list with one region per extracted subregion, or None

        Returns
        -------
        SpectrumPlus or list of SpectrumPlus
            A new SpectrumPlus object (or list of objects) restricted
            to the specified region(s).
        """
        fit_method = self.fit_method
        log_file = self.log_file

        region = to_spectral_region(region)
        N_regions = len(region.subregions)

        if continuum_region is not None:
            if not return_single_spectrum and N_regions > 1:
                if not isinstance(continuum_region, list):
                    raise ValueError(
                        'When extracting multiple separate spectra, '
                        'continuum_region must be a list with one region per subregion'
                    )
                if len(continuum_region) != N_regions:
                    raise ValueError(
                        f'continuum_region has {len(continuum_region)} elements '
                        f'but {N_regions} spectral regions were specified'
                    )
                continuum_region = [to_spectral_region(cr) for cr in continuum_region]
            else:
                continuum_region = to_spectral_region(continuum_region)

        # extract region
        new_spectrum = self._apply_region(
            self.spectrum,
            region,
            return_single_spectrum=return_single_spectrum
        )

        # get lower and upper bound of each subregion
        rmins = [r.lower.to_string() for r in region]
        rmaxs = [r.upper.to_string() for r in region]

        if return_single_spectrum or N_regions == 1:
            new_log = _copy_headers(log_file)
            log = (
                f'Extracting {N_regions} region(s) btwn : '
                f'{rmins[0]} - {rmaxs[-1]}'
            )
            _log_history(new_log, log)

            return SpectrumPlus(
                spectrum=new_spectrum,
                fit_method=fit_method,
                log_file=new_log,
                region=continuum_region
            )

        # convert each subregion to a SpectrumPlus
        spectrum_list = []
        for i, (spec, rmin, rmax) in enumerate(zip(new_spectrum, rmins, rmaxs)):
            new_log = _copy_headers(log_file)
            _log_history(new_log, f'Extracting region : {rmin} - {rmax}')

            continuum_reg = continuum_region[i] if continuum_region is not None else None

            spectrum_list.append(
                SpectrumPlus(
                    spectrum=spec,
                    fit_method=fit_method,
                    log_file=new_log,
                    region=continuum_reg
                )
            )

        return spectrum_list

    def remove_nonfinite(self, return_mask=False):
        '''
        Return a new SpectrumPlus with all samples removed
        where the flux is not finite (NaN, +inf, -inf).

        This reduces the length of the spectrum!

        Parameters
        ----------
        return_mask : bool, optional, default=False
            If True, return the boolean mask used
            to remove non finite values.

        Returns
        -------
        SpectrumPlus :
            New SpectrumPlus object with all non finite values
            removed from the object.
        finite : bool
            Boolean mask used to mask the spectrum spectral axis
            and flux. Only returned if `return_mask` is True.
        '''
        finite = np.isfinite(self.flux)

        finite_spec = SpectrumPlus(
            spectral_axis=self.spectral_axis[finite],
            flux=self.flux[finite]
        )

        if return_mask:
            return finite_spec, finite

        return finite_spec

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
        new_log = _copy_headers(self.log_file)
        _log_history(new_log, f'Replacing flux at masked locations')

        return SpectrumPlus(
            spectrum=new_spectrum,
            fit_method=fit_method,
            log_file=new_log
        )

    # helper functions
    # ----------------
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
        new_log = _copy_headers(self.log_file)

        _log_history(new_log, f'Sliced data with key : {key}')

        return SpectrumPlus(
            spectrum=sub_spectrum,
            fit_method=fit_method,
            log_file=new_log
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

        new_log = _copy_headers(self.log_file)
        _log_history(new_log, f'Multiplying flux by {factor}')

        return SpectrumPlus(
            spectrum=scaled_spectrum,
            fit_method=fit_method,
            region=region,
            log_file=new_log
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

        new_log = _copy_headers(self.log_file)
        _log_history(new_log, f'Dividing flux by {factor}')

        return SpectrumPlus(
            spectrum=scaled_spectrum,
            fit_method=fit_method,
            region=region,
            log_file=new_log
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

        Returns
        -------
        continuum : Quantity
            Quantity array of continuum values.
        '''
        from .spectra_utils import fit_continuum

        return fit_continuum(spectrum, fit_method, region)


    def _construct_spectrum(
        self, *, spectrum=None, spectral_axis=None,
        flux=None, log_file=None, **kwargs
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

            if log_file is not None:
                _log_history(log_file, "Initialize SpectrumPlus from a <Spectrum> object")

            return spectrum

        if spectral_axis is None or flux is None:
            raise ValueError(
                'spectral_axis and flux must both be provided if spectrum is None.'
            )

        if log_file is not None:
            _log_history(
                log_file, "Initialize SpectrumPlus with input arrays"
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
