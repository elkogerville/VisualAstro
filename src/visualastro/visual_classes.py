'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-19
Description:
    Visualastro core data classes.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - spectral_cube
    - specutils
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
    - ExtractedSpectrum
        Data class for extracted spectra.
    - FitsFile
        Lightweight data class for fits files.
    - Utility Functions
        Functions used by all classes.
'''

import os
import warnings
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.units import Quantity, Unit
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from spectral_cube import SpectralCube
from specutils.fitting import fit_generic_continuum, fit_continuum
from specutils.spectra import Spectrum1D
from .va_config import get_config_value, va_config, _default_flag

# DataCube
# ––––––––
class DataCube:
    '''
    Lightweight wrapper for handling 3D spectral_cubes
    or arrays with optional headers and error arrays.
    This class supports both `numpy.ndarray` and `SpectralCube`
    inputs and provides convenient access to cube statistics,
    metadata, and visualization methods.

    Parameters
    ––––––––––
    data : np.ndarray or SpectralCube
        The input data cube. Must be 3-dimensional (T, N, M).
    header : fits.Header, array-like of fits.Header, or None, optional, default=None
        Header(s) associated with the data cube. If provided as a list or array,
        its length must match the cube’s first dimension.
    error : np.ndarray or None, optional, default=None
        Array of uncertainties with the same shape as `data`.
    wcs : astropy.wcs.wcs.WCS, array-like of astropy.wcs.wcs.WCS, or None, optional, default=None
        WCS information associated with the data extension.
        Can also be an array-like of WCS objects. If None,
        DataCube will attempt to extract the WCS from the header
        attribute. If `header` is an array-like, DataCube will
        extract the WCS from each header.

    Attributes
    ––––––––––
    data : np.ndarray or SpectralCube
        Original data object.
    header : array-like of fits.Header or fits.Header
        Header(s) associated with the data cube.
    error : np.ndarray or None
        Error array if provided, else None.
    value : np.ndarray
        Raw numpy array of the cube values.
    unit : astropy.units.Unit or None
        Physical unit of the data if available.
    wcs : array-like of WCS or WCS
        WCS(s) associated with the data cube.

    Properties
    ––––––––––
    shape : tuple
        Shape of the cube (T, N, M).
    size : int
        Total number of elements in the cube.
    ndim : int
        Number of dimensions.
    dtype : np.dtype
        Data type of the array.
    len : int
        Length of the first axis (T dimension).
    has_nan : bool
        True if any element in the cube is NaN.
    itemsize : int
        Size of one array element in bytes.
    nbytes : int
        Total memory footprint of the data in bytes.

    Methods
    –––––––

    Raises
    ––––––
    TypeError
        If `data` or `header` are not of an expected type.
    ValueError
        If `data` is not 3D, or if the dimensions of `header` or `error` do not match `data`.
    Examples
    ––––––––
    Load DataCube from fits file
    >>> cube = load_fits(path)
    >>> cube.header
    >>> cube.inspect()
    '''
    def __init__(self, data, header=None, error=None, wcs=None):
        self._initialize(data, header, error, wcs)

    def _initialize(self, data, header, error, wcs):

        # type checks
        if not isinstance(data, (np.ndarray, Quantity, SpectralCube)):
            raise TypeError(
                "'data' must be a np.ndarray, Quantity, or SpectralCube, "
                f'got {type(data).__name__}.'
            )

        # extract array view for validation
        if isinstance(data, SpectralCube):
            array = data.unmasked_data[:].value
            unit = data.unit
        elif isinstance(data, Quantity):
            array = data.value
            unit = data.unit
        else:
            array = np.asarray(data)
            unit = None

        # shape validation
        if array.ndim != 3:
            raise ValueError(
                f"'data' must be 3D (T, N, M), got shape {array.shape}."
            )

        # error validation
        if error is not None:
            err = np.asarray(error)
            if err.shape != array.shape:
                raise ValueError(
                    f"'error' must match shape of 'data', got {err.shape} vs {array.shape}."
                )

        # header validation
        if header is not None and not isinstance(
            header, (list, Header, np.ndarray, tuple)
        ):
            raise TypeError(
                "'header' must be a fits.Header or array-like of fits.header, "
                f'got {type(header).__name__}.'
            )

        if isinstance(header, (list, np.ndarray, tuple)):
            if len(header) == 0:
                raise ValueError(
                    'Header list cannot be empty.'
                )
            if array.shape[0] != len(header):
                raise ValueError(
                    f'Mismatch between T dimension and number of header: '
                    f'T={array.shape}, header={len(header)}.'
                )
            primary_hdr = header[0]
        else:
            primary_hdr = header

        # try extracting unit from headers if not found earlier
        if unit is None and isinstance(primary_hdr, Header):
            try:
                unit = Unit(primary_hdr['BUNIT'])
            except (ValueError, KeyError) as e:
                pass

        # try extracting WCS from headers
        if wcs is None:
            if isinstance(header, Header):
                try:
                    wcs = WCS(header)
                except Exception:
                    pass
            # if a list of headers extract a list of wcs
            elif isinstance(header, (list, np.ndarray, tuple)):
                wcs = []
                for h in header:
                    if not isinstance(h, Header):
                        wcs.append(None)
                        continue
                    try:
                        wcs.append(WCS(h))
                    except Exception:
                        wcs.append(None)

        if unit is not None and not isinstance(data, (Quantity, SpectralCube)):
            data = array * unit

        # assign attributes
        self.data = data
        self.header = header
        self.error = error
        self.value = array
        self.unit = unit
        self.wcs = wcs

    # support slicing
    def __getitem__(self, key):
        '''
        Return a slice or sub-cube from the data.
        Parameters
        ––––––––––
        key : slice or tuple
            Index or slice to apply to the cube.
        Returns
        –––––––
        slice : same type as `data`
            The corresponding subset of the cube.
        '''
        return self.data[key]

    def to(self, unit, equivalencies=None):
        '''
        Convert the DataCube data to a new physical unit.
        This method supports Quantity objects. SpectralCubes
        are also supported but only for 'flux' units. To
        convert SpectralCube wavelength units use `.with_spectral_unit()`.

        Parameters
        ––––––––––
        unit : str or astropy.units.Unit
            Target unit.
        equivalencies : list, optional
            Astropy equivalencies for unit conversion (e.g. spectral).

        Returns
        –––––––
        DataCube
            New cube with converted units.
        '''
        # convert unit to astropy unit
        unit = Unit(unit)
        # check that data has a unit
        if not isinstance(self.data, (Quantity, SpectralCube)):
            raise TypeError(
                'DataCube.data has no unit. Cannot use .to() '
                'unless data is an astropy Quantity.\n'
                'For SpectralCubes, use the .to() for flux '
                'conversions and with_spectral_unit() for '
                'converting the spectral axis.'
            )

        try:
            new_data = self.data.to(unit, equivalencies=equivalencies)
        except Exception as e:
            raise TypeError(
                f'Unit conversion failed: {e}'
            )
        # convert errors if present
        if self.error is not None:
            if isinstance(self.error, Quantity):
                new_error = self.error.to(unit, equivalencies=equivalencies)
            else:
                raise TypeError(
                    'DataCube.error must be a Quantity to convert units safely.'
                )
        else:
            new_error = None

        return DataCube(
            data=new_data,
            header=self.header,
            error=new_error,
            wcs=self.wcs
        )

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        '''
        Convert the spectral axis of the DataCube to a new unit.

        Parameters
        ––––––––––
        unit : str or astropy.units.Unit
            Target spectral unit.
        velocity_convention : str, optional
            'radio', 'optical', 'relativistic', etc.
        rest_value : Quantity, optional
            Rest frequency/wavelength for Doppler conversion.

        Returns
        –––––––
        DataCube
            New cube with converted spectral axis.
        '''

        unit = Unit(unit)

        if not isinstance(self.data, SpectralCube):
            raise TypeError(
                "with_spectral_unit() can only be used when DataCube.data "
                "is a SpectralCube. For unit conversion of flux values, "
                "use .to()."
            )
        # convert spectral axis
        try:
            new_data = self.data.with_spectral_unit(
                unit,
                velocity_convention=velocity_convention,
                rest_value=rest_value
            )
        except Exception as e:
            raise TypeError(f'Spectral axis conversion failed: {e}')

        return DataCube(
            data=new_data,
            header=self.header,
            error=self.error,
            wcs=self.wcs
        )

    # support reshaping
    def reshape(self, *shape):
        '''
        Return a reshaped view of the cube data.
        Parameters
        ––––––––––
        *shape : int
            New shape for the data array.
        Returns
        –––––––
        np.ndarray
            Reshaped data array.
        '''
        return self.value.reshape(*shape)

    # support len()
    def __len__(self):
        '''
        Return the number of spectral slices along the first axis.
        Returns
        –––––––
        int
            Length of the first dimension (T).
        '''
        return len(self.value)

    # support numpy operations
    def __array__(self):
        '''
        Return the underlying data as a NumPy array.
        Returns
        –––––––
        np.ndarray
            The underlying 3D array representation.
        '''
        return self.value

    def __mul__(self, other):
        '''
        Multiply the data cube by a scalar, a Quantity, or another DataCube.
        This operation returns a new `DataCube` instance. The WCS and headers
        of the original cube are preserved. Errors are propagated according to
        standard Gaussian error propagation rules.

        Parameters
        ––––––––––
        other : scalar, `~astropy.units.Quantity`, or DataCube
            - If a scalar or Quantity, the cube data are multiplied by `other`
                and the uncertainties are scaled by `abs(other)`.
            - If another `DataCube`, the data arrays are multiplied
                element-wise. The two cubes must have matching shapes.

        Returns
        –––––––
        DataCube
            A new data cube containing the multiplied data and propagated
            uncertainties.

        Notes
        –––––
        - Error propagation**

            For multiplication of data `A` by `k`, a scalar, Quantity
            object, or a broadcastable array or quantity array:
                A' = kA
                σA' = |k|σA

            For the product of two cubes `A` and `B` with uncertainties
            `σA` and `σB` (assumed independent):

                C = AB
                σC = sqrt( (A σB)**2 + (B σA)**2 )

            If only one cube provides uncertainties, the missing uncertainties
            are assumed to be zero.

        - WCS information is kept intact.

        Examples
        ––––––––
        Multiply a cube by a scalar:
            cube2 = cube1 * 3

        Multiply by a Quantity:
            cube2 = cube1 * (5 * u.um)

        Multiply two cubes with uncertainty propagation:
            cube3 = cube1 * cube2
        '''

        A = self.data
        σA = self.error

        if (np.isscalar(other)) or (isinstance(other, Quantity) and other.ndim == 0):

            new_data = A * other

            if σA is not None:
                new_error = σA * np.abs(other)
            else:
                new_error = None

            return DataCube(
                data=new_data,
                header=self.header,
                error=new_error,
                wcs=self.wcs
            )
        elif isinstance(other, (np.ndarray, Quantity)):
            # try broadcasting
            try:
                new_data = A * other
            except Exception:
                raise TypeError(
                    'Array or Quantity array cannot be broadcast to cube shape.\n'
                    f'self.data.shape: {self.data.shape}, other.shape: {other.shape}.'
                )

            # other has no uncertainties
            if σA is not None:
                new_error = σA * np.abs(other)
            else:
                new_error = None

            return DataCube(
                data=new_data,
                header=self.header,
                error=new_error,
                wcs=self.wcs
            )
        elif hasattr(other, 'data'):

            B = other.data
            σB = getattr(other, 'error', None)

            if A.shape != B.shape:
                raise ValueError(
                    f'DataCube shapes do not match: '
                    f'{A.shape} vs {B.shape}'
                )

            new_data = A * B

            if (σA is not None) and (σB is not None):
                 new_error = np.sqrt(
                     (A * σB)**2 + (B * σA)**2
                 )
            elif σA is not None:
                new_error = σA * np.abs(B)
            elif σB is not None:
                new_error = σB * np.abs(A)
            else:
                new_error = None

            return DataCube(
                data=new_data,
                header=self.header,
                error=new_error,
                wcs=self.wcs
            )
        else:
            raise ValueError(f'Invalid input: {other}!')

    __rmul__ = __mul__

    def __truediv__(self, other):
        '''
        Divide the data cube by a scalar, a Quantity, or another DataCube.
        This operation returns a new `DataCube` instance. The WCS and headers
        of the original cube are preserved. Errors are propagated according to
        standard Gaussian error propagation rules.

        Parameters
        ––––––––––
        other : scalar, `~astropy.units.Quantity`, or DataCube
            - If a scalar or Quantity, the cube data are divided by `other`
                and the uncertainties are scaled by `abs(other)`.
            - If another `DataCube`, the data arrays are divided
                element-wise. The two cubes must have matching shapes.

        Returns
        –––––––
        DataCube
            A new data cube containing the divided data and propagated
            uncertainties.

        Notes
        –––––
        **Error propagation**

            For division of data `A` by a scalar or Quantity `k`:
                A' = kA
                σA' = |k|σA

            For the quotient of two cubes `A` and `B` with uncertainties
            `σA` and `σB` (assumed independent):

                C = AB
                σC = sqrt( (A σB)**2 + (B σA)**2 )

            If only one cube provides uncertainties, the missing uncertainties
            are assumed to be zero.

        **WCS Handling**

            The WCS is passed through unchanged.

        Examples
        ––––––––
        Divide a cube by a scalar:
            cube2 = cube1 / 3

        Divide by a Quantity:
            cube2 = cube1 / (5 * u.um)

        Divide two cubes with uncertainty propagation:
            cube3 = cube1 * cube2
        '''

        if np.isscalar(other) or isinstance(other, Quantity):

            new_data = self.data / other

            if self.error is not None:
                new_error = self.error / abs(other)
            else:
                new_error = None

            return DataCube(
                data=new_data,
                header=self.header,
                error=new_error,
                wcs=self.wcs
            )

        if hasattr(other, 'data'):

            A = self.data
            σA = self.error
            B = other.data
            σB = getattr(other, 'error', None)


            new_data = A / B

            if (σA is not None) and (σB is not None):
                 new_error = np.abs(new_data) * np.sqrt(
                     (σA / A)**2 + (σB / B)**2
                 )
            elif σA is not None:
                new_error = σA / abs(B)
            elif σB is not None:
                new_error = np.abs(A) * σB / (B**2)
            else:
                new_error = None

            return DataCube(
                data=new_data,
                header=self.header,
                error=new_error,
                wcs=self.wcs
            )

    def header_get(self, key):
        '''
        Retrieve a header value by key from one or multiple headers.
        Parameters
        ––––––––––
        key : str
            FITS header keyword to retrieve.
        Returns
        –––––––
        value : list or str
            Header value(s) corresponding to `key`.
        Raises
        ––––––
        ValueError
            If headers are of an unsupported type or `key` is not found.
        '''
        if isinstance(self.header, (list, np.ndarray, tuple)):
            return [h[key] for h in self.header]
        elif isinstance(self.header, Header):
            return self.header[key] # type: ignore
        else:
            raise ValueError(f"Unsupported header type or key '{key}' not found.")

    def view(self, idx, vmin=_default_flag, vmax=_default_flag,
             norm=_default_flag, percentile=_default_flag,
             cmap=None, style=None, figsize=None):
        '''
        Display a 3D data array using matplotlib's imshow with configurable
        normalization, percentile clipping, and style context.
        Parameters
        ––––––––––
        data : array-like
            2D array of image data to display.
        idx : int or list of int, optional, default=None
            Index for slicing along the first axis if 'datas'
            contains a cube.
            - i -> returns cube[i]
            - [i] -> returns cube[i]
            - [i, j] -> returns the sum of cube[i:j+1] along axis 0
            If 'datas' is a list of cubes, you may also pass a list of
            indeces.
            ex: passing indeces for 2 cubes-> [[i,j], k].
        vmin : float or None, default=`_default_flag`
            Lower bound of the display range. If `_default_flag`, uses
            `va_config.vmin`. If both `vmin` and `percentile` are specified,
            `vmin` takes precedence.
        vmax : float or None, default=`_default_flag`
            Upper bound of the display range. If `_default_flag`, uses
            `va_config.vmax`. If both `vmax` and `percentile` are specified,
            `vmax` takes precedence.
        norm : str or matplotlib.colors.Normalize, default=`_default_flag`
            Normalization to apply to the image (e.g., 'linear', 'log',
            'asinh'). If `_default_flag`, uses `va_config.norm`.
        percentile : tuple of two floats or None, default=`_default_flag`
            Percentile range `(low, high)` used to set `vmin` and `vmax`
            automatically. If `_default_flag`, uses `va_config.percentile`.
        cmap : str or matplotlib colormap, default=None
            Colormap to use. If `None`, uses `va_config.cmap`.
        style : str or None, default=None
            Matplotlib style to apply (e.g., 'astro', 'dark_background').
            If `None`, uses `va_config.style`.
        figsize : array-like of two floats or None, default=None
            Figure size `(width, height)` in inches. If `None`, uses
            `va_config.figsize`.
        '''
        cube = self.value
        data = _slice_cube(cube, idx)
        clabel = self.unit

        _view(data, vmin, vmax, norm, percentile, cmap, style, figsize, clabel)

    def with_mask(self, mask):
        '''
        Apply a boolean mask to the cube and return the masked data.
        Parameters
        ––––––––––
        mask : np.ndarray or Mask
            Boolean mask to apply. Must match the cube shape.
        Returns
        –––––––
        masked_data : same type as `data`
            Masked version of the data.
        Raises
        ––––––
        TypeError
            If masking is unsupported for the data type.
        '''
        if isinstance(self.data, SpectralCube):
            return self.data.with_mask(mask) # type: ignore
        elif isinstance(self.data, (np.ndarray, Quantity)):
            return self.data[mask]
        else:
            raise TypeError(f'Cannot apply mask to data of type {type(self.data)}')

    def inspect(self, figsize=(10,6), style=None):
        '''
        Plot the mean and standard deviation across each cube slice.
        Useful for quickly identifying slices of interest in the cube.
        Parameters
        ––––––––––
        figsize : tuple, optional, default=(8,4)
            Size of the output figure.
        style : str or None, optional, default=None
            Matplotlib style to use for plotting. If None,
            uses the default value set by `va_config.style`.
        Notes
        –––––
        This method visualizes the mean and standard deviation of flux across
        each 2D slice of the cube as a function of slice index.
        '''
        # get default va_config values
        style = get_config_value(style, 'style')

        cube = self.value
        # compute mean and std across wavelengths
        mean_flux = np.nanmean(cube, axis=(1, 2))
        std_flux  = np.nanstd(cube, axis=(1, 2))

        T = np.arange(mean_flux.shape[0])
        style = _return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(T, mean_flux, c='darkslateblue', label='Mean')
            ax.plot(T, std_flux, c='#D81B60', ls='--', label='Std Dev')

            ax.set_xlabel('Cube Slice Index')
            ax.set_ylabel('Counts')
            ax.set_xlim(np.nanmin(T), np.nanmax(T))

            ax.legend(loc='best')

            plt.show()

    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of `DataCube`.
        '''
        return (
            f'<DataCube: unit={self.unit}, shape={self.shape}, dtype={self.dtype}>'
        )

    # physical properties / statistics
    @property
    def max(self):
        '''
        Returns
        –––––––
        float : Maximum value in the cube, ignoring NaNs.
        '''
        return np.nanmax(self.value)
    @property
    def min(self):
        '''
        Returns
        –––––––
        float : Minimum value in the cube, ignoring NaNs.
        '''
        return np.nanmin(self.value)
    @property
    def mean(self):
        '''
        Returns
        –––––––
        float : Mean of all values in the cube, ignoring NaNs.
        '''
        return np.nanmean(self.value)
    @property
    def median(self):
        '''
        Returns
        –––––––
        float : Median of all values in the cube, ignoring NaNs.
        '''
        return np.nanmedian(self.value)
    @property
    def std(self):
        '''
        Returns
        –––––––
        float : Standard deviation of all values in the cube, ignoring NaNs.
        '''
        return np.nanstd(self.value)


# ExtractedSpectrum
# –––––––––––––––––
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
    unit : `~astropy.units.Unit` or None
        Flux unit inferred from `flux` or `spectrum1d`.
    wave_unit : `~astropy.units.Unit` or None
        Wavelength unit inferred from `wavelength` or `spectrum1d`.
    '''
    def __init__(self, wavelength=None, flux=None, spectrum1d=None,
                 normalized=None, continuum_fit=None):
        self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

    def _initialize(self, wavelength, flux, spectrum1d, normalized, continuum_fit):
        self.wavelength = wavelength
        self.flux = flux
        self.spectrum1d = spectrum1d
        self.normalized = normalized
        self.continuum_fit = continuum_fit

        # assign attributes
        unit = None
        for candidate in (getattr(flux, 'unit', None),
                          getattr(spectrum1d, 'unit', None)):
            if candidate is not None:
                unit = candidate
                break

        wave_unit = None
        for candidate in (getattr(wavelength, 'unit', None),
                          getattr(getattr(spectrum1d, 'spectral_axis', None), 'unit', None)):
            if candidate is not None:
                wave_unit = candidate
                break
        self.unit = unit
        self.wave_unit = wave_unit

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
            continuum_fit = _compute_continuum_fit(spectrum1d, fit_method, region)
            normalized = (spectrum1d / continuum_fit).flux

            self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

            return None

        wavelength = self.wavelength if wavelength is None else wavelength
        flux = self.flux if flux is None else flux

        # rebuild spectrum1d if wavelength or flux is passed in
        if not _allclose(wavelength, self.wavelength) or not _allclose(flux, self.flux):

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
                continuum_fit = _compute_continuum_fit(spectrum1d, fit_method, region)
            if normalized is None:
                normalized = (spectrum1d / continuum_fit).flux

        # use previous values unless provided / recomputed
        spectrum1d = self.spectrum1d if spectrum1d is None else spectrum1d
        normalized = self.normalized if normalized is None else normalized
        continuum_fit = self.continuum_fit if continuum_fit is None else continuum_fit

        self._initialize(wavelength, flux, spectrum1d, normalized, continuum_fit)

        return None

    def view(self, color=None, style=None, figsize=None):
        '''
        Plot the spectrum flux versus wavelength.
        Parameters
        ––––––––––
        color : str or None, default=None
            Line color used for plotting the spectrum. If `None`,
            defaults to 'darkslateblue'.
        style : str or None, default=None
            Matplotlib style to apply (e.g., 'astro', 'dark_background').
            If `None`, uses the default style set by `va_config.style`.
        figsize : array-like of two floats or None, default=None
            Figure size `(width, height)` in inches. If `None`, uses
            the default value set by `va_config.figsize`.
        '''
        color = 'darkslateblue' if color is None else color
        style = get_config_value(style, 'style')
        figsize = get_config_value(figsize, 'figsize')

        style = _return_stylename(style)
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)

            wavelength = self.wavelength
            wavelength = wavelength.value if isinstance(wavelength, Quantity) else wavelength
            flux = self.flux
            flux = flux.value if isinstance(flux, Quantity) else flux

            ax.plot(wavelength, flux, c=color)

            ax.set_xlim(np.nanmin(wavelength), np.nanmax(wavelength))

            wave_unit = f' [{self.wave_unit}]' if self.wave_unit is not None else ''
            unit = f' [{self.unit}]' if self.unit is not None else ''
            ax.set_xlabel(f'Wavelength{wave_unit}')
            ax.set_ylabel(f'Flux{unit}')

            plt.show()

    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of `ExtractedSpectrum`.
        '''
        return (
            f'<ExtractedSpectrum: wave_unit={self.wave_unit}, flux_unit={self.unit}, len={len(self.wavelength)}>'
        )


# FitsFile
# ––––––––
class FitsFile:
    '''
    Lightweight container for FITS image data and metadata.

    Parameters
    ––––––––––
    data : array-like or `~astropy.units.Quantity`
        The primary image data. Can be a NumPy array or an
        `astropy.units.Quantity` object.
    header : `~astropy.io.fits.Header`, optional
        FITS header associated with the data. Used to extract
        WCS information and physical units if available.
    error : array-like, optional
        Optional uncertainty or error map associated with the data.
    wcs : astropy.wcs.wcs.WCS or None, optional, default=None
        WCS information associated with the data extension.
        If None, FitsFile will attempt to extract the WCS
        from the header attribute.

    Attributes
    ––––––––––
    data : ndarray
        The numerical data array (converted from `Quantity` if needed).
    header : `~astropy.io.fits.Header` or None
        The FITS header provided at initialization.
    error : ndarray or None
        Associated error data, if provided.
    value : ndarray
        Alias for `.data`.
    unit : `~astropy.units.Unit` or None
        The physical unit of the data. Inferred from the input
        `Quantity` or from the FITS header keyword ``BUNIT``.
    wcs : `~astropy.wcs.WCS` or None
        The WCS transformation extracted from the header, if valid.

    shape : tuple
        Shape of the data array.
    size : int
        Total number of elements in the data array.
    ndim : int
        Number of dimensions.
    dtype : numpy.dtype
        Data type of the array elements.
    len : int
        Length of the first axis.
    has_nan : bool
        True if the data contains any NaN values.
    itemsize : int
        Size in bytes of each array element.
    nbytes : int
        Total memory footprint of the array in bytes.
    '''
    def __init__(self, data, header=None, error=None, wcs=None):
        self._initialize(data, header, error, wcs)

    def _initialize(self, data, header, error, wcs):

        array = np.asarray(data)

        if isinstance(data, Quantity):
            unit = data.unit # type: ignore
        else:
            unit = None
            if isinstance(header, Header) and 'BUNIT' in header:
                try:
                    unit = Unit(header['BUNIT'])
                except Exception:
                    pass

        if not isinstance(data, Quantity):
            if unit is not None:
                data = data * unit

         # try extracting WCS
        if wcs is None and isinstance(header, Header):
            try:
                wcs = WCS(header)
            except Exception:
                pass

        self.data = data
        self.header = header
        self.error = error
        self.value = array
        self.unit = unit
        self.wcs = wcs
        self.footprint = None

        # data attributes
        self.shape = data.shape
        self.size = data.size
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.len = len(data)
        self.has_nan = np.isnan(data).any()
        self.itemsize = data.itemsize
        self.nbytes = data.nbytes

    # magic functions for FitsFile to behave like a np.ndarray
    def __getitem__(self, key):
        '''
        Return a slice of the data.
        Parameters
        ––––––––––
        key : slice or tuple
            Index or slice to apply to the data.
        Returns
        –––––––
        slice : same type as `data`
            The corresponding subset of the data.
        '''
        return self.data[key]

    def reshape(self, *shape):
        '''
        Return a reshaped view of the data.
        Parameters
        ––––––––––
        *shape : int
            New shape for the data array.
        Returns
        –––––––
        np.ndarray
            Reshaped data array.
        '''
        return self.data.reshape(*shape)

    def __len__(self):
        '''
        Return the number of spectral slices along the first axis.
        Returns
        –––––––
        int
            Length of the first dimension.
        '''
        return len(self.data)

    def __array__(self):
        '''
        Return the underlying data as a NumPy array.
        Returns
        –––––––
        np.ndarray
            The underlying 2D array representation.
        '''
        return self.data

    def header_get(self, key):
        '''
        Retrieve a header value by key from a header.
        Parameters
        ––––––––––
        key : str
            FITS header keyword to retrieve.
        Returns
        –––––––
        value : list or str
            Header values corresponding to `key`.
        Raises
        ––––––
        ValueError
            If headers are of an unsupported type or `key` is not found.
        '''
        if isinstance(self.header, Header):
            return self.header[key] # type: ignore
        else:
            raise ValueError(f"Unsupported header type or key '{key}' not found.")

    def update(self, data=None, header=None, error=None, wcs=None):
        '''
        Update any of the FitsFile attributes. All internally stored
        values are recomputed.
        Parameters
        ––––––––––
        data : array-like or `~astropy.units.Quantity`
            The primary image data. Can be a NumPy array or an
            `astropy.units.Quantity` object.
        header : `~astropy.io.fits.Header`, optional
            FITS header associated with the data. Used to extract
            WCS information and physical units if available.
        error : array-like, optional
            Optional uncertainty or error map associated with the data.
        wcs : astropy.wcs.wcs.WCS or None, optional, default=None
            WCS information associated with the data extension.
            If None, FitsFile will attempt to extract the WCS
            from the header attribute.

        Returns
        –––––––
        None
        '''
        data = self.data if data is None else data
        header = self.header if header is None else header
        error = self.error if error is None else error
        wcs = self.wcs if wcs is None else wcs

        self._initialize(data, header, error, wcs)

        return None

    def view(self, vmin=_default_flag, vmax=_default_flag,
             norm=_default_flag, percentile=_default_flag,
             cmap=None, style=None, figsize=None):
        '''
        Display a 2D data array using matplotlib's imshow with configurable
        normalization, percentile clipping, and style context.
        Parameters
        ––––––––––
        data : array-like
            2D array of image data to display.
        vmin : float or None, default=`_default_flag`
            Lower bound of the display range. If `_default_flag`, uses
            `va_config.vmin`. If both `vmin` and `percentile` are specified,
            `vmin` takes precedence.
        vmax : float or None, default=`_default_flag`
            Upper bound of the display range. If `_default_flag`, uses
            `va_config.vmax`. If both `vmax` and `percentile` are specified,
            `vmax` takes precedence.
        norm : str or matplotlib.colors.Normalize, default=`_default_flag`
            Normalization to apply to the image (e.g., 'linear', 'log',
            'asinh'). If `_default_flag`, uses `va_config.norm`.
        percentile : tuple of two floats or None, default=`_default_flag`
            Percentile range `(low, high)` used to set `vmin` and `vmax`
            automatically. If `_default_flag`, uses `va_config.percentile`.
        cmap : str or matplotlib colormap, default=None
            Colormap to use. If `None`, uses `va_config.cmap`.
        style : str or None, default=None
            Matplotlib style to apply (e.g., 'astro', 'dark_background').
            If `None`, uses `va_config.style`.
        figsize : array-like of two floats or None, default=None
            Figure size `(width, height)` in inches. If `None`, uses
            `va_config.figsize`.
        '''
        data = self.data
        clabel = self.unit
        _view(data, vmin, vmax, norm, percentile, cmap, style, figsize, clabel)

    def crop(self, size, position=None, mode='trim', frame='icrs', origin_idx=0):
        '''
        Crop the FitsFile object around a given position using WCS.
        This method creates a `Cutout2D` from the data, centered on a
        specified position in either pixel or world (RA/Dec) coordinates.
        It automatically handles cases where the WCS axes have been swapped
        due to a data transpose, and applies the same cropping to the
        associated error map if available.

        Parameters
        ––––––––––
        size : `~astropy.units.Quantity`, float, int, or tuple
            The size of the cutout region. Can be a single
            `Quantity` or a tuple specifying height and width.
            If a float or int, will interpret as number of pixels
            from center. If float, will round to nearest int.
            Ex:
                - 6 * u.arcsec
                - (6*u.deg, 4*u.deg)
                - (7, 8)
        position : array-like, `~astropy.coordinates.SkyCoord`, optional, default=None
            The center of the cutout region. Accepted formats are:
            - `(x, y)` : pixel coordinates (integers or floats)
            - `(ra, dec)` : sky coordinates as `~astropy.units.Quantity` in angular units
            - `~astropy.coordinates.SkyCoord` : directly specify a coordinate object
            - If None, defaults to the center of the image.
        mode : {'trim', 'partial', 'strict'}, default='trim'
            Defines how the function handles edges that fall outside the image:
            - 'trim': Trim the cutout to fit within the image bounds.
            - 'partial': Include all pixels that overlap the image, padded with NaNs.
            - 'strict': Raise an error if any part of the cutout is outside the image.
        frame : str, default='icrs'
            Coordinate frame for interpreting RA/Dec values when creating the `SkyCoord`.
        origin_idx : int, default=0
            Origin index for pixel-to-world conversion (0 for 0-based, 1 for 1-based).

        Returns
        –––––––
        cropped : FitsFile
            A new `FitsFile` instance containing:
            - data : Cropped image as a `np.ndarray`
            - header : Original FITS header
            - error : Cropped error array (if available)
            - wcs : Updated WCS corresponding to the cutout region

        Raises
        ––––––
        ValueError
            If the WCS is missing (None) or cutout creation fails.
        TypeError
            If the position is not one of the accepted types.

        Notes
        –––––
        - If the data were transposed and the WCS was swapped via `wcs.swapaxes(0, 1)`,
            the method will automatically attempt to correct for inverted RA/Dec axes.
        - The same cutout region is applied to the error array if present.

        Examples
        ––––––––
        Crop by pixel coordinates:
            >>> cube.crop(size=100, position=(250, 300))

        Crop by pixel coordinates:
            >>> cube.crop(size=6*u.arcsec, position=(250, 300))

        Crop by sky coordinates:
            >>> from astropy import units as u
            >>> cube.crop(size=6*u.arcsec, position=(83.8667*u.deg, -69.2697*u.deg))

        Crop using a SkyCoord object:
            >>> from astropy.coordinates import SkyCoord
            >>> c = SkyCoord(ra=83.8667*u.deg, dec=-69.2697*u.deg, frame='icrs')
            >>> cube.crop(size=6*u.arcsec, position=c)
        '''
        data = self.data
        error = self.error
        wcs = self.wcs

        if wcs is None:
            raise ValueError (
                'WCS is None. Please crop using normal array indexing or enter WCS.'
            )
        # if no position passed in use center of image
        if position is None:
            ny, nx = data.shape
            position = [nx//2, ny//2]
        # if position is passed in as integers, assume pixel indeces
        # and convert to world coordinates
        if isinstance(position[0], (float, int)) and isinstance(position[1], (float, int)):
            ra, dec = wcs.wcs_pix2world(position[0], position[1], origin_idx)
            ra *= u.deg
            dec *= u.deg
        # if position passed in as Quantity, convert to degrees
        elif (
            len(position) == 2 and
            all(isinstance(p, Quantity) for p in position)
        ):
            ra = position[0].to(u.deg)
            dec = position[1].to(u.deg)
        # if position passed in as SkyCoord, use that
        elif isinstance(position, SkyCoord):
            center = position
        else:
            raise TypeError(
                'Position must be a (x, y) pixel tuple, (ra, dec) in degrees, or a SkyCoord.'
            )
        # crop image
        if not isinstance(position, SkyCoord):
            # try with ra=ra and dec=dec and if it fails; swap
            # this is because the user maybe performed wcs.swapaxes(0,1)
            try:
                center = SkyCoord(ra=ra, dec=dec, frame=frame)
                cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
            except ValueError:
                # fallback if WCS RA/Dec swapped
                center = SkyCoord(ra=dec, dec=ra, frame=frame)
                cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
        else:
            cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
        # also crop the errors if available
        if error is not None:
            error = Cutout2D(error, position=center, size=size, wcs=wcs, mode=mode).data

        return FitsFile(cutout.data, header=self.header, error=error, wcs=cutout.wcs)


    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of `FitsFile`.
        '''
        return (
            f'<FitsFile: unit={self.unit}, shape={self.shape}, dtype={self.dtype}>'
        )

    # physical properties / statistics
    @property
    def max(self):
        '''
        Returns
        –––––––
        float : Maximum value in the data, ignoring NaNs.
        '''
        return np.nanmax(self.data)
    @property
    def min(self):
        '''
        Returns
        –––––––
        float : Minimum value in the data, ignoring NaNs.
        '''
        return np.nanmin(self.data)
    @property
    def mean(self):
        '''
        Returns
        –––––––
        float : Mean of all values in the data, ignoring NaNs.
        '''
        return np.nanmean(self.data)
    @property
    def median(self):
        '''
        Returns
        –––––––
        float : Median of all values in the data, ignoring NaNs.
        '''
        return np.nanmedian(self.data)
    @property
    def std(self):
        '''
        Returns
        –––––––
        float : Standard deviation of all values in the data, ignoring NaNs.
        '''
        return np.nanstd(self.data)


# Utility Functions
# –––––––––––––––––
def _compute_continuum_fit(spectrum1d, fit_method='fit_continuum', region=None):
    '''
    Fit the continuum of a 1D spectrum using a specified method.
    Parameters
    ––––––––––
    spectrum1d : Spectrum1D or ExtractedSpectrum
        Input 1D spectrum object containing flux and spectral_axis.
        ExtractedSpectrum is supported only if it contains a
        spectrum1d object.
    fit_method : str, optional, default='generic'
        Method used for fitting the continuum.
        - 'fit_continuum': uses `fit_continuum` with a specified window
        - 'generic'      : uses `fit_generic_continuum`
    region : array-like, optional, default=None
        Wavelength or pixel region(s) to use when `fit_method='fit_continuum'`.
        Ignored for other methods. This allows the user to specify which
        regions to include in the fit. Removing strong peaks is preferable to
        avoid skewing the fit up or down.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
    Returns
    –––––––
    continuum_fit : np.ndarray
        Continuum flux values evaluated at `spectrum1d.spectral_axis`.
    Notes
    –––––
    - Warnings during the fitting process are suppressed.
    '''
    # if input spectrum is ExtractedSpectrum object
    # extract the spectrum1d attribute
    if isinstance(spectrum1d, ExtractedSpectrum):
        spectrum1d = spectrum1d.spectrum1d
    # extract spectral axis
    spectral_axis = spectrum1d.spectral_axis
    # suppress warnings during continuum fitting
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # fit continuum with selected method
        if fit_method=='fit_continuum':
            # convert region to default units
            if region is not None:
                # extract unit
                unit = spectral_axis.unit
                # convert each element to spectral axis units
                region = [(rmin.to(unit), rmax.to(unit)) for rmin, rmax in region]
            fit = fit_continuum(spectrum1d, window=region)
        else:
            fit = fit_generic_continuum(spectrum1d)
    # fit the continuum of the provided spectral axis
    continuum_fit = fit(spectral_axis)

    return continuum_fit


def _return_stylename(style):
    '''
    Returns the path to a visualastro mpl stylesheet for
    consistent plotting parameters. Matplotlib styles are
    also available (ex: 'classic').

    To add custom user defined mpl sheets, add files in:
    VisualAstro/visualastro/stylelib/
    Ensure the stylesheet follows the naming convention:
        mystylesheet.mplstyle
    Parameters
    ––––––––––
    style : str
        Name of the mpl stylesheet without the extension.
        ex: 'astro'
    Returns
    –––––––
    style_path : str
        Path to matplotlib stylesheet.
    Notes
    –––––
    This is the helper function variant of return_stylename
    used for visual_classes.
    '''
    # if style is a default matplotlib stylesheet
    if style in mplstyle.available:
        return style
    # if style is a visualastro stylesheet
    else:
        style = style + '.mplstyle'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        style_path = os.path.join(dir_path, 'stylelib', style)
        return style_path

def _view(data, vmin=_default_flag, vmax=_default_flag,
          norm=_default_flag, percentile=_default_flag,
          cmap=None, style=None, figsize=None, clabel=None):
    '''
    Display a 2D data array using matplotlib's imshow with configurable
    normalization, percentile clipping, and style context.
    Parameters
    ––––––––––
    data : array-like
        2D array of image data to display.
    vmin : float or None, default=`_default_flag`
        Lower bound of the display range. If `_default_flag`, uses
        `va_config.vmin`. If both `vmin` and `percentile` are specified,
        `vmin` takes precedence.
    vmax : float or None, default=`_default_flag`
        Upper bound of the display range. If `_default_flag`, uses
        `va_config.vmax`. If both `vmax` and `percentile` are specified,
        `vmax` takes precedence.
    norm : str or matplotlib.colors.Normalize, default=`_default_flag`
        Normalization to apply to the image (e.g., 'linear', 'log',
        'asinh'). If `_default_flag`, uses `va_config.norm`.
    percentile : tuple of two floats or None, default=`_default_flag`
        Percentile range `(low, high)` used to set `vmin` and `vmax`
        automatically. If `_default_flag`, uses `va_config.percentile`.
    cmap : str or matplotlib colormap, default=None
        Colormap to use. If `None`, uses `va_config.cmap`.
    style : str or None, default=None
        Matplotlib style to apply (e.g., 'astro', 'dark_background').
        If `None`, uses `va_config.style`.
    figsize : array-like of two floats or None, default=None
        Figure size `(width, height)` in inches. If `None`, uses
        `va_config.figsize`.
    clabel : str or None, default=None
        Colorbar label.
    '''
    # get default va_config values
    vmin = va_config.vmin if vmin is _default_flag else vmin
    vmax = va_config.vmax if vmax is _default_flag else vmax
    norm = va_config.norm if norm is _default_flag else norm
    percentile = va_config.percentile if percentile is _default_flag else percentile
    cmap = get_config_value(cmap, 'cmap')
    style = get_config_value(style, 'style')
    figsize = get_config_value(figsize, 'figsize')
    cbar_width = va_config.cbar_width
    cbar_pad = va_config.cbar_pad

    vmin, vmax = _set_vmin_vmax(data, percentile, vmin, vmax)
    img_norm = _return_imshow_norm(vmin, vmax, norm)

    style = _return_stylename(style)
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(data, norm=img_norm, cmap=cmap)

        # add colorbar axes
        cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                            cbar_width, ax.get_position().height])
        # add colorbar
        cbar = fig.colorbar(im, cax=cax, pad=0.04)
        # formatting and label
        cbar.ax.tick_params(which=va_config.cbar_tick_which, direction=va_config.cbar_tick_dir)
        if clabel is not None:
            cbar.set_label(fr'{clabel}')

        plt.show()

def _return_imshow_norm(vmin, vmax, norm, **kwargs):
    '''
    Return a matplotlib or astropy normalization object for image display.
    Parameters
    ––––––––––
    vmin : float or None
        Minimum value for normalization.
    vmax : float or None
        Maximum value for normalization.
    norm : str or None
        Normalization algorithm for colormap scaling.
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'linear' -> no normalization applied
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'none' -> no normalization applied

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `linear_width` : float, optional, default=`va_config.linear_width`
            The effective width of the linear region, beyond
            which the transformation becomes asymptotically logarithmic.
            Only used in 'asinhnorm'.
        - `gamma` : float, optional, default=`va_config.gamma`
            Power law exponent.
    Returns
    –––––––
    norm_obj : None or matplotlib.colors.Normalize or astropy.visualization.ImageNormalize
        Normalization object to pass to `imshow`. None if `norm` is 'none'.
    '''
    linear_width = kwargs.get('linear_width', va_config.linear_width)
    gamma = kwargs.get('gamma', va_config.gamma)

    # use linear stretch if plotting boolean array
    if vmin==0 and vmax==1:
        return None

    # ensure norm is a string
    norm = 'none' if norm is None else norm
    # ensure case insensitivity
    norm = norm.lower()
    # dict containing possible stretch algorithms
    norm_map = {
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()), # type: ignore
        'asinhnorm': AsinhNorm(vmin=vmin, vmax=vmax, linear_width=linear_width),
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'powernorm': PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
        'linear': None,
        'none': None
    }
    if norm not in norm_map:
        raise ValueError(
            f'ERROR: unsupported norm: {norm}. '
            f'\nsupported norms are {list(norm_map.keys())}'
        )

    return norm_map[norm]

def _set_vmin_vmax(data, percentile=_default_flag, vmin=None, vmax=None):
    '''
    Compute vmin and vmax for image display. By default uses the
    data nanpercentile using `percentile`, but optionally vmin and/or
    vmax can be set by the user. Setting percentile to None results in
    no stretch. Passing in a boolean array uses vmin=0, vmax=1. This
    function is used internally by plotting functions.
    Parameters
    ––––––––––
    data : array-like
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
    percentile : list or tuple of two floats, or None, default=`_default_flag`
        Percentile range '[pmin, pmax]' to compute vmin and vmax.
        If None, sets vmin and vmax to None. If `_default_flag`, uses
        default value from `va_config.percentile`.
    vmin : float or None, default=None
        If provided, overrides the computed vmin.
    vmax : float or None, default=None
        If provided, overrides the computed vmax.
    Returns
    –––––––
    vmin : float or None
        Minimum value for image scaling.
    vmax : float or None
        Maximum value for image scaling.
    '''
    percentile = va_config.percentile if percentile is _default_flag else percentile
    # check if data is an array
    data = _check_is_array(data)
    # check if data is boolean
    if data.dtype == bool:
        return 0, 1

    # by default use percentile range
    if percentile is not None:
        if vmin is None:
            vmin = np.nanpercentile(data, percentile[0])
        if vmax is None:
            vmax = np.nanpercentile(data, percentile[1])
    # if vmin or vmax is provided overide and use those instead
    elif vmin is None and vmax is None:
        vmin = None
        vmax = None

    return vmin, vmax

def _check_is_array(data, keep_units=False):
    '''
    Ensure array input is np.ndarray.
    Parameters
    ––––––––––
    data : np.ndarray, DataCube, FitsFile, or Quantity
        Array or DataCube object.
    keep_inits : bool, optional, default=False
        If True, keep astropy units attached if present.
    Returns
    –––––––
    data : np.ndarray
        Array or 'data' component of DataCube.
    '''
    if isinstance(data, DataCube):
        data = data.value
    elif isinstance(data, FitsFile):
        data = data.data
    if isinstance(data, Quantity):
        if keep_units:
            return data
        else:
            data = data.value

    return np.asarray(data)

def _slice_cube(cube, idx):
    '''
    Return a slice of a data cube along the first axis.
    Parameters
    ––––––––––
    cube : np.ndarray
        Input data cube, typically with shape (T, N, ...) where T is the first axis.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    Returns
    –––––––
    cube : np.ndarray
        Sliced cube with shape (N, ...).
    '''
    if isinstance(cube, DataCube):
        cube = cube.data
    elif isinstance(cube, FitsFile):
        cube = cube.data

    # if index is integer
    if isinstance(idx, int):
        return cube[idx]
    # if index is list of integers
    elif isinstance(idx, list):
        # list of len 1
        if len(idx) == 1:
            return cube[idx[0]]
        # list of len 2
        elif len(idx) == 2:
            start, end = idx
            return cube[start:end+1].sum(axis=0)

    raise ValueError("'idx' must be an int or a list of one or two integers")


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
    - Unlike `numpy.allclose`, this function does **not** attempt unit
        conversion. Quantities must already share identical units.
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
