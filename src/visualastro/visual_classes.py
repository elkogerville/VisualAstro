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
from astropy.io.fits import Header
from astropy.units import Quantity, Unit
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from matplotlib import cm
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from spectral_cube import SpectralCube
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
    headers : fits.Header or array-like of fits.Header, optional, default=None
        Header(s) associated with the data cube. If provided as a list or array,
        its length must match the cube’s first dimension.
    errors : np.ndarray, optional, default=None
        Array of uncertainties with the same shape as `data`.

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

    Raises
    ––––––
    TypeError
        If `data` or `headers` are not of an expected type.
    ValueError
        If `data` is not 3D, or if the dimensions of `headers` or `errors` do not match `data`.
    Examples
    ––––––––
    Load DataCube from fits file
    >>> cube = load_fits(path)
    >>> cube.header
    >>> cube.inspect()
    '''
    def __init__(self, data, headers=None, errors=None):
        # type checks
        if not isinstance(data, (np.ndarray, SpectralCube)):
            raise TypeError(
                f"'data' must be a numpy array or SpectralCube, got {type(data).__name__}."
            )
        if headers is not None and not isinstance(
            headers, (list, Header, np.ndarray, tuple)
        ):
            raise TypeError(
                f"'headers' must be a fits.Header or array-like of fits.header, got {type(headers).__name__}."
            )

        # extract array view for validation
        if isinstance(data, SpectralCube):
            array = data.unmasked_data[:].value
            unit = data.unit
        elif isinstance(data, Quantity):
            array = data.value
            unit = data.unit
        else:
            array = data
            unit = None

        # shape validation
        if array.ndim != 3:
            raise ValueError(f"'data' must be 3D (T, N, M), got shape {array.shape}.")

        if isinstance(data, np.ndarray) and isinstance(headers, (list, np.ndarray)):
            if array.shape[0] != len(headers):
                raise ValueError(
                    f"Mismatch between T dimension and number of headers: "
                    f"T={array.shape}, headers={len(headers)}."
                )

        if errors is not None and errors.shape != array.shape:
            raise ValueError(
                f"'errors' must match shape of 'data', got {errors.shape} vs {array.shape}."
            )

        if isinstance(headers, Header):
            header = headers
        elif isinstance(headers, (list, tuple, np.ndarray)) and len(headers) > 0:
            header = headers[0]
        else:
            header = None

        # try extracting unit from headers if not found earlier
        if unit is None and header is not None and 'BUNIT' in header:
            try:
                unit = Unit(header['BUNIT'])
            except Exception:
                pass

        # try extracting WCS
        wcs = None
        if isinstance(headers, Header):
            try:
                wcs = WCS(headers)
            except Exception:
                pass
        elif isinstance(headers, (list, np.ndarray, tuple)):
            wcs = []
            for h in headers:
                if not isinstance(h, Header):
                    wcs.append(None)
                    continue
                try:
                    wcs.append(WCS(h))
                except Exception:
                    wcs.append(None)

        # assign core attributes
        self.data = data
        self.header = headers
        self.error = errors
        self.value = array
        self.unit = unit
        self.wcs = wcs

        # data attributes
        self.shape = array.shape
        self.size = array.size
        self.ndim = array.ndim
        self.dtype = array.dtype
        self.len = len(array)
        self.has_nan = np.isnan(array).any()
        self.itemsize = array.itemsize
        self.nbytes = array.nbytes

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
        self.wavelength = wavelength
        self.flux = flux
        self.spectrum1d = spectrum1d
        self.normalized = normalized
        self.continuum_fit = continuum_fit

        # assign attributes
        unit = None
        for candidate in (getattr(flux, "unit", None),
                          getattr(spectrum1d, "unit", None)):
            if candidate is not None:
                unit = candidate
                break

        wave_unit = None
        for candidate in (getattr(wavelength, "unit", None),
                          getattr(getattr(spectrum1d, "spectral_axis", None), "unit", None)):
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
    def __init__(self, data, header=None, error=None):
        data = np.asarray(data)
        unit = None
        if isinstance(data, Quantity):
            unit = data.unit
        elif isinstance(header, Header) and 'BUNIT' in header:
            try:
                unit = Unit(header['BUNIT'])
            except:
                pass

         # try extracting WCS
        wcs = None
        if isinstance(header, Header):
            try:
                wcs = WCS(header)
            except Exception:
                pass

        self.data = data
        self.header = header
        self.error = error
        self.value = data
        self.unit = unit
        self.wcs = wcs

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
