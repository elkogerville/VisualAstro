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
'''

import os
from astropy.io import fits
from astropy.io.fits import Header
from astropy.units import Quantity, Unit
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from spectral_cube import SpectralCube
from specutils.spectra import Spectrum1D
from .va_config import va_config

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
            headers, (list, fits.Header, np.ndarray, tuple)
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

        # try extracting unit from headers
        if isinstance(headers, Header) and 'BUNIT' in headers:
            try:
                unit = Unit(headers['BUNIT'])
            except Exception:
                pass
        if isinstance(headers, list) and 'BUNIT' in headers[0]:
            try:
                unit = Unit(headers[0]['BUNIT'])
            except Exception:
                pass

        # assign
        self.data = data
        self.header = headers
        self.error = errors
        self.value = array
        self.unit = unit

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

    def inspect(self, figsize=(8,4), style=va_config.style):
        '''
        Plot the mean and standard deviation across each cube slice.
        Useful for quickly identifying slices of interest in the cube.
        Parameters
        ––––––––––
        figsize : tuple, optional
            Size of the output figure. Default = (8, 4).
        style : str, optional
            Matplotlib style to use for plotting. Default = 'astro'.
        Notes
        –––––
        This method visualizes the mean and standard deviation of flux across
        each 2D slice of the cube as a function of slice index.
        '''
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

    # physical properties / statistics
    @property
    def max(self):
        '''
        Returns
        –––––––
        float : Maximum value in the cube (ignoring NaNs).
        '''
        return np.nanmax(self.value)
    @property
    def min(self):
        '''
        Returns
        –––––––
        float : Minimum value in the cube (ignoring NaNs).
        '''
        return np.nanmin(self.value)
    @property
    def mean(self):
        '''
        Returns
        –––––––
        float : Mean of all values in the cube (ignoring NaNs).
        '''
        return np.nanmean(self.value)
    @property
    def median(self):
        '''
        Returns
        –––––––
        float : Median of all values in the cube (ignoring NaNs).
        '''
        return np.nanmedian(self.value)
    @property
    def std(self):
        '''
        Returns
        –––––––
        float : Standard deviation of all values in the cube (ignoring NaNs).
        '''
        return np.nanstd(self.value)


# ExtractedSpectrum
# –––––––––––––––––
class ExtractedSpectrum:
    def __init__(self, wavelength=None, flux=None, spectrum1d=None,
                 normalized=None, continuum_fit=None):
        self.wavelength = wavelength
        self.flux = flux
        self.spectrum1d = spectrum1d
        self.normalized = normalized
        self.continuum_fit = continuum_fit

    # support slicing
    def __getitem__(self, key):
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


# FitsFile
# ––––––––
class FitsFile:
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

        self.data = data
        self.header = header
        self.error = error
        self.unit = unit

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
        return self.data[key]

    def reshape(self, *shape):
            return self.data.reshape(*shape)

    def __len__(self):
        return len(self.data)

    def __array__(self):
        return self.data

    # physical properties / statistics
    @property
    def max(self):
        return np.nanmax(self.data)
    @property
    def min(self):
        return np.nanmin(self.data)
    @property
    def mean(self):
        return np.nanmean(self.data)
    @property
    def median(self):
        return np.nanmedian(self.data)
    @property
    def std(self):
        return np.nanstd(self.data)

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
