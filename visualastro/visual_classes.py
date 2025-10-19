import os
from astropy.io import fits
from astropy.io.fits import Header
from astropy.units import Quantity, Unit
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from spectral_cube import SpectralCube
from specutils.spectra import Spectrum1D


class DataCube:
    def __init__(self, data, headers=None, errors=None):
        # type checks
        if not isinstance(data, (np.ndarray, SpectralCube)):
            raise TypeError(
                f"'data' must be a numpy array or SpectralCube, got {type(data).__name__}."
            )
        if headers is not None and not isinstance(
            headers, (list, np.ndarray, fits.Header)
        ):
            raise TypeError(
                f"'headers' must be a list, array or fits.Header, got {type(headers).__name__}."
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
        return self.data[key]
    # support reshaping
    def reshape(self, *shape):
            return self.value.reshape(*shape)
    # support len()
    def __len__(self):
        return len(self.value)
    # support numpy operations
    def __array__(self):
        return self.value

    def header_get(self, key):
        if isinstance(self.header, (list, tuple)):
            return [h[key] for h in self.header]
        elif isinstance(self.header, Header):
            return self.header[key]
        else:
            raise ValueError(f"Unsupported header type or key '{key}' not found.")

    def with_mask(self, mask):
        if isinstance(self.data, SpectralCube):
            return self.data.with_mask(mask)
        elif isinstance(self.data, (np.ndarray, Quantity)):
            return self.data[mask]
        else:
            raise TypeError(f'Cannot apply mask to data of type {type(self.data)}')

    def inspect(self, figsize=(8,4), style='astro'):
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
        return np.nanmax(self.value)
    @property
    def min(self):
        return np.nanmin(self.value)
    @property
    def mean(self):
        return np.nanmean(self.value)
    @property
    def median(self):
        return np.nanmedian(self.value)
    @property
    def std(self):
        return np.nanstd(self.value)


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
