from astropy.io import fits
from astropy.io.fits import Header
from astropy.units import Quantity, Unit
from dask.array import isin
import numpy as np
from spectral_cube import SpectralCube


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
