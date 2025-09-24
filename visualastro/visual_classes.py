from astropy.io import fits
import numpy as np
from spectral_cube import SpectralCube

class DataCube:
    def __init__(self, data, headers=None, errors=None):
        # type checks
        if not isinstance(data, (np.ndarray, SpectralCube)):
            raise TypeError(
                f"'data' must be a numpy array or SpectralCube, got {type(data).__name__}."
            )
        if headers is not None and not isinstance(headers, (list, np.ndarray, fits.Header)):
            raise TypeError(
                f"'headers' must be a list, array or fits.Header, got {type(headers).__name__}."
            )

        # extract array view for validation
        if isinstance(data, SpectralCube):
            array = data.unmasked_data[:].value
        else:
            array = data

        if array.ndim != 3:
            raise ValueError(
                f"'data' must be 3D (T, N, M), got shape {array.shape}."
            )

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

        # assign
        self.data = data
        self.header = headers
        self.error = errors
        self.value = array

class ExtractedSpectrum:
    def __init__(self, wavelength=None, flux=None,
                 spectrum1d=None, normalized=None,
                 continuum_fit=None):

        self.wavelength = wavelength
        self.flux = flux
        self.spectrum1d = spectrum1d
        self.normalized = normalized
        self.continuum_fit = continuum_fit

class FitsFile:
    def __init__(self, data, header=None):

        self.data = data
        self.header = header
