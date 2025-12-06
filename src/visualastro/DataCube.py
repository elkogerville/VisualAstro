'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-12-05
Description:
    DataCube data structure for 3D SpectralCubes or
    time series data cubes.
Dependencies:
    - astropy
    - numpy
    - spectral_cube
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
'''

import warnings
from astropy.io.fits import Header
from astropy.units import Quantity, Unit
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
from .va_config import get_config_value

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
    min : float
        Minimum value in the cube, ignoring NaNs.
    max : float
        Maximum value in the cube, ignoring NaNs.
    mean : float
        Mean of all values in the cube, ignoring NaNs.
    median : float
        Median of all values in the cube, ignoring NaNs.
    sum : float
        Sum of all values in the cube, ignoreing NaNs.
    std : float
        Standard deviation of all values in the cube, ignoring NaNs.
    quantity : Quantity
        Quantity array of data values (values + astropy units).
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

        # ensure that units are consistent
        if hasattr(data, 'unit'):
            if unit is not None and data.unit != unit:
                raise ValueError(
                    f'Unit mismatch: data={data.unit}, header={unit}'
                )
        elif unit is not None:
            data = array * unit

        # validate error units
        if error is not None and hasattr(error, 'unit') and unit is not None:
            if error.unit != unit:
                warnings.warn(
                    f'Error units ({error.unit}) differ from data units ({unit})',
                    UserWarning
                )

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
                new_error = self.error / np.abs(other)
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
                new_error = σA / np.abs(B)
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

    def update(self, data=None, header=None, error=None, wcs=None):
        '''
        Update any of the DataCube attributes. All internally stored
        values are recomputed.
        Parameters
        ––––––––––
        data : array-like or `~astropy.units.Quantity`
            The primary image data. Can be a NumPy array or an
            `astropy.units.Quantity` object.
        header : fits.Header, array-like of fits.Header, or None, optional, default=None
            Header(s) associated with the data cube. If provided as a list or array,
            its length must match the cube’s first dimension.
        error : array-like, optional
            Optional uncertainty or error map associated with the data.
        wcs : astropy.wcs.wcs.WCS or None, optional, default=None
            WCS information associated with the data extension.
            If None, DataCube will attempt to extract the WCS
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
        from .plot_utils import return_stylename

        # get default va_config values
        style = get_config_value(style, 'style')

        cube = self.value
        # compute mean and std across wavelengths
        mean_flux = np.nanmean(cube, axis=(1, 2))
        std_flux  = np.nanstd(cube, axis=(1, 2))

        T = np.arange(mean_flux.shape[0])
        style = return_stylename(style)
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
    def min(self):
        '''
        Returns
        –––––––
        float : Minimum value in the cube, ignoring NaNs.
        '''
        return np.nanmin(self.value)
    @property
    def max(self):
        '''
        Returns
        –––––––
        float : Maximum value in the cube, ignoring NaNs.
        '''
        return np.nanmax(self.value)
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
    def sum(self):
        '''
        Returns
        –––––––
        float : sum of all values in the cube, ignoring NaNs.
        '''
        return np.nansum(self.value)
    @property
    def std(self):
        '''
        Returns
        –––––––
        float : Standard deviation of all values in the cube, ignoring NaNs.
        '''
        return np.nanstd(self.value)
    @property
    def quantity(self):
        '''
        Returns
        –––––––
        Quantity : Quantity array of data values (values + astropy units).
        '''
        if self.unit is None:
            return None
        return self.value * self.unit

    # array properties
    @property
    def shape(self):
        '''
        Returns
        –––––––
        tuple : Shape of cube data.
        '''
        return self.value.shape
    @property
    def size(self):
        '''
        Returns
        –––––––
        int : Size of cube data.
        '''
        return self.value.size
    @property
    def ndim(self):
        '''
        Returns
        –––––––
        int : Number of dimensions of cube data.
        '''
        return self.value.ndim
    @property
    def dtype(self):
        '''
        Returns
        –––––––
        np.dtype : Datatype of the cube data.
        '''
        return self.value.dtype
    @property
    def has_nan(self):
        '''
        Returns
        –––––––
        bool : Returns True if there are NaNs in the cube.
        '''
        return np.isnan(self.value).any()
    @property
    def itemsize(self):
        '''
        Returns
        –––––––
        int : Length of 1 array element in bytes.
        '''
        return self.value.itemsize
    @property
    def nbytes(self):
        '''
        Returns
        –––––––
        int : Total number of bytes used by the data array.
        '''
        return self.value.nbytes

    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of DataCube.
        '''
        if isinstance(self.data, SpectralCube):
            flux_unit = self.unit
            wave_unit = self.data.spectral_axis.unit
            return (
                f'<DataCube[SpectralCube]: wavelength={wave_unit}, '
                f'flux={flux_unit}, shape={self.shape}, dtype={self.dtype}>'
            )

        datatype = 'np.ndarray' if self.unit is None else 'Quantity'

        return (
            f'<DataCube[{datatype}]: unit={self.unit}, '
            f'shape={self.shape}, dtype={self.dtype}>'
        )
