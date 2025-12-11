'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-12-08
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

import copy
import warnings
from astropy.io.fits import Header
from astropy.units import Quantity, Unit, UnitsError
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
from .fits_utils import (
update_header_key, with_updated_header_key,
_get_history, _log_history, _transfer_history
)
from .units import _check_unit_equality, get_common_units
from .va_config import get_config_value, _default_flag
from .validation import _validate_type
from .wcs_utils import get_wcs, reproject_wcs


class DataCube:
    '''
    Lightweight wrapper for handling 3D spectral_cubes
    or arrays with optional headers and error arrays.
    This class supports both `numpy.ndarray` and `SpectralCube`
    inputs and provides convenient access to cube statistics,
    metadata, and visualization methods.

    Parameters
    ––––––––––
    data : np.ndarray, Quantity, or SpectralCube
        The input data cube. Must be 3-dimensional (T, N, M).
    header : fits.Header, array-like of fits.Header, or None, optional, default=None
        Header(s) associated with the data cube. If provided as a list or array,
        its length must match the cube’s first dimension.
    error : np.ndarray, Quantity, or None, optional, default=None
        Array of uncertainties with the same shape as `data`.
    wcs : astropy.wcs.wcs.WCS, array-like of astropy.wcs.wcs.WCS, or None, optional, default=None
        WCS information associated with the data extension.
        Can also be an array-like of WCS objects. If None,
        DataCube will attempt to extract the WCS from the header
        attribute. If `header` is an array-like, DataCube will
        extract the WCS from each header.

    Attributes
    ––––––––––
    data : np.ndarray, Quantity, or SpectralCube
        Original data object.
    primary_header : fits.Header
        Primary header for the DataCube. Cube-level operations
        (e.g., unit conversions) add `HISTORY` entries here. For
        time series data with multiple headers, this references
        `header[0]`. If no header, is a blank header for logging
        purposes. Otherwise points to `header`.
    header : array-like of fits.Header or fits.Header
        Header(s) associated with the data cube.
        If no header(s) are passed in, an empty
        `Header` is created for log purposes.
    error : np.ndarray, Quantity, or None
        Error array if provided, else None.
    unit : astropy.units.Unit or None
        Physical unit of the data if available.
    wcs : array-like of WCS or WCS
        WCS(s) associated with the data cube.

    Properties
    ––––––––––
    value : np.ndarray
        Raw numpy array of the cube values.
    quantity : Quantity
        Quantity array of data values (values + astropy units).
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
    shape : tuple
        Shape of the cube (T, N, M).
    size : int
        Total number of elements in the cube.
    ndim : int
        Number of dimensions.
    dtype : np.dtype
        Data type of the array.
    has_nan : bool
        True if any element in the cube is NaN.
    itemsize : int
        Size of one array element in bytes.
    nbytes : int
        Total memory footprint of the data in bytes.
    log : list of str
        List of each log output in primary_header['HISTORY']

    Methods
    –––––––
    header_get(key)
        Retrieve a header value by key from one or multiple headers.
        If a Header is missing a key, None is returned.
    inspect(figsize=(10,6), style=None)
        Plot the mean and standard deviation across each cube slice.
        Useful for quickly identifying slices of interest in the cube.
    to(unit, equivalencies=None)
        Convert the cube unit (flux unit). This method works for
        `Quantities`, as well as `SpectralCube` flux units. To convert
        spectral_units for `SpectralCubes` use `with_spectral_unit()`.
        Returns a new cube.
    update(data=None, header=None, error=None, wcs=None)
        Update any of the DataCube attributes in place. All internally
        stored values are recomputed. If data has units and header has
        BUNIT, BUNIT will be automatically updated to match the data units.
    with_mask(mask)
        Apply a boolean mask to the cube. Works for both `Quantities`
        and `SpectralCubes`. The original shape is preserved and
        masked values are replaced with NaNs. Returns a new cube.
    with_spectral_unit(unit, velocity_convention=None, rest_value=None)
        Convert the cube spectral unit (wavelength, frequency, speed...).
        This method only works for `SpectralCube` data. Returns a new cube.

    Array Interface
    –––––––––––––––
    __array__
        Return the underlying data as a Numpy array.
    __get_item__
        Return a slice of the data.
    __len__()
        Return the length of the first axis.
    reshape(*shape)
        Return a reshaped view of the data.

    Raises
    ––––––
    TypeError
        - If `data`, `header`, or `error` are not of an expected type.
    UnitsError
        - If `BUNIT` is inconsistent across headers in a header list.
        - If `BUNIT` in `header` does not match the unit of `data`.
        - If `error` units do not match `data` units.
    ValueError
        - If `data` is not 3D with shape (T,N,M).
        - If `error` shape does not match `data` shape.
        - If the header list is empty.
        - If length of the header list does not match data T dimension.

    Examples
    ––––––––
    Load DataCube from fits file
    >>> cube = load_fits(filepath)
    >>> cube.data
    >>> cube.header
    >>> cube.inspect()
    '''

    def __init__(self, data, header=None, error=None, wcs=None):
        self._initialize(data, header, error, wcs)

    def _initialize(self, data, header, error, wcs):
        '''
        Helper method to initialize the
        class and perform type checking.
        '''
        # type checks
        data = _validate_type(
            data, (np.ndarray, Quantity, SpectralCube),
            allow_none=False, name='data'
        )
        assert data is not None
        header = _validate_type(
            header, (list, Header, np.ndarray, tuple),
            default=Header(), allow_none=True, name='header'
        )
        error = _validate_type(
            error, (np.ndarray, Quantity), default=None,
            allow_none=True, name='error'
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

        if isinstance(header, (list, np.ndarray, tuple)):
            if len(header) == 0:
                raise ValueError(
                    'Header list cannot be empty.'
                )
            if array.shape[0] != len(header):
                raise ValueError(
                    f'Mismatch between T dimension and number of headers: '
                    f'T={array.shape[0]}, header={len(header)}.'
                )
            primary_hdr = header[0]
        else:
            primary_hdr = header

        # ensure that units are consistent across all headers
        hdr_unit = get_common_units(header)

        # check that both units are equal
        _check_unit_equality(unit, hdr_unit, 'data', 'header')

        # use BUNIT if unit is None
        if unit is None and hdr_unit is not None:
            unit = hdr_unit
            _log_history(
                primary_hdr, f'Using header BUNIT: {hdr_unit}'
            )

        # add BUNIT to header(s) if not there
        if unit is not None and 'BUNIT' not in primary_hdr:
            _log_history(
                primary_hdr, f'Using data unit: {unit}'
            )
            if isinstance(header, Header):
                header['BUNIT'] = unit.to_string()
                _log_history(
                    primary_hdr, f'Added missing BUNIT={unit} to header'
                )

            elif isinstance(header, (list, tuple, np.ndarray)):
                for hdr in header:
                    hdr['BUNIT'] = unit.to_string()
                _log_history(
                    primary_hdr, f'Added missing BUNIT={unit} to all header slices'
                )

        # attatch units to data if is bare numpy array
        if not isinstance(data, (Quantity, SpectralCube)):
            if unit is not None:
                data = array * unit
                _log_history(
                    primary_hdr, f'Attached unit to data: unit={unit}'
                )

        # error validation
        if error is not None:
            err = np.asarray(error)
            if err.shape != array.shape:
                raise ValueError(
                    f"'error' must match shape of 'data', got {err.shape} vs {array.shape}."
                )
            if isinstance(error, Quantity) and unit is not None:
                if error.unit != unit:
                    raise UnitsError (
                        f'Error units ({error.unit}) differ from data units ({unit})'
                    )

        # try extracting WCS from headers
        if wcs is None:
            wcs = get_wcs(header)

        # assign attributes
        self.data = data
        self.primary_header = primary_hdr
        self.header = header
        self.error = error
        self.unit = unit
        self.wcs = wcs
        self.footprint = None

    # Properties
    # ––––––––––
    @property
    def value(self):
        '''
        Returns
        –––––––
        np.ndarray : View of the underlying numpy array.
        '''
        if isinstance(self.data, SpectralCube):
            return self.data.unmasked_data[:].value
        else:
            return np.asarray(self.data)
    @property
    def quantity(self):
        '''
        Returns
        –––––––
        Quantity : Quantity array of data values (values + astropy units).
        '''
        if self.unit is None:
            return None
        return self.unit * self.value

    # statistical properties
    @property
    def min(self):
        '''
        Returns
        –––––––
        Quantity or float
            Minimum value in the cube, ignoring NaNs.
        '''
        return self._stat('min')
    @property
    def max(self):
        '''
        Returns
        –––––––
        Quantity or float
            Maximum value in the cube, ignoring NaNs.
        '''
        return self._stat('max')
    @property
    def mean(self):
        '''
        Returns
        –––––––
        Quantity or float
            Mean of all values in the cube, ignoring NaNs.
        '''
        return self._stat('mean')
    @property
    def median(self):
        '''
        Returns
        –––––––
        Quantity or float
            Median of all values in the cube, ignoring NaNs.
        '''
        return self._stat('median')
    @property
    def sum(self):
        '''
        Returns
        –––––––
        Quantity or float
            Sum of all values in the cube, ignoring NaNs.
        '''
        return self._stat('sum')
    @property
    def std(self):
        '''
        Returns
        –––––––
        Quantity or float
            Standard deviation of all values in the cube, ignoring NaNs.
        '''
        return self._stat('std')

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
    @property
    def log(self):
        '''
        Get the processing history from the FITS HISTORY cards.
        Returns
        –––––––
        list of str or None
            List of HISTORY entries, or None if no header exists.
        '''
        return _get_history(self.primary_header)

    # Methods
    # –––––––
    def header_get(self, key):
        '''
        Retrieve a header value by key from one or multiple headers.
        If a Header is missing a key, None is returned.

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
        # case 1: single Header
        if isinstance(self.header, Header):
            return self.header.get(key, None)
        # case 2: Header list
        elif isinstance(self.header, (list, np.ndarray, tuple)):
            return [h.get(key, None) for h in self.header]
        else:
            raise ValueError(f'Unsupported header type.')

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
            _, ax = plt.subplots(figsize=figsize)

            ax.plot(T, mean_flux, c='darkslateblue', label='Mean')
            ax.plot(T, std_flux, c='#D81B60', ls='--', label='Std Dev')

            ax.set_xlabel('Cube Slice Index')
            ax.set_ylabel('Counts')
            ax.set_xlim(np.nanmin(T), np.nanmax(T))

            ax.legend(loc='best')

            plt.show()


    def reproject(
        self,
        reference_wcs,
        method=None,
        return_footprint=None,
        parallel=None,
        block_size=_default_flag
    ):
        '''
        Reproject DataCube to a new WCS grid.

        Parameters
        ––––––––––
        reference_wcs : WCS or Header
            Target WCS or FITS header to reproject onto
        method : {'interp', 'exact'} or None
            Reprojection method
        return_footprint : bool or None
            Whether to return footprint array
        parallel : bool, int, or None
            Parallelization settings
        block_size : tuple, 'auto', or None
            Block size for reprojection

        Returns
        –––––––
        new_cube : DataCube
            Reprojected DataCube.

        Notes
        –––––
        - As of right now, the WCS and Header are correctly
          updated for reprojected SpectralCubes only! For any
          other data type, the old WCS and Header are passed down.

        '''
        method = get_config_value(method, 'reproject_method')

        if isinstance(self.data, SpectralCube):
            try:
                new_data = self.data.reproject(reference_wcs)
                new_header = Header() if new_data.header is None else new_data.header.copy()

                if isinstance(self.primary_header, Header):
                    new_header = _transfer_history(self.primary_header, new_header)

                _log_history(
                    new_header,
                    f'Reprojected data with SpectralCube.reproject'
                )

                if self.error is not None:
                   new_error, footprint = reproject_wcs(
                       (self.error, self.wcs), reference_wcs,
                       return_footprint=True, method=method,
                       parallel=parallel, block_size=block_size
                   )
                   _log_history(
                       new_header,
                       f'Reprojected errors with reproject_{method}'
                   )
                else:
                    new_error = None
                    footprint = None

                new_cube = DataCube(
                    new_data, new_header,
                    new_error, new_data.wcs
                )
                if return_footprint:
                    new_cube.footprint = footprint

                return new_cube

            except Exception as e:
                if isinstance(self.primary_header, Header):
                    warnings.warn(
                        f'\nSpectralCube.reproject() failed ({e}), '
                        f'falling back to manual reprojection!'
                        'WARNING: SpectralCube at .data is now a Quantity!'
                    )
                pass

        if self.wcs is None and self.header is None:
            raise ValueError(
                'Cannot reproject: DataCube has neither .wcs nor .header'
            )

        wcs_info = self.wcs if self.wcs is not None else self.header

        new_data, footprint = reproject_wcs(
            (self.data, wcs_info),
            reference_wcs,
            return_footprint=True,
            method=method,
            parallel=parallel,
            block_size=block_size,
            description='looping over data slices'
        )

        new_header = Header() if self.header is None else self.header.copy()
        if isinstance(self.primary_header, Header):
            new_header = _transfer_history(self.primary_header, new_header)

        _log_history(
            new_header,
            f'Reprojected data with reproject_{method}'
        )

        if self.error is not None:
            new_error = reproject_wcs(
                (self.error, self.wcs), reference_wcs,
                return_footprint=False, method=method,
                parallel=parallel, block_size=block_size,
                description='looping over error slices'
            )
            _log_history(
                new_header,
                f'Reprojected errors with reproject_{method}'
            )
        else:
            new_error = None

        new_wcs = None if self.wcs is None else copy.deepcopy(self.wcs)

        warnings.warn(
            f'\nWCS Header correction not yet implemented for manual reprojection! '
            f'Output cube Header and WCS are still mapped to the original cube! '
        )

        new_cube = DataCube(
            new_data, new_header,
            new_error, new_wcs
        )

        if return_footprint:
            new_cube.footprint = footprint

        return new_cube

    def to(self, unit, equivalencies=None):
        '''
        Convert the DataCube data to a new physical unit.
        This method supports Quantity objects as well as
        SpectralCubes, but only for 'flux' units. To convert
        SpectralCube wavelength units use `.with_spectral_unit()`.

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
            raise UnitsError(
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

        # update header BUNIT into copy and
        # transfer over pre-existing logs
        new_hdr = with_updated_header_key(
            'BUNIT', unit, self.header, self.primary_header
        )
        # update wcs
        new_wcs = None if self.wcs is None else copy.deepcopy(self.wcs)

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_wcs
        )

    def update(self, data=None, header=None, error=None, wcs=None):
        '''
        Update any of the DataCube attributes in place. All internally
        stored values are recomputed.

        If data has units and header has BUNIT, BUNIT will be automatically
        updated to match the data units.

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
        # get existing values if not passed in
        data = self.data if data is None else data
        error = self.error if error is None else error
        wcs = self.wcs if wcs is None else wcs

        # get unit
        unit = getattr(data, 'unit', None)
        if unit is not None:
            unit = Unit(unit)

        # if user did not provide a header
        # ensure that BUNIT is updated
        if header is None:
            if unit is not None and self.header is not None:
                hdr_unit = get_common_units(self.header)
                if hdr_unit is None or hdr_unit != unit:
                    update_header_key(
                        'BUNIT', unit, self.header, self.primary_header
                    )
            header = self.header

        self._initialize(data, header, error, wcs)

        return None

    def with_mask(self, mask):
        '''
        Apply a boolean mask to the cube and return the
        masked data. The shape of the cube is preserved
        and values are masked with NaNs.

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
        # check mask shape
        mask = np.asarray(mask)
        if mask.shape != self.shape:
            raise ValueError(
                'Mask shape must match cube shape!'
            )

        # case 1: mask SpectralCube
        if isinstance(self.data, SpectralCube):
            new_data = self.data.with_mask(mask)
        # case 2: mask ndarray or Quantity
        elif isinstance(self.data, (np.ndarray, Quantity)):
            new_data = self.data.copy()
            new_data[~mask] = np.nan
        else:
            raise TypeError(
                f'Cannot apply mask to data of type {type(self.data)}'
            )

        # mask errors
        if self.error is not None:
            new_error = self.error.copy()
            new_error[~mask] = np.nan
        else:
            new_error = None

        # copy header and wcs
        if isinstance(self.header, Header):
            new_header = self.header.copy()
            # add log
            _log_history(new_header, f'Applied boolean mask to cube')
        # case 2: header is list of Headers
        elif isinstance(self.header, (list, np.ndarray, tuple)):
            new_header = [hdr.copy() for hdr in self.header]
            # add log
            _log_history(new_header[0], f'Applied boolean mask to cube')
        else:
            new_header = None

        new_wcs = None if self.wcs is None else copy.deepcopy(self.wcs)

        return DataCube(
            new_data,
            new_header,
            new_error,
            new_wcs
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
        # get unit strings
        old_unit = self.data.spectral_axis.unit

        # convert spectral axis
        try:
            new_data = self.data.with_spectral_unit(
                unit,
                velocity_convention=velocity_convention,
                rest_value=rest_value
            )
        except Exception as e:
            raise TypeError(f'Spectral axis conversion failed: {e}')

        if isinstance(self.header, Header):
            new_hdr = self.header.copy()
            _log_history(
                new_hdr, f'Converted spectral axis: {old_unit} -> {unit}'
            )
        elif isinstance(self.header, (list, np.ndarray, tuple)):
            new_hdr = [hdr.copy() for hdr in self.header]
            _log_history(
                new_hdr[0], f'Converted spectral axis: {old_unit} -> {unit}'
            )
        else:
            new_hdr = None

        new_error = None if self.error is None else self.error.copy()

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_data.wcs
        )

    # Array Interface
    # –––––––––––––––
    def __array__(self):
        '''
        Return the underlying data as a NumPy array.
        Returns
        –––––––
        np.ndarray
            The underlying 3D array representation.
        '''
        return self.value

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
        if self.unit is None:
            return self.value[key]

        return self.unit * self.value[key]

    def __len__(self):
        '''
        Return the number of spectral slices along the first axis.
        Returns
        –––––––
        int
            Length of the first dimension (T).
        '''
        return len(self.value)

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

    # statistical property helper
    def _stat(self, func):
        '''
        Compute a statistical property of the data, handling Quantity and SpectralCube
        objects.

        This method evaluates one of several statistical operations on the underlying
        data. If the data is a `SpectralCube`, the corresponding lazy method on the
        cube is used. Otherwise, the operation is applied using the appropriate
        NumPy NaN-aware function. If the object carries physical units, the
        returned value is scaled accordingly.

        Parameters
        ––––––––––
        func : {'min', 'max', 'mean', 'median', 'std'}
            The name of the statistical quantity to compute.

        Returns
        –––––––
        value : float, `astropy.units.Quantity`, or scalar-like
            The computed statistical value. If the underlying data includes
            units, the returned value is a `Quantity`; otherwise it is a unitless
            NumPy scalar.

        Raises
        ––––––
        KeyError
            If an unsupported statistic name is provided.
        '''
        _STAT_FUNCS = {
            'min': np.nanmin,
            'max': np.nanmax,
            'mean': np.nanmean,
            'median': np.nanmedian,
            'std': np.nanstd,
        }

        # evaluate SpectralCubes with lazy methods
        # ex: self.data.min()
        if isinstance(self.data, SpectralCube):
            method = getattr(self.data, func)
            return method()
        # otherwise evaluate with numpy and re-attach unit
        np_func = _STAT_FUNCS[func]
        value = np_func(self.data)
        return value if self.unit is None else self.unit * value

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
