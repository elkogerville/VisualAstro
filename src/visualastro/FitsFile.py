'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-12-10
Description:
    FitsFile data structure for 2D Fits data.
Dependencies:
    - astropy
    - numpy
'''

import copy
from astropy.io.fits import Header
from astropy.units import Quantity, Unit, UnitsError
from astropy.wcs import WCS
import numpy as np
from .fits_utils import (
    _copy_headers, _get_history,
    _log_history, _transfer_history,
    _update_header_key
)
from .units import _check_unit_equality, _validate_units_consistency
from .validation import _check_shapes_match, _validate_type
from .wcs_utils import get_wcs, _is_valid_wcs_slice


class FitsFile:
    '''
    Lightweight container for FITS image data and metadata.

    Parameters
    ----------
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
    ----------
    data : ndarray or Quantity
        The Numpy or Quantity array of data.
    header : `~astropy.io.fits.Header` or None
        The FITS header provided at initialization.
    error : ndarray, Quantity, or None
        Associated error data, if provided.
    wcs : astropy.wcs.WCS or None
        The WCS transformation extracted from the header, if valid.

    Properties
    ----------
    value : np.ndarray
        Raw numpy array of the data values.
    quantity : Quantity
        Quantity array of data values (values + astropy units).
    unit : astropy.units.Unit or None
        The physical unit of the data. Inferred from the input
        `Quantity` or from the FITS header keyword `BUNIT`.
    min : float
        Minimum value in the data, ignoring NaNs.
    max : float
        Maximum value in the data, ignoring NaNs.
    mean : float
        Mean of all values in the data, ignoring NaNs.
    median : float
        Median of all values in the data, ignoring NaNs.
    sum : float
        Sum of all values in the data, ignoring NaNs.
    std : float
        Standard deviation of all values in the data, ignoring NaNs.
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
    log: list of str
        List of each log output in primary_header['HISTORY']

    Methods
    -------
    header_get(key)
        Retrieve a value from the fits Header.
    to(unit, equivalencies=None)
        Convert the data unit. This method works for
        `Quantities`, and returns a new FitsFile.
    update(data=None, header=None, error=None, wcs=None)
        Update any of the FitsFile attributes in place. All internally
        stored values are recomputed. If data has units and header has
        BUNIT, BUNIT will be automatically updated to match the data units.

    Array Interface
    ---------------
    __array__
        Return the underlying data as a Numpy array.
    __get_item__
        Return a slice of the data.
    __len__()
        Return the length of the first axis.
    reshape(*shape)
        Return a reshaped view of the data.

    Raises
    ------
    TypeError
        - If `data` or `header` are not of an expected type.
    UnitsError
        - If `BUNIT` in `header` does not match the unit of `data`.
        - If `error` units do not match `data` units.
    ValueError
        - If `error` shape does not match `data` shape.
    '''

    def __init__(self, data, header=None, error=None, wcs=None):

        data = _validate_type(
            data, (np.ndarray, Quantity), allow_none=False, name='data'
        )
        assert data is not None
        header = _validate_type(
            header, Header, default=Header(), allow_none=True, name='header'
        )
        error = _validate_type(
            error, (np.ndarray, Quantity), allow_none=True, name='error'
        )

        array = np.asarray(data)
        # extract unit
        unit = data.unit if isinstance(data, Quantity) else None

        # extract BUNIT from header(s)
        hdr_unit = _validate_units_consistency(header)

        # check that data and header units are equal
        _check_unit_equality(unit, hdr_unit, 'data', 'header')

        # use BUNIT if unit is None
        if unit is None and hdr_unit is not None:
            unit = hdr_unit
            _log_history(
                header, f'Using header BUNIT: {hdr_unit}'
            )

        # add BUNIT to header if missing
        if unit is not None and 'BUNIT' not in header:
            header['BUNIT'] = unit.to_string() # type: ignore
            _log_history(
                header, f'Added missing BUNIT={unit} to header'
            )

        # attatch units to data if bare numpy array
        if not isinstance(data, Quantity) and unit is not None:
            data = array * unit
            _log_history(
                header, f'Attached unit to data: unit={unit}'
            )

        # error validation
        if error is not None:
            _check_shapes_match(array, error, 'data', 'error')

            if isinstance(error, Quantity) and unit is not None:
                _check_unit_equality(error.unit, unit, 'error unit', 'data unit')

        # try extracting WCS
        if wcs is None:
            wcs = get_wcs(header)
            assert isinstance(wcs, WCS)

        self.data = data
        self.primary_header = header
        self.header = header
        self.error = error
        self.wcs = wcs
        self.footprint = None

    # Properties
    # ----------
    @property
    def value(self):
        '''
        Returns
        -------
        np.ndarray : View of the underlying numpy array.
        '''
        return np.asarray(self.data)
    @property
    def quantity(self):
        '''
        Returns
        -------
        Quantity : Quantity array of data values (values + astropy units).
        '''
        if self.unit is None:
            return None
        return self.unit * self.value
    @property
    def unit(self):
        '''
        Returns
        -------
        Unit : Astropy.Unit of the data.
        '''
        return getattr(self.data, 'unit', None)

    # statistical properties
    @property
    def min(self):
        '''
        Returns
        -------
        float : Minimum value in the data, ignoring NaNs.
        '''
        return np.nanmin(self.data)
    @property
    def max(self):
        '''
        Returns
        -------
        float : Maximum value in the data, ignoring NaNs.
        '''
        return np.nanmax(self.data)
    @property
    def mean(self):
        '''
        Returns
        -------
        float : Mean of all values in the data, ignoring NaNs.
        '''
        return np.nanmean(self.data)
    @property
    def median(self):
        '''
        Returns
        -------
        float : Median of all values in the data, ignoring NaNs.
        '''
        return np.nanmedian(self.data)
    @property
    def sum(self):
        '''
        Returns
        -------
        float : sum of all values in the data, ignoring NaNs.
        '''
        return np.nansum(self.data)
    @property
    def std(self):
        '''
        Returns
        -------
        float : Standard deviation of all values in the data, ignoring NaNs.
        '''
        return np.nanstd(self.data)

    # array properties
    @property
    def shape(self):
        '''
        Returns
        -------
        tuple : Shape of data.
        '''
        return self.value.shape
    @property
    def size(self):
        '''
        Returns
        -------
        int : Size of data.
        '''
        return self.value.size
    @property
    def ndim(self):
        '''
        Returns
        -------
        int : Number of dimensions of data.
        '''
        return self.value.ndim
    @property
    def dtype(self):
        '''
        Returns
        -------
        np.dtype : Datatype of the data.
        '''
        return self.value.dtype
    @property
    def has_nan(self):
        '''
        Returns
        -------
        bool : Returns True if there are NaNs.
        '''
        return np.isnan(self.value).any()
    @property
    def itemsize(self):
        '''
        Returns
        -------
        int : Length of 1 array element in bytes.
        '''
        return self.value.itemsize
    @property
    def nbytes(self):
        '''
        Returns
        -------
        int : Total number of bytes used by the data array.
        '''
        return self.value.nbytes
    @property
    def log(self):
        '''
        Get the processing history from the FITS HISTORY cards.

        Returns
        -------
        list of str or None
            List of HISTORY entries, or None if no header exists.
        '''
        return _get_history(self.primary_header)

    # Methods
    # -------
    def header_get(self, key):
        '''
        Retrieve a header value by key from a header.

        Parameters
        ----------
        key : str
            FITS header keyword to retrieve.

        Returns
        -------
        value : list or str
            Header values corresponding to `key`.

        Raises
        ------
        ValueError
            If headers are of an unsupported type or `key` is not found.
        '''
        if isinstance(self.header, Header):
            return self.header.get(key, None)
        else:
            raise ValueError(f"Unsupported header type or key '{key}' not found.")

    def to(self, unit, equivalencies=None):
        '''
        Convert the FitsFile data to a new physical unit.
        The `data` attribute must be a Quantity object.
        This returns a new FitsFile object.

        Parameters
        ----------
        unit : str or astropy.units.Unit
            Target unit.
        equivalencies : list, optional
            Astropy equivalencies for unit conversion (e.g. spectral).

        Returns
        -------
        FitsFile
            New cube with converted units.
        '''
        # convert unit to astropy unit
        unit = Unit(unit)

        _validate_type(
            self.data, Quantity, allow_none=False, name='data'
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
                    'FitsFile.error must be a Quantity to convert units safely.'
                )
        else:
            new_error = None

        # update header BUNIT
        new_hdr = _copy_headers(self.header)
        new_hdr = _update_header_key('BUNIT', unit, new_hdr)

        # update wcs
        new_wcs = None if self.wcs is None else copy.deepcopy(self.wcs)

        return FitsFile(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_wcs
        )

    # Array Interface
    # ---------------
    def __array__(self):
        '''
        Return the underlying data as a NumPy array.

        Returns
        -------
        np.ndarray
            The underlying 2D array representation.
        '''
        return self.value

    def __getitem__(self, key):
        '''
        Return a slice of the data.

        The method attempts to slice the WCS. The Header
        keywords affected by the slicing is then updated
        using the new WCS.

        Parameters
        ----------
        key : slice or tuple
            Index or slice to apply to the data.

        Returns
        -------
        slice : FitsFile
            The corresponding subset of the data.
        '''
        new_data = self.data[key]
        new_error = self.error[key] if self.error is not None else None
        new_hdr = _copy_headers(self.primary_header)
        new_wcs = None

        _log_history(new_hdr, f'Sliced data with key : {key}')

        if not _is_valid_wcs_slice(key):
            # create new header if invalid slice
            new_hdr = _transfer_history(new_hdr, Header())
            _log_history(new_hdr, f'Header and WCS dropped due to invalid slice')

        else:
            if self.wcs is not None:
                try:
                    new_wcs = self.wcs[key]
                    wcs_hdr = new_wcs.to_header()
                    for wcs_key in wcs_hdr:
                        new_hdr[wcs_key] = wcs_hdr[wcs_key]

                except (AttributeError, TypeError, ValueError) as e:
                    new_wcs = None
                    new_hdr = _transfer_history(new_hdr, Header())
                    _log_history(new_hdr, f'Header and WCS dropped due to {type(e).__name__}')

        return FitsFile(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_wcs
        )

    def __len__(self):
        '''
        Return the number of spectral slices along the first axis.
        Returns
        -------
        int
            Length of the first dimension.
        '''
        return len(self.data)

    def reshape(self, *shape):
        '''
        Return a reshaped view of the data.
        Parameters
        ----------
        *shape : int
            New shape for the data array.
        Returns
        -------
        np.ndarray
            Reshaped data array.
        '''
        return self.data.reshape(*shape)

    def __repr__(self):
        '''
        Returns
        -------
        str : String representation of `FitsFile`.
        '''
        datatype = 'np.ndarray' if self.unit is None else 'Quantity'

        return (
            f'<FitsFile[{datatype}]: unit={self.unit}, '
            f'shape={self.shape}, dtype={self.dtype}>'
        )
