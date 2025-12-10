'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-12-05
Description:
    FitsFile data structure for 2D Fits data.
Dependencies:
    - astropy
    - numpy
Module Structure:
    - FitsFile
        Lightweight data class for fits files.
'''

import copy
from astropy.io.fits import Header
from astropy.time import Time
from astropy.units import Quantity, Unit, UnitsError
from astropy.wcs import WCS
import numpy as np
from .data_class_utils import log_history, update_BUNIT

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
    data : ndarray or Quantity
        The Numpy or Quantity array of data.
    header : `~astropy.io.fits.Header` or None
        The FITS header provided at initialization.
    error : ndarray, Quantity, or None
        Associated error data, if provided.
    unit : astropy.units.Unit or None
        The physical unit of the data. Inferred from the input
        `Quantity` or from the FITS header keyword ``BUNIT``.
    wcs : astropy.wcs.WCS or None
        The WCS transformation extracted from the header, if valid.

    Properties
    ––––––––––
    value : np.ndarray
        Raw numpy array of the data values.
    quantity : Quantity
        Quantity array of data values (values + astropy units).
    min : float
        Minimum value in the data, ignoring NaNs.
    max : float
        Maximum value in the data, ignoring NaNs.
    mean : float
        Mean of all values in the data, ignoring NaNs.
    median : float
        Median of all values in the data, ignoring NaNs.
    sum : float
        Sum of all values in the data, ignoreing NaNs.
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
    –––––––
    header_get(key)
        Retrieve a value from the fits Header.
    update(data=None, header=None, error=None, wcs=None)
        Update any of the parameters of ExtractedSpectrum.
        Values not provided are preserved from the existing
        instance. Recomputes any stored values.

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
    '''
    def __init__(self, data, header=None, error=None, wcs=None):
        self._initialize(data, header, error, wcs)

    def _initialize(self, data, header, error, wcs):
        '''
        Helper method to initialize the class.
        '''
        # type checks
        if not isinstance(data, (np.ndarray, Quantity)):
            raise TypeError(
                "'data' must be a np.ndarray or Quantity, "
                f'got {type(data).__name__}.'
            )

        array = np.asarray(data)

        # header validation
        if header is None:
            header = Header()
        elif not isinstance(header, Header):
            raise TypeError(
                "'header' must be a fits.Header, "
                f'got {type(header).__name__}.'
            )

        # extract unit
        unit = data.unit if isinstance(data, Quantity) else None

        # extract BUNIT from header
        if 'BUNIT' in header:
            try:
                hdr_unit = Unit(header['BUNIT']) # type: ignore
            except ValueError:
                hdr_unit = None
        else:
            hdr_unit = None

        # check that both units are equal
        if unit is not None and hdr_unit is not None:
            if unit != hdr_unit:
                raise UnitsError(
                    'Unit in header does not match data unit! '
                    f'Data unit: {unit}, Header unit: {hdr_unit}'
                )

        # use BUNIT if unit is None
        if unit is None and hdr_unit is not None:
            unit = hdr_unit
            log_history(
                header, f'Using header BUNIT: {hdr_unit}'
            )

        # add BUNIT to header if missing
        if unit is not None and 'BUNIT' not in header:
            header['BUNIT'] = unit.to_string() # type: ignore
            log_history(
                header, f'Added missing BUNIT={unit} to header'
            )

        # attatch units to data if bare numpy array
        if not isinstance(data, Quantity) and unit is not None:
            data = array * unit
            log_history(
                header, f'Attached unit to data: unit={unit}'
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
                    raise ValueError (
                        f'Error units ({error.unit}) differ from data units ({unit})'
                    )

         # try extracting WCS
        if wcs is None and isinstance(header, Header):
            try:
                wcs = WCS(header)
            except Exception:
                pass

        self.data = data
        self.primary_header = header
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
        float : Minimum value in the data, ignoring NaNs.
        '''
        return np.nanmin(self.data)
    @property
    def max(self):
        '''
        Returns
        –––––––
        float : Maximum value in the data, ignoring NaNs.
        '''
        return np.nanmax(self.data)
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
    def sum(self):
        '''
        Returns
        –––––––
        float : sum of all values in the data, ignoring NaNs.
        '''
        return np.nansum(self.value)
    @property
    def std(self):
        '''
        Returns
        –––––––
        float : Standard deviation of all values in the data, ignoring NaNs.
        '''
        return np.nanstd(self.data)

    # array properties
    @property
    def shape(self):
        '''
        Returns
        –––––––
        tuple : Shape of data.
        '''
        return self.value.shape
    @property
    def size(self):
        '''
        Returns
        –––––––
        int : Size of data.
        '''
        return self.value.size
    @property
    def ndim(self):
        '''
        Returns
        –––––––
        int : Number of dimensions of data.
        '''
        return self.value.ndim
    @property
    def dtype(self):
        '''
        Returns
        –––––––
        np.dtype : Datatype of the data.
        '''
        return self.value.dtype
    @property
    def has_nan(self):
        '''
        Returns
        –––––––
        bool : Returns True if there are NaNs.
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
        if isinstance(self.header, Header) and 'HISTORY' in self.header:
            return list(self.header['HISTORY']) # type: ignore
        return None

    # Methods
    # –––––––
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
            return self.header.get(key, None)
        else:
            raise ValueError(f"Unsupported header type or key '{key}' not found.")

    def to(self, unit, equivalencies=None):
        '''
        Convert the FitsFile data to a new physical unit.
        The `data` attribute must be a Quantity object.
        This returns a new FitsFile object.

        Parameters
        ––––––––––
        unit : str or astropy.units.Unit
            Target unit.
        equivalencies : list, optional
            Astropy equivalencies for unit conversion (e.g. spectral).

        Returns
        –––––––
        FitsFile
            New cube with converted units.
        '''
        # convert unit to astropy unit
        unit = Unit(unit)

        # check that data has a unit
        if not isinstance(self.data, Quantity):
            raise TypeError(
                'FitsFile.data has no unit. Cannot use .to() '
                'unless data is an astropy Quantity.'
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
                    'FitsFile.error must be a Quantity to convert units safely.'
                )
        else:
            new_error = None

        # update header BUNIT and transfer over pre-existing logs
        new_hdr = update_BUNIT(unit, self.header, self.primary_header)
        # update wcs
        new_wcs = None if self.wcs is None else copy.deepcopy(self.wcs)

        return FitsFile(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_wcs
        )

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

    # Array Interface
    # –––––––––––––––
    def __array__(self):
        '''
        Return the underlying data as a NumPy array.
        Returns
        –––––––
        np.ndarray
            The underlying 2D array representation.
        '''
        return self.data

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

    def __len__(self):
        '''
        Return the number of spectral slices along the first axis.
        Returns
        –––––––
        int
            Length of the first dimension.
        '''
        return len(self.data)

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

    def __repr__(self):
        '''
        Returns
        –––––––
        str : String representation of `FitsFile`.
        '''
        return (
            f'<FitsFile: unit={self.unit}, shape={self.shape}, dtype={self.dtype}>'
        )
