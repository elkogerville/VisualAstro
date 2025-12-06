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

from astropy.io.fits import Header
from astropy.units import Quantity, Unit
from astropy.wcs import WCS
import numpy as np


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
        if isinstance(data, Quantity) or hasattr(data, 'unit'):
            unit = data.unit
        else:
            unit = None

        if isinstance(header, Header):
            if 'BUNIT' in header:
                BUNIT = header['BUNIT']

                try:
                    hdr_unit = Unit(BUNIT)
                except ValueError:
                    hdr_unit = None

                if unit is not None and hdr_unit is not None:
                    if unit != hdr_unit:
                        raise ValueError(
                            'Unit extracted from header does not match '
                            'unit attatched to the data!'
                            f'Data unit: {unit}, Header unit: {hdr_unit}'
                        )

                unit = hdr_unit if unit is None else unit

        # attatch units to data if bare numpy array
        if not isinstance(data, Quantity):
            data_unit = getattr(data, 'unit', None)
            if data_unit is None and unit is not None:
                data = np.asarray(data) * unit

        # validate error units
        if error is not None and hasattr(error, 'unit') and unit is not None:
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
        self.header = header
        self.error = error
        self.value = np.asarray(data)
        self.unit = unit
        self.wcs = wcs

    # Properties
    # ––––––––––
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
        float : sum of all values in the cube, ignoring NaNs.
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
