'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-01-15
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
from astropy.units import Quantity, Unit, UnitsError
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm
from .fits_utils import (
    _copy_headers, _get_history,
    _log_history, _remove_history,
    _transfer_history, _update_header_key,
)
from .units import _check_unit_equality, _validate_units_consistency
from .va_config import get_config_value, _default_flag
from .validation import (
    _check_shapes_match,
    _validate_iterable_type,
    _validate_type
)
from .wcs_utils import (
    get_wcs,
    _is_valid_wcs_slice,
    _reproject_wcs,
    _strip_wcs_from_header,
    _update_header_from_wcs
)


class DataCube:
    '''
    Lightweight wrapper for handling 3D spectral_cubes
    or arrays with optional headers and error arrays.
    This class supports both `numpy.ndarray` and `SpectralCube`
    inputs and provides convenient access to cube statistics,
    metadata, and visualization methods.

    Parameters
    ----------
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
    ----------
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
    wcs : array-like of WCS or WCS
        WCS(s) associated with the data cube.

    Properties
    ----------
    value : np.ndarray
        Raw numpy array of the cube values.
    quantity : Quantity
        Quantity array of data values (values + astropy units).
    unit : astropy.units.Unit or None
        Physical unit of the data if available.
    min : float
        Minimum value in the cube, ignoring NaNs.
    max : float
        Maximum value in the cube, ignoring NaNs.
    mean : float
        Mean of all values in the cube, ignoring NaNs.
    median : float
        Median of all values in the cube, ignoring NaNs.
    sum : float
        Sum of all values in the cube, ignoring NaNs.
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
    -------
    header_get(key)
        Retrieve a header value by key from one or multiple headers.
        If a Header is missing a key, None is returned.
    inspect(figsize=(10,6), style=None)
        Plot the mean and standard deviation across each cube slice.
        Useful for quickly identifying slices of interest in the cube.
    reproject(
        reference_wcs, method=None, return_footprint=None,
        parallel=None, block_size=_default_flag
    )
        Reproject the data onto a new target WCS and return a new cube.
    to(unit, equivalencies=None)
        Convert the cube unit (flux unit). This method works for
        `Quantities`, as well as `SpectralCube` flux units. To convert
        spectral_units for `SpectralCubes` use `with_spectral_unit()`.
        Returns a new cube.
    with_mask(mask)
        Apply a boolean mask to the cube. Works for both `Quantities`
        and `SpectralCubes`. The original shape is preserved and
        masked values are replaced with NaNs. Returns a new cube.
    with_spectral_unit(unit, velocity_convention=None, rest_value=None)
        Convert the cube spectral unit (wavelength, frequency, speed...).
        This method only works for `SpectralCube` data. Returns a new cube.

    Array Interface
    ---------------
    __array__
        Return the underlying data as a Numpy array.
    __getitem__
        Return a slice of the data.
    __len__()
        Return the length of the first axis.
    reshape(*shape)
        Return a reshaped view of the data.

    Raises
    ------
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
    --------
    Load DataCube from fits file
    >>> cube = load_fits(filepath)
    >>> cube.data
    >>> cube.header
    >>> cube.inspect()
    '''

    def __init__(self, data, header=None, error=None, wcs=None):

        data = _validate_type(
            data, (np.ndarray, Quantity, SpectralCube),
            allow_none=False, name='data'
        )
        assert data is not None
        header = _validate_type(
            header, (list, Header, np.ndarray, tuple),
            allow_none=True, name='header'
        )
        error = _validate_type(
            error, (np.ndarray, Quantity), default=None,
            allow_none=True, name='error'
        )
        wcs = _validate_type(
            wcs, (list, WCS), default=None,
            allow_none=True, name='wcs'
        )
        if isinstance(wcs, list):
            _validate_iterable_type(wcs, WCS, 'wcs')

        if header is None:
            if isinstance(wcs, list):
                header = [Header() for _ in wcs]
            else:
                header = Header()

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

        # header(s) validation
        if isinstance(header, (list, np.ndarray, tuple)):
            header = list(header)

            if len(header) == 0:
                raise ValueError(
                    'Header list cannot be empty.'
                )
            if array.shape[0] != len(header):
                raise ValueError(
                    f'Mismatch between T dimension and number of headers: '
                    f'T={array.shape[0]}, header={len(header)}.'
                )
            _validate_iterable_type(header, Header, 'header')

            primary_hdr = header[0]
        else:
            primary_hdr = header

        _log_history(primary_hdr, 'Initialized DataCube')

        # ensure that units are consistent across all headers
        hdr_unit = _validate_units_consistency(header)

        # check that both units are equal
        _check_unit_equality(unit, hdr_unit, 'data', 'header')

        # use BUNIT if unit is None
        if unit is None and hdr_unit is not None:
            unit = hdr_unit
            _log_history(
                primary_hdr, f'Using header BUNIT: {hdr_unit}'
            )

        # add BUNIT to header(s) if not there
        if (
            unit is not None and
            isinstance(primary_hdr, Header) and
            'BUNIT' not in primary_hdr
        ):
            _log_history(
                primary_hdr, f'Using data unit: {unit}'
            )
            _update_header_key('BUNIT', unit, header, primary_hdr)

        # attatch units to data if is bare numpy array
        if not isinstance(data, (Quantity, SpectralCube)):
            if unit is not None:
                data = array * unit
                _log_history(
                    primary_hdr, f'Attached unit to data: unit={unit}'
                )

        # error validation
        if error is not None:
            _check_shapes_match(array, error, 'data', 'error')

            if isinstance(error, Quantity) and unit is not None:
                _check_unit_equality(error.unit, unit, 'error unit', 'data unit')

        # try extracting WCS from headers
        if wcs is None:
            wcs = get_wcs(header)

        # ensure header and wcs are in sync
        if wcs is not None:
            if isinstance(header, list):
                if not isinstance(wcs, list):
                    raise ValueError(
                        'If header is a list, wcs must be a list of same length!'
                    )
                if len(header) != len(wcs):
                    raise ValueError(
                        f'Header list length ({len(header)}) must match '
                        f'WCS list length ({len(wcs)})'
                    )
                for hdr, w in zip(header, wcs):
                    _update_header_from_wcs(hdr, w)
            else:
                # single header case
                if isinstance(wcs, list):
                    raise ValueError(
                        'If wcs is a list, header must also be a list'
                    )
                _update_header_from_wcs(header, wcs)

        # extract non WCS info from header
        nowcs_header = _strip_wcs_from_header(header)
        _remove_history(nowcs_header)

        # assign attributes
        self.data = data
        self.primary_header = primary_hdr
        self.header = header
        self.nowcs_header = nowcs_header
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
        if isinstance(self.data, SpectralCube):
            return self.data.filled_data[:].value
        if isinstance(self.data, Quantity):
            return self.data.value
        else:
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
        Quantity or float
            Minimum value in the cube, ignoring NaNs.
        '''
        return self._stat('min')
    @property
    def max(self):
        '''
        Returns
        -------
        Quantity or float
            Maximum value in the cube, ignoring NaNs.
        '''
        return self._stat('max')
    @property
    def mean(self):
        '''
        Returns
        -------
        Quantity or float
            Mean of all values in the cube, ignoring NaNs.
        '''
        return self._stat('mean')
    @property
    def median(self):
        '''
        Returns
        -------
        Quantity or float
            Median of all values in the cube, ignoring NaNs.
        '''
        return self._stat('median')
    @property
    def sum(self):
        '''
        Returns
        -------
        Quantity or float
            Sum of all values in the cube, ignoring NaNs.
        '''
        return self._stat('sum')
    @property
    def std(self):
        '''
        Returns
        -------
        Quantity or float
            Standard deviation of all values in the cube, ignoring NaNs.
        '''
        return self._stat('std')

    # array properties
    @property
    def shape(self):
        '''
        Returns
        -------
        tuple : Shape of cube data.
        '''
        return self.value.shape
    @property
    def size(self):
        '''
        Returns
        -------
        int : Size of cube data.
        '''
        return self.value.size
    @property
    def ndim(self):
        '''
        Returns
        -------
        int : Number of dimensions of cube data.
        '''
        return self.value.ndim
    @property
    def dtype(self):
        '''
        Returns
        -------
        np.dtype : Datatype of the cube data.
        '''
        return self.value.dtype
    @property
    def has_nan(self):
        '''
        Returns
        -------
        bool : Returns True if there are NaNs in the cube.
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
        Retrieve a header value by key from one or multiple headers.
        If a Header is missing a key, None is returned.

        Parameters
        ----------
        key : str
            FITS header keyword to retrieve.

        Returns
        -------
        value : list or str
            Header value(s) corresponding to `key`.

        Raises
        ------
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
        ----------
        figsize : tuple, optional, default=(8,4)
            Size of the output figure.
        style : str or None, optional, default=None
            Matplotlib style to use for plotting. If None,
            uses the default value set by `va_config.style`.

        Notes
        -----
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

        All WCS related metadata is updated in the
        reprojected DataCube.

        Parameters
        ----------
        reference_wcs : WCS or Header
            Target WCS or FITS header to reproject onto.
        method : {'interp', 'exact'} or None
            Reprojection method:
                - 'interp' : use `reproject_interp`
                - 'exact' : use `reproject_exact`
            If None, uses the default value
            set by `va_config.reproject_method`.
        return_footprint : bool or None, optional
            If True, return both reprojected data and reprojection
            footprints. If False, return only the reprojected data.
            If None, uses the default value set by `va_config.return_footprint`.
        parallel : bool, int, or None, optional, default=None
            If True, the reprojection is carried out in parallel,
            and if a positive integer, this specifies the number
            of threads to use. The reprojection will be parallelized
            over output array blocks specified by `block_size` (if the
            block size is not set, it will be determined automatically).
            If None, uses the default value set by `va_config.reproject_parallel`.
        block_size : tuple, 'auto', or None
            The size of blocks in terms of output array pixels that each block
            will handle reprojecting. Extending out from (0,0) coords positively,
            block sizes are clamped to output space edges when a block would extend
            past edge. Specifying 'auto' means that reprojection will be done in
            blocks with the block size automatically determined. If `block_size` is
            not specified or set to None, the reprojection will not be carried out in blocks.
            If `_default_flag`, uses the default value set by `va_config.reproject_block_size`.

        Returns
        -------
        new_cube : DataCube
            Reprojected DataCube.

        Notes
        -----
        - A cube-level WCS is only attached when the dimensionality of the
          reprojected data matches the dimensionality of the reference WCS.
        - For slice-by-slice reprojection (e.g., 3D → 2D targets), the output
          DataCube will not carry a cube-level WCS.
        - In time-series mode, per-slice headers are updated consistently
          when applicable.
        - If `return_footprint=True`, the reprojection footprint is attached
          to the returned DataCube as the `.footprint` attribute.
        '''
        return_footprint = get_config_value(return_footprint, 'return_footprint')

        if self.wcs is None and self.header is None:
            raise ValueError(
                'Cannot reproject: DataCube has neither .wcs nor .header'
            )

        if isinstance(reference_wcs, Header):
            reference_wcs = WCS(reference_wcs)
        elif not isinstance(reference_wcs, WCS):
            raise TypeError(
                'reference_wcs must be a WCS or fits.Header'
            )

        wcs_info = self.wcs if self.wcs is not None else self.header

        # new header free of WCS
        new_header = _copy_headers(self.nowcs_header)
        new_header = _transfer_history(self.primary_header, new_header)
        _log_history(new_header, f'Reprojected DataCube')

        # timeseries case, 3d cube with a list of headers/wcs
        if isinstance(wcs_info, list):
            data = self.value
            if self.unit is not None:
                data = self.unit * data
            reprojected_cube = []
            footprint = []

            for i, wcs in tqdm(enumerate(wcs_info), desc='Reprojecting each data slice'):
                reprojected, fp = _reproject_wcs(
                    (data[i], wcs),
                    reference_wcs,
                    method=method,
                    return_footprint=True,
                    parallel=parallel,
                    block_size=block_size
                )

                reprojected_cube.append(reprojected)
                footprint.append(fp)

            new_data = np.stack(reprojected_cube, axis=0)
            footprint = np.stack(footprint, axis=0)
            _log_history(
                new_header,
                'Reprojected timeseries cube slice-by-slice'
            )

            if self.error is not None:
                reprojected_errors = []

                for i, wcs in enumerate(tqdm(wcs_info, desc='Reprojecting each error slice')):
                    error_reproj = _reproject_wcs(
                        (self.error[i], wcs),
                        reference_wcs,
                        method=method,
                        return_footprint=False,
                        parallel=parallel,
                        block_size=block_size
                    )
                    reprojected_errors.append(error_reproj)

                new_error = np.stack(reprojected_errors, axis=0)
            else:
                new_error = None

        else:
            new_data, footprint = _reproject_wcs(
                (self.data, wcs_info),
                reference_wcs,
                method=method,
                return_footprint=True,
                parallel=parallel,
                block_size=block_size,
                description='Reprojecting data slices',
                log_file=new_header
            )

            if self.error is not None:
                new_error = _reproject_wcs(
                    (self.error, wcs_info),
                    reference_wcs,
                    method=method,
                    return_footprint=False,
                    parallel=parallel,
                    block_size=block_size,
                    description='Reprojecting error slices'
                )
            else:
                new_error = None

        if self.error is not None:
            _log_history(new_header, f'Reprojected errors')

        # update new header with reference WCS
        if isinstance(new_header, list):
            # timeseries cube
            if reference_wcs.naxis == 2:
                for hdr in new_header:
                    _update_header_from_wcs(hdr, reference_wcs)
                _log_history(new_header, 'Assigned 2D reference WCS to all slices')
            else:
                warnings.warn(
                    'Reference WCS is not 2D; dropping WCS for timeseries cube.'
                )
                _log_history(new_header, 'Dropped WCS due to dim mismatch')
        else:
            # non-timeseries cube
            if reference_wcs.naxis == new_data.ndim:
                _update_header_from_wcs(new_header, reference_wcs)
                _log_history(new_header, 'Updated all WCS keys in header')
            else:
                warnings.warn(
                    'Reference WCS dimensionality does not match data; '
                    'cube-level WCS not assigned.'
                )
                _log_history(new_header, 'Dropped WCS due to dim mismatch')

        if np.all(np.isnan(new_data)):
            raise ValueError(
                'All values in reprojected data are NaN. '
                'This likely indicates a WCS round-tripping failure.'
            )

        if isinstance(self.data, SpectralCube):
            if isinstance(new_header, list):
                raise RuntimeError(
                    'SpectralCube reprojection resulted in per-slice headers; '
                    'this is not supported.'
                )

            new_wcs = WCS(new_header)

            new_data = SpectralCube(
                data=new_data,
                wcs=new_wcs,
                meta=self.data.meta
            )

            new_data._spectral_unit = self.data._spectral_unit
            new_data._spectral_scale = self.data._spectral_scale

        new_cube = DataCube(
            data=new_data,
            header=new_header,
            error=new_error
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
        ----------
        unit : str or astropy.units.Unit
            Target unit.
        equivalencies : list, optional
            Astropy equivalencies for unit conversion (e.g. spectral).

        Returns
        -------
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

        # update header BUNIT
        new_hdr = _copy_headers(self.header)
        _update_header_key('BUNIT', unit, new_hdr)
        _log_history(new_hdr, f'Converted cube unit to {unit.to_string()}')

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
        )

    def with_mask(self, mask):
        '''
        Apply a boolean mask to the cube and return the
        masked data as a new DataCube.

        The shape of the cube is preserved and
        values are masked with NaNs.

        Parameters
        ----------
        mask : np.ndarray or Mask
            Boolean mask to apply. Must match the cube shape.

        Returns
        -------
        masked_data : DataCube
            Masked version of the data.

        Raises
        ------
        TypeError
            If masking is unsupported for the data type.
        '''
        # check mask shape
        mask = np.asarray(mask)

        # ensure mask is 3D
        if mask.ndim == 2:
            mask = mask[None, :, :]
        elif mask.ndim != 3:
            raise ValueError(
                f'Mask must be 2D or 3D, got {mask.ndim}D with shape {mask.shape}'
            )

        if mask.shape != self.shape:
            # broadcast 2D mask (1xNxM) across 3D cube (TxNxM)
            if mask.shape[1:] == self.shape[1:] and mask.shape[0] == 1:
                mask = np.broadcast_to(mask, self.shape)
            else:
                raise ValueError(
                    'Mask shape does not match cube shape and is not broadcastable! '
                    'For a 3D datacube with shape TxNxM, mask should either have shapes '
                    f'NxM, 1xNxM, or TxNxM. \nMask shape: {mask.shape}, cube shape: {self.shape}'
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
        new_hdr = _copy_headers(self.header)
        _log_history(new_hdr, 'Applied boolean mask to cube')

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
        )

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        '''
        Convert the spectral axis of the DataCube to a new unit.

        Parameters
        ----------
        unit : str or astropy.units.Unit
            Target spectral unit.
        velocity_convention : str, optional
            'radio', 'optical', 'relativistic', etc.
        rest_value : Quantity, optional
            Rest frequency/wavelength for Doppler conversion.
            Required if output type is velocity.

        Returns
        -------
        DataCube
            New cube with converted spectral axis.
        '''
        if not isinstance(self.data, SpectralCube):
            raise TypeError(
                'with_spectral_unit() can only be used when DataCube.data '
                'is a SpectralCube. For unit conversion of flux values, '
                'use .to().'
            )

        unit = Unit(unit)
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

        new_hdr = _copy_headers(self.header)
        _log_history(new_hdr, f'Converted spectral axis: {old_unit} -> {unit}')

        new_error = None if self.error is None else self.error.copy()

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_data.wcs
        )

    # Array Interface
    # ---------------
    def __array__(self):
        '''
        Return the underlying data as a NumPy array.
        Returns
        ----------
        np.ndarray
            The underlying 3D array representation.
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
        slice : DataCube
            The corresponding subset of the data.
        '''
        new_data = self.data[key]
        new_error = self.error[key] if self.error is not None else None

        new_hdr = _copy_headers(self.header)
        _log_history(new_hdr, f'Sliced data with key : {key}')

        if isinstance(new_hdr, list):
            new_hdr = _transfer_history(new_hdr, Header())
            _log_history(new_hdr, 'Header list dropped due to slicing')

        if self.wcs is None:
            new_wcs = None
        elif isinstance(self.wcs, list):
            new_wcs = None
            _log_history(new_hdr, 'WCS list dropped due to slicing')
        elif not _is_valid_wcs_slice(key):
            new_wcs = None
            new_hdr = _transfer_history(new_hdr, Header())
            _log_history(new_hdr, f'Header and WCS dropped due to invalid slice')
        else:
            try:
                new_wcs = self.wcs[key]
                _update_header_from_wcs(new_hdr, new_wcs)

            except (AttributeError, TypeError, ValueError) as e:
                new_wcs = None
                new_hdr = _transfer_history(new_hdr, Header())
                _log_history(new_hdr, f'Header and WCS dropped due to {type(e).__name__}')

        return DataCube(
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
            Length of the first dimension (T).
        '''
        return len(self.value)

    def reshape(self, *shape):
        '''
        Return a reshaped view of the cube data.
        Parameters
        ----------
        *shape : int
            New shape for the data array.
        Returns
        -------
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
        ----------
        func : {'min', 'max', 'mean', 'median', 'std'}
            The name of the statistical quantity to compute.

        Returns
        -------
        value : float, `astropy.units.Quantity`, or scalar-like
            The computed statistical value. If the underlying data includes
            units, the returned value is a `Quantity`; otherwise it is a unitless
            NumPy scalar.

        Raises
        ------
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
        -------
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
