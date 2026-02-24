"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-02-18
Description:
    DataCube data structure for 3D SpectralCubes or
    time series data cubes.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - spectral_cube
    - specutils
    - tqdm
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
"""

import warnings
from astropy.io.fits import Header
from astropy.units import (
    Quantity, Unit, UnitBase, UnitsError
)
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from spectral_cube import SpectralCube
from specutils import SpectralRegion, Spectrum
from tqdm import tqdm
from .spectra_utils import (
    fit_continuum,
    mask_spectral_region,
    spectral_idx_2_world as _spectral_idx_2_world,
    spectral_world_2_idx as _spectral_world_2_idx
)
from .fits_utils import (
    _copy_headers, _get_history,
    _log_history, _remove_history,
    _transfer_history, _update_header_key,
)
from .config import get_config_value, _default_flag
from .units import (
    ensure_common_unit, _check_unit_equality,
    require_spectral_region, to_unit, unit_2_string
)
from .utils import _type_name
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
    """
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
    """

    def __init__(
        self,
        data: NDArray | Quantity | SpectralCube,
        header: list[Header] | Header | NDArray | tuple[Header] | None = None,
        error: NDArray | Quantity | None = None,
        wcs: list[WCS] | WCS | tuple[WCS] | None = None
    ):
        data = _validate_type(
            data, (np.ndarray, Quantity, SpectralCube),
            allow_none=False, name='data'
        )
        header = _validate_type(
            header, (list, Header, np.ndarray, tuple),
            allow_none=True, name='header'
        )
        error = _validate_type(
            error, (np.ndarray, Quantity), default=None,
            allow_none=True, name='error'
        )
        wcs = _validate_type(
            wcs, (list, WCS, tuple), default=None,
            allow_none=True, name='wcs'
        )
        if isinstance(wcs, (list, tuple)):
            wcs = _validate_iterable_type(wcs, WCS, 'wcs')

        if header is None:
            if isinstance(wcs, (list, tuple)):
                header = [Header() for _ in wcs]
            else:
                header = Header()

        # extract array view for validation
        array, unit = self._get_value(data)

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
            header = _validate_iterable_type(header, Header, 'header')

            primary_hdr = header[0]
        else:
            primary_hdr = header

        _log_history(primary_hdr, 'Initialized DataCube')

        # ensure that units are consistent across all headers
        hdr_unit = ensure_common_unit(
            header, on_mismatch='raise',
            label='header'
        )

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

        if wcs is not None:
            nowcs_header = _strip_wcs_from_header(header)
            _remove_history(nowcs_header)
        else:
            nowcs_header = header

        # assign attributes
        self.data: NDArray | Quantity | SpectralCube = data
        self.primary_header: Header = primary_hdr
        self.header: Header | list[Header] = header
        self.nowcs_header: Header | list[Header]  = nowcs_header
        self.error: NDArray | Quantity | None = error
        self.wcs: WCS | list[WCS] | tuple[WCS] | None = wcs
        self.footprint: NDArray | None = None

    # Properties
    # ----------
    @property
    def value(self) -> NDArray:
        """
        Returns
        -------
        np.ndarray : View of the underlying numpy array.
        """
        value, unit = self._get_value(self.data)
        return value

    @property
    def quantity(self) -> Quantity | None:
        """
        Returns
        -------
        Quantity or None :
            Quantity array of data values (values + astropy units),
            of None if no units.
        """
        if self.unit is None:
            return None
        return self.unit * self.value

    @property
    def spectral_axis(self) -> Quantity | None:
        """
        Returns
        -------
        Quantity or None:
            Spectral axis of the data or None if data
            is not a ``spectral_cube``.
        """
        if isinstance(self.data, SpectralCube):
            return self.data.spectral_axis
        return None

    @property
    def unit(self) -> UnitBase | None:
        """
        Returns
        -------
        UnitBase or None :
            Astropy.Unit of the data or None if no units found.
        """
        return getattr(self.data, 'unit', None)

    @property
    def spectral_unit(self) -> UnitBase | None:
        """
        Returns
        -------
        UnitBase or None:
            Astropy.Unit of the spectral axis or None if not found.
        """
        if isinstance(self.data, SpectralCube):
            return self.data.spectral_axis.unit
        return None

    # statistical properties
    @property
    def min(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Minimum value in the cube, ignoring NaNs.
        """
        return self._stat('min')

    @property
    def max(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Maximum value in the cube, ignoring NaNs.
        """
        return self._stat('max')

    @property
    def mean(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Mean of all values in the cube, ignoring NaNs.
        """
        return self._stat('mean')

    @property
    def median(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Median of all values in the cube, ignoring NaNs.
        """
        return self._stat('median')

    @property
    def sum(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Sum of all values in the cube, ignoring NaNs.
        """
        return self._stat('sum')

    @property
    def std(self) -> float | Quantity:
        """
        Returns
        -------
        Quantity or float
            Standard deviation of all values in the cube, ignoring NaNs.
        """
        return self._stat('std')

    # array properties
    @property
    def shape(self) -> tuple:
        """
        Returns
        -------
        tuple : Shape of cube data.
        """
        return self.value.shape

    @property
    def size(self) -> int:
        """
        Returns
        -------
        int : Size of cube data.
        """
        return self.value.size

    @property
    def ndim(self) -> int:
        """
        Returns
        -------
        int : Number of dimensions of cube data.
        """
        return self.value.ndim

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        np.dtype : Datatype of the cube data.
        """
        return self.value.dtype

    @property
    def has_nan(self) -> np.bool:
        """
        Returns
        -------
        bool : Returns True if there are NaNs in the cube.
        """
        return np.isnan(self.value).any()

    @property
    def itemsize(self) -> int:
        """
        Returns
        -------
        int : Length of 1 array element in bytes.
        """
        return self.value.itemsize

    @property
    def nbytes(self) -> int:
        """
        Returns
        -------
        int : Total number of bytes used by the data array.
        """
        return self.value.nbytes

    @property
    def log(self) -> list[str] | None:
        """
        Get the processing history from the FITS HISTORY cards.

        Returns
        -------
        list of str or None
            List of HISTORY entries, or None if no header exists.
        """
        return _get_history(self.primary_header)

    # Methods
    # -------
    def header_get(self, key):
        """
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
        """
        # case 1: single Header
        if isinstance(self.header, Header):
            return self.header.get(key, None)

        # case 2: Header list
        elif isinstance(self.header, (list, np.ndarray, tuple)):
            return [h.get(key, None) for h in self.header]

        else:
            raise ValueError(f'Unsupported header type.')

    def inspect(self, figsize=(10,6), style=None):
        """
        Plot the mean and standard deviation across each cube slice.
        Useful for quickly identifying slices of interest in the cube.

        Parameters
        ----------
        figsize : tuple, optional, default=(8,4)
            Size of the output figure.
        style : str or None, optional, default=None
            Matplotlib style to use for plotting. If None,
            uses the default value set by `config.style`.

        Notes
        -----
        This method visualizes the mean and standard deviation of flux across
        each 2D slice of the cube as a function of slice index.
        """
        from .plot_utils import return_stylename

        # get default config values
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
        """
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
            set by `config.reproject_method`.
        return_footprint : bool or None, optional
            If True, return both reprojected data and reprojection
            footprints. If False, return only the reprojected data.
            If None, uses the default value set by `config.return_footprint`.
        parallel : bool, int, or None, optional, default=None
            If True, the reprojection is carried out in parallel,
            and if a positive integer, this specifies the number
            of threads to use. The reprojection will be parallelized
            over output array blocks specified by `block_size` (if the
            block size is not set, it will be determined automatically).
            If None, uses the default value set by `config.reproject_parallel`.
        block_size : tuple, 'auto', or None
            The size of blocks in terms of output array pixels that each block
            will handle reprojecting. Extending out from (0,0) coords positively,
            block sizes are clamped to output space edges when a block would extend
            past edge. Specifying 'auto' means that reprojection will be done in
            blocks with the block size automatically determined. If `block_size` is
            not specified or set to None, the reprojection will not be carried out in blocks.
            If `_default_flag`, uses the default value set by `config.reproject_block_size`.

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
        """
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

    def spectral_idx_2_world(self, idx):
        """
        Return the spectral value at a given index or index range.

        Parameters
        ----------
        idx : int, list of int, or None
            Index or indices specifying the position along the spectral axis:
            - ``i``      -> returns ``spectral_axis[i]``
            - ``[i]``    -> returns ``spectral_axis[i]``
            - ``[i, j]`` -> returns ``(spectral_axis[i] + spectral_axis[j]) / 2``
            - ``None``   -> returns ``(spectral_axis[0] + spectral_axis[-1]) / 2``

        Returns
        -------
        float
            Spectral value at the specified index, in the units of the
            cube's spectral axis.

        Raises
        ------
        ValueError
            If ``self.data`` is not a ``SpectralCube``.
        ValueError
            If ``idx`` is not an int, a list of one or two ints, or None.

        See Also
        --------
        spectral_world_2_idx : Inverse operation, maps spectral value to index.
        """
        if not isinstance(self.data, SpectralCube):
            raise ValueError(
                'DataCube.data must be a SpectralCube!'
            )
        return _spectral_idx_2_world(
            self.data.spectral_axis, idx, keep_unit=True
        )

    def spectral_world_2_idx(self, value):
        """
        Return the index of the nearest spectral channel to a given value.
        If ``value`` has no unit, the cube unit is assumed.

        Parameters
        ----------
        value : Quantity
            Spectral value in the same units as the cube's spectral axis.

        Returns
        -------
        int :
            Index of the nearest spectral channel.

        Raises
        ------
        ValueError :
            If ``data`` attribute is not a ``SpectralCube``.
        """
        if not isinstance(self.data, SpectralCube):
            raise ValueError(
                'DataCube.data must be a SpectralCube!'
            )

        return _spectral_world_2_idx(
            self.data.spectral_axis, value
        )

    def subtract_continuum(
        self,
        region=None,
        fit_method=None,
        min_valid_pixels='auto',
        print_info=None,
        auto_percentile=10,
        minimum_floor=3
    ):
        """
        Subtract a fitted continuum from each spatial pixel in the cube.

        For each pixel, a continuum is fitted using only the spectral channels
        within the specified region(s), then subtracted from all channels in
        that pixel. This is useful for isolating emission or absorption features
        by removing the underlying continuum.

        The cube is first slabbed to the full spectral range encompassing all
        subregions before fitting. For example, with region = [(6.5, 6.7),
        (7.2, 7.5)] µm, the continuum fit uses only those two ranges but is
        applied across the entire 6.5-7.5 µm span.

        Note
        ----
        The error attribute of the DataCube is dropped during this operation.

        Parameters
        ----------
        region : SpectralRegion, region input, or None
            Spectral region(s) to use for continuum fitting. Can be:
            - SpectralRegion object
            - (low, high) * unit for single region
            - [(low, high), ...] * unit for multiple regions
            - [(low * unit, high * unit), ...] for single or multiple regions
            - None (uses the entire spectral_axis range)
            Regions outside emission/absorption features are typically chosen.
        fit_method : {'fit_continuum', 'generic'} or None, optional, default=None
            Method used for fitting the continuum.
            - 'fit_continuum': uses `fit_continuum` with a specified window
            - 'generic' : uses `fit_generic_continuum`
            If None, uses the default value from ``config.spectrum_continuum_fit_method``.
        min_valid_pixels : int or 'auto', optional, default='auto'
            Minimum valid flux data points needed in order to attempt a continuum fit.
            If ``'auto'``, will compute a percentile-based threshold for valid pixels
            along the spectral axis to use as the threshold.
        auto_percentile : float, optional, default=10
            Percentile of the nonzero valid-spectral-point counts used to automatically
            set ``min_valid_pixels`` when ``min_valid_pixels='auto'``. Lower values are
            more permissive; higher values are more restrictive. Must be between 0 and 100.
        minimum_floor : int, optional, default=3
            Lower limit for the minimum number of valid pixels. If the computed
            or user-provided ``min_valid_pixels`` falls below this value, a ``ValueError``
            is raised. This prevents continuum fitting attempts on pixels with
            insufficient spectral data.
        print_info : bool or None, optional, default=None
            If True, will print the value of ``min_valid_pixels``.
            If None, uses the default value set by ``config.print_info``.

        Returns
        -------
        DataCube
            A new DataCube with the fitted continuum subtracted from each pixel.
            The spectral axis covers the full range of the input region.

        Raises
        ------
        ValueError :
            If ``self.data`` is not a SpectralCube.
        UnitsError :
            If the cube's spectral axis and region units are incompatible.
        ValueError :
            If no valid pixels are found in the region.

        Examples
        --------
        >>> import astropy.units as u
        >>> # Subtract continuum fitted from two spectral windows
        >>> # i.e. excluding an emission peak between 6.7-7.2 um
        >>> region = [(6.5, 6.7), (7.2, 7.5)] * u.um
        >>> cube_sub = datacube.subtract_continuum(region)
        """
        fit_method = get_config_value(fit_method, 'spectrum_continuum_fit_method')
        print_info = get_config_value(print_info, 'print_info')

        cube = self.data
        if not isinstance(cube, SpectralCube):
            raise ValueError(
                'cube must be or contain a SpectralCube, '
                f'got {_type_name(cube)} instead!'
            )

        if region is None:
            region = SpectralRegion(
                cube.spectral_axis.min(), cube.spectral_axis.max()
            )
        region = require_spectral_region(region)
        ensure_common_unit([cube.spectral_axis, region.lower], on_mismatch='raise')

        spec_min = region.lower
        spec_max = region.upper
        subcube = cube.spectral_slab(spec_min, spec_max)
        continuum_cube = np.full(subcube.shape, np.nan) * subcube.unit

        ny, nx = subcube.shape[1:]
        region_mask = mask_spectral_region(subcube.spectral_axis, region)

        flux_data = subcube.filled_data[:]
        finite_flux = np.isfinite(flux_data)
        # 3D mask representing which x,y pixels are both
        # within the region and finite for each spectral channel
        valid_mask = finite_flux & region_mask[:, None, None]
        # sum across spectral channels to get number of valid voxels
        # along a single pixel column. this is a 2d map where each
        # pixel contains the number of valid voxels in that column
        N_valid = valid_mask.sum(axis=0)

        # compute median number of valid flux pixels in cube
        if str(min_valid_pixels).lower() == 'auto':

            N_valid_flat = N_valid.ravel()
            N_nonzero = N_valid_flat[N_valid_flat > 0]

            if N_nonzero.size == 0:
                raise ValueError(
                    'No valid spectral pixels found in region. '
                    'Continuum fitting is not possible.'
                )

            min_valid_pixels = int(
                np.ceil(np.percentile(N_nonzero, auto_percentile))
            )
        else:
            min_valid_pixels = int(min_valid_pixels)

        if min_valid_pixels < minimum_floor:
            raise ValueError(
                f'Auto min_valid_pixels={min_valid_pixels} is below '
                f'minimum_floor={minimum_floor}. '
                'Widen the region or lower minimum_floor.'
            )

        # check if any spatial pixels have enough valid spectral points
        # for continuum fitting. otherwise the output would be all NaNs.
        if (N_valid >= min_valid_pixels).sum() == 0:
            raise ValueError(
                f'No pixels have enough valid points for continuum fitting. '
                f'Required: {min_valid_pixels}, Maximum available: {N_valid.max()}. '
                f'Try widening the region, lowering min_valid_pixels, or '
                f'setting minimum_floor lower.'
            )

        if print_info:
            # all valid pixels (finite and inside region)
            n_any = (N_valid > 0).sum()
            # valid pixels passing threshold
            n_meeting = (N_valid >= min_valid_pixels).sum()

            N_flat = N_valid.ravel()
            N_nonzero = N_flat[N_flat > 0]

            print(f'Minimum pixel threshold: {min_valid_pixels}')
            print(f'Pixels with any valid data: {n_any}/{ny*nx} '
                  f'({100*n_any/(ny*nx):.1f}%)')
            print(f'Usable pixels passing threshold: {n_meeting}/{n_any} '
                  f'({100*n_meeting/n_any:.1f}%)')

            print(f'Valid spectral points per pixel (nonzero only): '
                  f'min={N_nonzero.min()}, '
                  f'mean={np.mean(N_nonzero):.0f}, '
                  f'median={np.median(N_nonzero):.0f}, '
                  f'max={N_nonzero.max()}')

        for j in tqdm(range(ny), desc='Fitting continuum'):
            for i in range(nx):

                if N_valid[j, i] < min_valid_pixels:
                    continue

                pixel_flux = subcube[:, j, i]
                spec = Spectrum(spectral_axis=subcube.spectral_axis, flux=pixel_flux)
                cont_spec = fit_continuum(spec, fit_method, region)
                continuum_cube[:, j, i] = cont_spec

        continuum_subtracted_cube = subcube - continuum_cube

        new_hdr = _copy_headers(self.header)
        wcs_new = continuum_subtracted_cube.wcs.to_header()
        # spectral axis only
        for key in ('CRPIX3', 'CRVAL3', 'CDELT3', 'CUNIT3', 'CTYPE3'):
            if key in wcs_new:
                new_hdr[key] = wcs_new[key] # type: ignore

        # PC/CD terms involving spectral axis
        for key, val in wcs_new.items():
            if key.startswith(('PC', 'CD')):
                if key.endswith('3') or '_3' in key:
                    new_hdr[key] = val

        _log_history(new_hdr, 'Subtracted continuum from cube')

        return DataCube(
            data=continuum_subtracted_cube,
            header=new_hdr
        )

    def to(self, unit, equivalencies=None):
        """
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
        """
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
        _log_history(new_hdr, f'Converted cube unit to {unit.to_string()}')
        _update_header_key('BUNIT', unit, new_hdr)

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
        )

    def with_mask(self, mask):
        """
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
        """
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
                f'Cannot apply mask to data of type {_type_name(self.data)}'
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
        """
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
        """
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
        """
        Return the underlying data as a NumPy array.
        Returns
        ----------
        np.ndarray
            The underlying 3D array representation.
        """
        return self.value

    def __getitem__(self, key):
        """
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
        """
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
                _log_history(new_hdr, f'Header and WCS dropped due to {_type_name(e)}')

        return DataCube(
            data=new_data,
            header=new_hdr,
            error=new_error,
            wcs=new_wcs
        )

    def __len__(self):
        """
        Return the number of spectral slices along the first axis.
        Returns
        -------
        int
            Length of the first dimension (T).
        """
        return len(self.value)

    def reshape(self, *shape):
        """
        Return a reshaped view of the cube data.
        Parameters
        ----------
        *shape : int
            New shape for the data array.
        Returns
        -------
        np.ndarray
            Reshaped data array.
        """
        return self.value.reshape(*shape)


    # HELPER FUNCTIONS
    # ----------------
    def _get_value(self, data):
        """
        Get the underlying array representation of the data.

        Parameters
        ----------
        data : array-like, Quantity, or SpectralCube
            Data object containing an np.ndarray

        Returns
        -------
        value :
            np.ndarray
        """
        if isinstance(data, SpectralCube):
            array = data.filled_data[:].value
            unit = to_unit(data.unit)
        elif isinstance(data, Quantity):
            array = data.value
            unit = to_unit(data.unit)
        else:
            array = np.asarray(data)
            unit = None

        return array, unit

    def _stat(self, func: str) -> float | Quantity:
        """
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
        value : float or `astropy.units.Quantity`
            The computed statistical value. If the underlying data includes
            units, the returned value is a `Quantity`; otherwise it is a unitless
            NumPy scalar.

        Raises
        ------
        KeyError
            If an unsupported statistic name is provided.
        """
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
        """
        Returns
        -------
        str : String representation of DataCube.
        """
        if isinstance(self.data, SpectralCube):

            flux_unit = self.unit
            ns, ny, nx = self.shape
            dtype = self.dtype

            wx, wy = self.data.world_extrema
            wx_unit = wx.unit.to_string()
            wy_unit = wy.unit.to_string()

            spec = self.data.spectral_axis
            spec_unit = spec.unit.to_string()

            return (
                f"DataCube[SpectralCube]: unit={flux_unit}, shape=({ns}, {ny}, {nx}), dtype={dtype}\n"
                f"  {'nx:':<6}{nx:>6}    {'unit:':<6}{wx_unit:<6}  "
                f"{'range:':<7}{wx[0]:>12.6f}, {wx[1]:>12.6f}\n"
                f"  {'ny:':<6}{ny:>6}    {'unit:':<6}{wy_unit:<6}  "
                f"{'range:':<7}{wy[0]:>12.6f}, {wy[1]:>12.6f}\n"
                f"  {'ns:':<6}{ns:>6}    {'unit:':<6}{spec_unit:<6}  "
                f"{'range:':<7}{spec.min():>12.4g}, {spec.max():>12.4g}"
            )

        if isinstance(self.data, Quantity):
            return (
                f'<DataCube[Quantity]: unit={self.unit}, shape={self.shape}, dtype={self.dtype}>'
            )

        return (
            f'<DataCube[NDArray]: shape={self.shape}, dtype={self.dtype}>'
        )
