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
