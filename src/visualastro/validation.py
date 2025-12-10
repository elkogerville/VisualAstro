'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for validating inputs and type checks.
Dependencies:
    - astropy
'''


def _validate_type(
    data, types, default=None, allow_none=True, name='data'
):
    '''
    Validate that `data` is an instance of one of the allowed types.

    Parameters
    ––––––––––
    data : object
        The object to validate.
    types : type or tuple of types
        A type or tuple of types that `data` is allowed to be.
        Ex: Quantity, (Quantity), or (Quantity, SpectralCube)
    default : object, optional, default=None
            Value to return if `data` is None. Use this to provide
            a default instance when None is passed.
    allow_none : bool, default=True
            If True, None is a valid input. If False, None will raise TypeError.
    name : str
        Name of object for error message.

    Raises
    ––––––
    TypeError
        If `data` is not an instance of any of the types in `types`.
    '''
    if data is None and allow_none:
        return default

    # make iterable
    if not isinstance(types, tuple):
        types = (types,)

    if not isinstance(data, types):
        allowed = ', '.join(t.__name__ for t in types)
        raise TypeError(
            f"'{name}' must be one of: {allowed}; got {type(data).__name__}."
        )

    return data
