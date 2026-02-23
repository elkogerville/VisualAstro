'''
Author: Elko Gerville-Reache
Date Created: 2026-01-16
Date Modified: 2026-01-16
Description:
    General utility functions.
'''


from typing import Any


def _unwrap_if_single(array):
    '''
    Unwrap an array-like object if it contains exactly one element.

    If the input has length 1, the sole element is returned.
    Otherwise, the input is returned unchanged. This is primarily
    intended for user-facing APIs that return either a single object
    or a collection depending on the number of results.

    Parameters
    ----------
    array : Sequence[T]
        A sequence-like object supporting `len()` and indexing.
        Must have at least one element.

    Returns
    -------
    T or Sequence[T]
        The sole element if `len(array) == 1`, otherwise the original
        input sequence.
    '''
    return array[0] if len(array) == 1 else array


def _type_name(obj: Any) -> str:
    """
    Get a readable, string representation of an object's type.

    This is used internally for error messages, and is an
    alias for:
        ``type(obj).__name__``.

    Parameters
    ----------
    obj : Any
        Object to return type from.

    Returns
    -------
    str :
        String of the object type.
    """
    return type(obj).__name__
