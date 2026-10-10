"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-10-10
Description:
    Sequence utility functions.
"""

from collections.abc import Sequence
from itertools import cycle, islice
from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray


T = TypeVar('T')


@overload
def _cycle(data: list[T], i: int) -> T: ...

@overload
def _cycle(data: tuple[T, ...], i: int) -> T: ...

@overload
def _cycle(data: NDArray, i: int) -> Any: ...

@overload
def _cycle(data: Sequence[T], i: int) -> T: ...

def _cycle(data, i):
    """
    Cycle through a list continuously. When
    the bounds are reached, the index is reset
    to zero.

    This function is meant to be called inside
    of a loop, when lists of different lengths
    need to be iterated upon concurently.

    Parameters
    ----------
    data : list[T]
        Input data list.
    i : int
        Loop index.
    j : int
        Offset to add onto `i`. For internal
        cycling uses.

    Returns
    -------
    T :
        `data` element.
    """
    if not isinstance(data, (Sequence, np.ndarray)):
        raise ValueError(
            'data must be a Sequence or NDArray! '
            f'got {type(data).__name__}'
        )
    return data[int(i) % len(data)]


def match_length(lst: list, reference_list: list) -> list:
    """
    Cycle or crop a list to match the length of a reference sequence.

    Parameters
    ----------
    lst : list
        Sequence to extend or crop.
    ref : list
        Reference sequence whose length `lst` is matched to.

    Returns
    -------
    list :
        `lst` cycled (if shorter) or cropped (if longer) to len(ref).
    """
    return list(islice(cycle(lst), len(reference_list)))


@overload
def _unwrap_if_single(
    array: list[T]
) -> T | list[T]: ...

@overload
def _unwrap_if_single(
    array: tuple[T, ...]
) -> T | tuple[T, ...]: ...

@overload
def _unwrap_if_single(
    array: Any
) -> Any: ...

def _unwrap_if_single(
    array: Sequence[T] | NDArray[Any]
) -> T | Sequence[T] | NDArray[Any]:
    """
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
    """
    if isinstance(array, (list, tuple, np.ndarray)):
        return array[0] if len(array) == 1 else array
    return array


def _roll(items: list[T], shift: int) -> list[T]:
    """
    Roll a list by `shift` positions.

    Parameters
    ----------
    items : list
        List to roll. Not modified.
    shift : int
        Positive shifts elements right, negative shifts left.

    Returns
    -------
    list
        New rolled list. Empty input returns an empty list.
    """
    n = len(items)
    if n == 0:
        return []
    shift = int(shift) % n
    return items[-shift:] + items[:-shift] if shift else list(items)
