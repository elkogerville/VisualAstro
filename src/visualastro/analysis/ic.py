"""
Author: Elko Gerville-Reache
Date Created: 2026-06-08
Date Modified: 2026-06-08
Description:
    Data creation functions.
"""

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

@overload
def blob(N: int, as_array: Literal[False] = ...) -> tuple[NDArray, NDArray, NDArray]: ...

@overload
def blob(N: int, as_array: Literal[True]) -> NDArray: ...

def blob(N: int, as_array: bool = False) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """make a 3D blob"""
    cov = np.array([
        [2.0, 1.2, 0.8],
        [1.2, 1.5, 0.6],
        [0.8, 0.6, 1.0]
    ])

    mean = np.array([0.0, 0.0, 0.0])
    data = np.random.multivariate_normal(mean, cov, N)

    if as_array:
        return data

    return data[:,0], data[:,1], data[:,2]
