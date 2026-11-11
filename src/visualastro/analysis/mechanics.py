"""
Author: Elko Gerville-Reache
Date Created: 2026-11-07
Date Modified: 2026-11-07
Description:
    Orbital mechanics analysis module.
"""

import numpy as np
from numpy.typing import NDArray


def center_of_mass(
    positions: NDArray,
    masses: NDArray | None = None,
) -> NDArray:
    """Compute the (weighted) center of mass.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, d).
    masses : np.ndarray | None, optional, default=None
        Masses of shape (N,). None → unit masses.

    Returns
    -------
    np.ndarray
        Center of mass of shape (d,).

    Raises
    ------
    ValueError
        If shapes are inconsistent, inputs are non-finite, or the
        total mass is zero.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or len(pos) == 0:
        raise ValueError('positions must have shape (N, d), N > 0!')
    if not np.isfinite(pos).all():
        raise ValueError('positions must be finite!')

    m = None
    if masses is not None:
        m = np.asarray(masses, dtype=float)
        if m.shape != (len(pos),):
            raise ValueError(f'masses must have shape ({len(pos)},)!')
        if not np.isfinite(m).all():
            raise ValueError('masses must be finite!')
        if m.sum() == 0.0:
            raise ValueError('total mass must be non-zero!')

    return np.average(pos, axis=0, weights=m)
