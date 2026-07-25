"""
Author: Elko Gerville-Reache
Date Created: 2026-06-11
Date Modified: 2026-06-11
Description:
    Numerical and computational functions.
"""

from collections.abc import Sequence
from typing import Callable, Literal, overload

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import KDTree
from scipy.special import gamma


@overload
def interpolate(
    xp: u.Quantity,
    yp: u.Quantity,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[u.Quantity, u.Quantity]: ...

@overload
def interpolate(
    xp: NDArray,
    yp: NDArray,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[NDArray, NDArray]: ...

def interpolate(
    xp: NDArray | u.Quantity,
    yp: NDArray | u.Quantity,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[NDArray | u.Quantity, NDArray | u.Quantity]:
    """
    Interpolate a 1D array over a specified range.

    Parameters
    ----------
    xp : ArrayLike
        The x-coordinates of the data points. Must be 1D.
        Must be convertable to an array with `np.asarray`.
    yp : ArrayLike
        The y-coordinates of the data points. Must be 1D.
        Must be convertable to an array with `np.asarray`.
    x_range : tuple of float
        The (min, max) range over which to interpolate.
    N_samples : int
        Number of points in the interpolated output.
    method : {'linear', 'cubic', 'cubic_spline'}, default='linear'
        Interpolation method. Options:

        - `'linear'` : linear interpolation
        - `'cubic'` : cubic interpolation using `interp1d`
        - `'cubic_spline'` : cubic spline interpolation using `CubicSpline`

    Returns
    -------
    x_interp : np.ndarray
        The evenly spaced x-coordinates over the specified range.
    y_interp : np.ndarray
        The interpolated y-values corresponding to `x_interp`.
    """
    x_unit = xp.unit if isinstance(xp, u.Quantity) else None
    y_unit = yp.unit if isinstance(yp, u.Quantity) else None
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("'xp' and 'yp' must be 1D arrays")

    if len(xp) != len(yp):
        raise ValueError(
            f"'xp' and 'yp' must have the same length. "
            f"Got xp: {len(xp)}, yp: {len(yp)}"
        )
    if len(xp) < 2:
        raise ValueError(f'need at least 2 points for interpolation, got{len(xp)}')

    if not isinstance(N_samples, (int, np.integer)) or N_samples < 2:
        raise ValueError(f"'N_samples' must be an integer >= 2, got {N_samples}")

    if not isinstance(x_range, (Sequence, np.ndarray)) or len(x_range) != 2:
        raise ValueError(f"'x_range' must be a tuple of (min, max), got {x_range}")
    if x_range[0] >= x_range[1]:
        raise ValueError(
            f"'x_range' must be (min, max) with min < max, "
            f'got ({x_range[0]}, {x_range[1]})'
        )

    valid_methods = {'linear', 'cubic', 'cubic_spline'}
    if method not in valid_methods:
        raise ValueError(
            f"'method' must be one of {valid_methods}, got '{method}'"
        )

    # generate new interpolation samples
    x_interp = np.linspace(x_range[0], x_range[1], N_samples)

    # get interpolation method
    if method == 'cubic_spline':
        f_interp = CubicSpline(xp, yp)
    else:
        kind = method if method in ['linear', 'cubic'] else 'linear'
        f_interp = interp1d(xp, yp, kind=kind)

    # interpolate over new samples
    y_interp = f_interp(x_interp)

    if x_unit is not None:
        x_interp *= x_unit
    if y_unit is not None:
        y_interp *= y_unit

    return x_interp, y_interp


def kde1d(
    x: NDArray,
    bw_method: Literal['scott', 'silverman'] | float | Callable = 'scott',
    gridsize: int = 200,
    padding: float = 0.2,
    xlim: tuple[float, float] | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Estimate the 1D density of a sample using a Gaussian KDE.

    Parameters
    ----------
    x : np.ndarray
        1D sample values.
    bw_method : {'scott', 'silverman'} | scalar | callable, optional
        Bandwidth selection method passed to scipy.stats.gaussian_kde.
    gridsize : int, optional
        Number of evaluation points.
    padding : float, optional
        Fractional padding added to the data range.
    xlim : tuple of float, optional
        Explicit evaluation limits.

    Returns
    -------
    xgrid : np.ndarray
        Evaluation coordinates.
    density : np.ndarray
        KDE evaluated on `xgrid`.
    """
    x = np.asarray(x)
    gridsize = int(gridsize)

    if xlim is None:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        dx = (xmax - xmin) * padding
        xmin -= dx
        xmax += dx
    else:
        xmin, xmax = xlim

    xgrid = np.linspace(xmin, xmax, gridsize)

    kernel = stats.gaussian_kde(x, bw_method=bw_method)
    density = kernel(xgrid)

    return xgrid, density


def kde2d(
    x: NDArray,
    y: NDArray,
    bw_method: Literal['scott', 'silverman'] | float | Callable = 'scott',
    gridsize: int = 200,
    padding: float = 0.2,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Estimate the 2D density of a set of particles using a Gaussian KDE.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates of shape (N,).
    y : np.ndarray
        1D array of y-coordinates of shape (N,).
    bw_method : {'scott', 'silverman'} | scalar | callable, optional, default='scott'
        The method used to calculate the bandwidth factor for the Gaussian KDE.
        Can be one of:
        - `'scott'` or `'silverman'`: use standard rules of thumb.
        - a scalar constant: directly used as the bandwidth factor.
        - a callable: should take a `scipy.stats.gaussian_kde` instance as its
            sole argument and return a scalar bandwidth factor.

    gridsize : int, optional, default=200
        Grid resolution for the KDE.
    padding : float, optional, default=0.2
        Fractional padding applied to the data range when generating
        the evaluation grid, expressed as a fraction of the total span
        along each axis. For example, a value of 0.2 expands the grid
        limits by 20% beyond the minimum and maximum of the data in both
        `x` and `y` directions. This helps capture the tails of the
        Gaussian kernel near the plot boundaries.

    Returns
    -------
    xgrid : np.ndarray
        2D array of x-coordinates for the evaluation grid (shape res×res).
    ygrid : np.ndarray
        2D array of y-coordinates for the evaluation grid (shape res×res).
    Z : np.ndarray
        2D array of estimated density values on the grid (shape res×res).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    gridsize = int(gridsize)

    # compute bounds with % padding
    if xlim is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        padding_x = (xmax - xmin) * padding
        xmin -= padding_x
        xmax += padding_x
    else:
        xmin, xmax = xlim

    if ylim is None:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        padding_y = (ymax - ymin) * padding
        ymin -= padding_y
        ymax += padding_y
    else:
        ymin, ymax = ylim

    xgrid, ygrid = np.mgrid[
        xmin:xmax:complex(gridsize),
        ymin:ymax:complex(gridsize)
    ]

    values = np.vstack([x, y])
    grid = np.vstack([xgrid.ravel(), ygrid.ravel()])
    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    Z = np.reshape(kernel(grid), xgrid.shape)

    return xgrid, ygrid, Z


def number_density(
    points: NDArray,
    k: int,
    log: bool = False
) -> NDArray:
    """
    Estimate local number density using a k-nearest neighbors (kNN) adaptive
    volume estimator.

    For each point, the distance to its k-th nearest neighbor defines a
    hypersphere of radius r_k. The density is estimated as:

        rho = k / V_D(r_k)

    where V_D is the volume of a D-dimensional ball:

        V_D(r) = (pi^(D/2) / Gamma(D/2 + 1)) * r^D

    Parameters
    ----------
    points : np.ndarray of shape (N, D)
        Input point coordinates. N is the number of points, D is the
        spatial dimension (typically 2 or 3).
    k : int
        Number of nearest neighbors used to define the local volume scale.
        Controls the bias–variance trade-off of the estimator:

        * small `k` → high spatial resolution, noisy estimate
        * large `k` → smoother estimate, reduced variance

    log : bool, default=False
        If `True`, returns `np.log10` of the density. Recommended when
        densities span multiple orders of magnitude.

    Returns
    -------
    rho : np.ndarray of shape (N,)
        Estimated number density at each point. If `log=True`, returns
        `np.log10(rho)`.

    Notes
    -----
    - The method generalizes to arbitrary dimensions via the volume formula
      of the N-dimensional ball.

    Examples
    --------
    >>> import numpy as np
    >>> import visualastro as va
    >>> import matplotlib.pyplot as plt
    >>> pos = np.random.rand(20000, 3)
    >>> rho = va.number_density(pos, k=10)
    >>> plt.scatter(pos[:,0], pos[:,1], c=rho, s=0.5)
    """
    tree = KDTree(np.asarray(points))
    distances, _ = tree.query(points, k=k+1)

    rk = distances[:, -1]

    n = points.shape[1]
    volume = ((np.pi**(n/2) / gamma(n/2 + 1)) * rk**n)
    rho = k / volume

    if log:
        rho = np.log10(rho)

    return rho
