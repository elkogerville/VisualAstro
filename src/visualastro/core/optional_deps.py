"""
Author: Elko Gerville-Reache
Date Created: 2026-07-17
Date Modified: 2026-07-17
Description:
    All optional dependencies are imported here.
    This centralizes the logic in visualastro for
    what packages are availble at runtime for a user.

    **For each optional dependency**

    Please fill out the `'flag'` and `'msg'` fields in
    `_OPTIONAL_DEPS` so that `_require_dependency` will
    recognize the dependency.
"""

try:
    from spectral_cube import SpectralCube
    from spectral_cube.lower_dimensional_structures import Slice
    from spectral_cube.wcs_utils import strip_wcs_from_header
    _HAS_SPECTRAL_CUBE = True
except ImportError:
    SpectralCube = None
    Slice = None
    _HAS_SPECTRAL_CUBE = False


_OPTIONAL_DEPS = {
    'spectral-cube': {
        'flag': _HAS_SPECTRAL_CUBE,
        'msg': """
            spectral-cube is required for this function.
            Install via:
                CONDA :
                    $ conda install conda-forge::spectral-cube
                PIP :
                    $ pip install spectral-cube
        """
    }
}


def _require_dependency(dependency) -> None:
    """
    Raise `ImportError` if `dependency` is not installed.

    Parameters
    ----------
    dependency : str
        Dependency name. Must be defined in `_OPTIONAL_DEPS`
        in `core.optional_deps._OPTIONAL_DEPS`.

    Raises
    ------
    ImportError :
        If `dependency` is not installed.
    ValueError :
        If `_OPTIONAL_DEPS.get(dependency)=None`.
    """
    dep_info = _OPTIONAL_DEPS.get(dependency, None)
    if dep_info is None:
        raise ValueError(
            'Please specify an optional dependency!'
        )
    has_dependency: bool = dep_info['flag']
    msg: str = dep_info['msg']
    if not has_dependency:
        raise ImportError(msg)
