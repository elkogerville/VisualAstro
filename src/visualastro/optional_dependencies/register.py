"""
Author: Elko Gerville-Reache
Date Created: 2026-07-17
Date Modified: 2026-07-27
Description:
    All optional dependencies are imported here.
    This centralizes the logic in visualastro for
    what packages are availble at runtime for a user.

    **For each optional dependency**

    Please fill out the `'flag'` and `'msg'` fields in
    `_OPTIONAL_DEPS` so that `_require_dependency` will
    recognize the dependency.
"""

from visualastro.optional_dependencies._dust_extinction import _DUST_EXTINCTION_DEP
from visualastro.optional_dependencies._specutils import _SPECUTILS_DEP
from visualastro.optional_dependencies._spectralcube import _SPECTRALCUBE_DEP


_OPTIONAL_DEPS = {
    'dust_extinction': _DUST_EXTINCTION_DEP,
    'spectral-cube': _SPECTRALCUBE_DEP,
    'specutils': _SPECUTILS_DEP,
}


def _require_dependency(*dependency: str) -> None:
    """
    Raise `ImportError` if `dependency` is not installed.

    Parameters
    ----------
    dependency : str
        Dependency name(s). Must be defined in `_OPTIONAL_DEPS`
        in `visualastro.optional_dependencies.register`.

    Raises
    ------
    ImportError :
        If `dependency` is not installed.
    ValueError :
        If `_OPTIONAL_DEPS.get(dependency)=None`.
    """
    for dep in dependency:
        dep_info = _OPTIONAL_DEPS.get(dep, None)
        if dep_info is None:
            raise ValueError(
                'Please specify an optional dependency!'
            )
        has_dependency: bool = dep_info['flag']
        if not has_dependency:
            msg: str = dep_info['msg']
            raise ImportError(msg)
