"""
Author: Elko Gerville-Reache, Audrey Whitmer
Date Created: 2026-07-17
Date Modified: 2026-07-27
Description:
    All optional dependencies are registered here.
    To register an optional dependency in VisualAstro,
    add it to `_OPTIONAL_DEPS`. Each optional dependency
    should have its own file.
"""
import warnings

from visualastro.optional_dependencies._colorspacious import _COLORSPACIOUS_DEP
from visualastro.optional_dependencies._dust_extinction import _DUST_EXTINCTION_DEP
from visualastro.optional_dependencies._regions import _REGIONS_DEP
from visualastro.optional_dependencies._specutils import _SPECUTILS_DEP
from visualastro.optional_dependencies._spectralcube import _SPECTRALCUBE_DEP
from visualastro.optional_dependencies._tqdm import _TQDM_DEP


_OPTIONAL_DEPS = {
    'colorspacious': _COLORSPACIOUS_DEP,
    'dust_extinction': _DUST_EXTINCTION_DEP,
    'regions': _REGIONS_DEP,
    'spectral-cube': _SPECTRALCUBE_DEP,
    'specutils': _SPECUTILS_DEP,
    'tqdm': _TQDM_DEP,
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

def _offer_dependency(*dependency: str) -> None:
    """
    Raise `ImportWarning` if `dependency` is not installed.

    Parameters
    ----------
    dependency : str
        Dependency name(s). Must be defined in `_OPTIONAL_DEPS`
        in `visualastro.optional_dependencies.register`.

    Raises
    ------
    ImportWarning :
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
            warnings.warn(msg, stacklevel=2)
