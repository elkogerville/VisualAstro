"""
Author: Elko Gerville-Reache
Date Created: 2026-06-29
Date Modified: 2026-10-10
Description:
    Functions for kwargs and aliases within visualastro.
"""

from types import SimpleNamespace
from typing import Any

from visualastro.core.config import _Unset, _UNSET, _resolve_default


# KWARGS
# ------
# define aliases here! each key represents a parameter used in visualastro functions,
# and their values are aliases of that parameter. this way any each set can be used
# in a function, provided that the function uses the `_resolve_kwargs` interface.
# ensure that the values are tuples even for single items, meaning: 'key': (value,),
KWARG_ALIASES: dict['str', tuple[str, ...]] = {
    'color': ('colors',),
    'edgecolor': ('edgecolors', 'ec'),
    'facecolor': ('facecolors', 'fc'),
    'marker': ('markers', 'm'),
    'size': ('sizes', 's'),
    'alpha': ('alphas', 'a'),
    'markeredgecolor': ('markeredgecolors', 'mec'),
    'limits': ('limit', 'lims', 'lim'),
    'label': ('labels', 'l'),
    'bbox_to_anchor': ('bbox',),
    'legend_handles': ('legend_handle',),
    'legend_labels': ('legend_label',),
    'linecolor': ('linecolors', 'lc'),
    'linestyle': ('linestyles', 'ls'),
    'linewidth': ('linewidths', 'lw'),
    'linealpha': ('linealphas', 'la'),
    'array_order': ('order',),
    'colorbar': ('colorbars', 'cbar', 'cbars'),
    'cbar_width': ('colorbar_width',),
    'cbar_pad': ('colorbar_pad',),
    'cbar_label': ('colorbar_label', 'colorbar_labels', 'cbar_labels'),
    'cbar_tick_which': ('colorbar_tick_which',),
    'cbar_tick_dir': ('colorbar_tick_dir', 'colorbar_tick_direction', 'cbar_tick_direction'),
    'cbar_tick_labelsize': (
        'colorbar_tick_labelsize',
        'colorbar_tick_fontsize',
        'cbar_tick_fontsize'
    ),
    'gridlines': ('gridline', 'grid_line', 'grid_lines', 'grid'),
    'ellipses': ('ellipse',),
    'points': ('point',),
    'text_color': ('textcolor',),
    'unit_fmt': ('unit_format', 'unit_label_fmt', 'unit_label_format'),
    'axis_style': ('axes_style',),
    'plot_contours': ('plot_contour',),
    'plot_contour_offset': ('plot_contour_offsets', 'contour_offsets', 'contour_offset'),
    'extract_mode': ('how',),
    'zorder': ('zorders', 'z'),
    'vlines': ('vline',),
    'hlines': ('hline',),
}

# reverse mapping of aliases to canonical key
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in KWARG_ALIASES.items()
    for alias in aliases
}


ParamSpec = tuple[str, Any, Any]
KwargSpec = tuple[str, Any]


def _resolve_kwargs(
    kwargs: dict,
    params: list[ParamSpec] | None = None,
    additional_kwargs: list[KwargSpec] | None = None,
    copy_kwargs: list[KwargSpec] | None = None
) -> SimpleNamespace:
    """
    Resolve keyword arguments into a namespace of normalized parameters.

    `params` follow the form: [_param('name', var, default), ...].
    `additional_kwargs` follow the form: [_kwarg('name', default), ...].

    Parameters defined in `params` are intended for arguments that also
    exist in the parent function signature. Their values are first processed
    through `_pop_kwargs` to handle aliases, then passed through
    `_resolve_default` so that `_UNSET` values are replaced with their
    configured defaults.

    Parameters defined in `additional_kwargs` are intended for optional
    keyword-only passthrough arguments that are not part of the parent
    function signature. These are resolved using `_pop_kwargs`.

    Both `params` and `additional_kwargs` are aware to aliases defined
    in `visualastro.core.io.KWARG_ALIASES`.

    The resolved values are returned as attributes on a
    :class:`types.SimpleNamespace`.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments to resolve. Resolved parameters
        are popped in place.
    params : list[ParamSpec] | None, optional, default=None
        Sequence of `(name, value, default)` tuples describing parameters
        defined in the parent function signature.

        Each parameter is resolved as:

            1. Retrieve the value from `kwargs` using `_pop_kwargs`
            2. Replace unset sentinel values using `_resolve_default`

    additional_kwargs : list[KwargSpec] | None, optional, default=None
        Sequence of `(name, default)` tuples describing optional keyword
        arguments that should be retrieved directly from `kwargs` using
        fallback defaults.
    copy_kwargs : list[KwargSpec] | None, optional, default=None
        Sequence of `(name, default)` tuples describing keyword arguments
        to preserve in `kwargs` without popping.

        Values are retrieved from `kwargs` using `_get_kwargs` (non-mutating),
        allowing the original key-value pairs to remain available after
        resolution. Useful when downstream functions need access to
        arguments that are also resolved into the returned namespace.

        Aliases defined in `KWARG_ALIASES` are respected during lookup.

    Returns
    -------
    types.SimpleNamespace
        Namespace containing all resolved parameters as attributes.

    Raises
    ------
    ValueError :
        If `params`, `additional_kwargs`, and `copy_kwargs` are None.

    Examples
    --------
    >>> params = _resolve_kwargs(
    ...     kwargs,
    ...     [
    ...         _param('alpha', alpha, config.alpha),
    ...         _param('color', color, config.color),
    ...     ],
    ...     [
    ...         _kwarg('label', None),
    ...         _kwarg('cmap', config.cmap),
    ...     ]
    ... )
    >>>
    >>> params.alpha
    0.8
    >>> params.cmap
    'viridis'

    Notes
    -----
    `params` should be used for arguments originating from the function
    signature, especially when `_UNSET` or aliases must be handled.

    `params`, `additional_kwargs`, and `copy_kwargs` do not need aliases defined.

    `additional_kwargs` should be used for optional passthrough keyword
    arguments that behave like standard `kwargs.pop` retrievals.

    See Also
    --------
    visualastro.core.io._pop_kwargs
    visualastro.core.config._resolve_default
    visualastro.core.io._param
    visualastro.core.io._kwarg
    """
    if params is None and additional_kwargs is None and copy_kwargs is None:
        raise ValueError(
            'params, additional_kwargs, and copy_kwargs cannot all be None!'
        )

    out = {}

    if copy_kwargs is not None:
        for name, default in copy_kwargs:
            out[name] = _get_kwargs(kwargs, name, default)

    if params is not None:
        for name, value, default in params:
            value = _pop_kwargs(kwargs, name, value)
            out[name] = _resolve_default(value, default)

    if additional_kwargs is not None:
        for name, default in additional_kwargs:
            if name in out:
                continue
            out[name] = _pop_kwargs(kwargs, name, default)

    return SimpleNamespace(**out)


def _get_kwargs(
    kwargs: dict[str, Any],
    name: str,
    default: Any = None
) -> Any:
    """
    Retrieve a keyword argument by canonical name or registered alias.

    Identical to `_pop_kwargs` but does not mutate `kwargs`.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments to query.
    name : str
        Canonical name corresponding to a key in `KWARG_ALIASES`.
    default : Any, optional
        Value returned if `name` and all aliases absent from `kwargs`.
        Default is None.

    Returns
    -------
    Any
        Value associated with `name` or its first matched alias.
        If no match found, returns `default`.
    """
    for key in (name, *KWARG_ALIASES.get(name, ())):
        if (value := kwargs.get(key, _UNSET)) is not _UNSET:
            return value

    return default


def _pop_kwargs(
    kwargs: dict[str, Any],
    name: str,
    default: Any = None
) -> Any:
    """
    Pop a keyword argument by canonical name or registered alias.

    Searches `kwargs` for the canonical `name` or any of its registered
    aliases (from `KWARG_ALIASES` in `visualastro.core.io`), removes the
    first match found, and returns its value.

    Identical to `_get_kwargs` but does mutate `kwargs`.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments to mutate.
    name : str
        Canonical name corresponding to a key in `KWARG_ALIASES`.
    default : Any, optional
        Value returned if `name` and all aliases absent from `kwargs`.
        Default is None.

    Returns
    -------
    Any
        Value associated with `name` or its first matched alias.
        If no match found, returns `default`.

    Notes
    -----
    Mutates `kwargs` by removing the matched key. Search order is:
    canonical name first, then aliases in order defined in `KWARG_ALIASES`.
    """
    for key in (name, *KWARG_ALIASES.get(name, ())):
        if (value := kwargs.get(key, _UNSET)) is not _UNSET:
            kwargs.pop(key)
            return value

    return default


def _pop_prefixed(
    kwargs: dict,
    prefix: str,
    *,
    base: str | None = None,
    default: Any | _Unset = _UNSET
) -> dict:
    """
    Pop all keys starting with `prefix` and strip the prefix.

    Each matched key is stripped of `prefix`, and the remainder is mapped to
    its canonical name via `_ALIAS_TO_CANONICAL` (built from `KWARG_ALIASES`).
    Stripped names that are not registered aliases are kept unchanged.


    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments. Mutated in place: all keys starting
        with `prefix`, and `base` (including its aliases) if given, are
        removed.
    prefix : str
        Prefix to match, including the trailing underscore (e.g. `'legend_'`).
    base : str | None, optional, default=None
        Canonical name of an unprefixed kwarg to pop alongside the prefixed
        ones (e.g. `'legend'` for `legend=True` next to `legend_loc=...`).
        Alias resolution is delegated to `_pop_kwargs`.
    default : Any | _Unset, optional, default=_UNSET
        Value used for `base` when it is absent from `kwargs`. If `_UNSET`,
        `base` is omitted from the output when absent. Ignored if `base` is
        None.


    Returns
    -------
    dict[str, Any]
        Mapping of canonical (unprefixed) name to value.

    Notes
    -----
    If several matched keys resolve to the same canonical name (e.g.
    `legend_bbox` and `legend_bbox_to_anchor`), the first one encountered in
    `kwargs` iteration order is kept. All matched keys are removed from
    `kwargs` regardless.

    Examples
    --------
    >>> kwargs = {'legend_bbox': (0.5, 1), 'legend_loc': 'best', 'lw': 2}
    >>> _pop_prefixed(kwargs, 'legend_')
    {'bbox_to_anchor': (0.5, 1), 'loc': 'best'}
    >>> kwargs
    {'lw': 2}

    With `base`:

    >>> kwargs = {'legend': True, 'legend_loc': 'best'}
    >>> _pop_prefixed(kwargs, 'legend_', base='legend')
    {'loc': 'best', 'legend': True}
    >>> kwargs
    {}
    """
    out = {}
    for key in [k for k in kwargs if k.startswith(prefix)]:
        stripped = key[len(prefix):]
        canonical = _ALIAS_TO_CANONICAL.get(stripped, stripped)
        out.setdefault(canonical, kwargs.pop(key))

    if isinstance(base, str):
        value = _pop_kwargs(kwargs, base, default)
        if value is not _UNSET:
            out[base] = value

    return out


def _pop_mapped(
    kwargs: dict,
    keys: tuple[str, ...],
    *,
    prefix: str | None = None,
    base: str | None = None,
    default: Any | _Unset = _UNSET
) -> dict:
    """
    Pop a set of kwargs, including prefixed and aliases.
    Absent keys are omitted from the returned dict.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments. Mutated in place.
    keys : tuple[str, ...]
        Canonical unprefixed names to pop. Aliases resolved via `_pop_kwargs`.
    prefix : str | None, optional, default=None
        If given, additionally pop all keys starting with `prefix` via
        `_pop_prefixed` (e.g. `'legend_'`).
    base : str | None, optional, default=None
        Canonical name of an unprefixed kwarg to pop with the prefixed group.
        Requires `prefix`.
    default : Any | _Unset, optional, default=_UNSET
        Value for `base` when absent. Applies to `base` only, never to
        `keys`. Requires `prefix`.

    Returns
    -------
    dict[str, Any]
        Only user-supplied entries, keyed by canonical name.

    Examples
    --------
    >>> kwargs = {'xlim': (0, 1), 'ypad': 0.1, 'color': 'red'}
    >>> _pop_mapped(kwargs, ('limits', 'xlim', 'ylim', 'xpad', 'ypad'))
    {'xlim': (0, 1), 'ypad': 0.1}
    >>> kwargs
    {'color': 'red'}

    Absent keys are omitted and not filled with a default:

    >>> kwargs = {'highlight': True}
    >>> _pop_mapped(kwargs, ('ellipses', 'plot_ellipse', 'highlight', 'text_loc'))
    {'highlight': True}

    With a prefix group:

    >>> kwargs = {'xlim': (0, 1), 'legend_loc': 'best', 'lw': 2}
    >>> _pop_mapped(kwargs, ('xlim',), prefix='legend_')
    {'xlim': (0, 1), 'loc': 'best'}
    >>> kwargs
    {'lw': 2}
    """
    out = {}
    for key in keys:
        value = _pop_kwargs(kwargs, key, _UNSET)
        if value is not _UNSET:
            out[key] = value

    if isinstance(prefix, str):
        out |= _pop_prefixed(kwargs, prefix, base=base, default=default)

    return out


def _extract_kwargs(
    kwargs: dict,
    params: list[ParamSpec] | None = None,
    additional_kwargs: list[KwargSpec] | None = None,
    copy_kwargs: list[KwargSpec] | None = None
) -> dict:
    """
    Helper function for to return the output of _resolve_kwargs
    as a `dict` instead of `SimpleNamespace`.

    See `visualastro.core.io._resolve_kwargs` for documentation.
    """
    return vars(_resolve_kwargs(kwargs, params, additional_kwargs, copy_kwargs))


def _param(name: str, value: Any, default: Any) -> ParamSpec:
    """
    Helper function for defining a parameter in `_resolve_kwargs`.

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : Any
        Value of the parameter.
    default : Any
        Fallback value if `value` is `_UNSET`.

    Returns
    -------
    tuple :
        Returns a tuple of name, value, and default, unchanged.
    """
    if not isinstance(name, str):
        raise ValueError(
            f'name must be a str! got: {type(name).__name__}'
        )
    return (name, value, default)


def _kwarg(name: str, default: Any) -> KwargSpec:
    if not isinstance(name, str):
        raise ValueError(
            f'name must be a str! got: {type(name).__name__}'
        )
    return (name, default)
