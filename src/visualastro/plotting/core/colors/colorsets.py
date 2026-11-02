"""
Author: Elko Gerville-Reache
Date Created: 2026-11-02
Date Modified: 2026-11-02
Description:
    VisualAstro colorset registry.
"""

from collections.abc import Callable, Iterator
from typing import Literal

import matplotlib as mpl

from visualastro.plotting.core.colors import (
    COLORSETS, COLORSET_ALIASES, plot_colors
)


class Colorsets:
    """Query and plot registered VisualAstro colorsets."""

    _SORT_KEYS: dict[str, Callable[[str, list], object]] = {
        'name': lambda name, colors: name.lower(),
        'length': lambda name, colors: len(colors),
    }

    def __init__(self):
        self._registry = mpl.color_sequences

    def _resolve(self, name: str) -> str:
        """
        Resolve an alias to its canonical colorset name.

        Raises
        ------
        KeyError
            If `name` is not a registered VisualAstro colorset or alias.
        """
        name = COLORSET_ALIASES.get(name, name)
        if name not in COLORSETS or name not in self._registry:
            raise KeyError(
                f'Unknown colorset {name!r}. Available: {self.available}'
            )
        return name

    def __getitem__(self, name: str) -> list:
        return self._registry[self._resolve(name)]

    def __contains__(self, name: str) -> bool:
        try:
            self._resolve(name)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.available)

    def __len__(self) -> int:
        return len(self.available)

    @property
    def available(self) -> list[str]:
        """
        Canonical names of the registered VisualAstro colorsets.

        Aliases and Matplotlib built-in colorsets (e.g. ``'tab10'``) are
        excluded. Use :attr:`aliases` to list alias names.

        Returns
        -------
        list of str
            Colorset names present in both ``COLORSETS`` and the Matplotlib
            color sequence registry, in registration order.

        See Also
        --------
        aliases : Mapping of alias names to canonical names.
        query : Filter and sort colorset names.
        """
        return [
            name for name in COLORSETS
            if name in self._registry and name not in COLORSET_ALIASES
        ]

    @property
    def aliases(self) -> dict[str, str]:
        """
        Mapping of colorset aliases to canonical names.

        Only aliases registered in the Matplotlib color sequence registry
        are included.

        Returns
        -------
        dict of str to str
            Keys are alias names, values are the canonical names in
            :attr:`available` they resolve to.

        See Also
        --------
        available : Canonical colorset names.
        """
        return {
            alias: name for alias, name in COLORSET_ALIASES.items()
            if alias in self._registry
        }

    def query(
        self,
        length : int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sort_by: Literal['name', 'length'] | None = None,
        reverse: bool = False,
    ) -> list[str]:
        """
        Filter and sort colorset names.

        Parameters
        ----------
        length : int | None, optional, default=None
            Absolute length of colors.
        min_length : int | None, optional, default=None
            Minimum number of colors (inclusive).
        max_length : int | None, optional, default=None
            Maximum number of colors (inclusive).
        sort_by : {'name', 'length'} | None, optional, default=None
            Sort criterion. If ``None``, registration order is kept.
        reverse : bool, optional, default=False
            Reverse the sort order.

        Returns
        -------
        names : list[str]
            Matching canonical colorset names.

        Raises
        ------
        ValueError
            If ``sort_by`` is not a valid criterion.
        """
        if length is not None and (min_length is not None or max_length is not None):
            raise ValueError(
                "'length' is mutually exclusive with 'min_length' and 'max_length'."
            )
        if (
            min_length is not None and max_length is not None
            and min_length > max_length
        ):
            raise ValueError(
                f'min_length ({min_length}) cannot exceed max_length ({max_length}).'
            )
        if sort_by is not None and sort_by not in self._SORT_KEYS:
            raise ValueError(
                f'sort_by must be one of {list(self._SORT_KEYS)}, got {sort_by!r}'
            )

        names = []
        for name in self.available:
            n = len(self._registry[name])
            if length is not None and n != length:
                continue
            if min_length is not None and n < min_length:
                continue
            if max_length is not None and n > max_length:
                continue
            names.append(name)

        if sort_by is not None:
            key = self._SORT_KEYS[sort_by]
            names.sort(key=lambda n: key(n, self._registry[n]), reverse=reverse)
        return names

    def plot(
        self,
        colorset: str | None = None,
        length : int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sort_by: Literal['name', 'length'] | None = None,
        reverse: bool = False,
        **kwargs
    ) -> None:
        """
        Plot the colorset(s), optionally with additional
        filters.

        Any of the filters in :meth:`query` are valid arguments
        if ``colorset`` is ``None``.

        Parameters
        ----------
        colorset : str | None, optional, default=None
            Colorset name or alias. If ``None``, plots all colorsets
            unless one or more of the filters are used.
        length : int | None, optional, default=None
            Absolute length of colors.
        min_length : int | None, optional, default=None
            Minimum number of colors (inclusive).
        max_length : int | None, optional, default=None
            Maximum number of colors (inclusive).
        sort_by : {'name', 'length'} | None, optional, default=None
            Sort criterion. If ``None``, registration order is kept.
        reverse : bool, optional, default=False
            Reverse the sort order.
        **kwargs
            Forwarded to `plot_colors`.

        Returns
        -------
        None
        """
        if colorset is not None:
            colorset = self._resolve(colorset)
            plot_colors(color=colorset, **kwargs)

        query = self.query(
            length=length,
            min_length=min_length,
            max_length=max_length,
            sort_by=sort_by,
            reverse=reverse,
        )
        plot_colors(color=query, **kwargs)

    def __repr__(self) -> str:
        names = ',\n'.join(f'    {name!r}' for name in self.available)
        return f'{type(self).__name__}([\n{names}\n])'
