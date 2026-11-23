"""
Author: Elko Gerville-Reache
Date Created: 2026-11-22
Date Modified: 2026-11-22
Description:
    VisualAstro color registry base class.
"""

from collections.abc import Iterator, Mapping
from typing import Generic, Literal, TypeVar

T = TypeVar('T')


class _ColorRegistry(Generic[T]):

    _names: Mapping[str, T]
    _alias_names: dict[str, str]
    _registry: Mapping[str, T]
    _registry_type: Literal['color', 'colorset']

    def _resolve(self, name: str) -> str:
        """
        Resolve an alias to its canonical name.

        Raises
        ------
        KeyError
            If `name` is not a registered VisualAstro color/colorset or alias.
        """
        name = self._alias_names.get(name, name)
        if name not in self._names or name not in self._registry:
            raise KeyError(
                f'Unknown {self._registry_type} {name!r}. Available: {self._available}'
            )
        return name

    @property
    def _available(self) -> list[str]:
        """Retrieve all available registered colors / colorsets."""
        return [
            n for n in self._names
            if n in self._registry and n not in self._alias_names
        ]

    @property
    def _aliases(self) -> dict[str, str]:
        """Retrieve all alias names."""
        return {a: n for a, n in self._alias_names.items() if a in self._registry}

    def __getitem__(self, name: str) -> T:
        return self._registry[self._resolve(name)]

    def __contains__(self, name: str) -> bool:
        try:
            self._resolve(name)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self._available)

    def __len__(self) -> int:
        return len(self._available)

    def __repr__(self) -> str:
        names = ',\n'.join(f'    {name!r}' for name in sorted(self._available))
        return f'{type(self).__name__}([\n{names}\n])'
