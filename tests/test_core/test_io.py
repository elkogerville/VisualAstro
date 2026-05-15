"""
Author: Elko Gerville-Reache
Date Created: 2026-05-11
Date Modified: 2026-05-11
Description:
    Tests for io module.
Dependencies:
    - pytest
"""

from types import SimpleNamespace
import pytest

from visualastro.core.config import config, _UNSET
from visualastro.core.io import _kwarg, _param, _pop_kwargs, _resolve_kwargs


class TestKwargs:

    def test_params(self):
        """Test _params utility function"""
        param = _param('name', 2, 3)
        assert isinstance(param, tuple)
        assert param[0] == 'name' and param[1] == 2 and param[2] == 3
        with pytest.raises(ValueError):
            _param(1, 2, 3)

    def test_pop_kwargs(self):
        """Test _pop_kwargs utility function"""
        kwargs = {
            'color': 'r',
            'lw': 2,
            'ls': '--',
        }
        value = _pop_kwargs(kwargs, 'linewidth', config.linewidth)

        assert kwargs == {'color': 'r', 'ls': '--'}
        assert value == 2

        value = _pop_kwargs(kwargs, 'nonsense', config.style)
        assert kwargs == {'color': 'r', 'ls': '--'}
        assert value == config.style

        kwargs = {'test': 1}
        value = _pop_kwargs(kwargs, 'test', None)
        assert value == 1
        assert kwargs == {}

class TestResolveKwargs:
    def test_param_assignment_from_default(self):
        linestyle = _UNSET
        kwargs = {'lw': 2, 'xpad': 5}
        params = [_param('linestyle', linestyle, 'solid')]
        out = _resolve_kwargs(kwargs, params)

        assert kwargs == {'lw': 2, 'xpad': 5}
        assert out.linestyle == 'solid'

    def test_param_assignment_from_kwargs(self):
        kwargs = {'ls': '--', 'xpad': 5}
        params = [_param('linestyle', _UNSET, 'solid')]
        out = _resolve_kwargs(kwargs, params)

        assert kwargs == {'xpad': 5}
        assert out.linestyle == '--'

    def test_param_assignment(self):
        kwargs = {'xpad': 5}
        params = [_param('linestyle', '-.', 'solid')]
        out = _resolve_kwargs(kwargs, params)

        assert kwargs == {'xpad': 5}
        assert out.linestyle == '-.'

    def test_multiple_param_assignment(self):
        kwargs = {'ls': '--', 'alpha': 0.5}

        out = _resolve_kwargs(
            kwargs,
            [
                _param('linestyle', _UNSET, 'solid'),
                _param('alpha', _UNSET, 1.0),
            ]
        )

        assert kwargs == {}
        assert out.linestyle == '--'
        assert out.alpha == 0.5

    def test_none_does_not_resolve_to_default(self):
        kwargs = {}

        out = _resolve_kwargs(
            kwargs,
            [_param('label', None, 'default')]
        )

        assert out.label is None

    def test_additional_kwarg_assignment(self):
        kwargs = {'label': 'data'}

        out = _resolve_kwargs(
            kwargs,
            additional_kwargs=[_kwarg('label', None)]
        )

        assert kwargs == {}
        assert out.label == 'data'

    def test_additional_kwarg_default(self):
        kwargs = {}

        out = _resolve_kwargs(
            kwargs,
            additional_kwargs=[_kwarg('label', 'default')]
        )

        assert kwargs == {}
        assert out.label == 'default'

    def test_unrelated_kwargs_preserved(self):
        kwargs = {'meow': 1, 'omghi': 2}

        _resolve_kwargs(
            kwargs,
            [_param('linestyle', _UNSET, 'solid')]
        )

        assert kwargs == {'meow': 1, 'omghi': 2}

    def test_consumed_alias_removed(self):
        kwargs = {'ls': '--'}

        _resolve_kwargs(
            kwargs,
            [_param('linestyle', _UNSET, 'solid')]
        )

        assert 'ls' not in kwargs

        kwargs = {'ls': '--'}

        _resolve_kwargs(
            kwargs,
            additional_kwargs=[_kwarg('linestyle', 'solid')]
        )

        assert 'ls' not in kwargs

    def test_returns_namespace(self):
        out = _resolve_kwargs(
            {},
            [_param('alpha', _UNSET, 1.0)]
        )

        assert isinstance(out, SimpleNamespace)
        assert out.alpha == 1.0

    def test_name_assignment(self):
        name = 'intruderalert'
        out = _resolve_kwargs(
            {},
            [_param(name, 'redspyisinthebase', [1,1,1,'uhhhh',1])]
        )
        assert out.intruderalert == 'redspyisinthebase'

    def test_copy_kwargs(self):

        kwargs = {
            'alpha': 0.5,
            'color': 'red',
            'label': 'test'
        }

        result = _resolve_kwargs(
            kwargs,
            params=[_param('alpha', _UNSET, 0.8)],
            additional_kwargs=[_kwarg('color', 'blue')],
            copy_kwargs=[_kwarg('label', None)]
        )

        assert result.alpha == 0.5, f"Expected alpha=0.5, got {result.alpha}"
        assert result.color == 'red', f"Expected color='red', got {result.color}"
        assert result.label == 'test', f"Expected label='test', got {result.label}"

        assert 'alpha' not in kwargs, 'alpha should be popped from kwargs'
        assert 'color' not in kwargs, 'color should be popped from kwargs'
        assert kwargs['label'] == 'test', f"label should remain in kwargs, got {kwargs.get('label')}"

        kwargs = {
            'alpha': 0.5,
            'color': 'red',
            'label': 'test'
        }
        kwargs_orig = kwargs.copy()

        result = _resolve_kwargs(
            kwargs,
            copy_kwargs=[
                _kwarg('alpha', None),
                _kwarg('color', None),
                _kwarg('label', None),
            ]
        )

        assert kwargs == kwargs_orig

    def test_empty_call_raises(self):
        with pytest.raises(ValueError):
            _resolve_kwargs({})
