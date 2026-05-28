#!/usr/bin/env python
"""
Tests for dandelion.set_backend and dandelion._bind_backend_symbols.

Exercises the runtime backend-switching mechanism: verifying no-op behaviour,
alias normalisation, symbol rebinding on the top-level package, and correct
module provenance after each switch.
"""

import pytest
import dandelion as ddl
from dandelion._backend import get_backend, _set_backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_normalised() -> str:
    """Return 'pandas' or 'polars' regardless of the raw _BACKEND value."""
    return "pandas" if get_backend() == "pandas" else "polars"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_backend():
    """Restore the original backend after every test in this module."""
    original = _current_normalised()
    yield
    _set_backend(original)
    ddl._bind_backend_symbols()


# ---------------------------------------------------------------------------
# set_backend
# ---------------------------------------------------------------------------


def test_set_backend_noop_polars():
    """set_backend('polars') is a no-op when polars is already active."""
    _set_backend("polars")
    ddl._bind_backend_symbols()
    pl_before = ddl.pl
    ddl.set_backend("polars")
    assert ddl.pl is pl_before


def test_set_backend_noop_pandas():
    """set_backend('pandas') is a no-op when pandas is already active."""
    _set_backend("pandas")
    ddl._bind_backend_symbols()
    pl_before = ddl.pl
    ddl.set_backend("pandas")
    assert ddl.pl is pl_before


def test_set_backend_switch_to_base():
    """set_backend('base') activates the pandas backend."""
    ddl.set_backend("base")
    assert get_backend() == "pandas"


def test_set_backend_pandas_alias():
    """'pandas' is accepted as an alias for 'base'."""
    ddl.set_backend("pandas")
    assert get_backend() == "pandas"


def test_set_backend_switch_to_polars():
    """set_backend('polars') activates the polars backend."""
    ddl.set_backend("base")  # ensure we start from pandas
    ddl.set_backend("polars")
    assert get_backend() == "polars"


def test_set_backend_unknown_string_treated_as_polars():
    """Any unrecognised mode string is normalised to 'polars'."""
    ddl.set_backend("base")
    ddl.set_backend("something_unrecognised")
    assert get_backend() == "polars"


# ---------------------------------------------------------------------------
# _bind_backend_symbols — module provenance after switch
# ---------------------------------------------------------------------------


def test_bind_symbols_base_modules():
    """After switching to base, ddl.pp/tl/pl originate from dandelion.base.*"""
    ddl.set_backend("base")
    assert ddl.pl.__name__ == "dandelion.base.plotting"
    assert ddl.pp.__name__ == "dandelion.base.preprocessing"
    assert ddl.tl.__name__ == "dandelion.base.tools"


def test_bind_symbols_polars_modules():
    """After switching to polars, ddl.pp/tl/pl originate from dandelion.polars.*"""
    ddl.set_backend("base")  # ensure a real switch occurs
    ddl.set_backend("polars")
    assert ddl.pl.__name__ == "dandelion.polars.plotting"
    assert ddl.pp.__name__ == "dandelion.polars.preprocessing"
    assert ddl.tl.__name__ == "dandelion.polars.tools"


def test_bind_symbols_dandelion_class_base():
    """ddl.Dandelion resolves to the base Dandelion class after base switch."""
    from dandelion.base.core import Dandelion as BaseDandelion

    ddl.set_backend("base")
    assert ddl.Dandelion is BaseDandelion


def test_bind_symbols_dandelion_class_polars():
    """ddl.Dandelion resolves to the polars Dandelion class after polars switch."""
    from dandelion.polars.core import Dandelion as PolarsDandelion

    ddl.set_backend("base")
    ddl.set_backend("polars")
    assert ddl.Dandelion is PolarsDandelion


def test_bind_symbols_io_functions_present():
    """Core IO functions remain bound after a backend switch."""
    ddl.set_backend("base")
    for attr in (
        "read",
        "read_ddl",
        "read_h5ddl",
        "read_10x_airr",
        "read_airr",
    ):
        assert hasattr(
            ddl, attr
        ), f"ddl.{attr} missing after set_backend('base')"
    ddl.set_backend("polars")
    for attr in (
        "read",
        "read_ddl",
        "read_h5ddl",
        "read_10x_airr",
        "read_airr",
    ):
        assert hasattr(
            ddl, attr
        ), f"ddl.{attr} missing after set_backend('polars')"


def test_bind_symbols_roundtrip():
    """Switching base → polars → base restores the original base symbols."""
    ddl.set_backend("base")
    pl_base = ddl.pl
    ddl.set_backend("polars")
    ddl.set_backend("base")
    assert ddl.pl is pl_base
