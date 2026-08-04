#!/usr/bin/env python
from dandelion.logging import (
    __author__,
    __email__,
    __classifiers__,
    __version__,
)
from dandelion import logging

from ._backend import (
    import_backend_class,
    import_backend_module,
    _set_backend,
    get_backend,
)


def _bind_backend_symbols():
    """(Re)bind all backend-dependent symbols on the dandelion package."""
    import sys
    import dandelion as _ddl

    _ddl.pp = import_backend_module("preprocessing")
    _ddl.tl = import_backend_module("tools")
    _ddl.pl = import_backend_module("plotting")
    _ddl.Dandelion = import_backend_class("core", "Dandelion")
    _ddl.read = import_backend_class("io", "read")
    _ddl.read_ddl = import_backend_class("io", "read_ddl")
    _ddl.read_h5ddl = import_backend_class("io", "read_h5ddl")
    try:
        _ddl.read_zipddl = import_backend_class("io", "read_zipddl")
    except AttributeError:
        pass
    _ddl.read_10x_airr = import_backend_class("io", "read_10x_airr")
    _ddl.read_10x_vdj = import_backend_class("io", "read_10x_vdj")
    _ddl.read_airr = import_backend_class("io", "read_airr")
    _ddl.read_bd_airr = import_backend_class("io", "read_bd_airr")
    _ddl.read_parse_airr = import_backend_class("io", "read_parse_airr")
    _ddl.read_seekgene_vdj = import_backend_class("io", "read_seekgene_vdj")


def set_backend(mode: str) -> None:
    """Override the active backend at runtime.

    Parameters
    ----------
    mode : str
        ``'polars'`` to use the Polars-based backend, or ``'base'`` / ``'pandas'``
        to use the pandas-based backend.  If the requested mode is already
        active, this is a no-op.

    Examples
    --------
    >>> import dandelion as ddl
    >>> ddl.set_backend(mode="base")    # switch to pandas backend
    >>> ddl.set_backend(mode="polars")  # switch to polars backend
    """
    normalized = "pandas" if mode.lower() in ("base", "pandas") else "polars"
    current = "pandas" if get_backend() == "pandas" else "polars"
    if normalized == current:
        return
    _set_backend(normalized)
    _bind_backend_symbols()


# Dynamically import modules/classes using backend manager
pp = import_backend_module("preprocessing")
tl = import_backend_module("tools")
pl = import_backend_module("plotting")

# Import main API symbols
Dandelion = import_backend_class("core", "Dandelion")
read = import_backend_class("io", "read")
read_ddl = import_backend_class("io", "read_ddl")
read_h5ddl = import_backend_class("io", "read_h5ddl")
try:
    read_zipddl = import_backend_class("io", "read_zipddl")
    __all__ = ["read_zipddl"]
except AttributeError:
    __all__ = []
    pass
read_10x_airr = import_backend_class("io", "read_10x_airr")
read_10x_vdj = import_backend_class("io", "read_10x_vdj")
read_vdj = read_10x_vdj
read_airr = import_backend_class("io", "read_airr")
read_bd_airr = import_backend_class("io", "read_bd_airr")
read_parse_airr = import_backend_class("io", "read_parse_airr")
read_seekgene_vdj = import_backend_class("io", "read_seekgene_vdj")


__all__ += [
    "__author__",
    "__classifiers__",
    "__email__",
    "__version__",
    "Dandelion",
    "get_backend",
    "logging",
    "set_backend",
    "pl",
    "pp",
    "read_airr",
    "read_ddl",
    "read_10x_airr",
    "read_10x_vdj",
    "read_parse_airr",
    "read_bd_airr",
    "read_seekgene_vdj",
    "read_vdj",
    "tl",
    "utl",
]
