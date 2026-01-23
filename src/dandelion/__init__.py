#!/usr/bin/env python
from dandelion.logging import (
    __author__,
    __email__,
    __classifiers__,
    __version__,
)
from dandelion import logging

from ._backend import import_backend_class, import_backend_module

# Dynamically import modules/classes using backend manager
pp = import_backend_module("preprocessing")
utl = import_backend_module("utilities")
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
read_airr = import_backend_class("io", "read_airr")
read_bd_airr = import_backend_class("io", "read_bd_airr")
read_parse_airr = import_backend_class("io", "read_parse_airr")


__all__ += [
    "__author__",
    "__classifiers__",
    "__email__",
    "__version__",
    "Dandelion",
    "logging",
    "pl",
    "pp",
    "read_airr",
    "read_ddl",
    "read_10x_airr",
    "read_10x_vdj",
    "read_parse_airr",
    "read_bd_airr",
    "read_ddl",
    "tl",
    "utl",
]
