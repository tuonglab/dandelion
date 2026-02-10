#!/usr/bin/env python
from setuptools_scm import get_version

try:
    __version__ = get_version().split("+")[0]
except LookupError:  # prama: no cover
    from importlib.metadata import version

    __version__ = version("sc-dandelion").split("+")[0]

__author__ = "Zewen Kelvin Tuong"
__email__ = "z.tuong@uq.edu.au"
__classifiers__ = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: R",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
]

__all__ = [
    "__author__",
    "__email__",
    "__classifiers__",
    "__version__",
]
