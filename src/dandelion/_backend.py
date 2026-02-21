"""
Backend API utility for dynamic import of polars or base modules/classes.
Usage:
    MyClass = import_backend_class('tools', 'MyClass')
    my_instance = MyClass(...)

Set the environment variable ``DANDELION_BACKEND`` to ``"pandas"`` to force
the base (pandas) backend even when polars is installed.
"""

import importlib
import os

_BACKEND = os.environ.get("DANDELION_BACKEND", "auto").lower()


def import_backend_class(module: str, class_name: str):
    """
    Try to import class from dandelion.polars first, fallback to dandelion.base.
    Args:
        module (str): Submodule name (e.g., 'tools', 'preprocessing')
        class_name (str): Class or function name to import
    Returns:
        type: Imported class or function
    """
    if _BACKEND != "pandas":
        mod = importlib.import_module(f"dandelion.polars.{module}")
    else:
        mod = importlib.import_module(f"dandelion.base.{module}")
    return getattr(mod, class_name)


def import_backend_module(module: str):
    """
    Try to import module from dandelion.polars first, fallback to dandelion.base.
    Args:
        module (str): Submodule name (e.g., 'tools', 'utilities')
    Returns:
        module: Imported module
    """
    if _BACKEND != "pandas":
        return importlib.import_module(f"dandelion.polars.{module}")
    return importlib.import_module(f"dandelion.base.{module}")
