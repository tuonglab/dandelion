"""
Backend API utility for dynamic import of polars or base modules/classes.
Usage:
    MyClass = import_backend_class('tools', 'MyClass')
    my_instance = MyClass(...)
"""

import importlib


def import_backend_class(module: str, class_name: str):
    """
    Try to import class from dandelion.polars first, fallback to dandelion.base.
    Args:
        module (str): Submodule name (e.g., 'tools', 'preprocessing')
        class_name (str): Class or function name to import
    Returns:
        type: Imported class or function
    """
    try:
        mod = importlib.import_module(f"dandelion.polars.{module}")
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
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
    try:
        return importlib.import_module(f"dandelion.polars.{module}")
    except ImportError:
        return importlib.import_module(f"dandelion.base.{module}")
