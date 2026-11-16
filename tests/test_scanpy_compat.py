"""Tests for Scanpy plotting import compatibility."""

import builtins
import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


COMPAT_PATH = (
    Path(__file__).resolve().parents[1] / "src/dandelion/utilities/_compat.py"
)


@pytest.mark.parametrize("legacy_available", [True, False])
def test_scanpy_plotting_import_layouts(monkeypatch, legacy_available):
    """Select the appropriate plotting imports for either Scanpy layout."""
    palettes = object()
    embedding = object()
    original_import = builtins.__import__
    imports = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Simulate the available Scanpy plotting layout."""
        if name.startswith("scanpy.plotting"):
            imports.append(name)
            if name.startswith("scanpy.plotting.legacy"):
                if not legacy_available:
                    raise ModuleNotFoundError(
                        "No module named 'scanpy.plotting.legacy'",
                        name="scanpy.plotting.legacy",
                    )
            elif legacy_available:
                pytest.fail(
                    "Used old Scanpy plotting layout when legacy exists"
                )
            return SimpleNamespace(palettes=palettes, embedding=embedding)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    compat = runpy.run_path(str(COMPAT_PATH))

    assert compat["palettes"] is palettes
    assert compat["embedding"] is embedding
    prefix = "scanpy.plotting.legacy" if legacy_available else "scanpy.plotting"
    assert imports[-2:] == [prefix, f"{prefix}._tools.scatterplots"]


def test_scanpy_plotting_does_not_hide_missing_dependencies(monkeypatch):
    """Propagate missing dependencies other than the legacy module."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Simulate an unrelated missing dependency during the legacy import."""
        if name == "scanpy.plotting.legacy":
            raise ModuleNotFoundError(
                "No module named 'missing_dependency'",
                name="missing_dependency",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        runpy.run_path(str(COMPAT_PATH))


@pytest.mark.parametrize("backend", ["base", "polars"])
def test_plotting_backends_use_compatible_scanpy_imports(backend):
    """Check that each backend uses the shared imports and can plot."""
    if backend == "polars":
        pytest.importorskip("polars")

    import matplotlib.pyplot as plt
    import numpy as np
    from anndata import AnnData

    from dandelion.utilities import _compat

    plotting = importlib.import_module(
        f"dandelion.{backend}.plotting._plotting"
    )
    assert plotting.palettes is _compat.palettes
    assert plotting.embedding is _compat.embedding

    adata = AnnData(np.ones((3, 1)))
    adata.obsm["X_vdj"] = np.array([[0, 0], [1, 0], [0, 1]])
    try:
        plotting.clone_network(adata, edges=False, show=False)
    finally:
        plt.close()
