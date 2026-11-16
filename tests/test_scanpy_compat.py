"""Tests for Scanpy compatibility."""

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


@pytest.mark.parametrize("modern", [True, False])
def test_pca_all_vars_supports_both_api_signatures(monkeypatch, modern):
    """Select the all-variable argument and forward PCA options."""
    from anndata import AnnData

    from dandelion.utilities import _compat

    calls = []

    def modern_pca(data, n_comps, *, mask="default", zero_center=True):
        """Simulate the Scanpy 1.13 PCA API."""
        calls.append((data, n_comps, mask, zero_center))

    def old_pca(data, n_comps, *, use_highly_variable=None, zero_center=True):
        """Simulate older Scanpy PCA APIs."""
        calls.append((data, n_comps, use_highly_variable, zero_center))

    monkeypatch.setattr(_compat.sc.pp, "pca", modern_pca if modern else old_pca)
    data = AnnData()
    _compat.pca_all_vars(data, n_comps=2, zero_center=False)
    assert calls == [(data, 2, None if modern else False, False)]


@pytest.mark.parametrize("arg", ["mask", "mask_var", "use_highly_variable"])
def test_pca_all_vars_rejects_variable_selection(arg):
    """Do not allow PCA options to override the all-variable invariant."""
    from anndata import AnnData

    from dandelion.utilities._compat import pca_all_vars

    with pytest.raises(TypeError, match=arg):
        pca_all_vars(AnnData(), n_comps=2, **{arg: True})


def test_pca_all_vars_propagates_pca_errors(monkeypatch):
    """Do not retry or hide errors raised by Scanpy PCA."""
    from anndata import AnnData

    from dandelion.utilities import _compat

    def failing_pca(data, n_comps, *, mask=None):
        """Simulate a PCA implementation that fails."""
        raise ValueError("PCA failed")

    monkeypatch.setattr(_compat.sc.pp, "pca", failing_pca)
    with pytest.raises(ValueError, match="PCA failed"):
        _compat.pca_all_vars(AnnData(), n_comps=2)


def test_pca_all_vars_rejects_unknown_api(monkeypatch):
    """Fail explicitly when Scanpy has no supported variable-mask argument."""
    from anndata import AnnData

    from dandelion.utilities import _compat

    def unsupported_pca(data, n_comps):
        """Simulate an unsupported PCA implementation."""
        pytest.fail("Unsupported PCA must not be called")

    monkeypatch.setattr(_compat.sc.pp, "pca", unsupported_pca)
    with pytest.raises(TypeError, match="Unsupported Scanpy PCA API"):
        _compat.pca_all_vars(AnnData(), n_comps=2)


def test_pca_all_vars_uses_all_variables():
    """Ignore the highly-variable annotation when running actual PCA."""
    import numpy as np
    from anndata import AnnData

    from dandelion.utilities._compat import pca_all_vars

    adata = AnnData(
        np.array([[1.0, 0, 1], [0, 1, 2], [1, 1, 0], [2, 0, 0]])
    )
    adata.var["highly_variable"] = [True, False, False]
    pca_all_vars(adata, n_comps=2)
    assert adata.obsm["X_pca"].shape == (4, 2)
    assert adata.varm["PCs"].shape == (3, 2)
