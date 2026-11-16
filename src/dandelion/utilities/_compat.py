"""Compatibility helpers for supported Scanpy versions."""

from inspect import signature

import scanpy as sc
from anndata import AnnData

try:
    from scanpy.plotting.legacy import palettes
    from scanpy.plotting.legacy._tools.scatterplots import embedding
except ModuleNotFoundError as e:
    if e.name != "scanpy.plotting.legacy":
        raise
    from scanpy.plotting import palettes
    from scanpy.plotting._tools.scatterplots import embedding


def pca_all_vars(adata: AnnData, n_comps: int, **kwargs) -> None:
    """Run Scanpy PCA using all variables across supported API versions."""
    for arg in ("use_highly_variable", "mask_var", "mask"):
        if arg in kwargs:
            raise TypeError(f"vj_usage_pca does not accept {arg!r}")

    parameters = signature(sc.pp.pca).parameters
    if "mask" in parameters:
        sc.pp.pca(adata, n_comps=n_comps, mask=None, **kwargs)
    elif "use_highly_variable" in parameters:
        sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=False, **kwargs)
    else:
        raise TypeError("Unsupported Scanpy PCA API: no variable mask argument")
