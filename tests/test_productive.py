#!/usr/bin/env python
from itertools import cycle
import pytest

from dandelion.base.io import read_10x_vdj
from dandelion.base.tools import productive_ratio, vj_usage_pca
from dandelion.base.plotting import productive_ratio as plot_productive_ratio
from dandelion.base.preprocessing import check_contigs

from dandelion.polars.io import read_10x_vdj as read_10x_vdj_polars
from dandelion.polars.tools import productive_ratio as productive_ratio_polars
from dandelion.polars.tools import vj_usage_pca as vj_usage_pca_polars
from dandelion.polars.preprocessing import check_contigs as check_contigs_polars
from dandelion.polars.plotting import (
    productive_ratio as plot_productive_ratio_polars,
)


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x_mouse", "dummy_adata_mouse"
)
def test_productive_ratio(
    create_testfolder, annotation_10x_mouse, dummy_adata_mouse
):
    """test_productive_ratio"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(annot_file, index=False)
    vdj = read_10x_vdj(create_testfolder, filename_prefix="filtered")
    vdj.data["ambiguous"] = "F"
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in dummy_adata_mouse.obs_names]
    dummy_adata_mouse.obs["group"] = groups
    productive_ratio(dummy_adata_mouse, vdj, groupby="group", locus="IGH")
    assert "productive_ratio" in dummy_adata_mouse.uns
    productive_ratio(
        dummy_adata_mouse,
        vdj,
        groupby="group",
        locus="IGH",
        groups=["A", "B", "C"],
    )
    assert "productive_ratio" in dummy_adata_mouse.uns
    plot_productive_ratio(dummy_adata_mouse)


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_mouse")
def test_vj_usage_pca(create_testfolder, dummy_adata_mouse):
    """Test vj usage pca."""
    vdj = read_10x_vdj(create_testfolder, filename_prefix="filtered")
    _, adata = check_contigs(vdj, dummy_adata_mouse)
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in adata.obs_names]
    groups2 = [next(group) for i in adata.obs_names]
    adata.obs["group"] = groups
    adata.obs["group2"] = groups2
    new_adata = vj_usage_pca(
        adata,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata.obsm
    adata2 = adata.copy()
    new_adata2 = vj_usage_pca(
        adata2,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata2.obsm


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x_mouse", "dummy_adata_mouse"
)
def test_productive_ratio_polars(
    create_testfolder, annotation_10x_mouse, dummy_adata_mouse
):
    """test_productive_ratio"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(annot_file, index=False)
    vdj = read_10x_vdj_polars(create_testfolder, filename_prefix="filtered")
    vdj.data["ambiguous"] = "F"
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in dummy_adata_mouse.obs_names]
    dummy_adata_mouse.obs["group"] = groups
    productive_ratio_polars(
        dummy_adata_mouse, vdj, groupby="group", locus="IGH"
    )
    assert "productive_ratio" in dummy_adata_mouse.uns
    productive_ratio_polars(
        dummy_adata_mouse,
        vdj,
        groupby="group",
        locus="IGH",
        groups=["A", "B", "C"],
    )
    assert "productive_ratio" in dummy_adata_mouse.uns
    plot_productive_ratio_polars(dummy_adata_mouse)


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_mouse")
def test_vj_usage_pca_polars(create_testfolder, dummy_adata_mouse):
    """Test vj usage pca."""
    vdj = read_10x_vdj_polars(create_testfolder, filename_prefix="filtered")
    _, adata = check_contigs_polars(vdj, dummy_adata_mouse)
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in adata.obs_names]
    groups2 = [next(group) for i in adata.obs_names]
    adata.obs["group"] = groups
    adata.obs["group2"] = groups2
    new_adata = vj_usage_pca_polars(
        adata,
        vdj,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata.obsm
    adata2 = adata.copy()
    new_adata2 = vj_usage_pca_polars(
        adata2,
        vdj,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata2.obsm
