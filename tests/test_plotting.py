#!/usr/bin/env python
import pytest
import scanpy as sc

from dandelion.base.preprocessing import check_contigs
from dandelion.base.core import Dandelion
from dandelion.base.tools import generate_network, transfer
from dandelion.base.io import read_h5ddl
from dandelion.base.plotting import (
    clone_bubbleplot,
    clone_network,
    barplot,
    stackedbarplot,
    spectratype,
)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated", "dummy_adata")
def test_setup(create_testfolder, airr_reannotated, dummy_adata):
    """test_setup"""
    vdj, adata = check_contigs(airr_reannotated, dummy_adata)
    assert airr_reannotated.shape[0] == 8
    assert vdj._data.shape[0] == 8
    assert vdj._metadata.shape[0] == 5
    assert adata.n_obs == 5
    vdj._data["clone_id"] = ["A", "A", "A", "A", "A", "A", "A", "A"]
    vdj = Dandelion(vdj._data)
    generate_network(vdj, layout_method="mod_fr")
    transfer(adata, vdj)
    assert "clone_id" in adata.obs
    assert "X_vdj" in adata.obsm
    f1 = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_h5ddl(f1)
    adata.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_plot_network(create_testfolder):
    """test_plot_network"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    clone_network(adata, color=["isotype"], show=False, return_fig=False)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "sort,norm",
    [
        pytest.param(True, True),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(False, False),
    ],
)
def test_plot_bar(create_testfolder, sort, norm):
    """test_plot_bar"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    ax = barplot(vdj, color="v_call_VDJ")
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", sort_descending=sort)
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", normalize=norm)
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_bar2(create_testfolder):
    """test_plot_bar2"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = barplot(
        vdj,
        color="v_call_VDJ",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
    )
    assert ax is not None
    ax = barplot(adata, color="v_call_VDJ")
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar(create_testfolder, norm):
    """test_plot_stackedbar"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    ax = stackedbarplot(
        vdj, color="v_call_VDJ", groupby="isotype", normalize=norm
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar2(create_testfolder, norm):
    """test_plot_stackedbar2"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = stackedbarplot(
        vdj,
        color="v_call_VDJ",
        groupby="isotype",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
        normalize=norm,
    )
    assert ax is not None
    ax = stackedbarplot(
        adata,
        color="v_call_VDJ",
        groupby="isotype",
        normalize=norm,
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_spectratype(create_testfolder):
    """test_plot_spectratype"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    ax = spectratype(
        vdj, color="junction_length", groupby="c_call", locus="IGH"
    )
    assert ax is not None
    ax = spectratype(
        vdj,
        color="junction_length",
        groupby="c_call",
        locus="IGH",
        hide_legend=False,
        width=1,
        xtick_rotation=90,
        title="test",
        labels="test",
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_clone_bubbleplot(create_testfolder):
    """test_plot_clone_bubbleplot"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    f2 = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f2)
    ax = clone_bubbleplot(vdj, groupby="isotype")
    assert ax is not None
    ax = clone_bubbleplot(adata, groupby="isotype")
    assert ax is not None
    # palette as a complete nested dict
    ax = clone_bubbleplot(
        adata,
        groupby="isotype",
        palette={
            "isotype": {"IgM": "#ff7f0e", "IgK": "#1f77b4", "IgL": "#2ca02c"}
        },
    )
    assert ax is not None
    # palette as a partial nested dict (missing keys get auto-assigned)
    ax = clone_bubbleplot(
        adata,
        groupby="isotype",
        palette={"isotype": {"IgM": "#ff0000"}},
    )
    assert ax is not None
    # palette as a list per level
    ax = clone_bubbleplot(
        adata,
        groupby="isotype",
        palette={"isotype": ["#ff7f0e", "#1f77b4", "#2ca02c"]},
    )
    assert ax is not None
    # list palette for both levels of nested hierarchy
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        palette={
            "group2": ["red", "blue"],
            "group3": ["green", "orange", "purple"],
        },
    )
    assert ax is not None
    # nested hierarchy with palette only for outer level
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        palette={"group2": {"a": "#ff7f0e", "b": "#1f77b4"}},
    )
    assert ax is not None
    # nested hierarchy with palette for both levels
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        palette={
            "group2": {"a": "#ff7f0e", "b": "#1f77b4"},
            "group3": {"a": "#2ca02c", "b": "#e377c2", "c": "#bcbd22"},
        },
    )
    assert ax is not None
    # AnnData with nested hierarchy, palette for outer level only
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        palette={"group2": {"a": "#ff7f0e", "b": "#1f77b4"}},
    )
    assert ax is not None
    # AnnData with palette dict for single level
    ax = clone_bubbleplot(
        adata,
        groupby="isotype",
        palette={"isotype": {"IgM": "#ff7f0e", "IgK": "#1f77b4"}},
    )
    assert ax is not None
    ax = clone_bubbleplot(
        vdj,
        groupby="isotype",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        show_group_labels=False,
        show_clone_labels=True,
        alpha=0.4,
        palette="tab10",
        show_legend=False,
    )
    assert ax is not None
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        legend_kwargs={"loc": "upper right"},
    )
    assert ax is not None
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        show_legend=["group2"],
    )
    assert ax is not None
    ax = clone_bubbleplot(
        adata,
        groupby=["group2", "group3"],
        show_legend="group2",
    )
    assert ax is not None
    ax = clone_bubbleplot(vdj, groupby="isotype", show_count_labels=True)
    assert ax is not None
    ax = clone_bubbleplot(
        vdj,
        groupby="isotype",
        show_clone_labels=True,
        show_count_labels=True,
    )
    assert ax is not None
    with pytest.raises(ValueError):
        clone_bubbleplot(vdj, groupby="isotype", min_clone_size=999)
