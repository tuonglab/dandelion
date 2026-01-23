#!/usr/bin/env python
import pytest
import scanpy as sc

from dandelion.polars.preprocessing import check_contigs
from dandelion.polars.core import Dandelion
from dandelion.polars.tools import generate_network, transfer
from dandelion.polars.io import read_ddl
from dandelion.polars.plotting import (
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
    f1 = create_testfolder / "test.ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_ddl(f1)
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
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    ax = barplot(vdj, color="v_call_VDJ")
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", sort_descending=sort)
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", normalize=norm)
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_bar2(create_testfolder):
    """test_plot_bar2"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
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
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    ax = stackedbarplot(
        vdj, color="v_call_VDJ", groupby="isotype", normalize=norm
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar2(create_testfolder, norm):
    """test_plot_stackedbar2"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
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
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
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
