import pytest
import json

import polars as pl
import scanpy as sc

from unittest.mock import patch

import dandelion as ddl
from dandelion.utilities._polars import (
    DandelionPolars,
    check_contigs,
    read_zipddl,
    read_10x_vdj_polars,
)
from dandelion.tools._tools_polars import (
    find_clones,
    clone_size,
    transfer,
    to_scirpy,
)
from dandelion.tools._network_polars import generate_network
from dandelion.tools._diversity_polars import clone_diversity, clone_rarefaction


@pytest.fixture
@pytest.mark.usefixtures("airr_reannotated")
def airr_polars(airr_reannotated):
    """Helper fixture to create DandelionPolars object from airr_reannotated fixture."""
    return DandelionPolars(airr_reannotated)


@pytest.fixture
@pytest.mark.usefixtures("airr_reannotated2")
def airr_polars2(airr_reannotated2):
    """Helper fixture to create DandelionPolars object from airr_reannotated2 fixture."""
    return DandelionPolars(airr_reannotated2)


# convert from airr_Reannotate to airr, replicate this, run scirpy chainqc, test mudata as well
@pytest.mark.usefixtures("create_testfolder", "dummy_adata")
def test_setup(create_testfolder, airr_polars, airr_polars2, dummy_adata):
    """test setup"""
    vdj, adata = check_contigs(airr_polars, dummy_adata)
    vdj2 = check_contigs(airr_polars2)
    assert vdj.n_contigs == 8
    assert vdj2.n_contigs == 14
    assert vdj.n_obs == 5
    assert vdj2.n_obs == 8
    assert adata.n_obs == 5
    f = create_testfolder / "test.zipddl"
    f2 = create_testfolder / "test2.zipddl"
    vdj.write_zipddl(f)
    vdj2.write_zipddl(f2)
    assert len(list(create_testfolder.iterdir())) == 2
    vdj3 = read_zipddl(f)
    assert vdj3.metadata is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    """test find clones"""
    f = create_testfolder / "test.zipddl"
    f2 = create_testfolder / "test2.zipddl"
    vdj = read_zipddl(f)
    vdj2 = read_zipddl(f2)
    find_clones(vdj)
    find_clones(vdj2)
    assert not vdj._data.collect().get_column("clone_id").is_empty()
    assert not vdj._metadata.collect().get_column("clone_id").is_empty()
    assert not vdj2._data.collect().get_column("clone_id").is_empty()
    assert not vdj2._metadata.collect().get_column("clone_id").is_empty()
    assert (
        vdj._metadata.collect().get_column("clone_id").drop_nulls().n_unique()
        == 5
    )
    assert (
        vdj2._metadata.collect().get_column("clone_id").drop_nulls().n_unique()
        == 5
    )
    vdj.write_zipddl(f)
    vdj2.write_zipddl(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_clone_size(create_testfolder):
    """test clone_size"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    clone_size(vdj)
    assert not vdj._metadata.collect().get_column("clone_id").is_empty()
    clone_size(vdj, max_size=3)
    assert not vdj._metadata.collect().get_column("clone_id").is_empty()


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "resample,expected", [pytest.param(None, 8), pytest.param(16, 16)]
)
def test_generate_network(create_testfolder, resample, expected):
    """test generate network"""
    f = create_testfolder / "test.zipddl"
    f2 = create_testfolder / "test2.zipddl"
    vdj = read_zipddl(f)
    vdj2 = read_zipddl(f2)
    # create anndata from here
    adata = to_scirpy(vdj, to_mudata=False)
    if resample is not None:
        vdj, adata = generate_network(
            vdj, adata=adata, sample=resample, layout_method="mod_fr"
        )
        assert vdj.n_obs == expected
        assert vdj.layout is not None
        assert vdj.graph is not None
    else:
        generate_network(vdj2, layout_method="mod_fr")
        assert vdj2.n_obs == expected
        assert vdj2.layout is not None
        assert vdj2.graph is not None
    _data = vdj._data.with_columns(pl.lit("1").alias("clone_id"))
    _data = _data.collect() if isinstance(_data, pl.LazyFrame) else _data
    vdj = DandelionPolars(_data)
    assert vdj._data.collect_schema()["clone_id"] == pl.String
    generate_network(vdj, layout_method="mod_fr")
    assert vdj.layout is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    """test different clone key"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    find_clones(vdj, key_added="test_clone")
    assert not vdj._metadata.collect().get_column("test_clone").is_empty()
    assert vdj._data.collect_schema()["test_clone"] == pl.String
    generate_network(vdj, clone_key="test_clone", layout_method="mod_fr")
    assert vdj.layout is not None
    assert vdj.graph is not None


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    """test transfer"""
    f = create_testfolder / "test2.zipddl"
    vdj = read_zipddl(f)
    vdj, adata = check_contigs(vdj, dummy_adata2)
    transfer(dummy_adata2, vdj)
    assert "clone_id" in dummy_adata2.obs
    generate_network(vdj, layout_method="mod_fr")
    transfer(dummy_adata2, vdj)
    assert "X_vdj" in dummy_adata2.obsm
    f2 = create_testfolder / "test2.h5ad"
    dummy_adata2.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "method",
    [
        "chao1",
        "shannon",
        "gini",
    ],
)
def test_diversity_anndata(create_testfolder, method):
    """test div anndata"""
    f = create_testfolder / "test2.h5ad"
    adata = sc.read_h5ad(f)
    res, _ = clone_diversity(
        adata,
        groupby="sample_id",
        method=method,
        n_boot=5,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "normalize",
    [True, False],
)
def test_diversity_shannon(create_testfolder, normalize):
    """test shannon"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata = (
        vdj._metadata.with_row_index("idx")
        .with_columns(
            sample_id=(pl.col("idx") % 3).cast(pl.String).str.concat("sample_")
        )
        .drop("idx")
    )
    vdj.update_data()
    res, _ = clone_diversity(
        vdj,
        groupby="sample_id",
        method="shannon",
        normalize=normalize,
        n_boot=5,
        verbose=True,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "method",
    ["shannon", "chao1", "gini"],
)
def test_diversity_min_size_too_small(create_testfolder, method):
    """test shannon"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata = (
        vdj._metadata.with_row_index("idx")
        .with_columns(
            sample_id=(pl.col("idx") % 3).cast(pl.String).str.concat("sample_")
        )
        .drop("idx")
    )
    vdj.update_data()
    with pytest.raises(ValueError):
        clone_diversity(
            vdj,
            groupby="sample_id",
            method=method,
            min_size=6,
            n_boot=5,
            verbose=True,
        )


@pytest.mark.parametrize(
    "method",
    ["shannon", "chao1", "gini"],
)
def test_diversity_min_size_ok(create_testfolder, method):
    """test shannon"""
    f = create_testfolder / "test2.zipddl"
    vdj = read_zipddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata = (
        vdj._metadata.with_row_index("idx")
        .with_columns(
            sample_id=(pl.col("idx") % 3).cast(pl.String).str.concat("sample_")
        )
        .drop("idx")
    )
    vdj.update_data()
    res, _ = clone_diversity(
        vdj,
        groupby="sample_id",
        method=method,
        min_size=3,
        n_boot=5,
        verbose=True,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6", "dummy_adata_cr6")
def test_setup2(create_testfolder, json_10x_cr6, dummy_adata_cr6):
    """test setup 2"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = read_10x_vdj_polars(create_testfolder)
    vdj, adata = check_contigs(vdj, dummy_adata_cr6)
    assert vdj.n_contigs == 19
    assert vdj.n_obs == 10
    find_clones(vdj)
    generate_network(vdj, key="sequence", layout_method="mod_fr")
    transfer(adata, vdj)
    f = create_testfolder / "test.zipddl"
    vdj.write_zipddl(f)
    f2 = create_testfolder / "test.h5ad"
    adata.write_h5ad(f2)


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction_ad(mock_show, create_testfolder):
    """test rarefaction"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    clone_rarefaction(adata, groupby="sample_id")
    clone_rarefaction(adata, groupby="sample_id", plot=True)


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction_ddl(mock_show, create_testfolder):
    """test rarefaction3"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    clone_rarefaction(vdj, groupby="sample_id")
    clone_rarefaction(vdj, groupby="sample_id", plot=True)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("use_network", [True, False])
def test_diversity_gini2(create_testfolder, use_network):
    """test gini more"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    res, _ = clone_diversity(
        vdj,
        groupby="sample_id",
        min_size=6,
        key="sequence",
        n_boot=5,
        method="gini",
        use_network=use_network,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "metric", ["clone_network", "clone_degree", "clone_centrality"]
)
def test_diversity_gini3(create_testfolder, metric):
    """test gini more"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    res, _ = clone_diversity(
        vdj,
        groupby="sample_id",
        min_size=6,
        key="sequence",
        n_boot=5,
        network_metric=metric,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2a(create_testfolder):
    """test div"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    res, _ = clone_diversity(
        vdj,
        groupby="sample_id",
        reconstruct_network=False,
        key="sequence",
        n_boot=5,
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2b(create_testfolder):
    """test div2"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    res, _ = clone_diversity(
        vdj, groupby="sample_id", use_contracted=True, key="sequence", n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2c(create_testfolder):
    """test div3"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    vdj._data = vdj._data.with_columns(pl.lit("sample_test").alias("sample_id"))
    vdj.update_metadata(
        retrieve=["sample_id"],
        split=False,
        unique=True,
    )
    res, _ = clone_diversity(
        vdj, groupby="sample_id", key="sequence", return_table=True, n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    """test edge weights"""
    f = create_testfolder / "test.zipddl"
    vdj = read_zipddl(f)
    x = ddl.tl.extract_edge_weights(vdj)
    assert x is None
    x = ddl.tl.extract_edge_weights(vdj, expanded_only=True)
    assert x is None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "method",
    [
        "chao1",
        "shannon",
    ],
)
def test_diversity_anndata2(create_testfolder, method):
    """test div4"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    res, _ = clone_diversity(
        adata, groupby="sample_id", method=method, n_boot=5
    )
    assert res
