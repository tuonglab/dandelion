import pytest
import json

import numpy as np
import pandas as pd
import scanpy as sc

from unittest.mock import patch
import dandelion as ddl

from dandelion.base.preprocessing import check_contigs
from dandelion.base.io import read_h5ddl, read_10x_vdj
from dandelion.base.core import Dandelion
from dandelion.base.tools import (
    find_clones,
    clone_size,
    to_scirpy,
    generate_network,
    transfer,
    clone_diversity,
    clone_rarefaction,
)
from dandelion.base.tools._diversity import (
    calculate_chao1,
    calculate_shannon_entropy,
    drop_nan_values,
    _bootstrap_diversity_iteration,
    _bootstrap_network,
)


# convert from airr_Reannotate to airr, replicate this, run scirpy chainqc, test mudata as well
@pytest.mark.usefixtures(
    "create_testfolder", "airr_reannotated", "airr_reannotated2", "dummy_adata"
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """test setup"""
    vdj, adata = check_contigs(airr_reannotated, dummy_adata)
    vdj2 = check_contigs(airr_reannotated2)
    assert airr_reannotated.shape[0] == 8
    assert airr_reannotated2.shape[0] == 15
    assert vdj._data.shape[0] == 8
    assert vdj2._data.shape[0] == 14
    assert vdj._metadata.shape[0] == 5
    assert vdj2._metadata.shape[0] == 8
    assert adata.n_obs == 5
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj.write_h5ddl(f)
    vdj2.write_h5ddl(f2)
    assert len(list(create_testfolder.iterdir())) == 2
    vdj3 = read_h5ddl(f)
    assert vdj3.metadata is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    """test find clones"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = read_h5ddl(f)
    vdj2 = read_h5ddl(f2)
    find_clones(vdj)
    find_clones(vdj2)
    assert not vdj._data.clone_id.empty
    assert not vdj._metadata.clone_id.empty
    assert not vdj2._data.clone_id.empty
    assert not vdj2._metadata.clone_id.empty
    assert len({x for x in vdj._metadata["clone_id"] if pd.notnull(x)}) == 5
    assert len({x for x in vdj2._metadata["clone_id"] if pd.notnull(x)}) == 5
    vdj.write_h5ddl(f)
    vdj2.write_h5ddl(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_clone_size(create_testfolder):
    """test clone_size"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    clone_size(vdj)
    assert not vdj._metadata.clone_id_size.empty
    clone_size(vdj, max_size=3)
    assert not vdj._metadata.clone_id_size.empty


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "resample,expected", [pytest.param(None, 8), pytest.param(16, 16)]
)
def test_generate_network(create_testfolder, resample, expected):
    """test generate network"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = read_h5ddl(f)
    vdj2 = read_h5ddl(f2)
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
    vdj._data["clone_id"] = "1"
    vdj = Dandelion(vdj._data)
    assert vdj._data.clone_id.dtype == "object"
    generate_network(vdj, layout_method="mod_fr")
    assert vdj.layout is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    """test different clone key"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    find_clones(vdj, key_added="test_clone")
    assert not vdj._metadata.test_clone.empty
    assert vdj._data.test_clone.dtype == "object"
    generate_network(vdj, clone_key="test_clone", layout_method="mod_fr")
    assert vdj.layout is not None
    assert vdj.graph is not None


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    """test transfer"""
    f = create_testfolder / "test2.h5ddl"
    vdj = read_h5ddl(f)
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
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
    f = create_testfolder / "test2.h5ddl"
    vdj = read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
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
    vdj = read_10x_vdj(create_testfolder)
    vdj, adata = check_contigs(vdj, dummy_adata_cr6)
    assert vdj._data.shape[0] == 19
    assert vdj._metadata.shape[0] == 10
    find_clones(vdj)
    generate_network(vdj, key="sequence", layout_method="mod_fr")
    transfer(adata, vdj)
    f = create_testfolder / "test.h5ddl"
    vdj.write_h5ddl(f)
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    clone_rarefaction(vdj, groupby="sample_id")
    clone_rarefaction(vdj, groupby="sample_id", plot=True)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("use_network", [True, False])
def test_diversity_gini2(create_testfolder, use_network):
    """test gini more"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
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
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = clone_diversity(
        vdj, groupby="sample_id", use_contracted=True, key="sequence", n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2c(create_testfolder):
    """test div3"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = clone_diversity(
        vdj, groupby="sample_id", key="sequence", return_table=True, n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    """test edge weights"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
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


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


def test_calculate_chao1_normal():
    """calculate_chao1 returns a non-negative float for typical clone sizes."""
    values = np.array([1, 1, 2, 5])
    result = calculate_chao1(values)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_chao1_no_singletons():
    """calculate_chao1 with no singletons still returns a non-negative float."""
    values = np.array([3, 4, 5])
    result = calculate_chao1(values)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_shannon_entropy_normalized_equal():
    """Normalized Shannon entropy equals 1.0 for equal-sized clones."""
    values = np.array([3, 3, 3])
    result = calculate_shannon_entropy(values, normalize=True)
    assert abs(result - 1.0) < 1e-9


def test_calculate_shannon_entropy_normalized_unequal():
    """Normalized Shannon entropy is in [0, 1] for unequal clone sizes."""
    values = np.array([5, 2, 1])
    result = calculate_shannon_entropy(values, normalize=True)
    assert 0.0 <= result <= 1.0


def test_calculate_shannon_entropy_single_value():
    """Normalized Shannon entropy returns 0 for a single-clone repertoire."""
    result = calculate_shannon_entropy(np.array([5]), normalize=True)
    assert result == 0


def test_calculate_shannon_entropy_unnormalized():
    """Unnormalized Shannon entropy returns a non-negative float."""
    values = np.array([3, 2, 1])
    result = calculate_shannon_entropy(values, normalize=False)
    assert isinstance(result, float)
    assert result >= 0


def test_drop_nan_values_string_nan():
    """drop_nan_values removes the string 'nan' key from a Series index."""
    s = pd.Series({"clone_A": 3, "nan": 1, "clone_B": 2})
    drop_nan_values(s)
    assert "nan" not in s.index


def test_drop_nan_values_float_nan():
    """drop_nan_values removes np.nan from a Series index."""
    s = pd.Series({np.nan: 1, "clone_A": 3})
    drop_nan_values(s)
    assert np.nan not in s.index


def test_drop_nan_values_no_nan():
    """drop_nan_values leaves a clean Series unchanged."""
    s = pd.Series({"clone_A": 3, "clone_B": 2})
    original_len = len(s)
    drop_nan_values(s)
    assert len(s) == original_len


def test_bootstrap_diversity_iteration_chao1():
    """_bootstrap_diversity_iteration returns a non-negative float for chao1."""
    dat = pd.DataFrame({"clone_id": ["A", "A", "A", "B", "B", "C", "D", "E"]})
    result = _bootstrap_diversity_iteration(dat, "clone_id", "chao1", True, 5)
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_shannon_normalized():
    """_bootstrap_diversity_iteration returns a float for normalized shannon."""
    dat = pd.DataFrame({"clone_id": ["A", "A", "A", "B", "B", "C", "D", "E"]})
    result = _bootstrap_diversity_iteration(dat, "clone_id", "shannon", True, 5)
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_shannon_unnormalized():
    """_bootstrap_diversity_iteration returns a float for unnormalized shannon."""
    dat = pd.DataFrame({"clone_id": ["A", "A", "A", "B", "B", "C", "D", "E"]})
    result = _bootstrap_diversity_iteration(
        dat, "clone_id", "shannon", False, 5
    )
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_gini():
    """_bootstrap_diversity_iteration returns a non-negative float for gini."""
    dat = pd.DataFrame({"clone_id": ["A", "A", "A", "B", "B", "C", "D", "E"]})
    result = _bootstrap_diversity_iteration(dat, "clone_id", "gini", True, 5)
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_network_clone_degree(airr_reannotated2):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_degree."""
    vdj = check_contigs(airr_reannotated2)
    find_clones(vdj)
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_degree",
        min_size=4,
        expanded_only=False,
        contracted=False,
        layout_method="mod_fr",
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)


def test_bootstrap_network_clone_centrality(airr_reannotated2):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_centrality."""
    vdj = check_contigs(airr_reannotated2)
    find_clones(vdj)
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_centrality",
        min_size=4,
        expanded_only=False,
        contracted=False,
        layout_method="mod_fr",
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)


def test_bootstrap_network_clone_network(airr_reannotated2):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_network."""
    vdj = check_contigs(airr_reannotated2)
    find_clones(vdj)
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_network",
        min_size=4,
        expanded_only=False,
        contracted=False,
        layout_method="mod_fr",
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)
