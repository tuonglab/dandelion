import pytest
import json

import pandas as pd
import scanpy as sc

from unittest.mock import patch

import dandelion as ddl


# convert from airr_Reannotate to airr, replicate this, run scirpy chainqc, test mudata as well
@pytest.mark.usefixtures(
    "create_testfolder", "airr_reannotated", "airr_reannotated2", "dummy_adata"
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """test setup"""
    vdj, adata = ddl.pp.check_contigs(airr_reannotated, dummy_adata)
    vdj2 = ddl.pp.check_contigs(airr_reannotated2)
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
    vdj3 = ddl.read_h5ddl(f)
    assert vdj3.metadata is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    """test find clones"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj2 = ddl.read_h5ddl(f2)
    ddl.tl.find_clones(vdj)
    ddl.tl.find_clones(vdj2)
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
    vdj = ddl.read_h5ddl(f)
    ddl.tl.clone_size(vdj)
    assert not vdj._metadata.clone_id_size.empty
    ddl.tl.clone_size(vdj, max_size=3)
    assert not vdj._metadata.clone_id_size.empty


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "resample,expected", [pytest.param(None, 8), pytest.param(16, 16)]
)
def test_generate_network(create_testfolder, resample, expected):
    """test generate network"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj2 = ddl.read_h5ddl(f2)
    # create anndata from here
    adata = ddl.tl.to_scirpy(vdj, to_mudata=False)
    if resample is not None:
        vdj, adata = ddl.tl.generate_network(
            vdj, adata=adata, sample=resample, layout_method="mod_fr"
        )
        assert vdj.n_obs == expected
        assert vdj.layout is not None
        assert vdj.graph is not None
    else:
        ddl.tl.generate_network(vdj2, layout_method="mod_fr")
        assert vdj2.n_obs == expected
        assert vdj2.layout is not None
        assert vdj2.graph is not None
    vdj._data["clone_id"] = "1"
    vdj = ddl.Dandelion(vdj._data)
    assert vdj._data.clone_id.dtype == "object"
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    assert vdj.layout is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    """test different clone key"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ddl.tl.find_clones(vdj, key_added="test_clone")
    assert not vdj._metadata.test_clone.empty
    assert vdj._data.test_clone.dtype == "object"
    ddl.tl.generate_network(vdj, clone_key="test_clone", layout_method="mod_fr")
    assert vdj.layout is not None
    assert vdj.graph is not None


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    """test transfer"""
    f = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj, adata = ddl.pp.check_contigs(vdj, dummy_adata2)
    ddl.tl.transfer(dummy_adata2, vdj)
    assert "clone_id" in dummy_adata2.obs
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.transfer(dummy_adata2, vdj)
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
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
    vdj.update_data()
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
    vdj.update_data()
    with pytest.raises(ValueError):
        ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    # create random 3 sample ids to vdj.metadata
    vdj._metadata["sample_id"] = [
        f"sample_{i%3}" for i in range(vdj._metadata.shape[0])
    ]
    vdj.update_data()
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_10x_vdj(create_testfolder)
    vdj, adata = ddl.pp.check_contigs(vdj, dummy_adata_cr6)
    assert vdj._data.shape[0] == 19
    assert vdj._metadata.shape[0] == 10
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="sequence", layout_method="mod_fr")
    ddl.tl.transfer(adata, vdj)
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
    ddl.tl.clone_rarefaction(adata, groupby="sample_id")
    ddl.tl.clone_rarefaction(adata, groupby="sample_id", plot=True)


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction_ddl(mock_show, create_testfolder):
    """test rarefaction3"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    ddl.tl.clone_rarefaction(vdj, groupby="sample_id")
    ddl.tl.clone_rarefaction(vdj, groupby="sample_id", plot=True)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("use_network", [True, False])
def test_diversity_gini2(create_testfolder, use_network):
    """test gini more"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = ddl.tl.clone_diversity(
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
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", use_contracted=True, key="sequence", n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2c(create_testfolder):
    """test div3"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj._data["sample_id"] = "sample_test"
    vdj.update_metadata(
        retrieve=["sample_id"],
        retrieve_mode=["merge and unique only"],
    )
    res, _ = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", key="sequence", return_table=True, n_boot=5
    )
    assert res


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    """test edge weights"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
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
    res, _ = ddl.tl.clone_diversity(
        adata, groupby="sample_id", method=method, n_boot=5
    )
    assert res
