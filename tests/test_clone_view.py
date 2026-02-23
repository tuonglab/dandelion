#!/usr/bin/env python
"""Comprehensive tests for dandelion clone_view."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from dandelion.polars.tools._tools import clone_view


# ---------------------------------------------------------------------------
# Synthetic AnnData helper
# ---------------------------------------------------------------------------


def _make_adata(n_obs: int = 4) -> ad.AnnData:
    """Build a minimal AnnData with all keys clone_view may access."""
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = ad.AnnData(obs=obs)

    conn = sp.eye(n_obs, format="csr")
    dist = sp.eye(n_obs, format="csr") * 0.5
    emb = np.arange(n_obs * 2, dtype=float).reshape(n_obs, 2)

    adata.uns["dandelion"] = {
        "vdj_connectivities_all": conn,
        "vdj_distances_all": dist,
        "X_vdj_all": emb,
        "vdj_connectivities_expanded": conn,
        "vdj_distances_expanded": dist,
        "X_vdj_expanded": emb,
        "vdj_connectivities_full": conn,
        "vdj_distances_full": dist,
        "gex_connectivities": conn,
        "gex_distances": dist,
    }
    # gex neighbors lives in top-level uns (not inside dandelion)
    adata.uns["gex_neighbors"] = {
        "connectivities_key": "gex_connectivities",
        "distances_key": "gex_distances",
        "params": {"n_neighbors": 10, "method": "umap"},
    }

    adata.obsp["connectivities"] = conn
    adata.obsp["distances"] = dist
    adata.obsm["X_vdj"] = emb.copy()
    adata.obsm["X_custom"] = np.ones((n_obs, 3))

    return adata


# ---------------------------------------------------------------------------
# mode='all'
# ---------------------------------------------------------------------------


class TestCloneViewModeAll:
    def test_sets_connectivities(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_connectivities_all"]
        clone_view(adata, mode="all")
        diff = adata.obsp["connectivities"] - expected
        assert diff.nnz == 0

    def test_sets_distances(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_distances_all"]
        clone_view(adata, mode="all")
        diff = adata.obsp["distances"] - expected
        assert diff.nnz == 0

    def test_sets_embedding(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["X_vdj_all"]
        clone_view(adata, mode="all")
        np.testing.assert_array_equal(adata.obsm["X_vdj"], expected)

    def test_sets_default_neighbors_dict(self):
        adata = _make_adata()
        clone_view(adata, mode="all")
        assert "neighbors" in adata.uns
        assert adata.uns["neighbors"]["connectivities_key"] == "connectivities"
        assert adata.uns["neighbors"]["distances_key"] == "distances"
        assert adata.uns["neighbors"]["params"]["metric"] == "precomputed"

    def test_missing_connectivities_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_connectivities_all"]
        with pytest.raises(KeyError, match="vdj_connectivities_all"):
            clone_view(adata, mode="all")

    def test_missing_distances_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_distances_all"]
        with pytest.raises(KeyError, match="vdj_distances_all"):
            clone_view(adata, mode="all")

    def test_missing_embedding_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["X_vdj_all"]
        with pytest.raises(KeyError, match="X_vdj_all"):
            clone_view(adata, mode="all")


# ---------------------------------------------------------------------------
# mode='expanded'
# ---------------------------------------------------------------------------


class TestCloneViewModeExpanded:
    def test_sets_connectivities(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_connectivities_expanded"]
        clone_view(adata, mode="expanded")
        diff = adata.obsp["connectivities"] - expected
        assert diff.nnz == 0

    def test_sets_distances(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_distances_expanded"]
        clone_view(adata, mode="expanded")
        diff = adata.obsp["distances"] - expected
        assert diff.nnz == 0

    def test_sets_embedding(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["X_vdj_expanded"]
        clone_view(adata, mode="expanded")
        np.testing.assert_array_equal(adata.obsm["X_vdj"], expected)

    def test_sets_default_neighbors_dict(self):
        adata = _make_adata()
        clone_view(adata, mode="expanded")
        assert adata.uns["neighbors"]["params"]["metric"] == "precomputed"

    def test_missing_connectivities_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_connectivities_expanded"]
        with pytest.raises(KeyError, match="vdj_connectivities_expanded"):
            clone_view(adata, mode="expanded")

    def test_missing_distances_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_distances_expanded"]
        with pytest.raises(KeyError, match="vdj_distances_expanded"):
            clone_view(adata, mode="expanded")

    def test_missing_embedding_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["X_vdj_expanded"]
        with pytest.raises(KeyError, match="X_vdj_expanded"):
            clone_view(adata, mode="expanded")


# ---------------------------------------------------------------------------
# mode='full'
# ---------------------------------------------------------------------------


class TestCloneViewModeFull:
    def test_sets_connectivities(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_connectivities_full"]
        clone_view(adata, mode="full")
        diff = adata.obsp["connectivities"] - expected
        assert diff.nnz == 0

    def test_sets_distances(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["vdj_distances_full"]
        clone_view(adata, mode="full")
        diff = adata.obsp["distances"] - expected
        assert diff.nnz == 0

    def test_does_not_change_embedding(self):
        """mode='full' has emb_key=None, so X_vdj must remain unchanged."""
        adata = _make_adata()
        original_emb = adata.obsm["X_vdj"].copy()
        clone_view(adata, mode="full")
        np.testing.assert_array_equal(adata.obsm["X_vdj"], original_emb)

    def test_sets_default_neighbors_dict(self):
        adata = _make_adata()
        clone_view(adata, mode="full")
        assert adata.uns["neighbors"]["connectivities_key"] == "connectivities"
        assert adata.uns["neighbors"]["distances_key"] == "distances"

    def test_missing_connectivities_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_connectivities_full"]
        with pytest.raises(KeyError, match="vdj_connectivities_full"):
            clone_view(adata, mode="full")

    def test_missing_distances_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["vdj_distances_full"]
        with pytest.raises(KeyError, match="vdj_distances_full"):
            clone_view(adata, mode="full")


# ---------------------------------------------------------------------------
# mode='gex'
# ---------------------------------------------------------------------------


class TestCloneViewModeGex:
    def test_sets_connectivities(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["gex_connectivities"]
        clone_view(adata, mode="gex")
        diff = adata.obsp["connectivities"] - expected
        assert diff.nnz == 0

    def test_sets_distances(self):
        adata = _make_adata()
        expected = adata.uns["dandelion"]["gex_distances"]
        clone_view(adata, mode="gex")
        diff = adata.obsp["distances"] - expected
        assert diff.nnz == 0

    def test_does_not_change_embedding(self):
        """mode='gex' has emb_key=None, so X_vdj must remain unchanged."""
        adata = _make_adata()
        original_emb = adata.obsm["X_vdj"].copy()
        clone_view(adata, mode="gex")
        np.testing.assert_array_equal(adata.obsm["X_vdj"], original_emb)

    def test_sets_neighbors_from_gex_neighbors_key(self):
        """mode='gex' copies adata.uns['gex_neighbors'] to adata.uns['neighbors']."""
        adata = _make_adata()
        clone_view(adata, mode="gex")
        assert (
            adata.uns["neighbors"]["connectivities_key"] == "gex_connectivities"
        )
        assert adata.uns["neighbors"]["distances_key"] == "gex_distances"

    def test_missing_connectivities_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["gex_connectivities"]
        with pytest.raises(KeyError, match="gex_connectivities"):
            clone_view(adata, mode="gex")

    def test_missing_distances_raises(self):
        adata = _make_adata()
        del adata.uns["dandelion"]["gex_distances"]
        with pytest.raises(KeyError, match="gex_distances"):
            clone_view(adata, mode="gex")


# ---------------------------------------------------------------------------
# mode=None (manual key specification)
# ---------------------------------------------------------------------------


class TestCloneViewModeNone:
    def test_sets_connectivities_from_key(self):
        adata = _make_adata()
        conn_key = "vdj_connectivities_all"
        dist_key = "vdj_distances_all"
        expected = adata.uns["dandelion"][conn_key]
        clone_view(
            adata,
            mode=None,
            connectivities_key=conn_key,
            distances_key=dist_key,
        )
        diff = adata.obsp["connectivities"] - expected
        assert diff.nnz == 0

    def test_sets_distances_from_key(self):
        adata = _make_adata()
        conn_key = "vdj_connectivities_all"
        dist_key = "vdj_distances_all"
        expected = adata.uns["dandelion"][dist_key]
        clone_view(
            adata,
            mode=None,
            connectivities_key=conn_key,
            distances_key=dist_key,
        )
        diff = adata.obsp["distances"] - expected
        assert diff.nnz == 0

    def test_invalid_connectivities_key_raises(self):
        adata = _make_adata()
        with pytest.raises(KeyError):
            clone_view(adata, mode=None, connectivities_key="nonexistent")

    def test_invalid_distances_key_raises(self):
        """Valid connectivities_key but nonexistent distances_key raises KeyError."""
        adata = _make_adata()
        with pytest.raises(KeyError):
            clone_view(
                adata,
                mode=None,
                connectivities_key="vdj_connectivities_all",
                distances_key="nonexistent",
            )

    def test_sets_embedding_from_key(self):
        adata = _make_adata()
        # Add a distinct embedding to differentiate it from X_vdj
        new_emb = np.zeros((4, 3))
        adata.obsm["X_new"] = new_emb
        clone_view(
            adata,
            mode=None,
            connectivities_key="vdj_connectivities_all",
            distances_key="vdj_distances_all",
            embedding_key="X_new",
        )
        np.testing.assert_array_equal(adata.obsm["X_vdj"], new_emb)

    def test_invalid_embedding_key_raises(self):
        adata = _make_adata()
        with pytest.raises(KeyError, match="nonexistent_emb"):
            clone_view(
                adata,
                mode=None,
                connectivities_key="vdj_connectivities_all",
                distances_key="vdj_distances_all",
                embedding_key="nonexistent_emb",
            )

    def test_no_embedding_key_leaves_x_vdj_unchanged(self):
        """embedding_key=None must not modify X_vdj."""
        adata = _make_adata()
        original_emb = adata.obsm["X_vdj"].copy()
        clone_view(
            adata,
            mode=None,
            connectivities_key="vdj_connectivities_all",
            distances_key="vdj_distances_all",
            embedding_key=None,
        )
        np.testing.assert_array_equal(adata.obsm["X_vdj"], original_emb)


# ---------------------------------------------------------------------------
# Integration tests using real VDJ fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vdj_with_network(airr_reannotated, dummy_adata):
    from dandelion.polars.core._core import DandelionPolars
    from dandelion.polars.preprocessing._preprocessing import check_contigs
    from dandelion.polars.tools._tools import find_clones, transfer
    from dandelion.polars.tools._network import generate_network

    vdj = DandelionPolars(airr_reannotated)
    vdj, adata = check_contigs(vdj, dummy_adata)
    find_clones(vdj)
    generate_network(vdj, layout_method="mod_fr")
    transfer(adata, vdj)
    return vdj, adata


@pytest.fixture
def vdj_with_full_network(airr_reannotated, dummy_adata):
    """Network built with distance_mode='full'; transfer uses main_view='full'.

    This is the only path that writes vdj_connectivities_full /
    vdj_distances_full into adata.uns['dandelion'].
    """
    from dandelion.polars.core._core import DandelionPolars
    from dandelion.polars.preprocessing._preprocessing import check_contigs
    from dandelion.polars.tools._tools import find_clones, transfer
    from dandelion.polars.tools._network import generate_network

    vdj = DandelionPolars(airr_reannotated)
    vdj, adata = check_contigs(vdj, dummy_adata)
    find_clones(vdj)
    generate_network(
        vdj,
        layout_method="mod_fr",
        distance_mode="full",
        sequential_chain=True,
        n_cpus=1,
    )
    transfer(adata, vdj, main_view="full")
    return vdj, adata


def test_integration_mode_all(vdj_with_network):
    """mode='all' with real data sets obsp + neighbors."""
    vdj, adata = vdj_with_network
    clone_view(adata, mode="all")
    assert "connectivities" in adata.obsp
    assert "distances" in adata.obsp
    assert "neighbors" in adata.uns
    assert adata.uns["neighbors"]["params"]["metric"] == "precomputed"


def test_integration_mode_full(vdj_with_full_network):
    """mode='full' with real data: obsp is updated, X_vdj is unchanged,
    and neighbors dict uses precomputed metric."""
    vdj, adata = vdj_with_full_network
    assert "vdj_connectivities_full" in adata.uns["dandelion"]
    assert "vdj_distances_full" in adata.uns["dandelion"]
    original_emb = adata.obsm.get("X_vdj", None)
    clone_view(adata, mode="full")
    assert "connectivities" in adata.obsp
    assert "distances" in adata.obsp
    assert adata.uns["neighbors"]["connectivities_key"] == "connectivities"
    assert adata.uns["neighbors"]["distances_key"] == "distances"
    assert adata.uns["neighbors"]["params"]["metric"] == "precomputed"
    if original_emb is not None:
        np.testing.assert_array_equal(adata.obsm["X_vdj"], original_emb)


def test_integration_mode_none_valid_keys(vdj_with_network):
    """mode=None with keys found in real dandelion uns sets obsp correctly."""
    vdj, adata = vdj_with_network
    ddl = adata.uns.get("dandelion", {})
    conn_keys = [k for k in ddl if k.startswith("vdj_connectivities")]
    dist_keys = [k for k in ddl if k.startswith("vdj_distances")]
    if not conn_keys or not dist_keys:
        pytest.skip("No vdj_connectivities/distances keys in test data")
    clone_view(
        adata,
        mode=None,
        connectivities_key=conn_keys[0],
        distances_key=dist_keys[0],
    )
    assert "connectivities" in adata.obsp
    assert "distances" in adata.obsp


def test_integration_mode_none_missing_key_raises(vdj_with_network):
    """mode=None with a missing key raises KeyError."""
    vdj, adata = vdj_with_network
    with pytest.raises(KeyError):
        clone_view(adata, mode=None, connectivities_key="definitely_not_there")
