#!/usr/bin/env python
"""Comprehensive tests for dandelion clone_size - group_by, AnnData, and MuData."""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from dandelion.polars.tools._tools import clone_size


# ---------------------------------------------------------------------------
# Synthetic AnnData helper
# ---------------------------------------------------------------------------
#
# Layout (8 cells):
#   cell_0  C1  S1       cell_4  C2  S2
#   cell_1  C1  S1       cell_5  C3  S2
#   cell_2  C1  S1       cell_6  NaN S2  <- no clone
#   cell_3  C2  S1       cell_7  C1  S2
#
# Global clone sizes:  C1=4, C2=2, C3=1
# Per-group sizes:
#   S1 (4 cells):  C1=3, C2=1
#   S2 (4 cells):  C1=1, C2=1, C3=1


def _make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "clone_id": ["C1", "C1", "C1", "C2", "C2", "C3", None, "C1"],
            "sample_id": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    return ad.AnnData(obs=obs)


# ---------------------------------------------------------------------------
# Helper: retrieve obs from an AnnData or MuData
# ---------------------------------------------------------------------------


def _obs(obj) -> pd.DataFrame:
    if hasattr(obj, "mod"):
        return obj.mod["airr"].obs
    return obj.obs


# ---------------------------------------------------------------------------
# Output columns written
# ---------------------------------------------------------------------------

_BASE_COLS = ["clone_id_size", "clone_id_size_prop", "clone_id_size_category"]


# ---------------------------------------------------------------------------
# AnnData – global (no group_by)
# ---------------------------------------------------------------------------


class TestCloneSizeAnnDataGlobal:
    def test_all_output_columns_written(self):
        adata = _make_adata()
        clone_size(adata)
        for col in _BASE_COLS:
            assert col in adata.obs.columns, f"missing {col}"

    def test_max_size_column_written(self):
        adata = _make_adata()
        clone_size(adata, max_size=3)
        assert "clone_id_size_max_3" in adata.obs.columns

    def test_no_max_size_column_when_not_requested(self):
        adata = _make_adata()
        clone_size(adata)
        assert not any(
            c.startswith("clone_id_size_max_") for c in adata.obs.columns
        )

    def test_correct_global_size_values(self):
        adata = _make_adata()
        clone_size(adata)
        obs = adata.obs
        # C1 appears 4 times globally
        assert obs.loc["cell_0", "clone_id_size"] == 4
        assert obs.loc["cell_7", "clone_id_size"] == 4
        # C2 appears 2 times globally
        assert obs.loc["cell_3", "clone_id_size"] == 2
        # C3 appears 1 time globally
        assert obs.loc["cell_5", "clone_id_size"] == 1

    def test_nan_clone_gets_nan_size(self):
        adata = _make_adata()
        clone_size(adata)
        assert pd.isna(adata.obs.loc["cell_6", "clone_id_size"])

    def test_size_prop_values(self):
        adata = _make_adata()
        clone_size(adata)
        # C1 proportion = 4/8 = 0.5
        assert abs(adata.obs.loc["cell_0", "clone_id_size_prop"] - 0.5) < 1e-9

    def test_max_size_clipping_above(self):
        adata = _make_adata()
        clone_size(adata, max_size=3)
        # C1 size=4 >= max_size=3  →  ">= 3"
        assert adata.obs.loc["cell_0", "clone_id_size_max_3"] == ">= 3"

    def test_max_size_clipping_below(self):
        adata = _make_adata()
        clone_size(adata, max_size=5)
        # C1 size=4 < max_size=5  →  "4"
        assert adata.obs.loc["cell_0", "clone_id_size_max_5"] == "4"

    def test_key_added_renames_output_columns(self):
        adata = _make_adata()
        clone_size(adata, key_added="vdj")
        assert "vdj_size" in adata.obs.columns
        assert "vdj_size_prop" in adata.obs.columns
        assert "vdj_size_category" in adata.obs.columns

    def test_custom_clone_key(self):
        adata = _make_adata()
        adata.obs["my_clone"] = adata.obs["clone_id"]
        clone_size(adata, clone_key="my_clone")
        assert "my_clone_size" in adata.obs.columns

    def test_missing_clone_key_raises(self):
        adata = _make_adata()
        with pytest.raises(KeyError):
            clone_size(adata, clone_key="nonexistent")


# ---------------------------------------------------------------------------
# AnnData – with group_by
# ---------------------------------------------------------------------------


class TestCloneSizeAnnDataGroupBy:
    def test_all_output_columns_written(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        for col in _BASE_COLS:
            assert col in adata.obs.columns, f"missing {col}"

    def test_per_group_sizes_differ_from_global(self):
        """cell_0 (C1/S1) has group size 3, not global size 4."""
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        assert adata.obs.loc["cell_0", "clone_id_size"] == 3  # C1 in S1
        assert adata.obs.loc["cell_7", "clone_id_size"] == 1  # C1 in S2

    def test_per_group_proportions(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        obs = adata.obs
        # C1 in S1: 3/4 = 0.75
        assert abs(obs.loc["cell_0", "clone_id_size_prop"] - 0.75) < 1e-9
        # C1 in S2: 1/4 = 0.25
        assert abs(obs.loc["cell_7", "clone_id_size_prop"] - 0.25) < 1e-9
        # C2 in S1: 1/4 = 0.25
        assert abs(obs.loc["cell_3", "clone_id_size_prop"] - 0.25) < 1e-9

    def test_different_groups_get_different_sizes(self):
        """Same clone in different groups gets different sizes."""
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        s1_c1_size = adata.obs.loc["cell_0", "clone_id_size"]  # C1 in S1 = 3
        s2_c1_size = adata.obs.loc["cell_7", "clone_id_size"]  # C1 in S2 = 1
        assert s1_c1_size != s2_c1_size

    def test_nan_clone_gets_nan_size(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        assert pd.isna(adata.obs.loc["cell_6", "clone_id_size"])

    def test_max_size_with_group_by(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id", max_size=3)
        assert "clone_id_size_max_3" in adata.obs.columns
        # C1 in S1: size=3, max_size=3  →  ">= 3"
        assert adata.obs.loc["cell_0", "clone_id_size_max_3"] == ">= 3"
        # C2 in S1: size=1, max_size=3  →  "1"
        assert adata.obs.loc["cell_3", "clone_id_size_max_3"] == "1"

    def test_key_added_with_group_by(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id", key_added="vdj")
        assert "vdj_size" in adata.obs.columns
        assert "vdj_size_prop" in adata.obs.columns

    def test_size_category_column_present(self):
        adata = _make_adata()
        clone_size(adata, group_by="sample_id")
        cats = adata.obs["clone_id_size_category"].dropna().unique().tolist()
        valid = {"Rare", "Small", "Medium", "Large", "Hyperexpanded"}
        assert set(cats).issubset(valid)


# ---------------------------------------------------------------------------
# AnnData – multi-clone cells (pipe-separated clone_id)
# ---------------------------------------------------------------------------


class TestCloneSizeMultiClone:
    def test_group_by_multi_clone_takes_largest(self):
        """Cell with C1|C2 gets the size of whichever is larger in its group."""
        obs = pd.DataFrame(
            {
                "clone_id": ["C1", "C1", "C1", "C2", "C1|C2"],
                "sample_id": ["S1", "S1", "S1", "S1", "S1"],
            },
            index=[f"cell_{i}" for i in range(5)],
        )
        adata = ad.AnnData(obs=obs)
        clone_size(adata, group_by="sample_id")
        # In S1: C1 appears in cell_0,1,2,4 => 4; C2 appears in cell_3,4 => 2
        # Multi-clone cell should pick C1 (size=4) over C2 (size=2)
        assert adata.obs.loc["cell_4", "clone_id_size"] == 4

    def test_global_multi_clone_takes_largest(self):
        obs = pd.DataFrame(
            {
                "clone_id": ["C1", "C1", "C1", "C2", "C1|C2"],
            },
            index=[f"cell_{i}" for i in range(5)],
        )
        adata = ad.AnnData(obs=obs)
        clone_size(adata)
        # C1 in cells 0,1,2,4_split → 4 occurrences; C2 in cells 3,4_split → 2
        # cell_4 multi-clone: picks C1 (size=4) > C2 (size=2)
        assert adata.obs.loc["cell_4", "clone_id_size"] == 4


# ---------------------------------------------------------------------------
# MuData
# ---------------------------------------------------------------------------


@pytest.fixture
def mudata_with_clones():
    """MuData with an 'airr' modality containing clone_id and sample_id."""
    mudata = pytest.importorskip("mudata")
    MuData = mudata.MuData

    adata_airr = _make_adata()
    adata_gex = ad.AnnData(obs=pd.DataFrame(index=adata_airr.obs_names))
    return MuData({"airr": adata_airr, "gex": adata_gex})


class TestCloneSizeMuData:
    def test_all_output_columns_written_to_airr(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata)
        airr_obs = mdata.mod["airr"].obs
        for col in _BASE_COLS:
            assert col in airr_obs.columns, f"missing {col}"

    def test_results_not_in_gex(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata)
        for col in _BASE_COLS:
            assert col not in mdata.mod["gex"].obs.columns

    def test_correct_global_size_values(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata)
        airr_obs = mdata.mod["airr"].obs
        assert airr_obs.loc["cell_0", "clone_id_size"] == 4  # C1 globally
        assert airr_obs.loc["cell_3", "clone_id_size"] == 2  # C2 globally

    def test_nan_clone_gets_nan_size(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata)
        assert pd.isna(mdata.mod["airr"].obs.loc["cell_6", "clone_id_size"])

    def test_with_group_by_all_columns_written(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, group_by="sample_id")
        airr_obs = mdata.mod["airr"].obs
        for col in _BASE_COLS:
            assert col in airr_obs.columns

    def test_group_by_per_group_sizes(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, group_by="sample_id")
        airr_obs = mdata.mod["airr"].obs
        assert airr_obs.loc["cell_0", "clone_id_size"] == 3  # C1 in S1
        assert airr_obs.loc["cell_7", "clone_id_size"] == 1  # C1 in S2

    def test_group_by_per_group_proportions(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, group_by="sample_id")
        prop = mdata.mod["airr"].obs.loc["cell_0", "clone_id_size_prop"]
        assert abs(prop - 0.75) < 1e-9  # 3/4 in S1

    def test_with_max_size(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, max_size=3)
        airr_obs = mdata.mod["airr"].obs
        assert "clone_id_size_max_3" in airr_obs.columns
        # C1 size=4 >= 3  →  ">= 3"
        assert airr_obs.loc["cell_0", "clone_id_size_max_3"] == ">= 3"

    def test_with_group_by_and_max_size(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, group_by="sample_id", max_size=3)
        airr_obs = mdata.mod["airr"].obs
        assert "clone_id_size_max_3" in airr_obs.columns
        # C1 in S1: size=3 >= max_size=3  →  ">= 3"
        assert airr_obs.loc["cell_0", "clone_id_size_max_3"] == ">= 3"
        # C2 in S1: size=1  →  "1"
        assert airr_obs.loc["cell_3", "clone_id_size_max_3"] == "1"

    def test_key_added(self, mudata_with_clones):
        mdata = mudata_with_clones
        clone_size(mdata, key_added="vdj")
        airr_obs = mdata.mod["airr"].obs
        assert "vdj_size" in airr_obs.columns
        assert "vdj_size_prop" in airr_obs.columns

    def test_missing_clone_key_raises(self, mudata_with_clones):
        mdata = mudata_with_clones
        with pytest.raises(KeyError):
            clone_size(mdata, clone_key="nonexistent")


# ---------------------------------------------------------------------------
# Integration tests using real DandelionPolars fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vdj_base_with_adata(airr_reannotated, dummy_adata):
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


def test_integration_anndata_group_by(vdj_base_with_adata):
    """AnnData group_by with real data writes all columns."""
    vdj, adata = vdj_base_with_adata
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    clone_size(adata, group_by="sample_id")
    for col in _BASE_COLS:
        assert col in adata.obs.columns


def test_integration_anndata_group_by_max_size(vdj_base_with_adata):
    """AnnData group_by + max_size with real data."""
    vdj, adata = vdj_base_with_adata
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    clone_size(adata, group_by="sample_id", max_size=2)
    assert "clone_id_size_max_2" in adata.obs.columns


def test_integration_dandelion_group_by(vdj_base_with_adata):
    """DandelionPolars group_by with real data writes all columns."""
    vdj, adata = vdj_base_with_adata
    import polars as pl

    vdj._metadata = vdj._metadata.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(pl.lit("S1"))
        .otherwise(pl.lit("S2"))
        .alias("sample_id")
    )
    clone_size(vdj, group_by="sample_id")
    schema = vdj._metadata.collect_schema().names()
    assert "clone_id_size" in schema
    assert "clone_id_size_prop" in schema
    assert "clone_id_size_category" in schema
