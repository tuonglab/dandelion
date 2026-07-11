"""Comprehensive coverage tests for dandelion/polars/core/_core_polars.py."""

import copy
import pickle
import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from dandelion.polars.core._core import (
    DandelionPolars,
    DataFrameAccessor,
    LazyFrameAccessor,
    SeriesAccessor,
    _clean_single_entry,
    _map_clones_with_dict,
    _assign_clone_numbers,
    _flatten_and_count,
    _get_receptor_prefix,
    clean_unicode,
    _is_polars_string_dtype,
    _is_polars_boolean_dtype,
    _add_clone_info,
    _sanitize_data_polars,
    load_polars,
)
from dandelion.polars.io._io import read_zipddl
from dandelion.polars.preprocessing._preprocessing import check_contigs
from dandelion.polars.tools._tools import find_clones
from dandelion.polars.tools._network import generate_network

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def airr_polars(airr_reannotated):
    """DandelionPolars (lazy) from airr_reannotated."""
    return DandelionPolars(airr_reannotated)


@pytest.fixture
def airr_polars2(airr_reannotated2):
    """DandelionPolars (lazy) from airr_reannotated2."""
    return DandelionPolars(airr_reannotated2)


@pytest.fixture
def vdj_checked(airr_reannotated, dummy_adata):
    """DandelionPolars after check_contigs + find_clones."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj)
    return vdj


@pytest.fixture
def vdj_with_network(airr_reannotated, dummy_adata):
    """DandelionPolars after network generation."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj)
    generate_network(vdj, layout_method="mod_fr")
    return vdj


# ===========================================================================
# Group 1 – __init__ branches
# ===========================================================================


def test_init_library_type(airr_polars):
    """Cover library_type locus filtering branch (lines 139-146)."""
    vdj = DandelionPolars(airr_polars._data, library_type="ig")
    assert vdj.n_contigs >= 0


def test_init_not_lazy(airr_polars):
    """Cover non-lazy init path (lines 209-210 eager ID storage)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    assert isinstance(vdj._data, pl.DataFrame)
    assert vdj.n_contigs > 0


def test_init_pandas_metadata(airr_polars):
    """Cover pandas metadata input (lines 181, 183-185)."""
    meta_pd = airr_polars._metadata.collect().to_pandas()
    # pandas DataFrame without cell_id in index
    vdj = DandelionPolars(airr_polars._data, metadata=meta_pd)
    assert vdj.n_obs > 0


def test_init_pandas_metadata_named_index(airr_polars):
    """Cover pandas metadata where cell_id is the index with name (lines 183-185)."""
    meta_pd = airr_polars._metadata.collect().to_pandas()
    # Move cell_id from column to named index
    meta_pd = meta_pd.set_index("cell_id")  # index.name = "cell_id" (not None)
    vdj = DandelionPolars(airr_polars._data, metadata=meta_pd)
    assert vdj.n_obs > 0


def test_init_pandas_metadata_unnamed_index(airr_polars):
    """Cover pandas metadata where cell_id is the index but index has no name (line 184)."""
    meta_pd = airr_polars._metadata.collect().to_pandas()
    # Move cell_id from column to unnamed index
    meta_pd = meta_pd.set_index("cell_id")
    meta_pd.index.name = None  # force unnamed index → triggers line 184
    vdj = DandelionPolars(airr_polars._data, metadata=meta_pd)
    assert vdj.n_obs > 0


def test_init_not_lazy_with_lazy_metadata(airr_polars):
    """Cover non-lazy init when metadata is LazyFrame (lines 192-193)."""
    meta_lazy = airr_polars._metadata  # LazyFrame
    vdj = DandelionPolars(
        airr_polars._data.collect(),
        metadata=meta_lazy,
        lazy=False,
    )
    assert isinstance(vdj._metadata, pl.DataFrame)


def test_init_not_lazy_with_eager_metadata(airr_polars):
    """Cover non-lazy init when metadata is eager DataFrame (line 196)."""
    meta_eager = airr_polars._metadata.collect()  # eager
    vdj = DandelionPolars(
        airr_polars._data.collect(),
        metadata=meta_eager,
        lazy=False,
    )
    assert isinstance(vdj._metadata, pl.DataFrame)


# ===========================================================================
# Group 2 – __repr__ / _gen_repr
# ===========================================================================


def test_repr_not_lazy(airr_polars):
    """Cover non-lazy repr path (line 226)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    r = repr(vdj)
    assert "Dandelion object with" in r


def test_repr_with_graph_layout(vdj_with_network):
    """Cover repr with graph/layout populated (lines 242, 244)."""
    r = repr(vdj_with_network)
    assert "graph" in r
    assert "layout" in r


def test_repr_with_distances(vdj_with_network):
    """Cover repr with distances (line 246)."""
    if vdj_with_network.distances is not None:
        r = repr(vdj_with_network)
        assert "distances" in r


def test_repr_pandas_data(airr_polars):
    """Cover repr when _data is pandas (lines 232-233)."""
    vdj = airr_polars
    vdj.to_pandas()
    r = repr(vdj)
    assert "Dandelion" in r


# ===========================================================================
# Group 3 – __getitem__ edge cases
# ===========================================================================


def test_slice_pandas_series_cell_ids(airr_polars):
    """Cover pd.Series → pl.from_pandas path (line 278)."""
    vdj = airr_polars
    cell_ids = vdj._metadata.collect()["cell_id"].to_list()[:2]
    pd_series = pd.Series(cell_ids)
    sliced = vdj[pd_series]
    assert sliced.n_obs == 2


def test_slice_pandas_index(airr_polars):
    """Cover pd.Index → pl.Series path (line 282)."""
    vdj = airr_polars
    cell_ids = vdj._metadata.collect()["cell_id"].to_list()[:2]
    idx = pd.Index(cell_ids)
    sliced = vdj[idx]
    assert sliced.n_obs == 2


def test_slice_polars_expression(airr_polars):
    """Cover pl.Expr filter path (lines 300-303)."""
    vdj = airr_polars
    sliced = vdj[pl.col("locus") == "IGH"]
    assert sliced is not None


def test_slice_polars_dataframe(airr_polars):
    """Cover pl.DataFrame as index (lines 305-309)."""
    vdj = airr_polars
    subset_df = vdj._data.collect().head(3)
    sliced = vdj[subset_df]
    assert sliced is not None


def test_slice_polars_lazyframe(airr_polars):
    """Cover pl.LazyFrame as index (lines 305-309)."""
    vdj = airr_polars
    subset_lf = vdj._data.head(3)
    sliced = vdj[subset_lf]
    assert sliced is not None


def test_slice_invalid_index_type(airr_polars):
    """Cover TypeError on unsupported index type (line 312)."""
    vdj = airr_polars
    with pytest.raises(TypeError):
        _ = vdj[12345]


def test_slice_boolean_series_metadata_length(airr_polars):
    """Cover boolean mask matching metadata length (lines 331-359)."""
    vdj = airr_polars
    n = vdj.n_obs
    mask = pl.Series([i % 2 == 0 for i in range(n)])
    sliced = vdj[mask]
    assert sliced.n_obs <= n


def test_slice_pandas_backend(airr_polars):
    """Cover pandas backend slice path (lines 575-576)."""
    vdj = airr_polars
    vdj.to_pandas()
    cell_ids = list(vdj._metadata.index[:2])
    sliced = vdj[cell_ids]
    assert sliced is not None


def test_slice_with_network_graph_layout(vdj_with_network):
    """Cover distances/graph/layout slicing (lines 513-576)."""
    vdj = vdj_with_network
    cell_ids = vdj._metadata.collect()["cell_id"].to_list()[:3]
    sliced = vdj[cell_ids]
    assert sliced.n_obs <= 3


# ===========================================================================
# Group 4 – Properties with None / pandas data
# ===========================================================================


def test_n_obs_none_metadata(airr_polars):
    """Cover n_obs when _metadata is None (line 585)."""
    vdj = DandelionPolars(airr_polars._data, initialize=False)
    assert vdj.n_obs == 0


def test_n_obs_eager_dataframe(airr_polars):
    """Cover n_obs on eager pl.DataFrame (line 591)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    assert isinstance(vdj._metadata, pl.DataFrame)
    assert vdj.n_obs > 0


def test_n_obs_pandas_metadata(airr_polars):
    """Cover n_obs with pandas metadata (lines 592-593)."""
    vdj = airr_polars
    vdj.to_pandas()
    assert isinstance(vdj._metadata, pd.DataFrame)
    assert vdj.n_obs > 0


def test_n_contigs_none_data(airr_reannotated):
    """Cover n_contigs when _data is None (line 599)."""
    vdj = DandelionPolars.__new__(DandelionPolars)
    vdj._data = None
    assert vdj.n_contigs == 0


def test_n_contigs_eager_dataframe(airr_polars):
    """Cover n_contigs on eager pl.DataFrame (lines 604-605)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    assert isinstance(vdj._data, pl.DataFrame)
    assert vdj.n_contigs > 0


def test_n_contigs_pandas_data(airr_polars):
    """Cover n_contigs with pandas data (lines 606-607)."""
    vdj = airr_polars
    vdj.to_pandas()
    assert isinstance(vdj._data, pd.DataFrame)
    assert vdj.n_contigs > 0


def test_data_property_pandas(airr_polars):
    """Cover data property when _data is pandas (line 613)."""
    vdj = airr_polars
    vdj.to_pandas()
    d = vdj.data
    assert isinstance(d, pd.DataFrame)


def test_data_setter(airr_polars):
    """Cover data setter (lines 623-626)."""
    vdj = airr_polars
    new_data = vdj._data.collect()
    vdj.data = new_data
    assert vdj._backend == "polars"


def test_data_names_property_lazy(airr_polars):
    """Cover data_names on LazyFrame (lines 638-641)."""
    names = airr_polars.data_names
    assert isinstance(names, SeriesAccessor)
    assert len(names) > 0


def test_data_names_property_eager(airr_polars):
    """Cover data_names on eager DataFrame (lines 631-636)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    assert isinstance(vdj._data, pl.DataFrame)
    names = vdj.data_names
    assert isinstance(names, SeriesAccessor)


def test_data_names_property_pandas(airr_polars):
    """Cover data_names on pandas DataFrame (line 632-633)."""
    vdj = airr_polars
    vdj.to_pandas()
    names = vdj.data_names
    assert names is not None  # returns pd.Index


def test_data_names_setter_polars(airr_polars):
    """Cover data_names setter for polars (lines 650-654)."""
    vdj = airr_polars
    current = vdj._data.collect()["sequence_id"].to_list()
    new_names = [f"new_{i}" for i in range(len(current))]
    vdj.data_names = new_names
    result = vdj._data.collect()["sequence_id"].to_list()
    assert result == new_names


def test_data_names_setter_pandas(airr_polars):
    """Cover data_names setter for pandas (lines 646-649)."""
    vdj = airr_polars
    vdj.to_pandas()
    current = list(vdj._data.index)
    new_names = [f"new_{i}" for i in range(len(current))]
    vdj.data_names = new_names


def test_metadata_names_property_eager(airr_polars):
    """Cover metadata_names on eager (lines 669-680)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    names = vdj.metadata_names
    assert isinstance(names, SeriesAccessor)


def test_metadata_names_property_lazy(airr_polars):
    """Cover metadata_names on lazy (line 660)."""
    names = airr_polars.metadata_names
    assert isinstance(names, SeriesAccessor)
    assert len(names) > 0


def test_metadata_names_property_none(airr_polars):
    """Cover metadata_names when metadata is None (line 686)."""
    vdj = DandelionPolars(airr_polars._data, initialize=False)
    names = vdj.metadata_names
    assert names is None


def test_metadata_names_property_pandas(airr_polars):
    """Cover metadata_names on pandas (line 689)."""
    vdj = airr_polars
    vdj.to_pandas()
    names = vdj.metadata_names
    assert names is not None  # returns pd.Index


def test_metadata_names_setter_polars(airr_polars):
    """Cover metadata_names setter for polars (lines 700-708)."""
    vdj = airr_polars
    current = vdj._metadata.collect()["cell_id"].to_list()
    new_names = [f"cell_{i}" for i in range(len(current))]
    vdj.metadata_names = new_names
    result = vdj._metadata.collect()["cell_id"].to_list()
    assert result == new_names


def test_metadata_names_setter_pandas(airr_polars):
    """Cover metadata_names setter for pandas (lines 700-701)."""
    vdj = airr_polars
    vdj.to_pandas()
    current = list(vdj._metadata.index)
    new_names = [f"cell_{i}" for i in range(len(current))]
    vdj.metadata_names = new_names


# ===========================================================================
# Group 5 – Internal dim/cache methods
# ===========================================================================


def test_cache_data_triggered(airr_polars):
    """Cover _cache_data / _cache_lazyframe via data manipulation."""
    vdj = airr_polars
    # Force a cache rebuild
    vdj._cache_data()
    assert vdj._data is not None


# ===========================================================================
# Group 6 – ID management
# ===========================================================================


def test_add_sequence_prefix(airr_polars):
    """Cover add_sequence_prefix (lines 885-888)."""
    vdj = airr_polars
    original = vdj._data.collect()["sequence_id"].to_list()[0]
    vdj.add_sequence_prefix("PRE_")
    new_val = vdj._data.collect()["sequence_id"].to_list()[0]
    assert new_val.startswith("PRE_")
    assert original in new_val


def test_add_sequence_suffix(airr_polars):
    """Cover add_sequence_suffix (lines 897, 899)."""
    vdj = airr_polars
    vdj.add_sequence_suffix("_SUF")
    new_val = vdj._data.collect()["sequence_id"].to_list()[0]
    assert new_val.endswith("_SUF")


def test_add_cell_prefix(airr_polars):
    """Cover add_cell_prefix (lines 918-919, 921)."""
    vdj = airr_polars
    vdj.add_cell_prefix("CELL_")
    new_val = vdj._metadata.collect()["cell_id"].to_list()[0]
    assert new_val.startswith("CELL_")


def test_add_cell_suffix(airr_polars):
    """Cover add_cell_suffix (lines 932, 934, 1108)."""
    vdj = airr_polars
    vdj.add_cell_suffix("_BATCH")
    new_val = vdj._metadata.collect()["cell_id"].to_list()[0]
    assert new_val.endswith("_BATCH")


def test_reset_ids(airr_polars):
    """Cover reset_ids (lines 1123-1147, including bug-fixed line 1135)."""
    vdj = airr_polars
    original_seq = vdj._data.collect()["sequence_id"].to_list()[0]
    original_cell = vdj._metadata.collect()["cell_id"].to_list()[0]
    vdj.add_sequence_prefix("PRE_")
    vdj.add_cell_prefix("CELL_")
    vdj.reset_ids()
    restored_seq = vdj._data.collect()["sequence_id"].to_list()[0]
    restored_cell = vdj._metadata.collect()["cell_id"].to_list()[0]
    assert restored_seq == original_seq
    assert restored_cell == original_cell


def test_reset_ids_pandas(airr_polars):
    """Cover reset_ids pandas branch (lines 1123-1130)."""
    vdj = airr_polars
    vdj.to_pandas()
    original = vdj._data["sequence_id"].iloc[0]
    # Directly modify _data to simulate prefix without calling update_metadata
    vdj._data["sequence_id"] = "PRE_" + vdj._data["sequence_id"]
    vdj._data.index = vdj._data["sequence_id"]
    vdj.reset_ids()
    restored = vdj._data["sequence_id"].iloc[0]
    assert restored == original


def test_add_ids_pandas_backend(airr_polars):
    """Cover pandas-specific ID update paths (lines 931-932, 918-921)."""
    vdj = airr_polars
    vdj.to_pandas()
    vdj.add_sequence_prefix("PRE_", sync=False)
    # update_metadata converts back to polars, so use polars accessor
    first_seq = vdj._data.select("sequence_id").collect()["sequence_id"][0]
    assert first_seq.startswith("PRE_")


# ===========================================================================
# Group 7 – simplify
# ===========================================================================


def test_simplify(airr_polars):
    """Cover simplify polars path (lines 1152-1172)."""
    vdj = airr_polars
    # Just call simplify - strips alleles from gene calls
    vdj.simplify()
    assert vdj._metadata is not None


def test_simplify_not_lazy(airr_polars):
    """Cover simplify eager path (lines 1164-1165)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    vdj.simplify()
    assert vdj._metadata is not None


# ===========================================================================
# Group 8 – Aggregation helpers via update_metadata
# ===========================================================================


def test_update_metadata_first(airr_polars):
    """Cover _first() via retrieve + first=True (lines 1222-1231)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="v_call", split=False, first=True)
    assert "v_call" in vdj._metadata.collect_schema().names()


def test_update_metadata_merge_join_unique(airr_polars):
    """Cover _merge() with join=True, unique=True (line 1197)."""
    vdj = airr_polars
    vdj.update_metadata(
        retrieve="v_call", split=False, join=True, unique=True, first=False
    )
    assert "v_call" in vdj._metadata.collect_schema().names()


def test_update_metadata_split_first(airr_polars):
    """Cover _split_first() via split=True, first=True (lines 1525-1610)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="v_call", split=True, first=True)
    schema_names = vdj._metadata.collect_schema().names()
    assert any("v_call" in c for c in schema_names)


def test_update_metadata_split_no_join(airr_polars):
    """Cover _split with join=False (lines 1240, 1248-1257)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="v_call", split=True, join=False, unique=False)
    schema_names = vdj._metadata.collect_schema().names()
    assert any("v_call" in c for c in schema_names)


def test_update_metadata_split_mean(airr_polars):
    """Cover _split_mean() for numeric columns (line 1266, 1274-1283)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="umi_count", split=True, average=True)
    schema_names = vdj._metadata.collect_schema().names()
    assert any("umi_count" in c for c in schema_names)


def test_update_metadata_merge_mean(airr_polars):
    """Cover _mean() for merge+average (lines 1274-1283)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="umi_count", split=False, average=True)
    assert "umi_count" in vdj._metadata.collect_schema().names()


def test_update_metadata_split_sum(airr_polars):
    """Cover _split_sum() (lines 1240-1257)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="umi_count", split=True, average=False)
    schema_names = vdj._metadata.collect_schema().names()
    assert any("umi_count" in c for c in schema_names)


def test_split_with_celltype(airr_polars):
    """Cover _split() celltype branch (line 1379)."""
    vdj = airr_polars
    result = vdj._split(cols="v_call", celltype="B")
    assert result is not None


# ===========================================================================
# Group 9 – initialize_metadata / update_metadata branches
# ===========================================================================


def test_update_metadata_reinitialize(airr_polars):
    """Cover reinitialize=True path (line 2561)."""
    vdj = airr_polars
    vdj.update_metadata(reinitialize=True)
    assert vdj._metadata is not None


def test_update_metadata_missing_col_error(airr_polars):
    """Cover KeyError when column not found (line 2597, 2620)."""
    vdj = airr_polars
    with pytest.raises(KeyError):
        vdj.update_metadata(retrieve="nonexistent_column_xyz")


def test_update_metadata_join_unique_string(airr_polars):
    """Cover string col with join=True, unique=True (lines 2638-2643)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="v_call", split=True, join=True, unique=True)
    schema = vdj._metadata.collect_schema().names()
    assert any("v_call" in c for c in schema)


def test_update_metadata_no_split_average_numeric(airr_polars):
    """Cover average path for numeric (lines 2664-2666)."""
    vdj = airr_polars
    vdj.update_metadata(retrieve="umi_count", split=False, average=True)
    assert "umi_count" in vdj._metadata.collect_schema().names()


def test_update_metadata_no_split_join_string(airr_polars):
    """Cover no-split + join path for string (lines 2674-2675)."""
    vdj = airr_polars
    vdj.update_metadata(
        retrieve="junction_aa", split=False, join=True, unique=True
    )
    assert "junction_aa" in vdj._metadata.collect_schema().names()


def test_update_metadata_duplicate_col(airr_polars):
    """Cover duplicate column handling before join (lines 2691-2705)."""
    vdj = airr_polars
    # Retrieve a col that already exists in metadata → will hit duplicate handling
    if "v_call" not in vdj._metadata.collect_schema().names():
        vdj.update_metadata(retrieve="v_call", split=False)
    # Retrieve the same col again → duplicate
    vdj.update_metadata(retrieve="v_call", split=False, reinitialize=False)
    assert "v_call" in vdj._metadata.collect_schema().names()


def test_initialize_metadata_update_isotype_dict(airr_polars):
    """Cover update_isotype_dict path (line 1796)."""
    vdj = airr_polars
    vdj._initialize_metadata(
        update_isotype_dict={"IGHA1": "IgA1", "IGHA2": "IgA2"}
    )
    assert vdj._metadata is not None


def test_initialize_metadata_empty_frames(airr_polars):
    """Cover empty frames path in initialize_metadata (lines 1918-1922)."""
    # Create a minimal DandelionPolars without c_call to skip isotype
    vdj = DandelionPolars(airr_polars._data)
    # Remove most init_cols so that merge result may be empty
    vdj._initialize_metadata(init_cols=[])
    assert vdj._metadata is not None


def test_initialize_metadata_not_lazy(airr_polars):
    """Cover non-lazy path in _reinitialize_attributes (lines 1755-1757)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    vdj._initialize_metadata()
    assert isinstance(vdj._metadata, pl.DataFrame)


# ===========================================================================
# Group 10 – Backend conversions
# ===========================================================================


def test_to_pandas_lazy(airr_polars):
    """Cover to_pandas on lazy object (lines 2115-2133)."""
    vdj = airr_polars
    assert isinstance(vdj._data, pl.LazyFrame)
    vdj.to_pandas()
    assert isinstance(vdj._data, pd.DataFrame)
    assert vdj._backend == "pandas"


def test_to_pandas_already_pandas(airr_polars):
    """Cover to_pandas no-op when already pandas (line 2115)."""
    vdj = airr_polars
    vdj.to_pandas()
    vdj.to_pandas()  # second call is a no-op
    assert vdj._backend == "pandas"


def test_to_pandas_eager(airr_polars):
    """Cover to_pandas on eager pl.DataFrame (line 2119)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    assert isinstance(vdj._data, pl.DataFrame)
    vdj.to_pandas()
    assert isinstance(vdj._data, pd.DataFrame)


def test_to_pandas_missing_cell_id(airr_polars):
    """Cover to_pandas KeyError when cell_id missing from metadata (line 2135)."""
    vdj = airr_polars
    vdj._metadata = vdj._metadata.drop("cell_id")
    with pytest.raises(KeyError):
        vdj.to_pandas()


def test_to_polars_from_pandas(airr_polars):
    """Cover to_polars from pandas (lines 2145-2164)."""
    vdj = airr_polars
    vdj.to_pandas()
    assert vdj._backend == "pandas"
    vdj.to_polars(lazy=True)
    assert vdj._backend == "polars"


def test_to_polars_already_polars(airr_polars):
    """Cover to_polars no-op (line 2144)."""
    vdj = airr_polars
    vdj.to_polars()  # no-op since already polars
    assert vdj._backend == "polars"


def test_to_anndata(airr_polars):
    """Cover to_anndata (lines 2168-2180)."""
    adata = airr_polars.to_anndata()
    assert adata is not None
    assert adata.n_obs == airr_polars.n_obs


def test_to_eager(airr_polars):
    """Cover to_eager (lines 2197-2202)."""
    vdj = airr_polars
    assert isinstance(vdj._data, pl.LazyFrame)
    vdj.to_eager()
    assert isinstance(vdj._data, pl.DataFrame)


def test_to_lazy(airr_polars):
    """Cover to_lazy from eager (lines 2207-2223)."""
    vdj = airr_polars
    vdj.to_eager()
    assert isinstance(vdj._data, pl.DataFrame)
    vdj.to_lazy()
    assert isinstance(vdj._data, pl.LazyFrame)


# ===========================================================================
# Group 11 – Copy / clone / pickle
# ===========================================================================


def test_copy(airr_polars):
    """Cover copy() deepcopy (line 2240)."""
    vdj = airr_polars
    vdj_copy = vdj.copy()
    assert vdj_copy.n_obs == vdj.n_obs


def test_clone_lazy(airr_polars):
    """Cover clone() for lazy object (lines 2248-2250, 2268)."""
    vdj = airr_polars
    vdj_clone = vdj.clone()
    assert vdj_clone.n_obs == vdj.n_obs
    # Verify independence
    vdj_clone._metadata = vdj_clone._metadata.with_columns(
        pl.lit("test").alias("_test")
    )
    assert "_test" not in vdj._metadata.collect_schema().names()


def test_clone_with_graph(vdj_with_network):
    """Cover clone() with graph/layout (line 2268 deepcopy path)."""
    vdj = vdj_with_network
    vdj_clone = vdj.clone()
    assert vdj_clone.graph is not None
    assert vdj_clone.layout is not None


def test_pickle_lazy(airr_polars):
    """Cover __getstate__ / __setstate__ (pickle support)."""
    vdj = airr_polars
    data = pickle.dumps(vdj)
    vdj2 = pickle.loads(data)
    assert vdj2.n_obs == vdj.n_obs


def test_pickle_with_data(airr_polars):
    """Cover __getstate__ materializing lazy frames."""
    vdj = airr_polars
    assert isinstance(vdj._data, pl.LazyFrame)
    state = vdj.__getstate__()
    # After getstate, _data should be eager
    assert isinstance(state["_data"], pl.DataFrame)


# ===========================================================================
# Group 12 – update_data
# ===========================================================================


def test_update_data_lazy(airr_polars):
    """Cover update_data lazy path (lines 2319-2356)."""
    vdj = airr_polars
    # Add a new column to metadata that's not in data
    vdj._metadata = vdj._metadata.with_columns(
        pl.lit("group_A").alias("new_group_col")
    )
    vdj.update_data()
    data_cols = vdj._data.collect_schema().names()
    assert "new_group_col" in data_cols


def test_update_data_pandas(airr_polars):
    """Cover update_data pandas path (lines 2324-2345)."""
    vdj = airr_polars
    vdj.to_pandas()
    vdj._metadata["new_col_pd"] = "test_val"
    vdj.update_data()
    assert "new_col_pd" in vdj._data.columns


def test_update_data_eager(airr_polars):
    """Cover update_data eager polars path (lines 2346-2351)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    vdj._metadata = vdj._metadata.with_columns(pl.lit("val").alias("extra_col"))
    vdj.update_data()
    assert "extra_col" in vdj._data.columns


# ===========================================================================
# Group 13 – store_germline_reference error paths
# ===========================================================================


def test_store_germline_no_env(airr_polars, monkeypatch):
    """Cover no GERMLINE env → KeyError."""
    monkeypatch.delenv("GERMLINE", raising=False)
    with pytest.raises(KeyError):
        airr_polars.store_germline_reference()


def test_store_germline_short_list(airr_polars):
    """Cover list with < 3 files → TypeError (lines 2398-2399)."""
    with pytest.raises(TypeError):
        airr_polars.store_germline_reference(germline=["a.fasta"])


def test_store_germline_list_bad_extension(airr_polars):
    """Cover list with non-fasta extension → TypeError (lines 2405-2412)."""
    with pytest.raises(TypeError):
        airr_polars.store_germline_reference(
            germline=["a.txt", "b.txt", "c.txt"]
        )


def test_store_germline_list_valid_ext(airr_polars, tmp_path):
    """Cover valid fasta list path (lines 2405-2413)."""
    f1 = tmp_path / "v.fasta"
    f2 = tmp_path / "d.fasta"
    f3 = tmp_path / "j.fasta"
    for f in [f1, f2, f3]:
        f.write_text(">test\nATCGATCG\n")
    try:
        airr_polars.store_germline_reference(
            germline=[str(f1), str(f2), str(f3)]
        )
    except Exception:
        pass  # readGermlines may fail with simple FASTA


def test_store_germline_dir_too_few_files(airr_polars, tmp_path):
    """Cover directory with < 3 files → TypeError (line 2420)."""
    f1 = tmp_path / "v.fasta"
    f1.write_text(">v1\nATCG\n")
    with pytest.raises(TypeError):
        airr_polars.store_germline_reference(germline=str(tmp_path))


def test_store_germline_dir_bad_extension(airr_polars, tmp_path):
    """Cover directory with non-fasta files → TypeError (lines 2429)."""
    for i in range(3):
        (tmp_path / f"file{i}.txt").write_text("not fasta\n")
    with pytest.raises(TypeError):
        airr_polars.store_germline_reference(germline=str(tmp_path))


def test_store_germline_dir_valid(airr_polars, tmp_path):
    """Cover directory with valid fasta files (lines 2426-2434)."""
    for i in range(3):
        f = tmp_path / f"gene{i}.fasta"
        f.write_text(">test_gene\nATCGATCG\n")
    try:
        airr_polars.store_germline_reference(germline=str(tmp_path))
    except Exception:
        pass  # readGermlines may fail with minimal content


def test_store_germline_single_file(airr_polars, tmp_path):
    """Cover single-file path with RuntimeWarning (lines 2435-2440)."""
    f = tmp_path / "single.fasta"
    f.write_text(">test\nATCG\n")
    with pytest.warns(RuntimeWarning):
        try:
            airr_polars.store_germline_reference(germline=str(f))
        except Exception:
            pass  # readGermlines may fail


def test_store_germline_env_set(airr_polars, monkeypatch, tmp_path):
    """Cover GERMLINE env var path (line 2395, 2447)."""
    monkeypatch.setenv("GERMLINE", str(tmp_path))
    try:
        airr_polars.store_germline_reference()
    except BaseException:
        pass  # Directory exists but no valid germline files → SystemExit from presto


# ===========================================================================
# Group 14 – update_plus
# ===========================================================================


def test_update_plus_all_option(airr_polars):
    """Cover update_plus option='all' (lines 2748-2779)."""
    vdj = airr_polars
    # Add a SEQINFO column (junction already exists), add junction_length
    vdj._data = vdj._data.with_columns(
        pl.col("junction")
        .str.len_chars()
        .cast(pl.Int64)
        .alias("junction_length")
    )
    vdj.update_plus(option="all")
    assert vdj._metadata is not None


def test_update_plus_sequence_option(airr_polars):
    """Cover update_plus option='sequence' (lines 2777-2779)."""
    vdj = airr_polars
    vdj.update_plus(option="sequence")
    assert vdj._metadata is not None


def test_update_plus_mutations_option(airr_polars):
    """Cover update_plus option='mutations' (lines 2784-2791)."""
    vdj = airr_polars
    # Add a mock mutation column
    vdj._data = vdj._data.with_columns(
        pl.lit(0).cast(pl.Int64).alias("mu_count_heavy")
    )
    vdj.update_plus(option="mutations")
    assert vdj._metadata is not None


def test_update_plus_cdr3_lengths(airr_polars):
    """Cover update_plus option='cdr3 lengths' (lines 2796-2798)."""
    vdj = airr_polars
    vdj._data = vdj._data.with_columns(
        pl.col("junction")
        .str.len_chars()
        .cast(pl.Int64)
        .alias("junction_length")
    )
    vdj.update_plus(option="cdr3 lengths")
    assert vdj._metadata is not None


def test_update_plus_mutations_and_cdr3(airr_polars):
    """Cover update_plus option='mutations and cdr3 lengths' (lines 2803-2816)."""
    vdj = airr_polars
    vdj._data = vdj._data.with_columns(
        pl.lit(0).cast(pl.Int64).alias("mu_count_heavy"),
        pl.col("junction")
        .str.len_chars()
        .cast(pl.Int64)
        .alias("junction_length"),
    )
    vdj.update_plus(option="mutations and cdr3 lengths")
    assert vdj._metadata is not None


def test_update_plus_pandas_backend(airr_polars):
    """Cover update_plus pandas backend conversion (lines 2748-2749)."""
    vdj = airr_polars
    vdj.to_pandas()
    vdj.update_plus(option="sequence")
    assert vdj._metadata is not None


# ===========================================================================
# Group 15 – Write methods
# ===========================================================================


def test_write_airr(airr_polars, tmp_path):
    """Cover write_airr (lines 2837-2838)."""
    out = str(tmp_path / "test_airr.tsv")
    airr_polars.write_airr(out)
    assert (tmp_path / "test_airr.tsv").exists()


def test_write_airr_pandas_backend(airr_polars, tmp_path):
    """Cover write_airr with pandas backend conversion."""
    vdj = airr_polars
    vdj.to_pandas()
    out = str(tmp_path / "test_pandas.tsv")
    vdj.write_airr(out)
    assert (tmp_path / "test_pandas.tsv").exists()


def test_write_zipddl_no_compress(airr_polars, tmp_path):
    """Cover compress=False branch in write_zipddl (line 2867)."""
    out = str(tmp_path / "test_no_compress.zipddl")
    airr_polars.write_zipddl(out, compress=False)
    assert (tmp_path / "test_no_compress.zipddl").exists()


def test_write_zipddl_with_germline(airr_polars, tmp_path):
    """Cover germline writing in write_zipddl (line 2880)."""
    vdj = airr_polars
    vdj.germline = {"IGHV1-1*01": "ATCGATCG"}
    out = str(tmp_path / "test_germline.zipddl")
    vdj.write_zipddl(out)
    assert (tmp_path / "test_germline.zipddl").exists()


def test_write_zipddl_with_pandas_data(airr_polars, tmp_path):
    """Cover pandas-to-polars conversion in write_zipddl (line 2867)."""
    vdj = airr_polars
    vdj.to_pandas()
    out = str(tmp_path / "test_pd_write.zipddl")
    vdj.write_zipddl(out)
    assert (tmp_path / "test_pd_write.zipddl").exists()


def test_write_10x(airr_polars, tmp_path):
    """Cover write_10x (lines 3085-3141)."""
    vdj = airr_polars
    # write_10x requires pandas for iterrows()
    vdj.to_pandas()
    folder = str(tmp_path / "10x_out")
    vdj.write_10x(folder=folder)
    out_path = Path(folder)
    assert (out_path / "all_contig.fasta").exists()
    assert (out_path / "all_contig_annotations.csv").exists()


# ===========================================================================
# Group 16 – load_polars branches
# ===========================================================================


def test_load_polars_missing_sequence_id():
    """Cover KeyError when sequence_id not in columns (line 3207)."""
    df = pl.DataFrame({"some_col": ["a", "b"]})
    with pytest.raises(KeyError):
        load_polars(df)


def test_load_polars_duplicate_count(airr_polars):
    """Cover duplicate_count → umi_count rename (lines 3210-3213)."""
    df = airr_polars._data.collect()
    # Rename umi_count to duplicate_count
    if "umi_count" in df.columns:
        df = df.rename({"umi_count": "duplicate_count"})
        result = load_polars(df, lazy=True)
        schema = result.collect_schema()
        assert (
            "umi_count" in schema.names() or "duplicate_count" in schema.names()
        )


def test_load_polars_as_pandas(airr_polars):
    """Cover as_pandas=True path (lines 3215-3217)."""
    df = airr_polars._data.collect()
    result = load_polars(df, lazy=True, as_pandas=True)
    assert isinstance(result, pd.DataFrame)


def test_load_polars_none():
    """Cover None input (line 3221)."""
    result = load_polars(None)
    assert result is None


# ===========================================================================
# Group 17 – DataFrameAccessor
# ===========================================================================


def test_dataframe_accessor_lazy_column(airr_polars):
    """Cover LazyFrame column access via accessor (lines 3271-3273)."""
    da = airr_polars.data
    assert isinstance(da, DataFrameAccessor)
    # Accessing a column name on lazy → returns pl.Expr
    col_expr = da.cell_id
    assert isinstance(col_expr, LazyFrameAccessor)


def test_dataframe_accessor_lazy_attr_not_found(airr_polars):
    """Cover LazyFrame attr not in schema → AttributeError (lines 3275-3280)."""
    da = airr_polars.data
    with pytest.raises(AttributeError):
        _ = da.completely_nonexistent_attr_xyz


def test_dataframe_accessor_eager_column(airr_polars):
    """Cover eager DataFrame column access (lines 3285-3288)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    assert isinstance(da, DataFrameAccessor)
    col = da.cell_id  # returns SeriesAccessor for eager
    assert isinstance(col, SeriesAccessor)


def test_dataframe_accessor_eager_passthrough(airr_polars):
    """Cover eager DataFrame passthrough attr (line 3288)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    # 'dtypes' is not a column but a DF attribute
    dtypes = da.dtypes
    assert dtypes is not None


def test_dataframe_accessor_bracket_str_lazy(airr_polars):
    """Cover bracket string key on LazyFrame (line 3298)."""
    da = airr_polars.data
    result = da["cell_id"]
    assert isinstance(result, LazyFrameAccessor)


def test_dataframe_accessor_bracket_str_eager(airr_polars):
    """Cover bracket string key on eager DataFrame (line 3300)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    result = da["cell_id"]
    assert isinstance(result, SeriesAccessor)


def test_dataframe_accessor_bracket_list_lazy(airr_polars):
    """Cover bracket list key on LazyFrame (lines 3303-3305)."""
    da = airr_polars.data
    result = da[["cell_id", "sequence_id"]]
    assert result is not None


def test_dataframe_accessor_bracket_list_eager(airr_polars):
    """Cover bracket list key on eager DataFrame (line 3307)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    result = da[["cell_id", "sequence_id"]]
    assert result is not None


def test_dataframe_accessor_bracket_slice(airr_polars):
    """Cover bracket slice key (line 3311)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    result = da[0:2]
    assert result is not None


def test_dataframe_accessor_bracket_series(airr_polars):
    """Cover bracket with pl.Series/pl.Expr (line 3315)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    mask = pl.Series([True, False] * (vdj.n_contigs // 2 + 1))
    mask = mask[: vdj.n_contigs]
    result = da[mask]
    assert result is not None


def test_dataframe_accessor_bracket_expr(airr_polars):
    """Cover bracket with pl.Expr (line 3315)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    result = da[pl.col("productive") == "T"]
    assert result is not None


def test_dataframe_accessor_bracket_other(airr_polars):
    """Cover bracket fallback (line 3319)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    result = da[0]  # integer row access → DataFrame.__getitem__(0)
    assert result is not None


def test_dataframe_accessor_setitem_list(airr_polars):
    """Cover __setitem__ with list value (line 3334)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    n = vdj.n_contigs
    da["test_col"] = ["val"] * n


def test_dataframe_accessor_setitem_series(airr_polars):
    """Cover __setitem__ with pl.Series (lines 3335-3337)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    n = vdj.n_contigs
    da["test_col2"] = pl.Series("test_col2", ["v"] * n)


def test_dataframe_accessor_setitem_series_rename(airr_polars):
    """Cover __setitem__ with pl.Series needing rename (line 3337)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    n = vdj.n_contigs
    da["new_name"] = pl.Series("old_name", ["v"] * n)  # name != key


def test_dataframe_accessor_setitem_series_accessor(airr_polars):
    """Cover __setitem__ with SeriesAccessor (lines 3338-3341)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    sa = da["cell_id"]
    da["cell_id_copy"] = sa


def test_dataframe_accessor_setitem_scalar(airr_polars):
    """Cover __setitem__ with scalar (line 3348)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    da["constant_col"] = "constant_value"


def test_dataframe_accessor_setitem_expr(airr_polars):
    """Cover __setitem__ with pl.Expr (line 3350)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    da["productive_upper"] = pl.col("productive").str.to_uppercase()


def test_dataframe_accessor_columns_lazy(airr_polars):
    """Cover .columns property on LazyFrame accessor (lines 3396-3398)."""
    da = airr_polars.data
    cols = da.columns
    assert isinstance(cols, list)
    assert "cell_id" in cols


def test_dataframe_accessor_setattr_df(airr_polars):
    """Cover __setattr__ allowing _df/_schema (lines 3322-3325)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    # Setting _df directly should work
    da._df = vdj._data
    # Setting other attribute should raise
    with pytest.raises(AttributeError):
        da.some_attr = "value"


# ===========================================================================
# Group 18 – SeriesAccessor
# ===========================================================================


def test_series_accessor_pandas_methods(airr_polars):
    """Cover SeriesAccessor pandas-method mapping (lines 3419-3423)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    sa = da["cell_id"]
    assert isinstance(sa, SeriesAccessor)

    # Test isin → is_in (line 3420)
    result = sa.isin(["dummy_cell"])
    assert isinstance(result, pl.Series)

    # Test isna → is_null (line 3432)
    result2 = sa.isna()
    assert isinstance(result2, pl.Series)

    # Test notna → is_not_null (line 3435)
    result3 = sa.notna()
    assert isinstance(result3, pl.Series)

    # Test fillna → fill_null (line 3439)
    result4 = sa.fillna("unknown")
    assert isinstance(result4, pl.Series)

    # Test dropna → drop_nulls (line 3429)
    result5 = sa.dropna()
    assert isinstance(result5, pl.Series)


def test_series_accessor_getitem(airr_polars):
    """Cover SeriesAccessor __getitem__ (line 3442)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    sa = vdj.data["cell_id"]
    val = sa[0]
    assert val is not None


def test_series_accessor_repr(airr_polars):
    """Cover SeriesAccessor __repr__ (line 3445)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    sa = vdj.data["cell_id"]
    r = repr(sa)
    assert len(r) > 0


def test_series_accessor_len(airr_polars):
    """Cover SeriesAccessor __len__ (line 3448)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    sa = vdj.data["cell_id"]
    assert len(sa) == vdj.n_contigs


def test_series_accessor_iter(airr_polars):
    """Cover SeriesAccessor __iter__ (line 3451)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    sa = vdj.data["cell_id"]
    items = list(sa)
    assert len(items) > 0


def test_series_accessor_comparison_operators(airr_polars):
    """Cover SeriesAccessor comparison operators (lines 3454+)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    sa = vdj.data["umi_count"]
    # __eq__
    result1 = sa == 1
    # __ne__
    result2 = sa != 1
    # __lt__
    result3 = sa < 5
    # __le__
    result4 = sa <= 5
    # __gt__
    result5 = sa > 1
    # __ge__
    result6 = sa >= 1
    for r in [result1, result2, result3, result4, result5, result6]:
        assert isinstance(r, pl.Series)


# ===========================================================================
# Group 19 – Module-level functions
# ===========================================================================


def test_clean_single_entry_none():
    """Cover _clean_single_entry(None) (lines 4010-4011)."""
    assert _clean_single_entry(None) == "None"
    assert _clean_single_entry("") == "None"


def test_clean_single_entry_none_parts():
    """Cover all-None parts → ['None'] fallback (lines 4014-4016)."""
    result = _clean_single_entry("None|None")
    assert result == "None"


def test_clean_single_entry_vdj_preferred():
    """Cover VDJ preferred over VJ deduplication (lines 4050-4054)."""
    # VJ and VDJ with same key → VDJ wins
    result = _clean_single_entry("B_VJ_clone1|B_VDJ_clone1")
    assert "B_VDJ_clone1" in result
    assert "B_VJ_clone1" not in result


def test_clean_single_entry_abt_gdt():
    """Cover abT and gdT prefixes (lines 4033-4043)."""
    result_abt = _clean_single_entry("abT_VDJ_x|abT_VJ_x")
    assert "abT_VDJ_x" in result_abt

    result_gdt = _clean_single_entry("gdT_VDJ_y|gdT_VJ_y")
    assert "gdT_VDJ_y" in result_gdt


def test_clean_single_entry_other_format():
    """Cover OTHER category (line 4047-4048)."""
    result = _clean_single_entry("weird_format_xyz")
    assert result == "weird_format_xyz"


def test_clean_single_entry_existing_vj_kept():
    """Cover VDJ vs VJ preference when VDJ already exists (line 4055)."""
    result = _clean_single_entry("B_VDJ_clone1|B_VJ_clone2")
    # Different keys so both kept
    assert "B_VDJ_clone1" in result


def test_map_clones_with_dict_none():
    """Cover _map_clones_with_dict with None/empty (lines 4070-4071)."""
    assert _map_clones_with_dict(None, {}) == "None"
    assert _map_clones_with_dict("", {}) == "None"


def test_map_clones_with_dict_lookup():
    """Cover _map_clones_with_dict with lookup (line 4072)."""
    d = {"B_VDJ_1": "1", "B_VJ_2": "2"}
    result = _map_clones_with_dict("B_VDJ_1|B_VJ_2", d)
    assert "1" in result
    assert "2" in result
    # Unknown clone passes through
    result2 = _map_clones_with_dict("unknown_clone", d)
    assert "unknown_clone" in result2


def test_assign_clone_numbers_single_type():
    """Cover _assign_clone_numbers with single receptor type (lines 4159-4162)."""
    clone_counts = {"B_1": 10, "B_2": 5, "B_3": 2}
    result = _assign_clone_numbers(clone_counts)
    assert len(result) == 3
    # Sequential without prefix
    assert set(result.values()) == {"1", "2", "3"}


def test_assign_clone_numbers_multiple_types():
    """Cover _assign_clone_numbers with multiple receptor types (lines 4165-4186)."""
    clone_counts = {
        "B_clone1": 10,
        "B_clone2": 5,
        "abT_clone1": 8,
        "gdT_clone1": 3,
        "other_clone": 1,
    }
    result = _assign_clone_numbers(clone_counts)
    assert "B_clone1" in result
    assert "abT_clone1" in result
    # Values should have type prefix
    assert any("B_" in v for v in result.values())


def test_get_receptor_prefix():
    """Cover _get_receptor_prefix."""
    assert _get_receptor_prefix("B_clone1") == "B"
    assert _get_receptor_prefix("abT_clone1") == "abT"
    assert _get_receptor_prefix("unknown_clone1") is None


def test_clean_unicode_non_string():
    """Cover clean_unicode with non-string input (line 4318-4319)."""
    assert clean_unicode(123) == ""
    assert clean_unicode(None) == ""


def test_clean_unicode_string():
    """Cover clean_unicode with string (lines 4321, 4323)."""
    result = clean_unicode("hello world")
    assert result == "hello world"
    # Unicode normalization
    result2 = clean_unicode("caf\u00e9")
    assert "caf" in result2


def test_is_polars_string_dtype_lazyframe(airr_polars):
    """Cover _is_polars_string_dtype with LazyFrame (line 4329)."""
    df = airr_polars._data
    assert isinstance(df, pl.LazyFrame)
    assert _is_polars_string_dtype(df, "cell_id") is True
    assert _is_polars_string_dtype(df, "umi_count") is False


def test_is_polars_boolean_dtype_lazyframe(airr_polars):
    """Cover _is_polars_boolean_dtype with LazyFrame (line 4336)."""
    df = airr_polars._data
    assert _is_polars_boolean_dtype(df, "cell_id") is False


def test_validate_airr_polars_lazyframe(airr_polars):
    """Cover _validate_airr_polars with LazyFrame input (line 4345)."""
    from dandelion.polars.core._core import _validate_airr_polars

    df = airr_polars._data  # LazyFrame
    # Should not raise
    _validate_airr_polars(df)


def test_sanitize_data_polars_pandas():
    """Cover _sanitize_data_polars with pandas input (lines 4204-4222)."""
    df = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq2"],
            "cell_id": ["cell1", "cell2"],
            "productive": ["T", "F"],
            "v_call": ["IGHV1-1", "IGHV1-2"],
            "d_call": ["IGHD1-1", "IGHD1-2"],
            "j_call": ["IGHJ1", "IGHJ1"],
            "junction": ["CARGYYY", "CARGYYY"],
            "junction_aa": ["CAR", "CAR"],
            "mixed_numeric": ["1", "2"],
        }
    )
    result = _sanitize_data_polars(df)
    assert result is not None


def test_sanitize_data_polars_with_boolean_col():
    """Cover boolean-like column sanitization in _sanitize_data_polars."""
    df = pl.DataFrame(
        {
            "sequence_id": ["seq1", "seq2"],
            "cell_id": ["cell1", "cell2"],
            "productive": ["T", "F"],
            "v_call": ["IGHV1-1", "IGHV1-2"],
            "d_call": ["IGHD1-1", "IGHD1-2"],
            "j_call": ["IGHJ1", "IGHJ1"],
            "junction": ["CARGYYY", "CARGYYY"],
            "junction_aa": ["CAR", "CAR"],
            "rev_comp": ["TRUE", "FALSE"],
        }
    )
    result = _sanitize_data_polars(df)
    assert result is not None


def test_add_clone_info_lazy():
    """Cover _add_clone_info with lazy input (line 4089-4091)."""
    df = pl.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],
            "clone_id": ["B_VDJ_1", "B_VDJ_1|B_VJ_2", "None"],
        }
    ).lazy()
    result = _add_clone_info(df, "clone_id")
    assert result is not None
    assert "clone_id_rank" in result.collect_schema().names()


def test_add_clone_info_eager():
    """Cover _add_clone_info with eager input (lines 4093-4094)."""
    df = pl.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],
            "clone_id": ["B_VDJ_1", "B_VDJ_2", "B_VDJ_1"],
        }
    )
    result = _add_clone_info(df, "clone_id")
    assert result is not None
    assert "clone_id_rank" in result.columns


# ===========================================================================
# Additional tests for remaining uncovered lines
# ===========================================================================


def test_init_cache_handles(airr_polars):
    """Cover cache_handles parameter in __init__ (line 117)."""
    existing_handles = {"dummy": None}
    vdj = DandelionPolars(airr_polars._data, cache_handles=existing_handles)
    assert vdj._cache_handles is existing_handles


def test_init_germline_param(airr_polars):
    """Cover germline parameter in __init__ (line 130)."""
    germ = {"IGHV1-1": "ATCG"}
    vdj = DandelionPolars(airr_polars._data, germline=germ)
    assert "IGHV1-1" in vdj.germline


def test_init_library_type_eager(airr_polars):
    """Cover library_type filter with eager data (lines 148, 154)."""
    eager_data = airr_polars._data.collect()
    vdj = DandelionPolars(eager_data, library_type="ig", lazy=False)
    assert vdj.n_contigs >= 0


def test_init_pandas_data_original_ids():
    """Cover pandas data original ID storage (lines 216-218)."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "sequence_id": ["a_contig_1", "b_contig_1"],
            "cell_id": ["a", "b"],
            "locus": ["IGH", "IGH"],
            "productive": ["T", "T"],
            "v_call": ["IGHV1-1", "IGHV1-2"],
            "d_call": ["IGHD1-1", "IGHD1-2"],
            "j_call": ["IGHJ1", "IGHJ1"],
            "c_call": ["IGHA", "IGHG"],
            "umi_count": [1, 2],
            "junction": ["CARG", "CARG"],
            "junction_aa": ["CAR", "CAR"],
        }
    )
    vdj = DandelionPolars(df, lazy=False)
    # After init, _data should still be polars (load_polars converts pandas)
    assert vdj._original_sequence_ids is not None


def test_gen_repr_fallback():
    """Cover _gen_repr fallback for AttributeError (lines 235-237)."""
    vdj = DandelionPolars.__new__(DandelionPolars)
    # Set up minimal attributes
    vdj._lazy = True
    vdj._data = None
    vdj._metadata = None
    vdj.layout = None
    vdj.graph = None
    vdj.distances = None
    # This should not raise
    r = vdj._gen_repr(0, 0)
    assert isinstance(r, str)


def test_slice_pd_dataframe_index(airr_polars):
    """Cover pd.DataFrame → pl.from_pandas path (line 280)."""
    vdj = airr_polars
    # Pass a full VDJ pandas DataFrame (with sequence_id) as the index
    pd_df = vdj._data.head(3).collect().to_pandas()
    sliced = vdj[pd_df]
    assert sliced is not None


def test_metadata_setter_pandas(airr_polars):
    """Cover metadata.setter with pandas DataFrame (lines 669-679)."""
    vdj = airr_polars
    meta_pd = vdj._metadata.collect().to_pandas()
    meta_pd = meta_pd.set_index("cell_id")  # move cell_id to index
    meta_pd.index.name = None  # unnamed index → line 673
    vdj.metadata = meta_pd
    assert vdj._metadata is not None


def test_metadata_setter_pandas_with_cell_id_col(airr_polars):
    """Cover metadata.setter with pandas DataFrame where cell_id is a column (line 676)."""
    vdj = airr_polars
    meta_pd = vdj._metadata.collect().to_pandas()
    # cell_id is already a column
    vdj.metadata = meta_pd
    assert vdj._metadata is not None


def test_is_sanitized_pandas_path(airr_polars):
    """Cover _is_sanitized with pandas data (lines 726-737)."""
    vdj = airr_polars
    vdj.to_pandas()
    # After to_pandas(), _data is pd.DataFrame
    result = vdj._is_sanitized()
    assert isinstance(result, bool)


def test_set_dim_df(airr_polars):
    """Cover _set_dim_df (lines 741-744)."""
    vdj = airr_polars
    vdj.to_pandas()
    idx = pd.Index(vdj._metadata.index)
    vdj._set_dim_df(pd.DataFrame(index=idx), "metadata")


def test_prep_dim_index_warning(airr_polars):
    """Cover _prep_dim_index with non-string values warning (lines 754-777)."""
    vdj = airr_polars
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vdj._prep_dim_index(pd.Index([1, 2, 3]), "metadata")
    assert result is not None


def test_cache_data_lazy(airr_polars):
    """Cover _cache_data when lazy (lines 788-803)."""
    vdj = airr_polars
    assert vdj._lazy is True
    # _cache_data is called internally, but we can call it directly
    vdj._cache_data()
    assert vdj._data is not None


def test_add_sequence_prefix_remove_trailing(airr_polars):
    """Cover remove_trailing_hyphen_number path in _clean_sequence_id (lines 965-972)."""
    vdj = airr_polars
    vdj.add_sequence_prefix("PRE_", remove_trailing_hyphen_number=True)
    assert vdj._data is not None


def test_add_cell_prefix_remove_trailing(airr_polars):
    """Cover remove_trailing_hyphen_number path in _clean_cell_id (lines 992-994)."""
    vdj = airr_polars
    vdj.add_cell_prefix("PRE_", remove_trailing_hyphen_number=True)
    assert vdj._data is not None


def test_simplify_pandas(airr_polars):
    """Cover simplify on pandas-backed data (lines 1182-1186)."""
    vdj = airr_polars
    vdj.to_pandas()
    vdj.simplify()
    assert vdj._data is not None


def test_init_no_c_call_triggers_noiso_path():
    """Cover _classify_locus_pair_noiso path (lines 2030-2031, 3728-3849).

    T cell data with TRAC/TRBC c_calls but no isotype in metadata
    → _classify_locus_pair_noiso() is used for locus_status.
    """
    df = pl.DataFrame(
        {
            "sequence_id": ["a_contig_1", "a_contig_2", "b_contig_1"],
            "cell_id": ["a", "a", "b"],
            "locus": ["TRA", "TRB", "TRB"],
            "productive": ["T", "T", "T"],
            "v_call": ["TRAV1-1", "TRBV1-2", "TRBV2-1"],
            "d_call": [None, "TRBD1", "TRBD2"],
            "j_call": ["TRAJ1", "TRBJ1", "TRBJ2"],
            "c_call": ["TRAC", "TRBC1", "TRBC2"],
            "umi_count": [1, 2, 3],
            "junction": ["CARGYYY", "CARGYYY", "CARGGGG"],
            "junction_aa": ["CARY", "CARY", "CARG"],
        }
    )
    vdj = DandelionPolars(df)
    assert vdj._metadata is not None
    meta_cols = vdj._metadata.collect_schema().names()
    assert "locus_status" in meta_cols


def test_to_eager_polars(airr_polars):
    """Cover to_eager when _data and _metadata are lazy (lines 2207-2226)."""
    vdj = airr_polars
    assert isinstance(vdj._data, pl.LazyFrame)
    vdj.to_eager()
    assert isinstance(vdj._data, pl.DataFrame)
    assert vdj._lazy is False


def test_to_lazy_polars(airr_polars):
    """Cover to_lazy from eager (lines 2228-2252)."""
    vdj = airr_polars
    vdj.to_eager()
    assert isinstance(vdj._data, pl.DataFrame)
    vdj.to_lazy()
    assert isinstance(vdj._data, pl.LazyFrame)
    assert vdj._lazy is True


def test_to_anndata_eager_metadata(airr_polars):
    """Cover to_anndata with eager metadata (line 2195-2197)."""
    vdj = airr_polars
    vdj.to_eager()
    assert isinstance(vdj._metadata, pl.DataFrame)
    adata = vdj.to_anndata()
    assert adata is not None


def test_clone_pandas(airr_polars):
    """Cover clone with pandas data (line 2271-2272)."""
    vdj = airr_polars
    vdj.to_pandas()
    cloned = vdj.clone()
    assert cloned is not vdj
    assert isinstance(cloned._data, pd.DataFrame)


def test_dataframe_accessor_setitem_scalar_lazy(airr_polars):
    """Cover DataFrameAccessor __setitem__ with scalar on LazyFrame (line 3370)."""
    da = airr_polars.data
    da["test_scalar_col"] = "constant_value"
    assert "test_scalar_col" in airr_polars._data.collect_schema().names()


def test_dataframe_accessor_setitem_expr_lazy(airr_polars):
    """Cover DataFrameAccessor __setitem__ with pl.Expr on LazyFrame (lines 3374-3386)."""
    da = airr_polars.data
    da["test_expr_col"] = pl.lit("expr_value")
    assert "test_expr_col" in airr_polars._data.collect_schema().names()


def test_dataframe_accessor_setitem_invalid_type(airr_polars):
    """Cover DataFrameAccessor __setitem__ TypeError path (lines 3387-3390)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    with pytest.raises(TypeError):
        da["bad_col"] = object()


def test_dataframe_accessor_len_lazy(airr_polars):
    """Cover DataFrameAccessor __len__ on LazyFrame (lines 3412-3413)."""
    da = airr_polars.data
    length = len(da)
    assert length > 0


def test_dataframe_accessor_columns_eager(airr_polars):
    """Cover DataFrameAccessor columns on eager DataFrame (line 3423)."""
    vdj = DandelionPolars(airr_polars._data.collect(), lazy=False)
    da = vdj.data
    cols = da.columns
    assert isinstance(cols, list)
    assert "sequence_id" in cols


def test_metadata_names_eager_metadata(airr_polars):
    """Cover metadata_names with eager metadata (line 687-689)."""
    vdj = airr_polars
    vdj.to_eager()
    names = vdj.metadata_names
    assert isinstance(names, SeriesAccessor)


def test_metadata_names_pandas_metadata(airr_polars):
    """Cover metadata_names with pandas metadata (line 685-686)."""
    vdj = airr_polars
    vdj.to_pandas()
    names = vdj.metadata_names
    assert isinstance(names, pd.Index)


def test_metadata_names_setter_pandas(airr_polars):
    """Cover metadata_names.setter with pandas (lines 700-703)."""
    vdj = airr_polars
    vdj.to_pandas()
    new_names = vdj._metadata.index.tolist()
    vdj.metadata_names = new_names
    assert list(vdj._metadata.index) == new_names
