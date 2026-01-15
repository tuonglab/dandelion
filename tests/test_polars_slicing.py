"""Test slicing/indexing of DandelionPolars objects."""
import pytest
import polars as pl
import numpy as np
from dandelion.utilities._polars import DandelionPolars


@pytest.fixture
def vdj_polars(vdj_small):
    """Convert vdj_small to DandelionPolars."""
    # vdj_small is a pandas-based Dandelion object
    # Convert its data to polars format
    return DandelionPolars(vdj_small.data)


# Test slicing by cell IDs (list, tuple, set, array)

def test_slice_by_list(vdj_polars):
    """Test slicing with list of cell IDs."""
    cell_ids = vdj_polars.metadata_names[:3].to_list()
    result = vdj_polars[cell_ids]
    assert result.n_obs == 3
    assert result.n_contigs > 0


def test_slice_by_tuple(vdj_polars):
    """Test slicing with tuple of cell IDs."""
    cell_ids = tuple(vdj_polars.metadata_names[:3].to_list())
    result = vdj_polars[cell_ids]
    assert result.n_obs == 3


def test_slice_by_set(vdj_polars):
    """Test slicing with set of cell IDs."""
    cell_ids = set(vdj_polars.metadata_names[:3].to_list())
    result = vdj_polars[cell_ids]
    assert result.n_obs == 3


def test_slice_by_array(vdj_polars):
    """Test slicing with numpy array of cell IDs."""
    cell_ids = np.array(vdj_polars.metadata_names[:3].to_list())
    result = vdj_polars[cell_ids]
    assert result.n_obs == 3


# Test slicing with polars Expressions (from data columns)

def test_slice_by_data_expression(vdj_polars):
    """Test slicing with expression on data column."""
    result = vdj_polars[pl.col("productive") == "T"]
    assert result.n_obs > 0
    assert result.n_contigs > 0


def test_slice_by_multiple_conditions(vdj_polars):
    """Test slicing with multiple conditions on data columns."""
    result = vdj_polars[
        (pl.col("productive") == "T")
        & (pl.col("locus").is_in(["IGH", "IGK", "IGL"]))
    ]
    assert result.n_obs >= 0


def test_slice_by_locus_expression(vdj_polars):
    """Test slicing by locus expression."""
    result = vdj_polars[pl.col("locus") == "IGH"]
    assert result.n_contigs >= 0


# Test slicing with metadata column expressions

def test_slice_by_metadata_chain_status(vdj_polars):
    """Test slicing by chain_status metadata column."""
    # Get all unique values first to find one that exists
    if isinstance(vdj_polars._metadata, pl.LazyFrame):
        unique_statuses = (
            vdj_polars._metadata.select("chain_status")
            .collect()
            .to_series()
            .unique()
            .to_list()
        )
    else:
        unique_statuses = (
            vdj_polars._metadata.select("chain_status")
            .to_series()
            .unique()
            .to_list()
        )

    if unique_statuses:  # If there are values
        test_status = unique_statuses[0]
        result = vdj_polars[vdj_polars.metadata.chain_status == test_status]
        assert result.n_obs > 0
        assert result.n_obs <= vdj_polars.n_obs


def test_slice_by_metadata_isotype_status(vdj_polars):
    """Test slicing by isotype_status metadata column."""
    if "isotype_status" in vdj_polars.metadata.columns:
        result = vdj_polars[vdj_polars.metadata.isotype_status != "Multi"]
        assert result.n_obs >= 0


def test_slice_by_metadata_productive(vdj_polars):
    """Test slicing by productive_VDJ metadata column."""
    if "productive_VDJ" in vdj_polars.metadata.columns:
        result = vdj_polars[vdj_polars.metadata["productive_VDJ"] == "T"]
        assert result.n_obs >= 0


def test_slice_by_metadata_inequality(vdj_polars):
    """Test slicing with inequality operator on metadata."""
    result = vdj_polars[vdj_polars.metadata.chain_status != "Single pair"]
    assert result.n_obs >= 0


def test_slice_by_metadata_isin(vdj_polars):
    """Test slicing with is_in operator on metadata."""
    valid_statuses = ["Single pair", "Orphan VDJ"]
    result = vdj_polars[
        vdj_polars.metadata.chain_status.is_in(valid_statuses)
    ]
    assert result.n_obs >= 0


# Test slicing with Series (cell IDs or boolean mask)

def test_slice_by_cell_id_series(vdj_polars):
    """Test slicing with Series of cell IDs."""
    cell_ids = vdj_polars.metadata_names[:3]
    result = vdj_polars[cell_ids]
    assert result.n_obs == 3


def test_slice_by_boolean_series_data(vdj_polars):
    """Test slicing with boolean Series from data."""
    if isinstance(vdj_polars._data, pl.DataFrame):
        mask = vdj_polars._data["productive"].cast(pl.Boolean)
        result = vdj_polars[mask]
        assert result.n_contigs > 0


# Test slicing with DataFrame/LazyFrame

def test_slice_by_filtered_lazyframe(vdj_polars):
    """Test slicing with a filtered LazyFrame."""
    if isinstance(vdj_polars._data, pl.LazyFrame):
        filtered_data = vdj_polars._data.filter(pl.col("productive") == "T")
    else:
        filtered_data = vdj_polars._data.filter(pl.col("productive") == "T")
    result = vdj_polars[filtered_data]
    assert result.n_contigs > 0


# Test that slicing maintains data/metadata consistency

def test_data_metadata_consistency(vdj_polars):
    """Test that data and metadata remain in sync after slicing."""
    result = vdj_polars[vdj_polars.metadata.chain_status == "Single pair"]

    # Collect cell_ids from data
    if isinstance(result._data, pl.LazyFrame):
        data_cell_ids = set(
            result._data.select("cell_id")
            .collect(engine="streaming")
            .to_series()
            .unique()
            .to_list()
        )
    else:
        data_cell_ids = set(
            result._data.select("cell_id").to_series().unique().to_list()
        )

    # Get cell_ids from metadata
    if isinstance(result._metadata, pl.LazyFrame):
        meta_cell_ids = set(
            result._metadata.select("cell_id")
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
    else:
        meta_cell_ids = set(
            result._metadata.select("cell_id").to_series().to_list()
        )

    assert (
        data_cell_ids == meta_cell_ids
    ), "Data and metadata cell_ids don't match"


def test_slice_preserves_original(vdj_polars):
    """Test that slicing doesn't modify the original object."""
    original_n_obs = vdj_polars.n_obs
    original_n_contigs = vdj_polars.n_contigs

    _ = vdj_polars[vdj_polars.metadata.chain_status == "Single pair"]

    assert vdj_polars.n_obs == original_n_obs
    assert vdj_polars.n_contigs == original_n_contigs


def test_slice_multiple_times(vdj_polars):
    """Test slicing the sliced object."""
    result1 = vdj_polars[vdj_polars.metadata.chain_status == "Single pair"]
    result2 = result1[pl.col("productive") == "T"]

    assert result2.n_obs <= result1.n_obs
    assert result2.n_contigs <= result1.n_contigs


# Test edge cases and error handling

def test_slice_empty_result(vdj_polars):
    """Test slicing that results in empty data."""
    # Filter for a very unlikely combination
    result = vdj_polars[
        (pl.col("productive") == "T") & (pl.col("locus") == "ZZZZZ")
    ]
    assert result.n_contigs == 0


def test_slice_all_cells(vdj_polars):
    """Test slicing that includes all cells."""
    all_cells = vdj_polars.metadata_names.to_list()
    result = vdj_polars[all_cells]
    assert result.n_obs == vdj_polars.n_obs


def test_invalid_index_type(vdj_polars):
    """Test that invalid index type raises TypeError."""
    with pytest.raises(TypeError):
        _ = vdj_polars[12.5]  # Invalid type


# Test slicing behavior with lazy vs eager frames

def test_slice_lazy_object(vdj_polars):
    """Test slicing lazy Dandelion object."""
    # Convert to lazy
    if hasattr(vdj_polars, "lazy"):
        original_lazy = vdj_polars.lazy
        vdj_polars.lazy = True
        result = vdj_polars[
            vdj_polars.metadata.chain_status == "Single pair"
        ]
        assert isinstance(result._data, (pl.LazyFrame, pl.DataFrame))
        vdj_polars.lazy = original_lazy


def test_slice_eager_object(vdj_polars):
    """Test slicing eager Dandelion object."""
    # Ensure eager
    if hasattr(vdj_polars, "lazy"):
        original_lazy = vdj_polars.lazy
        vdj_polars.lazy = False
        result = vdj_polars[
            vdj_polars.metadata.chain_status == "Single pair"
        ]
        assert isinstance(result._data, (pl.LazyFrame, pl.DataFrame))
        vdj_polars.lazy = original_lazy
