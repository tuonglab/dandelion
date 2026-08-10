"""Test vdj_sample implementation."""

import pandas as pd
import polars as pl

from dandelion.polars.core._core import DandelionPolars
from dandelion.polars.tools._tools import vdj_sample


def test_vdj_sample_without_replacement():
    """Test vdj_sample without replacement (size < total cells)."""
    # Create mock VDJ data
    mock_data = {
        "sequence_id": [
            f"cell_{i}_contig_{j}" for i in range(100) for j in range(2)
        ],
        "cell_id": [f"cell_{i}" for i in range(100) for _ in range(2)],
        "locus": ["IGH", "IGK"] * 100,
        "productive": ["T"] * 200,
        "v_call": ["IGHV1-1*01"] * 100 + ["IGKV1-1*01"] * 100,
        "d_call": [""] * 200,
        "j_call": ["IGHJ1*01"] * 100 + ["IGKJ1*01"] * 100,
        "c_call": ["IGHM"] * 100 + ["IGKC"] * 100,
        "junction": ["CAAAAA" + str(i) for i in range(100) for _ in range(2)],
        "junction_aa": ["QAAAA" + str(i) for i in range(100) for _ in range(2)],
    }

    df = pl.DataFrame(mock_data)
    vdj = DandelionPolars(df)

    # Sample without replacement
    vdj_sampled = vdj_sample(vdj, size=30, random_state=42)

    # Verify results - convert to eager if lazy
    if isinstance(vdj_sampled._metadata, pl.LazyFrame):
        vdj_sampled.to_eager()
    assert (
        vdj_sampled._metadata.shape[0] == 30
    ), "Should have 30 cells in metadata"
    assert not isinstance(
        vdj_sampled._metadata, pd.DataFrame
    ), "Metadata should be polars DataFrame"


def test_vdj_sample_with_replacement():
    """Test vdj_sample with replacement (size > total cells)."""
    # Create mock VDJ data
    mock_data = {
        "sequence_id": [
            f"cell_{i}_contig_{j}" for i in range(100) for j in range(2)
        ],
        "cell_id": [f"cell_{i}" for i in range(100) for _ in range(2)],
        "locus": ["IGH", "IGK"] * 100,
        "productive": ["T"] * 200,
        "v_call": ["IGHV1-1*01"] * 100 + ["IGKV1-1*01"] * 100,
        "d_call": [""] * 200,
        "j_call": ["IGHJ1*01"] * 100 + ["IGKJ1*01"] * 100,
        "c_call": ["IGHM"] * 100 + ["IGKC"] * 100,
        "junction": ["CAAAAA" + str(i) for i in range(100) for _ in range(2)],
        "junction_aa": ["QAAAA" + str(i) for i in range(100) for _ in range(2)],
    }

    df = pl.DataFrame(mock_data)
    vdj = DandelionPolars(df)

    # Sample with replacement (size > total)
    vdj_sampled = vdj_sample(vdj, size=150, random_state=42)

    # Verify results - convert to eager if lazy
    if isinstance(vdj_sampled._metadata, pl.LazyFrame):
        vdj_sampled.to_eager()
    assert (
        vdj_sampled._metadata.shape[0] == 150
    ), "Should have 150 cells in metadata"
    assert not isinstance(
        vdj_sampled._metadata, pd.DataFrame
    ), "Metadata should be polars DataFrame"

    # Check if suffixes were added correctly
    cell_ids = vdj_sampled._data["cell_id"].to_list()
    has_suffixes = any("-" in str(cid) for cid in cell_ids)
    assert has_suffixes, "Should have suffixed cell IDs for duplicates"


def test_vdj_sample_forced_replacement():
    """Test vdj_sample with forced replacement."""
    # Create mock VDJ data
    mock_data = {
        "sequence_id": [
            f"cell_{i}_contig_{j}" for i in range(100) for j in range(2)
        ],
        "cell_id": [f"cell_{i}" for i in range(100) for _ in range(2)],
        "locus": ["IGH", "IGK"] * 100,
        "productive": ["T"] * 200,
        "v_call": ["IGHV1-1*01"] * 100 + ["IGKV1-1*01"] * 100,
        "d_call": [""] * 200,
        "j_call": ["IGHJ1*01"] * 100 + ["IGKJ1*01"] * 100,
        "c_call": ["IGHM"] * 100 + ["IGKC"] * 100,
        "junction": ["CAAAAA" + str(i) for i in range(100) for _ in range(2)],
        "junction_aa": ["QAAAA" + str(i) for i in range(100) for _ in range(2)],
    }

    df = pl.DataFrame(mock_data)
    vdj = DandelionPolars(df)

    # Sample with forced replacement
    vdj_sampled = vdj_sample(vdj, size=30, force_replace=True, random_state=42)

    # Verify results - convert to eager if lazy
    if isinstance(vdj_sampled._metadata, pl.LazyFrame):
        vdj_sampled.to_eager()
    assert (
        vdj_sampled._metadata.shape[0] == 30
    ), "Should have 30 cells in metadata"
    assert not isinstance(
        vdj_sampled._metadata, pd.DataFrame
    ), "Metadata should be polars DataFrame"


def test_vdj_sample_preserves_structure():
    """Test that vdj_sample preserves data structure."""
    # Create mock VDJ data
    mock_data = {
        "sequence_id": [
            f"cell_{i}_contig_{j}" for i in range(50) for j in range(2)
        ],
        "cell_id": [f"cell_{i}" for i in range(50) for _ in range(2)],
        "locus": ["IGH", "IGK"] * 50,
        "productive": ["T"] * 100,
        "v_call": ["IGHV1-1*01"] * 50 + ["IGKV1-1*01"] * 50,
        "d_call": [""] * 100,
        "j_call": ["IGHJ1*01"] * 50 + ["IGKJ1*01"] * 50,
        "c_call": ["IGHM"] * 50 + ["IGKC"] * 50,
        "junction": ["CAAAAA" + str(i) for i in range(50) for _ in range(2)],
        "junction_aa": ["QAAAA" + str(i) for i in range(50) for _ in range(2)],
    }

    df = pl.DataFrame(mock_data)
    vdj = DandelionPolars(df)

    # Sample
    vdj_sampled = vdj_sample(vdj, size=20, random_state=42)

    # Verify structure is preserved
    assert isinstance(
        vdj_sampled, DandelionPolars
    ), "Should return DandelionPolars"
    assert isinstance(
        vdj_sampled._data, (pl.DataFrame, pl.LazyFrame)
    ), "Data should be polars DataFrame or LazyFrame"
    # Convert to eager if lazy to check metadata
    if isinstance(vdj_sampled._metadata, pl.LazyFrame):
        vdj_sampled.to_eager()
    assert not isinstance(
        vdj_sampled._metadata, pd.DataFrame
    ), "Metadata should be polars DataFrame"

    # Verify all sampled cells have sequences
    cell_ids = vdj_sampled._metadata["cell_id"].to_list()
    for cell_id in cell_ids:
        assert (
            vdj_sampled._data.filter(pl.col("cell_id") == cell_id).shape[0] > 0
        ), f"Cell {cell_id} should have sequences"


def test_vdj_sample_accepts_readonly_probability_array():
    """vdj_sample should handle read-only numpy arrays for p."""
    mock_data = {
        "sequence_id": [
            f"cell_{i}_contig_{j}" for i in range(20) for j in range(2)
        ],
        "cell_id": [f"cell_{i}" for i in range(20) for _ in range(2)],
        "locus": ["IGH", "IGK"] * 20,
        "productive": ["T"] * 40,
        "v_call": ["IGHV1-1*01"] * 20 + ["IGKV1-1*01"] * 20,
        "d_call": [""] * 40,
        "j_call": ["IGHJ1*01"] * 20 + ["IGKJ1*01"] * 20,
        "c_call": ["IGHM"] * 20 + ["IGKC"] * 20,
        "junction": ["CAAAAA" + str(i) for i in range(20) for _ in range(2)],
        "junction_aa": ["QAAAA" + str(i) for i in range(20) for _ in range(2)],
    }

    vdj = DandelionPolars(pl.DataFrame(mock_data))
    probs = (
        vdj._metadata.select(pl.lit(1.0 / vdj.n_obs).alias("p"))
        .collect(engine="streaming")
        .get_column("p")
        .to_numpy()
    )
    probs.setflags(write=False)

    vdj_sampled = vdj_sample(vdj, size=10, p=probs, random_state=42)

    if isinstance(vdj_sampled._metadata, pl.LazyFrame):
        vdj_sampled.to_eager()
    assert vdj_sampled._metadata.shape[0] == 10
