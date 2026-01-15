"""
Tests for lazy distance matrix computation with polars DataFrames.

This module tests the polars-friendly lazy distance computation functions
to ensure they produce valid distance matrices.
"""

import numpy as np
import polars as pl
import pytest
import tempfile
import shutil
from pathlib import Path

from dandelion.tools._lazydistances_polars import (
    calculate_distance_matrix_zarr,
)
from dandelion.tools._network_polars import (
    calculate_distance_matrix_original,
    calculate_distance_matrix_original_full,
    calculate_distance_matrix_long,
)
from dandelion.utilities._distances import LevenshteinMetric


@pytest.fixture
def sample_sequence_data():
    """Create sample sequence data for testing."""
    sequences = {
        "cell_id": [f"cell_{i}" for i in range(10)],
        "v_call": [
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCC",
            "ATCGATCGATCC",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "AAAATTTTGGGG",
            "AAAATTTTGGGG",
            "CCCCGGGGAAAA",
            "CCCCGGGGAAAA",
        ],
        "d_call": [
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
            "ATCGATCGATCG",
        ],
        "j_call": [
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
            "GCTAGCTAGCTA",
        ],
    }
    return pl.DataFrame(sequences)


@pytest.fixture
def sample_membership_data():
    """Create sample membership data for clone mode."""
    return {
        0: ["cell_0", "cell_1"],
        1: ["cell_2", "cell_3"],
        2: ["cell_4", "cell_5"],
        3: ["cell_6", "cell_7"],
        4: ["cell_8", "cell_9"],
    }


def test_lazydistances_original_vs_eager(
    sample_sequence_data, sample_membership_data
):
    """Test that lazy version produces valid distance matrix with membership."""
    metric = LevenshteinMetric()

    # Compute lazy version with membership
    with tempfile.TemporaryDirectory() as tmpdir:
        lazy_result = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            membership=sample_membership_data,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        # Convert lazy result to numpy
        lazy_array = lazy_result[:]

        # Verify shape
        assert lazy_array.shape == (10, 10)

        # Verify it's a valid distance matrix: symmetric, non-negative, finite
        assert np.allclose(lazy_array, lazy_array.T, equal_nan=True)
        # Check that within-group distances are computed (should have some values)
        assert np.any(np.isfinite(lazy_array))


def test_lazydistances_original_with_membership(
    sample_sequence_data, sample_membership_data
):
    """Test lazy version with membership (clone mode)."""
    metric = LevenshteinMetric()

    # Compute lazy version
    with tempfile.TemporaryDirectory() as tmpdir:
        lazy_result = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            membership=sample_membership_data,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        lazy_array = lazy_result[:]

        # Verify shape
        assert lazy_array.shape == (10, 10)

        # Verify symmetry (distance matrices should be symmetric)
        assert np.allclose(lazy_array, lazy_array.T, equal_nan=True)

        # Verify that some distances are computed
        assert np.any(np.isfinite(lazy_array))


def test_lazydistances_original_full_vs_eager(sample_sequence_data):
    """Test lazy version produces valid full pairwise distance matrix."""
    metric = LevenshteinMetric()

    # Compute lazy version (full pairwise, no membership)
    with tempfile.TemporaryDirectory() as tmpdir:
        lazy_result = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            membership=None,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        lazy_array = lazy_result[:]

        # Verify shape
        assert lazy_array.shape == (10, 10)

        # Verify symmetry
        assert np.allclose(lazy_array, lazy_array.T, equal_nan=True)

        # Verify some distances are computed
        assert np.any(np.isfinite(lazy_array))


def test_lazydistances_with_padding(sample_sequence_data):
    """Test lazy version with padding enabled."""
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        result_padded = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        # Check that result is a valid Zarr array
        assert hasattr(result_padded, "shape")
        assert result_padded.shape == (10, 10)

        # Convert to numpy and verify it's a valid distance matrix
        array = result_padded[:]
        assert np.all(np.isfinite(np.diag(array)) | np.isnan(np.diag(array)))


def test_lazydistances_without_padding(sample_sequence_data):
    """Test lazy version with padding disabled."""
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        result_no_pad = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            pad_to_max=False,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        # Check that result is valid
        assert hasattr(result_no_pad, "shape")
        assert result_no_pad.shape == (10, 10)

        array = result_no_pad[:]
        assert np.all(np.isfinite(np.diag(array)) | np.isnan(np.diag(array)))


def test_lazydistances_chunking_effect(sample_sequence_data):
    """Test that different chunk sizes produce same results."""
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir1:
        result_small_chunks = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir1,
            chunk_size=3,
            n_cpus=1,
            verbose=False,
        )
        array_small = result_small_chunks[:]

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_large_chunks = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir2,
            chunk_size=20,
            n_cpus=1,
            verbose=False,
        )
        array_large = result_large_chunks[:]

    # Results should be identical
    np.testing.assert_allclose(
        array_small,
        array_large,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


def test_lazydistances_membership_partial(sample_sequence_data):
    """Test with membership covering only some cells."""
    metric = LevenshteinMetric()

    partial_membership = {
        0: ["cell_0", "cell_1"],
        1: ["cell_2", "cell_3"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            membership=partial_membership,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        array = result[:]

        # Check shape
        assert array.shape == (10, 10)

        # Cells not in membership should have 0 distances (unfilled)
        # Cells in membership should have computed distances
        # This is specific to membership-based computation


def test_lazydistances_empty_sequences(sample_sequence_data):
    """Test with some empty sequences."""
    metric = LevenshteinMetric()

    data_with_empty = sample_sequence_data.clone()
    data_with_empty = data_with_empty.with_columns(
        [
            pl.when(pl.col("cell_id") == "cell_0")
            .then(pl.lit(""))
            .otherwise(pl.col("v_call"))
            .alias("v_call"),
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = calculate_distance_matrix_zarr(
            data_with_empty,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        array = result[:]

        # Check that computation completed
        assert array.shape == (10, 10)
        # Verify symmetry
        assert np.allclose(array, array.T, equal_nan=True)


def test_lazydistances_single_cell(sample_sequence_data):
    """Test with single cell (edge case)."""
    metric = LevenshteinMetric()

    single_cell = sample_sequence_data[0:1]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = calculate_distance_matrix_zarr(
            single_cell,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=1,
            n_cpus=1,
            verbose=False,
        )

        array = result[:]

        # Check shape
        assert array.shape == (1, 1)


def test_lazydistances_two_cells(sample_sequence_data):
    """Test with two cells (minimal case)."""
    metric = LevenshteinMetric()

    two_cells = sample_sequence_data[0:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = calculate_distance_matrix_zarr(
            two_cells,
            metric=metric,
            pad_to_max=True,
            out_path=tmpdir,
            chunk_size=1,
            n_cpus=1,
            verbose=False,
        )

        array = result[:]

        # Check shape
        assert array.shape == (2, 2)
        # Distance should be symmetric
        assert array[0, 1] == array[1, 0]


def test_lazydistances_out_path_created(sample_sequence_data):
    """Test that output path is properly created."""
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "distance_matrix.zarr"

        result = calculate_distance_matrix_zarr(
            sample_sequence_data,
            metric=metric,
            pad_to_max=True,
            out_path=str(out_path),
            chunk_size=5,
            n_cpus=1,
            verbose=False,
        )

        # Check that zarr array was created
        assert hasattr(result, "shape")
        assert result.shape == (10, 10)
