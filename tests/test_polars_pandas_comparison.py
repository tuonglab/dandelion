"""
Test that polars and pandas implementations produce identical distance matrices.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tempfile

from dandelion.tools._lazydistances import (
    calculate_distance_matrix_zarr as calculate_distance_matrix_zarr_pandas,
)
from dandelion.tools._lazydistances_polars import (
    calculate_distance_matrix_zarr as calculate_distance_matrix_zarr_polars,
)
from dandelion.utilities._distances import LevenshteinMetric


@pytest.fixture
def sample_data():
    """Create sample data in both pandas and polars formats."""
    data_dict = {
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
        "d_call": ["ATCGATCGATCG"] * 10,
        "j_call": ["GCTAGCTAGCTA"] * 10,
    }

    # Pandas version with explicit index
    df_pandas = pd.DataFrame(data_dict, index=[f"cell_{i}" for i in range(10)])

    # Polars version with cell_id column
    data_with_cell_id = data_dict.copy()
    data_with_cell_id["cell_id"] = [f"cell_{i}" for i in range(10)]
    df_polars = pl.DataFrame(data_with_cell_id)

    return df_pandas, df_polars


@pytest.fixture
def sample_membership():
    """Create sample membership data."""
    return {
        0: ["cell_0", "cell_1"],
        1: ["cell_2", "cell_3"],
        2: ["cell_4", "cell_5"],
        3: ["cell_6", "cell_7"],
        4: ["cell_8", "cell_9"],
    }


def test_full_pairwise_comparison(sample_data):
    """Test that full pairwise distance matrices are identical."""
    df_pandas, df_polars = sample_data
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir1:
        result_pandas = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=True,
            membership=None,
            zarr_path=tmpdir1 + "/pandas",
            chunk_size=5,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_polars = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=True,
            membership=None,
            out_path=tmpdir2,
            chunk_size=5,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_array = result_polars[:]

    # Compare results
    np.testing.assert_allclose(
        result_pandas,
        result_polars_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
        err_msg="Full pairwise distance matrices should be identical",
    )


def test_membership_mode_comparison(sample_data, sample_membership):
    """Test that membership mode produces identical results."""
    df_pandas, df_polars = sample_data
    metric = LevenshteinMetric()

    with tempfile.TemporaryDirectory() as tmpdir1:
        result_pandas = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=True,
            membership=sample_membership,
            zarr_path=tmpdir1 + "/pandas",
            chunk_size=5,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_polars = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=True,
            membership=sample_membership,
            out_path=tmpdir2,
            chunk_size=5,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_array = result_polars[:]

    # Compare results
    np.testing.assert_allclose(
        result_pandas,
        result_polars_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
        err_msg="Membership mode distance matrices should be identical",
    )

    # Additional checks for membership mode
    # Verify that only within-group distances are computed
    for clone_id, members in sample_membership.items():
        # Get indices for members
        member_indices = [int(m.split("_")[1]) for m in members]

        # Check that within-group distances exist
        for i in member_indices:
            for j in member_indices:
                if i != j:
                    assert np.isfinite(
                        result_pandas[i, j]
                    ), f"Missing distance for clone {clone_id}: {i}, {j}"
                    assert np.isfinite(
                        result_polars_array[i, j]
                    ), f"Missing distance for clone {clone_id}: {i}, {j}"


def test_padding_comparison(sample_data):
    """Test that padding parameter produces identical results."""
    df_pandas, df_polars = sample_data
    metric = LevenshteinMetric()

    # Test with padding
    with tempfile.TemporaryDirectory() as tmpdir1:
        result_pandas_pad = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=True,
            membership=None,
            zarr_path=tmpdir1 + "/pandas",
            chunk_size=5,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_polars_pad = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=True,
            membership=None,
            out_path=tmpdir2,
            chunk_size=5,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_pad_array = result_polars_pad[:]

    np.testing.assert_allclose(
        result_pandas_pad,
        result_polars_pad_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    # Test without padding
    with tempfile.TemporaryDirectory() as tmpdir3:
        result_pandas_nopad = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=False,
            membership=None,
            zarr_path=tmpdir3 + "/pandas",
            chunk_size=5,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir4:
        result_polars_nopad = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=False,
            membership=None,
            out_path=tmpdir4,
            chunk_size=5,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_nopad_array = result_polars_nopad[:]

    np.testing.assert_allclose(
        result_pandas_nopad,
        result_polars_nopad_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
