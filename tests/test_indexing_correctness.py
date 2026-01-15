"""
Test to verify that the indexing is correct in the polars implementation.
This test specifically checks that the distance matrix rows/columns correspond
to the correct cell_ids.
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


def test_indexing_correctness():
    """Test that distance matrix indices correspond to correct cell_ids."""

    # Create data with distinct sequences so we can verify indexing
    data_dict = {
        "v_call": [
            "AAAA",  # cell_0
            "TTTT",  # cell_1
            "GGGG",  # cell_2
            "CCCC",  # cell_3
            "ATAT",  # cell_4
        ],
        "d_call": ["XXXX"] * 5,
        "j_call": ["YYYY"] * 5,
    }

    cell_ids = [f"cell_{i}" for i in range(5)]

    # Pandas version with explicit index
    df_pandas = pd.DataFrame(data_dict, index=cell_ids)

    # Polars version with cell_id column
    data_with_cell_id = data_dict.copy()
    data_with_cell_id["cell_id"] = cell_ids
    df_polars = pl.DataFrame(data_with_cell_id)

    metric = LevenshteinMetric()

    # Compute distance matrices
    with tempfile.TemporaryDirectory() as tmpdir1:
        result_pandas = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=False,
            membership=None,
            zarr_path=tmpdir1 + "/pandas",
            chunk_size=3,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_polars = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=False,
            membership=None,
            out_path=tmpdir2,
            chunk_size=3,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_array = result_polars[:]

    # Test 1: Matrices should be identical
    np.testing.assert_allclose(
        result_pandas,
        result_polars_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
        err_msg="Distance matrices should be identical",
    )

    # Test 2: Verify specific distances
    # Distance from cell_0 (AAAA) to cell_1 (TTTT) should be 4
    assert (
        result_pandas[0, 1] == 4
    ), f"Expected distance 4, got {result_pandas[0, 1]}"
    assert (
        result_polars_array[0, 1] == 4
    ), f"Expected distance 4, got {result_polars_array[0, 1]}"

    # Distance from cell_0 (AAAA) to cell_2 (GGGG) should be 4
    assert (
        result_pandas[0, 2] == 4
    ), f"Expected distance 4, got {result_pandas[0, 2]}"
    assert (
        result_polars_array[0, 2] == 4
    ), f"Expected distance 4, got {result_polars_array[0, 2]}"

    # Distance from cell_1 (TTTT) to cell_2 (GGGG) should be 4
    assert (
        result_pandas[1, 2] == 4
    ), f"Expected distance 4, got {result_pandas[1, 2]}"
    assert (
        result_polars_array[1, 2] == 4
    ), f"Expected distance 4, got {result_polars_array[1, 2]}"

    # Diagonal should be NaN
    for i in range(5):
        assert np.isnan(
            result_pandas[i, i]
        ), f"Diagonal [{i}, {i}] should be NaN"
        assert np.isnan(
            result_polars_array[i, i]
        ), f"Diagonal [{i}, {i}] should be NaN"

    # Test 3: Symmetry
    for i in range(5):
        for j in range(i + 1, 5):
            assert (
                result_pandas[i, j] == result_pandas[j, i]
            ), f"Not symmetric at [{i}, {j}]"
            assert (
                result_polars_array[i, j] == result_polars_array[j, i]
            ), f"Not symmetric at [{i}, {j}]"


def test_indexing_with_membership():
    """Test that membership mode respects correct cell_id indexing."""

    # Create data
    data_dict = {
        "v_call": [
            "AAAA",  # cell_0 - clone_0
            "AAAT",  # cell_1 - clone_0
            "TTTT",  # cell_2 - clone_1
            "TTTG",  # cell_3 - clone_1
            "GGGG",  # cell_4 - clone_2
            "GGGC",  # cell_5 - clone_2
        ],
        "d_call": ["XXXX"] * 6,
        "j_call": ["YYYY"] * 6,
    }

    cell_ids = [f"cell_{i}" for i in range(6)]

    # Pandas version
    df_pandas = pd.DataFrame(data_dict, index=cell_ids)

    # Polars version
    data_with_cell_id = data_dict.copy()
    data_with_cell_id["cell_id"] = cell_ids
    df_polars = pl.DataFrame(data_with_cell_id)

    # Define membership
    membership = {
        "clone_0": ["cell_0", "cell_1"],
        "clone_1": ["cell_2", "cell_3"],
        "clone_2": ["cell_4", "cell_5"],
    }

    metric = LevenshteinMetric()

    # Compute distance matrices
    with tempfile.TemporaryDirectory() as tmpdir1:
        result_pandas = calculate_distance_matrix_zarr_pandas(
            df_pandas,
            metric=metric,
            pad_to_max=False,
            membership=membership,
            zarr_path=tmpdir1 + "/pandas",
            chunk_size=3,
            n_cpus=1,
            compress=False,
            lazy=False,
            verbose=False,
        )

    with tempfile.TemporaryDirectory() as tmpdir2:
        result_polars = calculate_distance_matrix_zarr_polars(
            df_polars,
            metric=metric,
            pad_to_max=False,
            membership=membership,
            out_path=tmpdir2,
            chunk_size=3,
            n_cpus=1,
            compress=False,
            verbose=False,
        )
        result_polars_array = result_polars[:]

    # Test 1: Matrices should be identical
    np.testing.assert_allclose(
        result_pandas,
        result_polars_array,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
        err_msg="Distance matrices with membership should be identical",
    )

    # Test 2: Within-clone distances should be computed
    # cell_0 (AAAA) to cell_1 (AAAT) - distance should be 1
    assert (
        result_pandas[0, 1] == 1
    ), f"Expected distance 1, got {result_pandas[0, 1]}"
    assert (
        result_polars_array[0, 1] == 1
    ), f"Expected distance 1, got {result_polars_array[0, 1]}"

    # cell_2 (TTTT) to cell_3 (TTTG) - distance should be 1
    assert (
        result_pandas[2, 3] == 1
    ), f"Expected distance 1, got {result_pandas[2, 3]}"
    assert (
        result_polars_array[2, 3] == 1
    ), f"Expected distance 1, got {result_polars_array[2, 3]}"

    # cell_4 (GGGG) to cell_5 (GGGC) - distance should be 1
    assert (
        result_pandas[4, 5] == 1
    ), f"Expected distance 1, got {result_pandas[4, 5]}"
    assert (
        result_polars_array[4, 5] == 1
    ), f"Expected distance 1, got {result_polars_array[4, 5]}"

    # Test 3: Between-clone distances should be 0 (not computed)
    # cell_0 to cell_2 (different clones)
    assert (
        result_pandas[0, 2] == 0
    ), f"Expected distance 0, got {result_pandas[0, 2]}"
    assert (
        result_polars_array[0, 2] == 0
    ), f"Expected distance 0, got {result_polars_array[0, 2]}"

    # cell_0 to cell_4 (different clones)
    assert (
        result_pandas[0, 4] == 0
    ), f"Expected distance 0, got {result_pandas[0, 4]}"
    assert (
        result_polars_array[0, 4] == 0
    ), f"Expected distance 0, got {result_polars_array[0, 4]}"


if __name__ == "__main__":
    test_indexing_correctness()
    test_indexing_with_membership()
    print("All indexing tests passed!")
