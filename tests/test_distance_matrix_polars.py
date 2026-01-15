"""
Test to verify that the polars-based calculate_distance_matrix_original
produces the same results as the pandas-based version.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polyleven import levenshtein

from dandelion.tools._network import (
    calculate_distance_matrix_original as calc_dist_pandas,
    calculate_distance_matrix_original_full as calc_dist_full_pandas,
    calculate_distance_matrix_long as calc_dist_long_pandas,
)
from dandelion.tools._network_polars import (
    calculate_distance_matrix_original as calc_dist_polars,
    calculate_distance_matrix_original_full as calc_dist_full_polars,
    calculate_distance_matrix_long as calc_dist_long_polars,
)
from dandelion.utilities._distances import LevenshteinMetric


def test_calculate_distance_matrix_original_polars_vs_pandas():
    """
    Test that polars and pandas implementations produce identical results.
    """
    # Create sample data with cell_id and sequence columns
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
            "CASSLGQAYEQY",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CALSEDRSGGSYIPTF",
        ],
    }

    # Create membership dict: clone_id -> list of cell_ids
    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3"],
        "clone_2": ["cell_4", "cell_5"],
    }

    # Create pandas DataFrame (for old function)
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Create polars DataFrame (for new function)
    df_polars = pl.DataFrame(data)

    # Get metric (use levenshtein directly)
    metric = LevenshteinMetric()

    # Run both functions
    result_pandas = calc_dist_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    result_polars = calc_dist_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    # Compare results (NaN values should be in the same positions)
    # Check that NaN positions match
    assert np.array_equal(
        np.isnan(result_pandas), np.isnan(result_polars)
    ), "NaN positions don't match between pandas and polars implementations"

    # Check that non-NaN values are close (allowing for floating point precision)
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Distance matrices differ between pandas and polars implementations",
    )

    print("✓ Pandas and Polars implementations produce identical results!")


def test_calculate_distance_matrix_with_padding():
    """
    Test with pad_to_max=True.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3"],
        "VDJ": ["CASSLGQAY", "CASSLGQ", "CASSYGQAYEQY"],
    }

    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=True,
        verbose=False,
    )

    result_polars = calc_dist_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=True,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Padding test passed!")


def test_calculate_distance_matrix_with_empty_sequences():
    """
    Test with some empty/None sequences.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
        "VDJ": ["CASSLGQAYEQY", "", "CASSYGQAYEQY", None],
        "VJ": ["CALSEDRSGGSYIPTF", "CALSEDRSGGSYIPTF", None, ""],
    }

    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3", "cell_4"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    result_polars = calc_dist_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Empty sequences test passed!")


def test_calculate_distance_matrix_multiple_clones():
    """
    Test with multiple clones of varying sizes.
    """
    data = {
        "cell_id": [f"cell_{i}" for i in range(10)],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",  # clone_1
            "CASSLTGELF",
            "CASSLTGELF",  # clone_2
            "CASRPGLAGGRPEQY",
            "CASRPGLAGGRPEQY",
            "CASRPGLAGGRPEQY",  # clone_3
            "CASSQETQY",
            "CASSQETQY",  # clone_4
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CALSEDRTGQKLVF",
            "CAMRVGDNSKLIW",
            "CAMRVGDNSKLIW",
            "CAMRVGDNSKLIW",
            "CAMREDSNYQLIW",
            "CAMREDSNYQLIW",
        ],
    }

    membership = {
        "clone_1": ["cell_0", "cell_1", "cell_2"],
        "clone_2": ["cell_3", "cell_4"],
        "clone_3": ["cell_5", "cell_6", "cell_7"],
        "clone_4": ["cell_8", "cell_9"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    result_polars = calc_dist_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Multiple clones test passed!")


def test_calculate_distance_matrix_original_full_polars_vs_pandas():
    """
    Test that polars and pandas implementations of _full produce identical results.
    """
    # Create sample data with cell_id and sequence columns
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
            "CASSLGQAYEQY",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CALSEDRSGGSYIPTF",
        ],
    }

    # Create pandas DataFrame (for old function)
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Create polars DataFrame (for new function)
    df_polars = pl.DataFrame(data)

    # Get metric
    metric = LevenshteinMetric()

    # Run both functions
    result_pandas = calc_dist_full_pandas(
        df_pandas, metric=metric, pad_to_max=False, n_cpus=1, verbose=False
    )

    result_polars = calc_dist_full_polars(
        df_polars, metric=metric, pad_to_max=False, n_cpus=1, verbose=False
    )

    # Compare results
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Distance matrices differ between pandas and polars implementations (full mode)",
    )

    print(
        "✓ Pandas and Polars implementations (full mode) produce identical results!"
    )


def test_calculate_distance_matrix_full_with_empty_sequences():
    """
    Test full mode with some empty/None sequences.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
        "VDJ": ["CASSLGQAYEQY", "", "CASSYGQAYEQY", None],
        "VJ": ["CALSEDRSGGSYIPTF", "CALSEDRSGGSYIPTF", None, ""],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_full_pandas(
        df_pandas, metric=metric, pad_to_max=False, n_cpus=1, verbose=False
    )

    result_polars = calc_dist_full_polars(
        df_polars, metric=metric, pad_to_max=False, n_cpus=1, verbose=False
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Empty sequences test passed (full mode)!")


def test_calculate_distance_matrix_full_with_padding():
    """
    Test full mode with pad_to_max=True.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3"],
        "VDJ": ["CASSLGQAY", "CASSLGQ", "CASSYGQAYEQY"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_full_pandas(
        df_pandas, metric=metric, pad_to_max=True, n_cpus=1, verbose=False
    )

    result_polars = calc_dist_full_polars(
        df_polars, metric=metric, pad_to_max=True, n_cpus=1, verbose=False
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Padding test passed (full mode)!")


def test_calculate_distance_matrix_full_multicore():
    """
    Test full mode with multiple cores (n_cpus > 1).
    """
    data = {
        "cell_id": [f"cell_{i}" for i in range(8)],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
            "CASRPGLAGGRPEQY",
            "CASRPGLAGGRPEQY",
            "CASSQETQY",
            "CASSQETQY",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CAMRVGDNSKLIW",
            "CAMRVGDNSKLIW",
            "CAMREDSNYQLIW",
            "CAMREDSNYQLIW",
        ],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_full_pandas(
        df_pandas, metric=metric, pad_to_max=False, n_cpus=2, verbose=False
    )

    result_polars = calc_dist_full_polars(
        df_polars, metric=metric, pad_to_max=False, n_cpus=2, verbose=False
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Multicore test passed (full mode)!")


def test_calculate_distance_matrix_long_clone_mode_polars_vs_pandas():
    """
    Test that polars and pandas implementations of _long produce identical results in clone mode.
    """
    # Create sample data with cell_id and sequence columns
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
            "CASSLGQAYEQY",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CALSEDRSGGSYIPTF",
        ],
    }

    # Create membership dict: clone_id -> list of cell_ids
    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3"],
        "clone_2": ["cell_4", "cell_5"],
    }

    # Create pandas DataFrame (for old function)
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Create polars DataFrame (for new function)
    df_polars = pl.DataFrame(data)

    # Get metric
    metric = LevenshteinMetric()

    # Run both functions
    result_pandas = calc_dist_long_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    result_polars = calc_dist_long_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    # Compare results
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Distance matrices differ between pandas and polars implementations (long mode with clones)",
    )

    print(
        "✓ Pandas and Polars implementations (long mode with clones) produce identical results!"
    )


def test_calculate_distance_matrix_long_full_mode_polars_vs_pandas():
    """
    Test that polars and pandas implementations of _long produce identical results in full mode (no membership).
    """
    # Create sample data
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
        ],
    }

    # Create pandas DataFrame (for old function)
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Create polars DataFrame (for new function)
    df_polars = pl.DataFrame(data)

    # Get metric
    metric = LevenshteinMetric()

    # Run both functions with membership=None
    result_pandas = calc_dist_long_pandas(
        df_pandas,
        membership=None,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    result_polars = calc_dist_long_polars(
        df_polars,
        membership=None,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    # Compare results
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Distance matrices differ between pandas and polars implementations (long mode full)",
    )

    print(
        "✓ Pandas and Polars implementations (long mode full) produce identical results!"
    )


def test_calculate_distance_matrix_long_with_padding():
    """
    Test long mode with pad_to_max=True.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3"],
        "VDJ": ["CASSLGQAY", "CASSLGQ", "CASSYGQAYEQY"],
    }

    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_long_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=True,
        n_cpus=1,
        verbose=False,
    )

    result_polars = calc_dist_long_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=True,
        n_cpus=1,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Padding test passed (long mode)!")


def test_calculate_distance_matrix_long_multicore():
    """
    Test long mode with multiple cores (n_cpus > 1) in full mode.
    """
    data = {
        "cell_id": [f"cell_{i}" for i in range(6)],
        "VDJ": [
            "CASSLGQAYEQY",
            "CASSLGQAYEQY",
            "CASSYGQAYEQY",
            "CASSLTGELF",
            "CASRPGLAGGRPEQY",
            "CASSQETQY",
        ],
        "VJ": [
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRSGGSYIPTF",
            "CALSEDRTGQKLVF",
            "CAMRVGDNSKLIW",
            "CAMREDSNYQLIW",
        ],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_long_pandas(
        df_pandas,
        membership=None,
        metric=metric,
        pad_to_max=False,
        n_cpus=2,
        verbose=False,
    )

    result_polars = calc_dist_long_polars(
        df_polars,
        membership=None,
        metric=metric,
        pad_to_max=False,
        n_cpus=2,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Multicore test passed (long mode)!")


def test_calculate_distance_matrix_long_with_empty_sequences():
    """
    Test long mode with some empty/None sequences in clone mode.
    """
    data = {
        "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
        "VDJ": ["CASSLGQAYEQY", "", "CASSYGQAYEQY", None],
        "VJ": ["CALSEDRSGGSYIPTF", "CALSEDRSGGSYIPTF", None, ""],
    }

    membership = {
        "clone_1": ["cell_1", "cell_2", "cell_3", "cell_4"],
    }

    # Pandas version
    df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = pl.DataFrame(data)

    metric = LevenshteinMetric()

    result_pandas = calc_dist_long_pandas(
        df_pandas,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    result_polars = calc_dist_long_polars(
        df_polars,
        membership=membership,
        metric=metric,
        pad_to_max=False,
        n_cpus=1,
        verbose=False,
    )

    # Compare
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Empty sequences test passed (long mode)!")


if __name__ == "__main__":
    test_calculate_distance_matrix_original_polars_vs_pandas()
    test_calculate_distance_matrix_with_padding()
    test_calculate_distance_matrix_with_empty_sequences()
    test_calculate_distance_matrix_multiple_clones()
    test_calculate_distance_matrix_original_full_polars_vs_pandas()
    test_calculate_distance_matrix_full_with_empty_sequences()
    test_calculate_distance_matrix_full_with_padding()
    test_calculate_distance_matrix_full_multicore()
    test_calculate_distance_matrix_long_clone_mode_polars_vs_pandas()
    test_calculate_distance_matrix_long_full_mode_polars_vs_pandas()
    test_calculate_distance_matrix_long_with_padding()
    test_calculate_distance_matrix_long_multicore()
    test_calculate_distance_matrix_long_with_empty_sequences()
    print("\n✅ All tests passed!")
    print("\n✅ All tests passed!")
