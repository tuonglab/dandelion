"""
Test to verify that the polars-based calculate_distance_matrix_original
produces the same results as the pandas-based version.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dandelion.utilities._io import read_10x_vdj
from dandelion.utilities._polars import read_10x_vdj_polars
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


def _build_membership_from_metadata(dat, clone_key: str = "clone_id"):
    """Create clone membership mapping from metadata."""
    membership: dict[str, list[str]] = {}

    # Handle different metadata types
    if isinstance(dat._metadata, pl.LazyFrame):
        clone_data = dat._metadata.select(["cell_id", clone_key]).collect(
            engine="streaming"
        )
        clone_data = clone_data.to_pandas()
    elif isinstance(dat._metadata, pl.DataFrame):
        clone_data = dat._metadata.select(["cell_id", clone_key]).to_pandas()
    else:
        # Pandas DataFrame - index is cell_id
        clone_data = dat._metadata[[clone_key]].copy()
        clone_data["cell_id"] = clone_data.index

    for idx, row in clone_data.iterrows():
        cell_name = row["cell_id"]
        clone_str = row[clone_key]

        if pd.notna(clone_str) and clone_str != "None":
            clone_ids = str(clone_str).split("|")
            for clone_id in clone_ids:
                clone_id = clone_id.strip()
                if clone_id and clone_id != "None":
                    if clone_id not in membership:
                        membership[clone_id] = []
                    if cell_name not in membership[clone_id]:
                        membership[clone_id].append(cell_name)
    return membership


@pytest.fixture
def mouse_vdj_membership(annotation_10x_mouse):
    """Parsed mouse VDJ data plus clone membership mapping."""
    dat = read_10x_vdj_polars(annotation_10x_mouse)
    membership = _build_membership_from_metadata(dat, clone_key="clone_id")

    # Process data to get sequence columns only (like the actual function does)
    # Filter to relevant loci
    dat_filtered = dat[
        dat.data.locus.is_in(["IGH", "TRB", "TRD", "IGK", "IGL", "TRA", "TRG"])
    ]

    # Split by junction to get sequence columns - use join=True to get strings
    dat_seq = dat_filtered._split("junction", explode=True)

    return dat_seq, membership


def _assert_distance_matrix_original_matches(data, membership):
    """Helper to compare pandas vs polars original distance matrices."""
    # Handle polars LazyFrame/DataFrame
    if isinstance(data, (pl.LazyFrame, pl.DataFrame)):
        if isinstance(data, pl.LazyFrame):
            data_collected = data.collect(engine="streaming")
        else:
            data_collected = data

        # For pandas, convert and split pipe-delimited columns
        df_pandas = data_collected.to_pandas()

        # Split pipe-delimited columns into separate columns for pandas
        new_cols = {}
        cols_to_drop = []
        for col in df_pandas.columns:
            if col != "cell_id":
                # Check if column contains pipe-delimited strings
                first_val = None
                for val in df_pandas[col]:
                    if pd.notna(val) and val:
                        first_val = val
                        break

                if (
                    first_val is not None
                    and isinstance(first_val, str)
                    and "|" in first_val
                ):
                    # Split by pipe
                    split_df = df_pandas[col].str.split("|", expand=True)
                    for i in range(split_df.shape[1]):
                        new_cols[f"{col}_{i}"] = split_df[i].fillna("")
                    cols_to_drop.append(col)

        # Add new columns
        for col_name, col_data in new_cols.items():
            df_pandas[col_name] = col_data

        # Drop original pipe-delimited columns
        if cols_to_drop:
            df_pandas = df_pandas.drop(columns=cols_to_drop)

        df_pandas.set_index("cell_id", inplace=True)

        # For polars, use as is
        df_polars = data_collected
    else:
        # Already pandas
        df_pandas = pd.DataFrame(data)
        df_pandas.set_index("cell_id", inplace=True)
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

    assert np.array_equal(
        np.isnan(result_pandas), np.isnan(result_polars)
    ), "NaN positions don't match between pandas and polars implementations"

    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Distance matrices differ between pandas and polars implementations",
    )


def test_calculate_distance_matrix_original_polars_vs_pandas(
    mouse_vdj_membership,
):
    """
    Test that polars and pandas implementations produce identical results.
    """
    data, membership = mouse_vdj_membership
    _assert_distance_matrix_original_matches(data, membership)
    print("✓ Pandas and Polars implementations produce identical results!")


def test_calculate_distance_matrix_with_padding(mouse_vdj_membership):
    """
    Test with pad_to_max=True.
    """
    data, membership = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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


def test_calculate_distance_matrix_multiple_clones(mouse_vdj_membership):
    """
    Test with multiple clones of varying sizes.
    """
    data, membership = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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


def test_calculate_distance_matrix_full_with_padding(mouse_vdj_membership):
    """
    Test full mode with pad_to_max=True.
    """
    data, _ = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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


def test_calculate_distance_matrix_full_multicore(mouse_vdj_membership):
    """
    Test full mode with multiple cores (n_cpus > 1).
    """
    data, _ = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True, drop=True)
    # remove cell_id from pandas for comparison

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

    metric = LevenshteinMetric()

    assert np.array_equal(
        df_pandas.values, df_polars.select(pl.exclude("cell_id")).to_numpy()
    )

    seq_cols = [col for col in df_pandas.columns]
    seq_colsp = [
        col for col in df_polars.collect_schema().names() if col != "cell_id"
    ]
    assert np.array_equal(np.array(seq_cols), np.array(seq_colsp))

    result_pandas = calc_dist_full_pandas(
        df_pandas, metric=metric, pad_to_max=False, n_cpus=2, verbose=False
    )

    result_polars = calc_dist_full_polars(
        df_polars, metric=metric, pad_to_max=False, n_cpus=2, verbose=False
    )
    # Compare
    assert np.array_equal(result_pandas, result_polars, equal_nan=True)
    assert np.array_equal(np.isnan(result_pandas), np.isnan(result_polars))
    non_nan_mask = ~np.isnan(result_pandas)
    np.testing.assert_allclose(
        result_pandas[non_nan_mask],
        result_polars[non_nan_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    print("✓ Multicore test passed (full mode)!")


def test_calculate_distance_matrix_long_clone_mode_polars_vs_pandas(
    mouse_vdj_membership,
):
    """
    Test that polars and pandas implementations of _long produce identical results in clone mode.
    """
    data, membership = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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
    assert np.array_equal(result_pandas, result_polars, equal_nan=True)

    print(
        "✓ Pandas and Polars implementations (long mode with clones) produce identical results!"
    )


def test_calculate_distance_matrix_long_with_padding(mouse_vdj_membership):
    """
    Test long mode with pad_to_max=True.
    """
    data, membership = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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


def test_calculate_distance_matrix_long_multicore(mouse_vdj_membership):
    """
    Test long mode with multiple cores (n_cpus > 1) in full mode.
    """
    data, _ = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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


def test_calculate_distance_matrix_long_with_empty_sequences(
    mouse_vdj_membership,
):
    """
    Test long mode with some empty/None sequences in clone mode.
    """
    data, membership = mouse_vdj_membership

    # For pandas, convert and handle pipe-delimited columns
    if isinstance(data, pl.DataFrame):
        df_pandas = data.to_pandas()
    else:
        df_pandas = pd.DataFrame(data)
    df_pandas.set_index("cell_id", inplace=True)

    # Polars version
    df_polars = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

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
    from tests.fixtures.fixtures_mouse import (
        annotation_10x_mouse as annotation_10x_mouse_fixture,
    )

    # Create the fixture data
    _dat_polars = read_10x_vdj_polars(annotation_10x_mouse_fixture())
    _membership = _build_membership_from_metadata(
        _dat_polars, clone_key="clone_id"
    )
    _dat_filtered = _dat_polars[
        _dat_polars.data.locus.is_in(
            ["IGH", "TRB", "TRD", "IGK", "IGL", "TRA", "TRG"]
        )
    ]
    _dat_seq = _dat_filtered._split("junction", explode=True)
    _fixture = (_dat_seq, _membership)

    test_calculate_distance_matrix_original_polars_vs_pandas(_fixture)
    test_calculate_distance_matrix_with_padding(_fixture)
    test_calculate_distance_matrix_with_empty_sequences(_fixture)
    test_calculate_distance_matrix_multiple_clones(_fixture)
    test_calculate_distance_matrix_original_full_polars_vs_pandas(_fixture)
    test_calculate_distance_matrix_full_with_empty_sequences(_fixture)
    test_calculate_distance_matrix_full_with_padding(_fixture)
    test_calculate_distance_matrix_full_multicore(_fixture)
    test_calculate_distance_matrix_long_clone_mode_polars_vs_pandas(_fixture)
    test_calculate_distance_matrix_long_full_mode_polars_vs_pandas(_fixture)
    test_calculate_distance_matrix_long_with_padding(_fixture)
    test_calculate_distance_matrix_long_multicore(_fixture)
    test_calculate_distance_matrix_long_with_empty_sequences(_fixture)
    print("\n✅ All tests passed!")
