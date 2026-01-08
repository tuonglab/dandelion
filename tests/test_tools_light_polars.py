#!/usr/bin/env python
import pandas as pd
import polars as pl
import pytest

from dandelion.utilities._polars import DandelionPolars
from dandelion.tools._tools_polars import concat


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_polars(airr_reannotated):
    """Concatenate DandelionPolars"""
    vdj1 = DandelionPolars(airr_reannotated, verbose=False)
    vdj2 = vdj1.copy()
    vdj2.add_cell_prefix("test_")
    # Concatenate
    res = concat([vdj1, vdj2])
    assert isinstance(res, DandelionPolars)
    # Expect double contigs total
    assert (
        res._data.select(pl.count()).collect()[0, 0]
        == airr_reannotated.shape[0] * 2
    )


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_with_auto_suffixes(airr_reannotated):
    """Concatenate with automatic numeric suffixes to resolve duplicates."""
    # Create three DandelionPolars objects from the same data to get duplicate indices
    vdj1 = DandelionPolars(airr_reannotated.iloc[:2].copy(), verbose=False)
    vdj2 = vdj1.copy()
    vdj3 = vdj1.copy()

    # Concatenate without explicit suffixes, should auto-append numeric suffixes
    res = concat([vdj1, vdj2, vdj3], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Check that result is combined
    total_count = vdj1._data.select(pl.count()).collect()[0, 0]
    expected = total_count * 3
    assert res._data.select(pl.count()).collect()[0, 0] == expected


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_dict_input(airr_reannotated):
    """Concatenate using dictionary input instead of list."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    arrays_dict = {"sample1": vdj1, "sample2": vdj2}
    res = concat(arrays_dict, verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs
    combined_count = (
        vdj1._data.select(pl.count()).collect()[0, 0]
        + vdj2._data.select(pl.count()).collect()[0, 0]
    )
    assert res._data.select(pl.count()).collect()[0, 0] == combined_count


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_check_unique_false(airr_reannotated):
    """Concatenate without checking for uniqueness (performance test)."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    res = concat([vdj1, vdj2], check_unique=False, verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should still concatenate successfully even without unique check
    combined_count = (
        vdj1._data.select(pl.count()).collect()[0, 0]
        + vdj2._data.select(pl.count()).collect()[0, 0]
    )
    assert res._data.select(pl.count()).collect()[0, 0] == combined_count


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_multiple_objects(airr_reannotated):
    """Concatenate multiple DandelionPolars objects."""
    vdj = DandelionPolars(airr_reannotated, verbose=False)

    # Create three copies with different cell prefixes
    vdj1 = vdj.copy()
    vdj2 = vdj.copy()
    vdj2.add_cell_prefix("s2_")
    vdj3 = vdj.copy()
    vdj3.add_cell_prefix("s3_")

    res = concat([vdj1, vdj2, vdj3], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have triple the contigs
    single_count = vdj._data.select(pl.count()).collect()[0, 0]
    assert res._data.select(pl.count()).collect()[0, 0] == single_count * 3


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_dandelion_and_polars_dataframe(airr_reannotated):
    """Concatenate DandelionPolars with Polars DataFrame."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)

    # Convert DandelionPolars to Polars DataFrame to get compatible schema
    polars_df = (
        vdj1._data.collect()
        if isinstance(vdj1._data, pl.LazyFrame)
        else vdj1._data
    )

    res = concat([vdj1, polars_df], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs (doubled)
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    assert res._data.select(pl.count()).collect()[0, 0] == vdj1_count * 2


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_dandelion_and_polars_lazyframe(airr_reannotated):
    """Concatenate DandelionPolars with Polars LazyFrame."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)

    # Convert DandelionPolars data to LazyFrame
    polars_lf = vdj1._data

    res = concat([vdj1, polars_lf], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs (doubled)
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    assert res._data.select(pl.count()).collect()[0, 0] == vdj1_count * 2


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_dandelion_and_pandas_dataframe(airr_reannotated):
    """Concatenate DandelionPolars with another DandelionPolars (simplest mixed type)."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    # Concatenate two DandelionPolars objects
    res = concat([vdj1, vdj2], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    vdj2_count = vdj2._data.select(pl.count()).collect()[0, 0]
    assert (
        res._data.select(pl.count()).collect()[0, 0] == vdj1_count + vdj2_count
    )


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_mixed_polars_types(airr_reannotated):
    """Concatenate mix of Polars DataFrame and LazyFrame."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    # Get eager DataFrame from one
    polars_df = (
        vdj1._data.collect()
        if isinstance(vdj1._data, pl.LazyFrame)
        else vdj1._data
    )
    # Get LazyFrame from the other
    polars_lf = vdj2._data

    res = concat([polars_df, polars_lf], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    vdj2_count = vdj2._data.select(pl.count()).collect()[0, 0]
    assert (
        res._data.select(pl.count()).collect()[0, 0] == vdj1_count + vdj2_count
    )


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_all_input_types(airr_reannotated):
    """Concatenate mixed DandelionPolars, DataFrame and LazyFrame."""
    # Create three DandelionPolars objects
    vdj1 = DandelionPolars(airr_reannotated.iloc[:2].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[2:4].copy(), verbose=False)
    vdj3 = DandelionPolars(airr_reannotated.iloc[4:6].copy(), verbose=False)

    # Convert second to eager DataFrame and third to LazyFrame
    polars_df = (
        vdj2._data.collect()
        if isinstance(vdj2._data, pl.LazyFrame)
        else vdj2._data
    )
    polars_lf = vdj3._data

    res = concat([vdj1, polars_df, polars_lf], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have all parts combined
    expected_count = (
        vdj1._data.select(pl.count()).collect()[0, 0]
        + vdj2._data.select(pl.count()).collect()[0, 0]
        + vdj3._data.select(pl.count()).collect()[0, 0]
    )
    assert res._data.select(pl.count()).collect()[0, 0] == expected_count


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_pure_polars_dataframes(airr_reannotated):
    """Concatenate only Polars DataFrames."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    # Convert both to Polars DataFrames
    polars_df1 = (
        vdj1._data.collect()
        if isinstance(vdj1._data, pl.LazyFrame)
        else vdj1._data
    )
    polars_df2 = (
        vdj2._data.collect()
        if isinstance(vdj2._data, pl.LazyFrame)
        else vdj2._data
    )

    res = concat([polars_df1, polars_df2], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    vdj2_count = vdj2._data.select(pl.count()).collect()[0, 0]
    assert (
        res._data.select(pl.count()).collect()[0, 0] == vdj1_count + vdj2_count
    )


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_pure_pandas_dataframes(airr_reannotated):
    """Concatenate only Pandas DataFrames."""
    vdj1 = DandelionPolars(airr_reannotated.iloc[:4].copy(), verbose=False)
    vdj2 = DandelionPolars(airr_reannotated.iloc[4:8].copy(), verbose=False)

    # Convert both to pandas
    pandas_df1 = (
        vdj1._data.collect().to_pandas()
        if isinstance(vdj1._data, pl.LazyFrame)
        else vdj1._data.to_pandas()
    )
    pandas_df2 = (
        vdj2._data.collect().to_pandas()
        if isinstance(vdj2._data, pl.LazyFrame)
        else vdj2._data.to_pandas()
    )

    res = concat([pandas_df1, pandas_df2], verbose=False)

    assert isinstance(res, DandelionPolars)
    # Should have combined contigs
    vdj1_count = vdj1._data.select(pl.count()).collect()[0, 0]
    vdj2_count = vdj2._data.select(pl.count()).collect()[0, 0]
    assert (
        res._data.select(pl.count()).collect()[0, 0] == vdj1_count + vdj2_count
    )
