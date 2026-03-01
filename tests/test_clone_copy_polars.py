#!/usr/bin/env python
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from dandelion.polars.core import Dandelion
from dandelion.polars.tools import find_clones
from dandelion.polars.tools import generate_network


@pytest.mark.usefixtures("vdj_smaller")
@pytest.mark.parametrize("full_check", [True, False])
def test_clone_and_copy_consistency(vdj_smaller, full_check):
    """clone() and copy() should produce identical, independent objects."""

    def _to_pandas(df):
        if df is None:
            return None
        if isinstance(df, pl.LazyFrame):
            return df.collect(engine="streaming").to_pandas()
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    # Build a Polars-backed object from fixture
    base = Dandelion(vdj_smaller._data)
    if full_check:
        find_clones(base)
        generate_network(base, key="junction")

    cloned = base.clone()
    copied = base.copy()

    # Consistency of content
    assert_frame_equal(
        _to_pandas(base._data), _to_pandas(cloned._data), check_like=True
    )
    assert_frame_equal(
        _to_pandas(base._data), _to_pandas(copied._data), check_like=True
    )
    if base._metadata is not None:
        assert_frame_equal(
            _to_pandas(base._metadata),
            _to_pandas(cloned._metadata),
            check_like=True,
        )
        assert_frame_equal(
            _to_pandas(base._metadata),
            _to_pandas(copied._metadata),
            check_like=True,
        )

    # Independence: mutate clones and ensure originals unchanged
    cloned._data = (
        cloned._data.with_columns(pl.lit("x").alias("_tmp_col"))
        .collect(engine="streaming")
        .lazy()
    )
    if cloned._metadata is not None:
        cloned._metadata = (
            cloned._metadata.with_columns(pl.lit("y").alias("_tmp_meta_col"))
            .collect(engine="streaming")
            .lazy()
        )

    assert "_tmp_col" not in _to_pandas(base._data).columns
    if base._metadata is not None:
        assert "_tmp_meta_col" not in _to_pandas(base._metadata).columns

    # Cache handles are not shared
    assert base._cache_handles is not cloned._cache_handles
    assert base._cache_handles is not copied._cache_handles
