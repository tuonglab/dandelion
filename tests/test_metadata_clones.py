import pytest
import pandas as pd
import polars as pl


def test_clone_count_metadata(annotation_10x_mouse):
    """Check if metadata clone counts match between pandas and polars"""
    from dandelion.tools import find_clones
    from dandelion.tools._tools_polars import find_clones as find_clones_polars
    from dandelion.utilities._io import read_10x_vdj
    from dandelion.utilities._polars import read_10x_vdj_polars
    from dandelion.utilities._core import load_data

    # Load data using proper readers
    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    # Run find_clones
    vdj_pd = find_clones(vdj_pd, verbose=False)
    vdj_pl = find_clones_polars(vdj_pl, verbose=False)

    print("\n=== CLONE COUNT COMPARISON ===")

    # Count clones in vdj data
    pd_data = load_data(vdj_pd._data)
    pl_data = (
        vdj_pl._data.collect()
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    pd_clones = pd_data["clone_id"].nunique()
    pl_clones = pl_data["clone_id"].n_unique()

    print(f"\nVDJ Data Clone Counts:")
    print(f"  Pandas: {pd_clones}")
    print(f"  Polars: {pl_clones}")
    print(f"  Match: {pd_clones == pl_clones}")

    # Count clones in metadata
    pd_meta = vdj_pd._metadata
    pl_meta = vdj_pl._metadata
    if isinstance(pl_meta, pl.LazyFrame):
        pl_meta = pl_meta.collect()

    pd_meta_clones = pd_meta["clone_id"].nunique()
    pl_meta_clones = pl_meta["clone_id"].n_unique()

    print(f"\nMetadata Clone Counts:")
    print(f"  Pandas: {pd_meta_clones}")
    print(f"  Polars: {pl_meta_clones}")
    print(f"  Match: {pd_meta_clones == pl_meta_clones}")

    # Count cells in metadata
    print(f"\nMetadata Structure:")
    print(f"  Pandas rows: {pd_meta.shape[0]}")
    print(f"  Polars rows: {pl_meta.shape[0]}")

    # Check unique cells
    if "cell_id" in pd_meta.columns:
        pd_cells = pd_meta["cell_id"].nunique()
        pl_cells = pl_meta["cell_id"].n_unique()
        print(f"  Pandas unique cell_ids: {pd_cells}")
        print(f"  Polars unique cell_ids: {pl_cells}")

    # Main assertion - clone counts in vdj data should match
    assert (
        pd_clones == pl_clones
    ), f"Clone counts don't match in vdj data: {pd_clones} vs {pl_clones}"
    assert (
        pd_meta_clones == pl_meta_clones
    ), f"Clone counts don't match in metadata: {pd_meta_clones} vs {pl_meta_clones}"
