import pytest
import pandas as pd
import polars as pl


def test_metadata_structure(annotation_10x_mouse):
    """Compare metadata structure between pandas and polars implementations"""
    from dandelion.tools import find_clones
    from dandelion.tools._tools_polars import find_clones as find_clones_polars
    from dandelion.utilities._io import read_10x_vdj
    from dandelion.utilities._polars import read_10x_vdj_polars

    # Load data using proper readers
    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    # Run find_clones
    vdj_pd = find_clones(vdj_pd, verbose=False)
    vdj_pl = find_clones_polars(vdj_pl, verbose=False)

    print("\n=== METADATA STRUCTURE COMPARISON ===")

    # Check vdj data shape and columns
    print(f"\nVDJ Data:")
    print(f"  Pandas vdj shape: {vdj_pd._data.shape}")
    print(f"  Polars vdj shape: {vdj_pl._data.shape}")
    print(f"  Pandas columns: {sorted(vdj_pd._data.columns.tolist())}")
    print(f"  Polars columns: {sorted(vdj_pl._data.columns)}")

    # Check metadata structure
    print(f"\nMetadata (_metadata):")
    pd_meta = vdj_pd._metadata
    pl_meta = vdj_pl._metadata
    if isinstance(pl_meta, pl.LazyFrame):
        pl_meta = pl_meta.collect()

    print(f"  Pandas metadata shape: {pd_meta.shape}")
    print(f"  Polars metadata shape: {pl_meta.shape}")
    print(f"  Pandas columns: {sorted(pd_meta.columns.tolist())}")
    print(f"  Polars columns: {sorted(pl_meta.columns)}")

    # Check what's in metadata
    print(f"\nMetadata content:")
    print(f"  Pandas unique clone_ids: {pd_meta['clone_id'].nunique()}")
    print(f"  Polars unique clone_ids: {pl_meta['clone_id'].n_unique()}")

    # Check if metadata rows correspond to cells or clones
    print(f"\nMetadata interpretation:")
    print(f"  Pandas metadata index name: {pd_meta.index.name}")
    print(f"  Pandas metadata index (first 5): {pd_meta.index[:5].tolist()}")
    if "cell_id" in pd_meta.columns:
        print(f"  Pandas cell_id unique: {pd_meta['cell_id'].nunique()}")

    # Compare metadata row counts and content
    print(f"\nMetadata row analysis:")
    print(
        f"  Difference: Pandas {pd_meta.shape[0]} vs Polars {pl_meta.shape[0]} = {pd_meta.shape[0] - pl_meta.shape[0]} missing rows in Polars"
    )

    # If there's a difference, show sample rows
    if pd_meta.shape[0] != pl_meta.shape[0]:
        print(f"\nDifference in metadata rows:")
        print(f"  Pandas has {pd_meta.shape[0]} rows")
        print(f"  Polars has {pl_meta.shape[0]} rows")
        print(f"  Difference: {pd_meta.shape[0] - pl_meta.shape[0]}")

        # Find rows in pandas not in polars using index
        pd_set = set(zip(pd_meta.index, pd_meta["clone_id"]))
        pl_set = (
            set(zip(pl_meta.to_pandas().index, pl_meta.to_pandas()["clone_id"]))
            if isinstance(pl_meta, pl.DataFrame)
            else set(zip(pl_meta.index, pl_meta["clone_id"]))
        )

        in_pd_not_pl = pd_set - pl_set
        in_pl_not_pd = pl_set - pd_set

        print(f"\n  Rows in Pandas not in Polars: {len(in_pd_not_pl)}")
        if in_pd_not_pl:
            for idx, clone in list(in_pd_not_pl)[:5]:
                print(f"    - {idx}: {clone}")

        print(f"\n  Rows in Polars not in Pandas: {len(in_pl_not_pd)}")
        if in_pl_not_pd:
            for idx, clone in list(in_pl_not_pd)[:5]:
                print(f"    - {idx}: {clone}")
