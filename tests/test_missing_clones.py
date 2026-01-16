import pytest
import pandas as pd
import polars as pl


def test_missing_clones_in_metadata(annotation_10x_mouse):
    """Find which clones are missing from polars metadata"""
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

    print("\n=== MISSING CLONES ANALYSIS ===")

    # Get clone IDs from both
    pd_data = load_data(vdj_pd._data)
    pl_data = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    pd_vdj_clones = set(pd_data["clone_id"].unique())
    pl_vdj_clones = set(pl_data["clone_id"].to_list())

    pd_meta = vdj_pd._metadata
    pl_meta = vdj_pl._metadata
    if isinstance(pl_meta, pl.LazyFrame):
        pl_meta = pl_meta.collect(engine="streaming")

    pd_meta_clones = set(pd_meta["clone_id"].unique())
    pl_meta_clones = set(pl_meta["clone_id"].to_list())

    print(f"VDJ Data clones:")
    print(f"  Pandas: {len(pd_vdj_clones)}")
    print(f"  Polars: {len(pl_vdj_clones)}")

    print(f"\nMetadata clones:")
    print(f"  Pandas: {len(pd_meta_clones)}")
    print(f"  Polars: {len(pl_meta_clones)}")

    # Find missing clones
    clones_in_vdj_not_meta_pd = pd_vdj_clones - pd_meta_clones
    clones_in_vdj_not_meta_pl = pl_vdj_clones - pl_meta_clones

    print(f"\nClones in VDJ data but NOT in metadata:")
    print(f"  Pandas: {len(clones_in_vdj_not_meta_pd)} missing")
    if clones_in_vdj_not_meta_pd:
        for clone in list(clones_in_vdj_not_meta_pd)[:5]:
            print(f"    - {clone}")

    print(f"  Polars: {len(clones_in_vdj_not_meta_pl)} missing")
    if clones_in_vdj_not_meta_pl:
        for clone in list(clones_in_vdj_not_meta_pl)[:5]:
            print(f"    - {clone}")

    # Check if polars is missing different clones than pandas
    if clones_in_vdj_not_meta_pl != clones_in_vdj_not_meta_pd:
        print(f"\nClones missing in Polars but not Pandas:")
        missing_only_pl = clones_in_vdj_not_meta_pl - clones_in_vdj_not_meta_pd
        if missing_only_pl:
            for clone in missing_only_pl:
                print(f"    - {clone}")

        print(f"\nClones missing in Pandas but not Polars:")
        missing_only_pd = clones_in_vdj_not_meta_pd - clones_in_vdj_not_meta_pl
        if missing_only_pd:
            for clone in missing_only_pd:
                print(f"    - {clone}")
