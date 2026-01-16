import pytest
import pandas as pd
import polars as pl


def test_missing_cells_in_metadata(annotation_10x_mouse):
    """Find which cells are missing from polars metadata"""
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

    print("\n=== MISSING CELLS ANALYSIS ===")

    # Get cell IDs from both
    pd_meta = vdj_pd._metadata
    pl_meta = vdj_pl._metadata
    if isinstance(pl_meta, pl.LazyFrame):
        pl_meta = pl_meta.collect(engine="streaming")

    pd_cells = set(pd_meta.index.tolist())
    pl_cells = set(pl_meta["cell_id"].to_list())

    print(f"Pandas metadata cells: {len(pd_cells)}")
    print(f"Polars metadata cells: {len(pl_cells)}")

    # Find missing cells
    cells_in_pd_not_pl = pd_cells - pl_cells
    cells_in_pl_not_pd = pl_cells - pd_cells

    print(f"\nCells in Pandas metadata but NOT in Polars metadata:")
    print(f"  Count: {len(cells_in_pd_not_pl)}")
    if cells_in_pd_not_pl:
        for cell in list(cells_in_pd_not_pl)[:10]:
            print(f"    - {cell}")
            # Check if this cell has clone_id in pandas
            pd_clone = (
                pd_meta.loc[cell, "clone_id"]
                if "clone_id" in pd_meta.columns
                else None
            )
            print(f"      Pandas clone_id: {pd_clone}")

    print(f"\nCells in Polars metadata but NOT in Pandas metadata:")
    print(f"  Count: {len(cells_in_pl_not_pd)}")
    if cells_in_pl_not_pd:
        for cell in list(cells_in_pl_not_pd)[:10]:
            print(f"    - {cell}")
