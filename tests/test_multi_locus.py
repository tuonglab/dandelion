import pytest
import pandas as pd
import polars as pl


def test_multi_locus_cells(annotation_10x_mouse):
    """Check if Multi locus cells are being filtered out"""
    from dandelion.utilities._io import read_10x_vdj
    from dandelion.utilities._polars import read_10x_vdj_polars

    # Load data
    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    missing_cells = ["AGCCTAAGTGTTTGTG-1", "GACGCGTCAGTAGAGC-1"]

    print("\n=== MULTI LOCUS ANALYSIS ===")

    # Check pandas
    pd_data = vdj_pd._data
    print(f"\nPandas data:")
    for cell in missing_cells:
        seqs = pd_data[pd_data["cell_id"] == cell]
        print(f"  Cell {cell}: {len(seqs)} sequences")
        if len(seqs) > 0:
            print(f"    locus: {seqs['locus'].tolist()}")

    pd_meta = vdj_pd._metadata
    print(f"\nPandas metadata: {pd_meta.shape[0]} cells")
    for cell in missing_cells:
        if cell in pd_meta.index:
            print(f"  Cell {cell}: EXISTS in pandas metadata")
        else:
            print(f"  Cell {cell}: MISSING from pandas metadata")

    # Check polars
    pl_data = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )
    print(f"\nPolars data:")
    for cell in missing_cells:
        seqs = pl_data.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        if seqs.shape[0] > 0:
            print(f"    locus: {seqs['locus'].to_list()}")

    pl_meta = (
        vdj_pl._metadata.collect(engine="streaming")
        if isinstance(vdj_pl._metadata, pl.LazyFrame)
        else vdj_pl._metadata
    )
    print(f"\nPolars metadata: {pl_meta.shape[0]} cells")
    for cell in missing_cells:
        if cell in pl_meta["cell_id"].to_list():
            print(f"  Cell {cell}: EXISTS in polars metadata")
        else:
            print(f"  Cell {cell}: MISSING from polars metadata")

    # Check all Multi locus sequences
    pl_multi = pl_data.filter(pl.col("locus") == "Multi")
    print(f"\n\nTotal sequences with locus='Multi': {pl_multi.shape[0]}")
    multi_cells = pl_multi["cell_id"].unique().to_list()
    print(f"Unique cells with Multi locus sequences: {len(multi_cells)}")

    # Check how many Multi cells are in metadata
    multi_in_meta = [
        c for c in multi_cells if c in pl_meta["cell_id"].to_list()
    ]
    print(
        f"Multi cells in polars metadata: {len(multi_in_meta)} / {len(multi_cells)}"
    )

    missing_multi = [
        c for c in multi_cells if c not in pl_meta["cell_id"].to_list()
    ]
    if missing_multi:
        print(f"\nCells with Multi locus MISSING from metadata:")
        for c in missing_multi[:10]:
            print(f"  - {c}")
