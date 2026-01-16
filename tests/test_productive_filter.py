import pytest
import pandas as pd
import polars as pl


def test_productive_filtering(annotation_10x_mouse):
    """Check if productive filtering is causing the issue"""
    from dandelion.utilities._polars import read_10x_vdj_polars

    missing_cells = ["AGCCTAAGTGTTTGTG-1", "GACGCGTCAGTAGAGC-1"]

    print("\n=== PRODUCTIVE FILTERING CHECK ===")

    # Load data
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse, remove_malformed=False)
    pl_data = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    print(f"\nBefore initialize_metadata:")
    for cell in missing_cells:
        seqs = pl_data.filter(pl.col("cell_id") == cell)
        print(f"\n  Cell {cell}: {seqs.shape[0]} sequences")
        for row in seqs.iter_rows(named=True):
            print(f"    {row['sequence_id']}")
            print(f"      locus: {row['locus']}")
            print(f"      productive: {row['productive']}")

    # Check what happens with productive_only filtering
    productive_data = pl_data.filter(
        pl.col("productive")
        .cast(pl.String)
        .str.to_uppercase()
        .is_in(["TRUE", "T", "1", "YES", "Y"])
    )

    print(f"\nAfter productive_only=True filter:")
    for cell in missing_cells:
        seqs = productive_data.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        if seqs.shape[0] > 0:
            for row in seqs.iter_rows(named=True):
                print(
                    f"    {row['sequence_id']}: productive={row['productive']}"
                )

    # Get unique cells from productive data
    unique_productive_cells = productive_data.select("cell_id").unique()
    print(
        f"\nTotal unique cells in productive data: {unique_productive_cells.shape[0]}"
    )

    for cell in missing_cells:
        if cell in unique_productive_cells["cell_id"].to_list():
            print(f"  Cell {cell}: IN productive cells")
        else:
            print(
                f"  Cell {cell}: NOT in productive cells (will be missing from metadata!)"
            )
