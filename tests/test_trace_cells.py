import pytest
import pandas as pd
import polars as pl


def test_trace_missing_cells(annotation_10x_mouse):
    """Trace where the 2 cells get lost"""
    from dandelion.tools._tools_polars import find_clones as find_clones_polars
    from dandelion.utilities._polars import read_10x_vdj_polars

    # Load data
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    missing_cells = ["AGCCTAAGTGTTTGTG-1", "GACGCGTCAGTAGAGC-1"]

    print("\n=== TRACING MISSING CELLS ===")

    # Check if cells exist in initial data
    data_before = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )
    print(f"\nBefore find_clones:")
    print(f"  Total sequences: {data_before.shape[0]}")
    for cell in missing_cells:
        seqs = data_before.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        if seqs.shape[0] > 0:
            print(f"    sequence_ids: {seqs['sequence_id'].to_list()}")
            print(f"    locus: {seqs['locus'].to_list()}")

    # Check metadata before find_clones
    if vdj_pl._metadata is not None:
        meta_before = (
            vdj_pl._metadata.collect(engine="streaming")
            if isinstance(vdj_pl._metadata, pl.LazyFrame)
            else vdj_pl._metadata
        )
        print(f"\n  Metadata before: {meta_before.shape[0]} cells")
        for cell in missing_cells:
            if cell in meta_before["cell_id"].to_list():
                print(f"    Cell {cell}: EXISTS in metadata")
            else:
                print(f"    Cell {cell}: MISSING from metadata")

    # Run find_clones
    find_clones_polars(vdj_pl, verbose=False)

    # Check after find_clones
    data_after = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )
    print(f"\nAfter find_clones:")
    print(f"  Total sequences: {data_after.shape[0]}")
    for cell in missing_cells:
        seqs = data_after.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        if seqs.shape[0] > 0:
            print(f"    sequence_ids: {seqs['sequence_id'].to_list()}")
            print(f"    clone_ids: {seqs['clone_id'].to_list()}")

    # Check metadata after
    meta_after = (
        vdj_pl._metadata.collect(engine="streaming")
        if isinstance(vdj_pl._metadata, pl.LazyFrame)
        else vdj_pl._metadata
    )
    print(f"\n  Metadata after: {meta_after.shape[0]} cells")
    for cell in missing_cells:
        if cell in meta_after["cell_id"].to_list():
            cell_data = meta_after.filter(pl.col("cell_id") == cell)
            print(f"    Cell {cell}: EXISTS in metadata")
            print(f"      clone_id: {cell_data['clone_id'].to_list()}")
        else:
            print(f"    Cell {cell}: MISSING from metadata")
