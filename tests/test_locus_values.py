import pytest
import pandas as pd
import polars as pl


def test_locus_values_before_filter(annotation_10x_mouse):
    """Check locus values before remove_malformed filter"""
    from dandelion.utilities._polars import read_10x_vdj_polars
    from dandelion.utilities._io import read_10x_vdj

    missing_cells = ["AGCCTAAGTGTTTGTG-1", "GACGCGTCAGTAGAGC-1"]

    print("\n=== LOCUS VALUES BEFORE FILTERING ===")

    # Check what's in the raw fixture
    print(f"\nRaw fixture data:")
    for cell in missing_cells:
        seqs = annotation_10x_mouse[annotation_10x_mouse["barcode"] == cell]
        print(f"  Cell {cell}: {len(seqs)} sequences")
        for idx, row in seqs.iterrows():
            print(f"    {row['contig_id']}: chain={row.get('chain', 'N/A')}")

    # Load with polars WITHOUT remove_malformed
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse, remove_malformed=False)
    pl_data = (
        vdj_pl._data.collect()
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    print(f"\nPolars (remove_malformed=False):")
    for cell in missing_cells:
        seqs = pl_data.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        for row in seqs.iter_rows(named=True):
            print(f"    {row['sequence_id']}: locus={row['locus']}")

    # Load with polars WITH remove_malformed (default)
    vdj_pl_filtered = read_10x_vdj_polars(
        annotation_10x_mouse, remove_malformed=True
    )
    pl_data_filtered = (
        vdj_pl_filtered._data.collect()
        if isinstance(vdj_pl_filtered._data, pl.LazyFrame)
        else vdj_pl_filtered._data
    )

    print(f"\nPolars (remove_malformed=True - default):")
    for cell in missing_cells:
        seqs = pl_data_filtered.filter(pl.col("cell_id") == cell)
        print(f"  Cell {cell}: {seqs.shape[0]} sequences")
        if seqs.shape[0] > 0:
            for row in seqs.iter_rows(named=True):
                print(f"    {row['sequence_id']}: locus={row['locus']}")

    # Load with pandas
    vdj_pd = read_10x_vdj(annotation_10x_mouse, remove_malformed=True)
    pd_data = vdj_pd._data

    print(f"\nPandas (remove_malformed=True - default):")
    for cell in missing_cells:
        seqs = pd_data[pd_data["cell_id"] == cell]
        print(f"  Cell {cell}: {len(seqs)} sequences")
        if len(seqs) > 0:
            for idx, row in seqs.iterrows():
                print(f"    {row['sequence_id']}: locus={row['locus']}")
