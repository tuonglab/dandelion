import pytest
import pandas as pd
import polars as pl


def test_pandas_productive_check(annotation_10x_mouse):
    """Check how pandas handles empty productive values"""
    from dandelion.utilities._io import read_10x_vdj

    missing_cells = ["AGCCTAAGTGTTTGTG-1", "GACGCGTCAGTAGAGC-1"]

    print("\n=== PANDAS PRODUCTIVE CHECK ===")

    print(f"\nRaw fixture productive values:")
    for cell in missing_cells:
        seqs = annotation_10x_mouse[annotation_10x_mouse["barcode"] == cell]
        print(f"\n{cell}:")
        for idx, row in seqs.iterrows():
            prod_val = row.get("productive", "N/A")
            print(
                f"  {row['contig_id']}: productive='{prod_val}' (type: {type(prod_val)})"
            )

    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    pd_data = vdj_pd._data

    print(f"\n\nPandas after read_10x_vdj:")
    for cell in missing_cells:
        seqs = pd_data[pd_data["cell_id"] == cell]
        print(f"\n{cell}: {len(seqs)} sequences")
        for idx, row in seqs.iterrows():
            print(f"  {row['sequence_id']}: productive='{row['productive']}'")

    pd_meta = vdj_pd._metadata
    print(f"\n\nPandas metadata: {pd_meta.shape[0]} cells")
    for cell in missing_cells:
        if cell in pd_meta.index:
            print(f"  {cell}: EXISTS in metadata")
        else:
            print(f"  {cell}: MISSING from metadata")
