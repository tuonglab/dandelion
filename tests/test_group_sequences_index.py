"""Debug group_sequences index issue."""

import pytest
import polars as pl
import pandas as pd
from dandelion.utilities._io import read_10x_vdj
from dandelion.utilities._polars import read_10x_vdj_polars, load_polars
from dandelion.utilities._core import load_data
from dandelion.tools._tools import group_sequences


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_group_sequences_index(annotation_10x_mouse):
    """Debug group_sequences index."""
    print("\n" + "=" * 60)
    print("group_sequences Index Debug")
    print("=" * 60)

    # Load as pandas
    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    pd_data = load_data(vdj_pd._data)

    # Load as polars converted to pandas
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)
    dat_ = load_polars(vdj_pl._data)
    if isinstance(dat_, pl.LazyFrame):
        dat_ = dat_.collect(engine="streaming")
    if isinstance(dat_, pl.DataFrame):
        dat_ = dat_.to_pandas()

    # Get VDJ data for both
    vdj_loci = ["IGH"]
    pd_vdj = pd_data[pd_data["locus"].isin(vdj_loci)].copy()
    pl_vdj = dat_[dat_["locus"].isin(vdj_loci)].copy()

    print(f"\nPandas VDJ index type: {type(pd_vdj.index)}")
    print(f"Pandas VDJ first 3 index values: {pd_vdj.index[:3].tolist()}")
    print(
        f"Pandas VDJ 'sequence_id' column first 3: {pd_vdj['sequence_id'].iloc[:3].tolist()}"
    )

    print(f"\nPolars VDJ index type: {type(pl_vdj.index)}")
    print(f"Polars VDJ first 3 index values: {pl_vdj.index[:3].tolist()}")
    print(
        f"Polars VDJ 'sequence_id' column first 3: {pl_vdj['sequence_id'].iloc[:3].tolist()}"
    )

    # Group sequences for both
    print(f"\nGrouping sequences...")
    pd_vj_len, pd_seq_grp = group_sequences(
        pd_vdj,
        junction_key="junction_aa",
        recalculate_length=True,
        by_alleles=False,
        locus="ig",
    )

    pl_vj_len, pl_seq_grp = group_sequences(
        pl_vdj,
        junction_key="junction_aa",
        recalculate_length=True,
        by_alleles=False,
        locus="ig",
    )

    # Check what keys are in the trees
    pd_key = list(pd_seq_grp.keys())[0]
    pl_key = list(pl_seq_grp.keys())[0]

    pd_length_key = list(pd_seq_grp[pd_key].keys())[0]
    pl_length_key = list(pl_seq_grp[pl_key].keys())[0]

    print(f"\nPandas seq_grp sample keys at ({pd_key}, {pd_length_key}):")
    pd_seqs = list(pd_seq_grp[pd_key][pd_length_key].keys())[:3]
    for seq in pd_seqs:
        print(f"  {repr(seq)}")

    print(f"\nPolars seq_grp sample keys at ({pl_key}, {pl_length_key}):")
    pl_seqs = list(pl_seq_grp[pl_key][pl_length_key].keys())[:3]
    for seq in pl_seqs:
        print(f"  {repr(seq)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
