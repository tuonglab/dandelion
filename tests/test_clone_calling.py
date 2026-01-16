"""Check if clone naming convention is identical between pandas and polars."""

import pytest
import polars as pl
from sklearn.metrics import adjusted_rand_score

from dandelion.tools._tools import find_clones
from dandelion.tools._tools_polars import find_clones as find_clones_polars
from dandelion.utilities._io import read_10x_vdj
from dandelion.utilities._polars import read_10x_vdj_polars, load_polars
from dandelion.utilities._core import load_data


@pytest.fixture
@pytest.mark.usefixtures("annotation_10x_mouse")
def vdj_pl(annotation_10x_mouse):
    """Load and process data in polars version."""
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)
    find_clones_polars(vdj_pl, verbose=False)
    return vdj_pl


@pytest.fixture
@pytest.mark.usefixtures("annotation_10x_mouse")
def vdj_pd(annotation_10x_mouse):
    """Load and process data in pandas version."""
    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    find_clones(vdj_pd, verbose=False)
    return vdj_pd


def _norm(label: str) -> str:
    """Normalize clone labels"""
    if label is None or label == "":
        return ""
    parts = [p for p in str(label).split("|") if p]
    parts.sort()
    return "|".join(parts)


def test_clone_consistency(vdj_pl, vdj_pd):
    """Check if clone calling is consistent between pandas and polars."""
    pd_data = load_data(vdj_pd._data)
    pl_data = load_polars(vdj_pl._data, lazy=False)
    # Get data aligned by sequence_id
    pd_seqs = pd_data["sequence_id"].tolist()
    pl_seqs = pl_data["sequence_id"].to_list()

    pd_clones = pd_data["clone_id"].tolist()
    pl_clones = pl_data["clone_id"].to_list()

    pd_dict = {seq: clone for seq, clone in zip(pd_seqs, pd_clones)}
    pl_dict = {seq: clone for seq, clone in zip(pl_seqs, pl_clones)}

    common = sorted(set(pd_seqs) & set(pl_seqs))

    # Adjusted Rand Index check (contig-level, order-insensitive)
    pd_labels = [_norm(pd_dict[seq]) for seq in common]
    pl_labels = [_norm(pl_dict[seq]) for seq in common]

    ari = adjusted_rand_score(pd_labels, pl_labels)
    assert ari == 1.0


def test_clone_count_metadata(vdj_pl, vdj_pd):
    """Check if metadata clone counts match between pandas and polars"""
    print("\n=== CLONE COUNT COMPARISON ===")

    # Count clones in vdj data
    pd_data = load_data(vdj_pd._data)
    pl_data = load_polars(vdj_pl._data, lazy=False)
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
        pl_meta = pl_meta.collect(engine="streaming")

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
