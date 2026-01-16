"""Final phases for find_clones debugging - split phase 8."""

import pytest
import polars as pl
from dandelion.tools._tools import find_clones
from dandelion.tools._tools_polars import find_clones as find_clones_polars
from dandelion.utilities._io import read_10x_vdj
from dandelion.utilities._polars import read_10x_vdj_polars
from dandelion.utilities._core import load_data
from sklearn.metrics import adjusted_rand_score


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_phase8a_pandas_find_clones(annotation_10x_mouse):
    """Phase 8a: Test pandas find_clones alone."""
    print("\n" + "=" * 60)
    print("PHASE 8a: Pandas find_clones")
    print("=" * 60)

    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    print(
        f"✓ Loaded pandas Dandelion: {vdj_pd._data.shape if hasattr(vdj_pd._data, 'shape') else len(vdj_pd._data)}"
    )

    vdj_pd = find_clones(vdj_pd, verbose=False)
    print(f"✓ find_clones completed")
    print(f"  Result is Dandelion: {vdj_pd is not None}")

    if vdj_pd is not None:
        pd_data = load_data(vdj_pd._data)
        pd_clones = len(set(c for c in pd_data["clone_id"] if c))
        print(f"  Unique clones: {pd_clones}")
        assert pd_clones > 0, "Pandas should have clones"


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_phase8b_polars_find_clones(annotation_10x_mouse):
    """Phase 8b: Test polars find_clones alone."""
    print("\n" + "=" * 60)
    print("PHASE 8b: Polars find_clones")
    print("=" * 60)

    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)
    print(
        f"✓ Loaded polars Dandelion: {vdj_pl._data.collect(engine="streaming").shape if hasattr(vdj_pl._data, 'collect') else vdj_pl._data.shape}"
    )

    vdj_pl = find_clones_polars(vdj_pl, verbose=False)
    print(f"✓ find_clones_polars completed")
    print(f"  Result is DandelionPolars: {vdj_pl is not None}")

    if vdj_pl is not None:
        pl_data = (
            vdj_pl._data.collect(engine="streaming")
            if isinstance(vdj_pl._data, pl.LazyFrame)
            else vdj_pl._data
        )
        pl_clones = len(set(c for c in pl_data["clone_id"].to_list() if c))
        print(f"  Unique clones: {pl_clones}")
        assert pl_clones > 0, "Polars should have clones"


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_phase8c_compare_outputs(annotation_10x_mouse):
    """Phase 8c: Compare outputs after both complete."""
    print("\n" + "=" * 60)
    print("PHASE 8c: Output Comparison")
    print("=" * 60)

    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    vdj_pd = find_clones(vdj_pd, verbose=False)
    vdj_pl = find_clones_polars(vdj_pl, verbose=False)

    pd_data = load_data(vdj_pd._data)
    pl_data = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    pd_clones = len(set(c for c in pd_data["clone_id"] if c))
    pl_clones = len(set(c for c in pl_data["clone_id"].to_list() if c))

    print(f"Pandas unique clones: {pd_clones}")
    print(f"Polars unique clones: {pl_clones}")

    # Check common sequences
    pd_seqs = set(pd_data["sequence_id"].tolist())
    pl_seqs = set(pl_data["sequence_id"].to_list())
    common = pd_seqs & pl_seqs
    print(f"Common sequences: {len(common)}")

    assert len(common) > 0, "Should have common sequences"


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_phase8d_ari_comparison(annotation_10x_mouse):
    """Phase 8d: ARI score comparison."""
    print("\n" + "=" * 60)
    print("PHASE 8d: ARI Score")
    print("=" * 60)

    vdj_pd = read_10x_vdj(annotation_10x_mouse)
    vdj_pl = read_10x_vdj_polars(annotation_10x_mouse)

    vdj_pd = find_clones(vdj_pd, verbose=False)
    vdj_pl = find_clones_polars(vdj_pl, verbose=False)

    pd_data = load_data(vdj_pd._data)
    pl_data = (
        vdj_pl._data.collect(engine="streaming")
        if isinstance(vdj_pl._data, pl.LazyFrame)
        else vdj_pl._data
    )

    # Get aligned clone lists
    pd_seqs = pd_data["sequence_id"].tolist()
    pl_seqs = pl_data["sequence_id"].to_list()

    pd_clones = pd_data["clone_id"].tolist()
    pl_clones = pl_data["clone_id"].to_list()

    pd_dict = {seq: clone for seq, clone in zip(pd_seqs, pd_clones)}
    pl_dict = {seq: clone for seq, clone in zip(pl_seqs, pl_clones)}

    common = sorted(set(pd_seqs) & set(pl_seqs))
    aligned_pd = [pd_dict[seq] for seq in common]
    aligned_pl = [pl_dict[seq] for seq in common]

    ari = adjusted_rand_score(aligned_pd, aligned_pl)
    print(f"ARI score: {ari:.6f}")
    print(f"Expected: 1.0 for perfect match")

    assert ari > 0.95, f"ARI {ari:.6f} is too low (expected > 0.95)"
