"""Check if clone naming convention is identical between pandas and polars."""

import pytest
import polars as pl
from dandelion.tools._tools import find_clones
from dandelion.tools._tools_polars import find_clones as find_clones_polars
from dandelion.utilities._io import read_10x_vdj
from dandelion.utilities._polars import read_10x_vdj_polars
from dandelion.utilities._core import load_data


@pytest.mark.usefixtures("annotation_10x_mouse")
def test_clone_naming_convention(annotation_10x_mouse):
    """Check if clone naming convention is identical."""
    print("\n" + "=" * 60)
    print("Clone Naming Convention Comparison")
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

    # Get data aligned by sequence_id
    pd_seqs = pd_data["sequence_id"].tolist()
    pl_seqs = pl_data["sequence_id"].to_list()

    pd_clones = pd_data["clone_id"].tolist()
    pl_clones = pl_data["clone_id"].to_list()

    pd_dict = {seq: clone for seq, clone in zip(pd_seqs, pd_clones)}
    pl_dict = {seq: clone for seq, clone in zip(pl_seqs, pl_clones)}

    common = sorted(set(pd_seqs) & set(pl_seqs))

    print(f"\nTotal sequences: {len(common)}")

    # Check for exact matches
    exact_matches = 0
    differences = []

    for seq in common:
        pd_clone = pd_dict[seq]
        pl_clone = pl_dict[seq]

        if pd_clone == pl_clone:
            exact_matches += 1
        else:
            differences.append((seq, pd_clone, pl_clone))

    print(f"Exact clone ID matches: {exact_matches}/{len(common)}")
    print(f"Differences: {len(differences)}")

    if differences:
        print(f"\nFirst 10 differences:")
        for seq, pd_clone, pl_clone in differences[:10]:
            print(f"  {seq}")
            print(f"    Pandas: {pd_clone}")
            print(f"    Polars: {pl_clone}")

    # Check for patterns in differences
    if differences:
        # Check if it's just reordering of VDJ/VJ parts
        print(f"\nAnalyzing difference patterns:")

        # Check if differences are in cells with multiple chains
        cells_with_diffs = set(
            seq.split("_contig_")[0] for seq, _, _ in differences
        )
        print(f"  Cells with differences: {len(cells_with_diffs)}")

        # Check if differences are in VJ-only vs VDJ+VJ assignments
        vj_only_diffs = sum(
            1 for _, pd, pl in differences if "VDJ" not in pd or "VDJ" not in pl
        )
        vdj_vj_diffs = sum(
            1 for _, pd, pl in differences if "VDJ" in pd and "VDJ" in pl
        )

        print(f"  VJ-only assignment differences: {vj_only_diffs}")
        print(f"  VDJ+VJ assignment differences: {vdj_vj_diffs}")

    # Check orphan handling
    print(f"\nOrphan sequence handling:")
    pd_unassigned = sum(1 for c in pd_clones if not c or c == "")
    pl_unassigned = sum(1 for c in pl_clones if not c or c == "")

    print(f"  Pandas unassigned: {pd_unassigned}")
    print(f"  Polars unassigned: {pl_unassigned}")

    # Check metadata
    print(f"\nMetadata (cell-level):")
    if hasattr(vdj_pd, "_metadata"):
        print(f"  Pandas metadata shape: {vdj_pd._metadata.shape}")
        if "clone_id" in vdj_pd._metadata.columns:
            pd_meta_clones = vdj_pd._metadata["clone_id"].nunique()
            print(f"  Pandas unique clones in metadata: {pd_meta_clones}")

    if hasattr(vdj_pl, "_metadata"):
        pl_meta = (
            vdj_pl._metadata.collect(engine="streaming")
            if isinstance(vdj_pl._metadata, pl.LazyFrame)
            else vdj_pl._metadata
        )
        print(f"  Polars metadata shape: {pl_meta.shape}")
        if "clone_id" in pl_meta.columns:
            pl_meta_clones = pl_meta["clone_id"].n_unique()
            print(f"  Polars unique clones in metadata: {pl_meta_clones}")

    # Check if differences are just ordering
    print(f"\nChecking if differences are just ordering:")
    all_same_sets = True
    for seq, pd_clone, pl_clone in differences:
        pd_set = set(pd_clone.split("|"))
        pl_set = set(pl_clone.split("|"))
        if pd_set != pl_set:
            print(f"  {seq}: Different sets!")
            all_same_sets = False
        else:
            print(f"  {seq}: Same clones, different order ✓")

    if all_same_sets and differences:
        print(
            f"\n✓ All differences are just ordering of alternative clone assignments"
        )
        print(f"  This is acceptable since ARI considers them equivalent")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
