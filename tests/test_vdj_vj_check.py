#!/usr/bin/env python
from dandelion.base.preprocessing._preprocessing import check_productive_chain

UMI_FOLDCHANGE_CUTOFF = 2
CON_FOLDCHANGE_CUTOFF = 5


def test_vdj_contig_scenario_1():
    """When there is a clear winner in the umi counts."""
    umi_counts = {"c1": 10, "c2": 3, "c3": 1}
    consensus_counts = {"c1": 50, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 1
    assert len(extra) == 1
    assert len(ambiguous) == 1


def test_vdj_contig_scenario_2():
    """When there is a clear winner in the consensus counts."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 1
    assert len(extra) == 0
    assert len(ambiguous) == 2


def test_vdj_contig_scenario_3():
    """When there are no clear winners."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 2, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 0
    assert len(extra) == 0
    assert len(ambiguous) == 3


def test_vdj_contig_scenario_5():
    """When there is a clear winner in the consensus counts and the second is far."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 4, "c3": 3}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 1
    assert len(extra) == 1
    assert len(ambiguous) == 1


def test_vdj_contig_scenario_4():
    """When there is a clear winner in the consensus counts and the second is close."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 5, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 1
    assert len(extra) == 0
    assert len(ambiguous) == 2


def test_vdj_contig_scenario_6():
    """When there is a clear loser."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 3}
    consensus_counts = {"c1": 100, "c2": 100, "c3": 3}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        1,
    )
    assert len(keep) == 1
    assert len(extra) == 1
    assert len(ambiguous) == 1


def test_vj_contig_scenario_1():
    """When there is a clear winner in the UMI counts."""
    umi_counts = {"c1": 10, "c2": 3, "c3": 1}
    consensus_counts = {"c1": 50, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 1


def test_vj_contig_scenario_2():
    """When there is a tie in UMI counts but a clear winner in consensus counts."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 1
    assert "c1" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 2


def test_vj_contig_scenario_3():
    """When there are no clear winners in either UMI or consensus counts."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 2, "c2": 1, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 0
    assert len(extra) == 0
    assert len(ambiguous) == 3


def test_vj_contig_scenario_4():
    """When two contigs have similar UMI counts but one is far ahead in consensus counts."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 4, "c3": 3}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 1
    assert "c3" in ambiguous


def test_vj_contig_scenario_5():
    """When UMI counts suggest a tie but consensus counts resolve it."""
    umi_counts = {"c1": 3, "c2": 3, "c3": 3}
    consensus_counts = {"c1": 50, "c2": 5, "c3": 1}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 1
    assert "c3" in ambiguous


def test_vj_contig_scenario_6():
    """When there is a clear loser among three contigs."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 3}
    consensus_counts = {"c1": 100, "c2": 100, "c3": 3}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 1
    assert "c3" in ambiguous


def test_vj_contig_scenario_7():
    """When there is a clear loser among three contigs but not ambiguous."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 10}
    consensus_counts = {"c1": 100, "c2": 100, "c3": 30}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 1
    assert "c3" in extra
    assert len(ambiguous) == 0


def test_vj_contig_scenario_8():
    """When there is a clear loser among three contigs and explicitly ambiguous."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 1}
    consensus_counts = {"c1": 100, "c2": 100, "c3": 5}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 0
    assert len(ambiguous) == 1
    assert "c3" in ambiguous


def test_vj_contig_scenario_9():
    """When there is a clear loser among three contigs but not really ambiguous."""
    umi_counts = {"c1": 30, "c2": 30, "c3": 9}
    consensus_counts = {"c1": 100, "c2": 100, "c3": 5}
    keep, extra, ambiguous = check_productive_chain(
        umi_counts,
        consensus_counts,
        UMI_FOLDCHANGE_CUTOFF,
        CON_FOLDCHANGE_CUTOFF,
        2,
    )
    assert len(keep) == 2
    assert "c1" in keep and "c2" in keep
    assert len(extra) == 1
    assert "c3" in extra
    assert len(ambiguous) == 0
