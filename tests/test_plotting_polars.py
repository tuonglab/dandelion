#!/usr/bin/env python
import pytest
import scanpy as sc

from dandelion.polars.preprocessing import check_contigs
from dandelion.polars.core import Dandelion
from dandelion.polars.tools import generate_network, transfer
from dandelion.polars.io import read_ddl
from dandelion.polars.plotting import (
    clone_circlepackplot,
    clone_network,
    barplot,
    stackedbarplot,
    spectratype,
)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated", "dummy_adata")
def test_setup(create_testfolder, airr_reannotated, dummy_adata):
    """test_setup"""
    vdj, adata = check_contigs(airr_reannotated, dummy_adata)
    vdj.data["clone_id"] = ["A", "A", "A", "A", "A", "A", "A", "A"]
    vdj = Dandelion(vdj._data)
    generate_network(vdj, layout_method="mod_fr")
    transfer(adata, vdj)
    assert "clone_id" in adata.obs
    assert "X_vdj" in adata.obsm
    f1 = create_testfolder / "test.ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_ddl(f1)
    adata.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_plot_network(create_testfolder):
    """test_plot_network"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    clone_network(adata, color=["isotype"], show=False, return_fig=False)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "sort,norm",
    [
        pytest.param(True, True),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(False, False),
    ],
)
def test_plot_bar(create_testfolder, sort, norm):
    """test_plot_bar"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    ax = barplot(vdj, color="v_call_VDJ")
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", sort_descending=sort)
    assert ax is not None
    ax = barplot(vdj, color="v_call_VDJ", normalize=norm)
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_bar2(create_testfolder):
    """test_plot_bar2"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = barplot(
        vdj,
        color="v_call_VDJ",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
    )
    assert ax is not None
    ax = barplot(adata, color="v_call_VDJ")
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar(create_testfolder, norm):
    """test_plot_stackedbar"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    ax = stackedbarplot(
        vdj, color="v_call_VDJ", group_by="isotype", normalize=norm
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar2(create_testfolder, norm):
    """test_plot_stackedbar2"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = stackedbarplot(
        vdj,
        color="v_call_VDJ",
        group_by="isotype",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
        normalize=norm,
    )
    assert ax is not None
    ax = stackedbarplot(
        adata,
        color="v_call_VDJ",
        group_by="isotype",
        normalize=norm,
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_spectratype(create_testfolder):
    """test_plot_spectratype"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    ax = spectratype(
        vdj, color="junction_length", group_by="c_call", locus="IGH"
    )
    assert ax is not None
    ax = spectratype(
        vdj,
        color="junction_length",
        group_by="c_call",
        locus="IGH",
        hide_legend=False,
        width=1,
        xtick_rotation=90,
        title="test",
        labels="test",
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_clone_circlepackplot(create_testfolder):
    """test_plot_clone_circlepackplot"""
    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    f2 = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f2)
    ax = clone_circlepackplot(vdj, group_by="isotype")
    assert ax is not None
    ax = clone_circlepackplot(adata, group_by="isotype")
    assert ax is not None
    # palette as a complete nested dict
    ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        palette={
            "isotype": {"IgM": "#ff7f0e", "IgK": "#1f77b4", "IgL": "#2ca02c"}
        },
    )
    assert ax is not None
    # palette as a partial nested dict (missing keys get auto-assigned)
    ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        palette={"isotype": {"IgM": "#ff0000"}},
    )
    assert ax is not None
    # palette as a list per level
    ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        palette={"isotype": ["#ff7f0e", "#1f77b4", "#2ca02c"]},
    )
    assert ax is not None
    # list palette for both levels of nested hierarchy
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        palette={
            "group2": ["red", "blue"],
            "group3": ["green", "orange", "purple"],
        },
    )
    assert ax is not None
    # nested hierarchy with palette only for outer level
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        palette={"group2": {"a": "#ff7f0e", "b": "#1f77b4"}},
    )
    assert ax is not None
    # nested hierarchy with palette for both levels
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        palette={
            "group2": {"a": "#ff7f0e", "b": "#1f77b4"},
            "group3": {"a": "#2ca02c", "b": "#e377c2", "c": "#bcbd22"},
        },
    )
    assert ax is not None
    # AnnData with nested hierarchy, palette for outer level only
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        palette={"group2": {"a": "#ff7f0e", "b": "#1f77b4"}},
    )
    assert ax is not None
    # AnnData with palette dict for single level
    ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        palette={"isotype": {"IgM": "#ff7f0e", "IgK": "#1f77b4"}},
    )
    assert ax is not None
    ax = clone_circlepackplot(
        vdj,
        group_by="isotype",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        show_group_labels=False,
        show_clone_labels=True,
        alpha=0.4,
        palette="tab10",
        show_legend=False,
    )
    assert ax is not None
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        legend_kwargs={"loc": "upper right"},
    )
    assert ax is not None
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_legend=["group2"],
    )
    assert ax is not None
    ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_legend="group2",
    )
    assert ax is not None
    ax = clone_circlepackplot(vdj, group_by="isotype", show_count_labels=True)
    assert ax is not None
    ax = clone_circlepackplot(
        vdj,
        group_by="isotype",
        show_clone_labels=True,
        show_count_labels=True,
    )
    assert ax is not None
    with pytest.raises(ValueError):
        clone_circlepackplot(vdj, group_by="isotype", min_clone_size=999)


@pytest.mark.usefixtures("create_testfolder")
def test_plot_clone_circlepackplot_new_features(create_testfolder):
    """Tests for as_subplots, scale_*, outer_ring_color, and new count-label behaviour."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    f = create_testfolder / "test.ddl"
    vdj = read_ddl(f)
    f2 = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f2)

    # --- Return-type contracts -----------------------------------------------

    # Single-panel returns (Figure, Axes)
    fig, ax = clone_circlepackplot(adata, group_by=["group2", "group3"])
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # as_subplots returns (Figure, list[Axes])
    fig, axes = clone_circlepackplot(
        adata, group_by=["group2", "group3"], as_subplots=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(axes, list)
    assert all(isinstance(a, Axes) for a in axes)
    # group2 has two unique values ("a", "b") → two subplots
    assert len(axes) == 2

    # --- as_subplots with single-level group_by --------------------------------
    fig, axes = clone_circlepackplot(
        adata, group_by="isotype", as_subplots=True
    )
    assert isinstance(axes, list)
    assert len(axes) >= 1

    # --- n_col / n_row grid layout --------------------------------------------
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        n_col=1,
        n_row=2,
    )
    assert len(axes) == 2
    # 1-col × 2-row grid → 2 total axes
    assert len(fig.get_axes()) == 2

    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        n_col=1,
        n_row=3,  # oversized: 1 padding cell
    )
    assert len(axes) == 2  # returned list only has active subplots
    assert len(fig.get_axes()) == 3  # full grid including the off padding cell

    # --- scale_subplots=False -------------------------------------------------
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        scale_subplots=False,
    )
    assert len(axes) == 2

    # --- scale_factor (single-panel and subplot) ------------------------------
    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], scale_factor=1.5
    )
    assert isinstance(ax, Axes)

    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        scale_factor=0.7,
    )
    assert len(axes) == 2

    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        scale_subplots=False,
        scale_factor=0.8,
    )
    assert len(axes) == 2

    # --- outer_ring_color in single-panel mode --------------------------------
    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], outer_ring_color="blue"
    )
    blue_rgba = tuple(mcolors.to_rgba("blue"))
    ring_patches = [
        p
        for p in ax.patches
        if isinstance(p, mpatches.Circle)
        and not p.get_fill()
        and p.get_linewidth() >= 2.0
    ]
    assert len(ring_patches) >= 1
    # All level-0 (outermost) rings should be blue
    blue_patches = [
        p for p in ring_patches if tuple(p.get_edgecolor()) == blue_rgba
    ]
    assert len(blue_patches) >= 1

    # --- outer_ring_color in subplot mode ------------------------------------
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        outer_ring_color="black",
    )
    black_rgba = tuple(mcolors.to_rgba("black"))
    for ax in axes:
        # The manually drawn outer ring is always centred at (0, 0)
        outer_at_origin = [
            p
            for p in ax.patches
            if isinstance(p, mpatches.Circle)
            and not p.get_fill()
            and abs(p.center[0]) < 1e-9
            and abs(p.center[1]) < 1e-9
        ]
        assert len(outer_at_origin) >= 1
        assert tuple(outer_at_origin[0].get_edgecolor()) == black_rgba

    # --- outer_ring_color with vdj (DandelionPolars / non-AnnData) -----------
    fig, ax = clone_circlepackplot(
        vdj, group_by="isotype", outer_ring_color="red"
    )
    red_rgba = tuple(mcolors.to_rgba("red"))
    ring_patches = [
        p
        for p in ax.patches
        if isinstance(p, mpatches.Circle)
        and not p.get_fill()
        and p.get_linewidth() >= 2.0
    ]
    assert len(ring_patches) >= 1
    assert all(tuple(p.get_edgecolor()) == red_rgba for p in ring_patches)

    # --- show_count_labels: group totals outside ring, individual inside ------
    # Single-panel: texts should be present for both groups and leaves
    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], show_count_labels=True
    )
    assert len(ax.texts) > 0

    # show_clone_labels + show_count_labels together
    fig, ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_clone_labels=True,
        show_count_labels=True,
    )
    assert len(ax.texts) > 0

    # Group count labels are placed BELOW the ring (at y - r, va="top"),
    # so their y-coordinate should be strictly below the group circle centre.
    fig, ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_count_labels=True,
        show_group_labels=False,  # suppress name labels so texts = count only
    )
    group_unfilled = [
        p
        for p in ax.patches
        if isinstance(p, mpatches.Circle) and not p.get_fill()
    ]
    # For every group circle its count text must sit below its centre
    for patch in group_unfilled:
        cx, cy, cr = patch.center[0], patch.center[1], patch.get_radius()
        # Find a text near (cx, cy - cr) — the expected label position
        nearby = [
            t
            for t in ax.texts
            if abs(t.get_position()[0] - cx) < 1e-6
            and abs(t.get_position()[1] - (cy - cr - 0.05)) < 1e-6
        ]
        assert len(nearby) == 1, (
            f"Expected one count-label below group circle at "
            f"({cx:.3f}, {cy - cr:.3f}); found {len(nearby)}"
        )

    # --- show_count_labels in subplot mode: outer ring gets total label ------
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        show_count_labels=True,
    )
    for ax in axes:
        # At least the outer-ring total text should be present
        assert len(ax.texts) > 0

    # --- show_enclosure_label=False: outer ring text suppressed --------------
    # Single-panel: no text should appear below the unfilled group rings
    fig, ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_count_labels=True,
        show_enclosure_label=False,
        show_group_labels=False,
    )
    group_unfilled = [
        p
        for p in ax.patches
        if isinstance(p, mpatches.Circle) and not p.get_fill()
    ]
    for patch in group_unfilled:
        cx, cy, cr = patch.center[0], patch.center[1], patch.get_radius()
        below = [
            t
            for t in ax.texts
            if abs(t.get_position()[0] - cx) < 1e-6
            and abs(t.get_position()[1] - (cy - cr - 0.05)) < 1e-6
        ]
        assert len(below) == 0

    # Subplot mode: show_enclosure_label=False removes exactly the outer ring
    # total text (leaf clone count texts are separate and unaffected).
    fig, axes_with = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        show_count_labels=True,
        show_enclosure_label=True,
        show_group_labels=False,
    )
    fig, axes_without = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        show_count_labels=True,
        show_enclosure_label=False,
        show_group_labels=False,
    )
    for ax_with, ax_without in zip(axes_with, axes_without):
        assert len(ax_without.texts) == len(ax_with.texts) - 1

    # --- title as suptitle in subplot mode, as ax.title in single-panel ------
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        title="Subplot Title",
    )
    assert any("Subplot Title" in t.get_text() for t in fig.texts)

    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], title="Single Title"
    )
    assert ax.get_title() == "Single Title"

    # --- legend control -------------------------------------------------------
    # show_legend=False: no legend on any axis
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        show_legend=False,
    )
    for ax in axes:
        assert ax.get_legend() is None

    # Default (show_legend=None): legend goes on the last active subplot
    fig, axes = clone_circlepackplot(
        adata, group_by=["group2", "group3"], as_subplots=True
    )
    assert axes[-1].get_legend() is not None

    # --- palette offset: level-0 and level-1 auto-colours are distinct --------
    # With palette=None and two group levels, run to completion without errors.
    fig, ax = clone_circlepackplot(adata, group_by=["group2", "group3"])
    # Collect edgecolors of all unfilled group-ring patches
    all_ring_colors = [
        tuple(p.get_edgecolor())
        for p in ax.patches
        if isinstance(p, mpatches.Circle) and not p.get_fill()
    ]
    # With the palette offset fix, the two level-0 rings and the inner
    # level-1 rings should not all share the same colour.
    assert len(set(all_ring_colors)) > 1

    # --- named string palette with offset ------------------------------------
    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], palette="tab10"
    )
    assert ax is not None

    # --- palette dict with auto-fill for missing values ----------------------
    fig, ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        palette={"group2": {"a": "red"}, "group3": {"a": "green"}},
    )
    assert ax is not None

    # --- ValueError when no clones remain ------------------------------------
    with pytest.raises(ValueError):
        clone_circlepackplot(
            adata, group_by="isotype", as_subplots=True, min_clone_size=999
        )

    # --- packer="packcircles" (skip if not installed) -----------------------
    import importlib

    if importlib.util.find_spec("packcircles") is not None:
        fig, ax = clone_circlepackplot(
            adata, group_by="isotype", packer="packcircles"
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        fig, axes = clone_circlepackplot(
            adata,
            group_by=["group2", "group3"],
            as_subplots=True,
            packer="packcircles",
        )
        assert len(axes) == 2

    # Invalid packer name raises ValueError
    with pytest.raises(ValueError, match="Unknown packer"):
        clone_circlepackplot(adata, group_by="isotype", packer="unknown_packer")

    # --- aggregate_by_size ---------------------------------------------------
    # Basic: completes without error
    fig, ax = clone_circlepackplot(
        adata, group_by="isotype", aggregate_by_size=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        aggregate_by_size=True,
    )
    assert isinstance(axes, list)
    assert len(axes) == 2

    # show_clone_labels=True: aggregate node labels use "n=<size>" format
    fig, ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        aggregate_by_size=True,
        show_clone_labels=True,
    )
    assert len(ax.texts) > 0
    agg_labels = [t for t in ax.texts if t.get_text().startswith("n=")]
    assert len(agg_labels) > 0

    # show_count_labels=True with aggregate_by_size: texts are present
    fig, ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        aggregate_by_size=True,
        show_count_labels=True,
    )
    assert len(ax.texts) > 0

    # aggregate_by_size + as_subplots + show_count_labels
    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        aggregate_by_size=True,
        show_count_labels=True,
    )
    for ax in axes:
        assert len(ax.texts) > 0

    # --- max_clones_per_group ------------------------------------------------
    # Basic: completes without error
    fig, ax = clone_circlepackplot(
        adata, group_by="isotype", max_clones_per_group=2
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, axes = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        as_subplots=True,
        max_clones_per_group=1,
    )
    assert isinstance(axes, list)
    assert len(axes) == 2

    # Limited plot has no more labels than unlimited
    fig_limited, ax_limited = clone_circlepackplot(
        adata,
        group_by="isotype",
        max_clones_per_group=1,
        show_clone_labels=True,
        show_count_labels=False,
        show_group_labels=False,
    )
    fig_unlimited, ax_unlimited = clone_circlepackplot(
        adata,
        group_by="isotype",
        show_clone_labels=True,
        show_count_labels=False,
        show_group_labels=False,
    )
    assert len(ax_limited.texts) <= len(ax_unlimited.texts)

    # max_clones_per_group + aggregate_by_size combined
    fig, ax = clone_circlepackplot(
        adata,
        group_by="isotype",
        max_clones_per_group=2,
        aggregate_by_size=True,
    )
    assert isinstance(ax, Axes)
