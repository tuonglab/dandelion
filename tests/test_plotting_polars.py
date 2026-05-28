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
    import numpy as np
    import matplotlib.collections as mcollections

    fig, ax = clone_circlepackplot(
        adata, group_by=["group2", "group3"], outer_ring_color="blue"
    )
    blue_rgba = np.array(mcolors.to_rgba("blue"))
    # In single-panel mode, rings are drawn via PatchCollection (ax.collections),
    # not as individual patches (ax.patches).
    ring_colls = [
        c
        for c in ax.collections
        if isinstance(c, mcollections.PatchCollection)
        and np.all(np.isclose(c.get_linewidth(), 2.0))
    ]
    assert len(ring_colls) >= 1
    # At least one edge colour across all ring patches should be blue
    all_ec = np.concatenate([c.get_edgecolor() for c in ring_colls], axis=0)
    assert any(np.allclose(row, blue_rgba) for row in all_ec)

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
    red_rgba = np.array(mcolors.to_rgba("red"))
    ring_colls = [
        c
        for c in ax.collections
        if isinstance(c, mcollections.PatchCollection)
        and np.all(np.isclose(c.get_linewidth(), 2.0))
    ]
    assert len(ring_colls) >= 1
    all_ec = np.concatenate([c.get_edgecolor() for c in ring_colls], axis=0)
    assert all(np.allclose(row, red_rgba) for row in all_ec)

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

    # With show_group_labels=False all rendered texts are count labels
    fig, ax = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_count_labels=True,
        show_group_labels=False,
    )
    assert len(ax.texts) > 0

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
    # Single-panel: disabling enclosure_label should reduce total text count
    fig_enc_on, ax_enc_on = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_count_labels=True,
        show_enclosure_label=True,
        show_group_labels=False,
    )
    fig_enc_off, ax_enc_off = clone_circlepackplot(
        adata,
        group_by=["group2", "group3"],
        show_count_labels=True,
        show_enclosure_label=False,
        show_group_labels=False,
    )
    assert len(ax_enc_off.texts) < len(ax_enc_on.texts)

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
    # In single-panel mode rings live in PatchCollection (ax.collections)
    fig, ax = clone_circlepackplot(adata, group_by=["group2", "group3"])
    ring_colls = [
        c
        for c in ax.collections
        if isinstance(c, mcollections.PatchCollection)
        and np.all(np.isclose(c.get_linewidth(), 2.0))
    ]
    assert len(ring_colls) >= 1
    all_ring_colors = [
        tuple(row) for c in ring_colls for row in c.get_edgecolor()
    ]
    # With the palette offset fix, level-0 and level-1 ring colours differ
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
    import pandas as _pd
    import anndata as _ad

    # Synthetic data: IgA has 3 distinct clones so that packcircles'
    # leaf-packing path (which requires N >= 3 radii) is fully exercised.
    _pc_obs = _pd.DataFrame(
        {
            "clone_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
            "isotype": ["IgA", "IgA", "IgA", "IgA", "IgA", "IgA", "IgG", "IgG"],
            "group2": ["a", "a", "a", "b", "b", "b", "a", "b"],
            "group3": ["x", "x", "y", "x", "y", "y", "x", "y"],
        },
        index=[f"pc_{i}" for i in range(8)],
    )
    for _col in ("isotype", "group2", "group3"):
        _pc_obs[_col] = _pc_obs[_col].astype("category")
    _adata_pc = _ad.AnnData(X=np.zeros((8, 5)), obs=_pc_obs)

    if importlib.util.find_spec("packcircles") is not None:
        # IgA has 3 distinct clones → pc.pack(radii) is called with N=3
        fig, ax = clone_circlepackplot(
            _adata_pc, group_by="isotype", packer="packcircles"
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Multi-level subplots
        fig, axes = clone_circlepackplot(
            _adata_pc,
            group_by=["group2", "group3"],
            as_subplots=True,
            packer="packcircles",
        )
        assert isinstance(axes, list)
        assert len(axes) == 2
        # aggregate_by_size + packcircles combined
        fig, ax = clone_circlepackplot(
            _adata_pc,
            group_by="isotype",
            aggregate_by_size=True,
            packer="packcircles",
        )
        assert isinstance(ax, Axes)

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
