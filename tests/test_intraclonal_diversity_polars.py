import networkx as nx
import pandas as pd
import polars as pl
import pytest

from types import SimpleNamespace

from dandelion.polars.tools._intraclonal_diversity import (
    _expanded_clone_assignments,
    intraclonal_diversity,
    intraclonal_metrics_per_clone,
)


def _toy_global_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ("c1", "c2", 1.0),
            ("c1", "c3", 1.0),
            ("c2", "c3", 1.0),
            ("c5", "c6", 2.0),
        ]
    )
    G.add_node("c4")
    return G


def _toy_polars_data(lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    frame = pl.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A|B", "A", "B", "C|D", "E|F", "E|F"],
        }
    )
    return frame.lazy() if lazy else frame


def test_expanded_clone_assignments_polars_lazy():
    vdj_obj = SimpleNamespace(data=_toy_polars_data(lazy=True), graph=(None, None))
    expanded = _expanded_clone_assignments(vdj_obj)

    assert isinstance(expanded, pd.DataFrame)
    by_cell = expanded.groupby("cell_id")["clone_id"].apply(list).to_dict()
    assert set(by_cell["c1"]) == {"A", "B"}
    assert by_cell["c4"] == ["C|D"]


def test_intraclonal_diversity_polars_default_graph_slot_and_wrapper():
    G = _toy_global_graph()
    vdj_obj = SimpleNamespace(data=_toy_polars_data(lazy=False), graph=(None, G))

    out = intraclonal_diversity(vdj_obj, min_cells=1, top_n=None, fast=False)
    out_wrapper = intraclonal_metrics_per_clone(
        vdj_obj,
        min_cells=1,
        top_n=None,
        fast=False,
    )

    assert not out.empty
    assert out.columns[:2].tolist() == ["clone_id", "clone_size"]
    assert set(out["mode"]) == {"exact"}

    pd.testing.assert_frame_equal(
        out.sort_values("clone_id").reset_index(drop=True),
        out_wrapper.sort_values("clone_id").reset_index(drop=True),
    )


def test_intraclonal_diversity_missing_graph_raises_polars():
    vdj_obj = SimpleNamespace(data=_toy_polars_data(lazy=False), graph=(None, None))
    with pytest.raises(ValueError, match="requires vdj_obj.graph\[1\]"):
        intraclonal_diversity(vdj_obj)


def test_intraclonal_diversity_group_by_polars():
    frame = pl.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A", "A", "A", "A", "B", "B"],
            "sample_id": ["s1", "s1", "s2", "s2", "s1", "s2"],
        }
    )
    vdj_obj = SimpleNamespace(data=frame.lazy(), graph=(None, _toy_global_graph()))

    out = intraclonal_diversity(
        vdj_obj,
        group_by="sample_id",
        min_cells=1,
        top_n=None,
        fast=False,
    )

    assert "sample_id" in out.columns
    assert set(out["sample_id"]) == {"s1", "s2"}


def test_intraclonal_diversity_group_by_missing_column_raises_polars():
    vdj_obj = SimpleNamespace(data=_toy_polars_data(lazy=False), graph=(None, _toy_global_graph()))
    with pytest.raises(ValueError, match="Missing required columns"):
        intraclonal_diversity(vdj_obj, group_by="sample_id")
