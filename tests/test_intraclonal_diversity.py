import networkx as nx
import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from dandelion.base.tools._intraclonal_diversity import (
    _expanded_clone_assignments,
    _intraclonal_diversity_metrics,
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


def _toy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A|B", "A", "B", "C|D", "E|F", "E|F"],
        }
    )


def _toy_vdj_obj() -> SimpleNamespace:
    G = _toy_global_graph()
    return SimpleNamespace(data=_toy_data(), graph=(None, G))


def test_expanded_clone_assignments_smart_split_base():
    vdj_obj = _toy_vdj_obj()
    expanded = _expanded_clone_assignments(vdj_obj)

    by_cell = expanded.groupby("cell_id")["clone_id"].apply(list).to_dict()
    assert set(by_cell["c1"]) == {"A", "B"}
    assert by_cell["c4"] == ["C|D"]
    assert by_cell["c5"] == ["E|F"]


def test_intraclonal_diversity_column_order_and_wrapper_base():
    vdj_obj = _toy_vdj_obj()

    out = intraclonal_diversity(
        vdj_obj,
        min_cells=1,
        top_n=None,
        fast=False,
    )
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


def test_intraclonal_diversity_with_graph_dict_base():
    data = _toy_data()
    graph_obj = {
        "A|B": nx.Graph([("c1", "c2"), ("c1", "c3")]),
        "A": nx.Graph([("c2", "c1")]),
        "B": nx.Graph([("c3", "c1")]),
        "C|D": nx.Graph(),
        "E|F": nx.Graph([("c5", "c6")]),
    }
    graph_obj["C|D"].add_node("c4")
    vdj_obj = SimpleNamespace(data=data, graph=(None, _toy_global_graph()))
    vdj_obj.graph = (None, graph_obj)

    out = intraclonal_diversity(
        vdj_obj,
        min_cells=1,
        top_n=None,
        fast=True,
        n_sources=2,
        betweenness_k=2,
        random_state=7,
    )

    assert not out.empty
    assert {"A", "B", "C|D", "E|F"}.issubset(set(out["clone_id"]))


def test_intraclonal_diversity_metrics_edge_cases_base():
    empty_metrics = _intraclonal_diversity_metrics(nx.Graph(), fast=True)
    assert empty_metrics["n_nodes"] == 0
    assert empty_metrics["n_edges"] == 0
    assert empty_metrics["mode"] == "fast"
    assert np.isnan(empty_metrics["largest_community_fraction"])

    G = nx.Graph()
    G.add_weighted_edges_from([("x", "y", 1.0), ("y", "z", 1.0)])
    fast_metrics = _intraclonal_diversity_metrics(
        G,
        fast=True,
        n_sources=2,
        betweenness_k=2,
        random_state=1,
    )

    assert fast_metrics["n_nodes"] == 3
    assert fast_metrics["mode"] == "fast"


def test_intraclonal_diversity_empty_after_filter_base():
    vdj_obj = _toy_vdj_obj()
    out = intraclonal_diversity(vdj_obj, min_cells=99)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_intraclonal_diversity_missing_graph_raises_base():
    vdj_obj = SimpleNamespace(data=_toy_data(), graph=(None, None))
    with pytest.raises(ValueError, match="requires vdj_obj.graph\[1\]"):
        intraclonal_diversity(vdj_obj)


def test_intraclonal_diversity_group_by_base():
    data = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A", "A", "A", "A", "B", "B"],
            "sample_id": ["s1", "s1", "s2", "s2", "s1", "s2"],
        }
    )
    vdj_obj = SimpleNamespace(data=data, graph=(None, _toy_global_graph()))

    out = intraclonal_diversity(
        vdj_obj,
        group_by="sample_id",
        min_cells=1,
        top_n=None,
        fast=False,
    )

    assert "sample_id" in out.columns
    assert set(out["sample_id"]) == {"s1", "s2"}
    assert set(zip(out["sample_id"], out["clone_id"])) == {
        ("s1", "A"),
        ("s1", "B"),
        ("s2", "A"),
        ("s2", "B"),
    }


def test_intraclonal_diversity_group_by_missing_column_raises_base():
    vdj_obj = _toy_vdj_obj()
    with pytest.raises(ValueError, match="Missing required columns"):
        intraclonal_diversity(vdj_obj, group_by="sample_id")
