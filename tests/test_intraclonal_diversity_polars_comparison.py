import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from types import SimpleNamespace

from dandelion.base.tools._intraclonal_diversity import (
    intraclonal_diversity as intraclonal_diversity_base,
)
from dandelion.polars.tools._intraclonal_diversity import (
    intraclonal_diversity as intraclonal_diversity_polars,
)


def _toy_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            ("c1", "c2", 1.0),
            ("c1", "c3", 1.0),
            ("c2", "c3", 1.0),
            ("c3", "c4", 1.5),
            ("c5", "c6", 2.0),
        ]
    )
    return G


def test_intraclonal_diversity_base_vs_polars_match():
    data_pd = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A|B", "A", "B", "B", "C", "C"],
        }
    )
    data_pl = pl.DataFrame(data_pd)
    G = _toy_graph()

    vdj_base = SimpleNamespace(data=data_pd, graph=(None, G))
    vdj_polars = SimpleNamespace(data=data_pl, graph=(None, G))

    out_base = intraclonal_diversity_base(
        vdj_base,
        min_cells=1,
        top_n=None,
        fast=False,
    )
    out_polars = intraclonal_diversity_polars(
        vdj_polars,
        min_cells=1,
        top_n=None,
        fast=False,
    )

    out_base = out_base.sort_values("clone_id").reset_index(drop=True)
    out_polars = out_polars.sort_values("clone_id").reset_index(drop=True)

    assert out_base["clone_id"].tolist() == out_polars["clone_id"].tolist()
    assert out_base["clone_size"].tolist() == out_polars["clone_size"].tolist()

    numeric_cols = [
        c
        for c in out_base.columns
        if c not in {"clone_id", "mode"}
        and pd.api.types.is_numeric_dtype(out_base[c])
    ]
    for col in numeric_cols:
        np.testing.assert_allclose(
            out_base[col].to_numpy(),
            out_polars[col].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"Column mismatch for {col}",
        )

    assert out_base["mode"].tolist() == out_polars["mode"].tolist()


def test_intraclonal_diversity_base_vs_polars_match_group_by():
    data_pd = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "clone_id": ["A", "A", "A", "B", "B", "B"],
            "sample_id": ["s1", "s1", "s2", "s1", "s2", "s2"],
        }
    )
    data_pl = pl.DataFrame(data_pd)
    G = _toy_graph()

    vdj_base = SimpleNamespace(data=data_pd, graph=(None, G))
    vdj_polars = SimpleNamespace(data=data_pl, graph=(None, G))

    out_base = intraclonal_diversity_base(
        vdj_base,
        group_by="sample_id",
        min_cells=1,
        top_n=None,
        fast=False,
    )
    out_polars = intraclonal_diversity_polars(
        vdj_polars,
        group_by="sample_id",
        min_cells=1,
        top_n=None,
        fast=False,
    )

    out_base = out_base.sort_values(["sample_id", "clone_id"]).reset_index(
        drop=True
    )
    out_polars = out_polars.sort_values(["sample_id", "clone_id"]).reset_index(
        drop=True
    )

    assert out_base["sample_id"].tolist() == out_polars["sample_id"].tolist()
    assert out_base["clone_id"].tolist() == out_polars["clone_id"].tolist()
    assert out_base["clone_size"].tolist() == out_polars["clone_size"].tolist()
