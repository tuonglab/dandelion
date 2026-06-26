import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from collections.abc import Hashable

from dandelion.polars.core._core import DandelionPolars


def _distance_to_similarity(distance: float, eps: float = 1e-9) -> float:
    """Convert distance-like edge weights to similarity-like weights."""
    return 1.0 / (float(distance) + eps)


def _with_similarity_weights(
    G: nx.Graph,
    distance_key: str = "weight",
    similarity_key: str = "similarity_weight",
) -> nx.Graph:
    """Copy graph and add similarity edge weights derived from distance weights."""
    H = G.copy()
    for _, _, data in H.edges(data=True):
        distance = data.get(distance_key, 1.0)
        try:
            data[similarity_key] = _distance_to_similarity(distance)
        except (TypeError, ValueError):
            data[similarity_key] = _distance_to_similarity(1.0)
    return H


def _largest_connected_component(G: nx.Graph) -> nx.Graph:
    """Return largest connected component as an independent graph copy."""
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G.copy()
    largest_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def _sample_pairwise_distance_summary(
    G: nx.Graph,
    distance_key: str = "weight",
    n_sources: int = 32,
    random_state: int = 42,
) -> tuple[float, float, dict[Hashable, dict[Hashable, float]]]:
    """Estimate average shortest path and diameter from sampled sources."""
    if G.number_of_nodes() <= 1:
        return 0.0, 0.0, {n: {n: 0.0} for n in G.nodes()}

    rng = np.random.default_rng(random_state)
    nodes = list(G.nodes())
    n_pick = min(n_sources, len(nodes))
    source_nodes = rng.choice(nodes, size=n_pick, replace=False).tolist()

    lengths: dict[Hashable, dict[Hashable, float]] = {}
    dists: list[float] = []
    for src in source_nodes:
        src_lengths = nx.single_source_dijkstra_path_length(
            G, src, weight=distance_key
        )
        lengths[src] = src_lengths
        for dst, dist in src_lengths.items():
            if dst != src:
                dists.append(float(dist))

    diameter_est = float(max(dists)) if dists else 0.0
    avg_shortest_path_est = float(np.mean(dists)) if dists else 0.0
    return diameter_est, avg_shortest_path_est, lengths


def _exact_pairwise_distance_summary(
    G: nx.Graph, distance_key: str = "weight"
) -> tuple[float, float, dict[Hashable, dict[Hashable, float]]]:
    """Compute exact diameter and average shortest path from all pairs."""
    if G.number_of_nodes() <= 1:
        return 0.0, 0.0, {n: {n: 0.0} for n in G.nodes()}

    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=distance_key))
    nodes = list(lengths.keys())
    dists: list[float] = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            dists.append(lengths[u][v])

    diameter = float(max(dists)) if dists else 0.0
    avg_shortest_path = float(np.mean(dists)) if dists else 0.0
    return diameter, avg_shortest_path, lengths


def _greedy_community_summary(
    G: nx.Graph, similarity_key: str = "similarity_weight"
) -> tuple[int, float, float]:
    """Compute greedy-modularity community summaries."""
    if G.number_of_nodes() <= 1:
        return 1, 0.0, 1.0
    if G.number_of_edges() == 0:
        n_nodes = G.number_of_nodes()
        return n_nodes, 0.0, float(1.0 / n_nodes)

    communities = list(
        nx.algorithms.community.greedy_modularity_communities(
            G, weight=similarity_key
        )
    )
    if not communities:
        return 0, np.nan, np.nan

    modularity = nx.algorithms.community.modularity(
        G, communities, weight=similarity_key
    )
    largest_comm_size = max((len(c) for c in communities), default=0)
    largest_comm_fraction = float(largest_comm_size / G.number_of_nodes())
    return int(len(communities)), float(modularity), largest_comm_fraction


def _intraclonal_diversity_metrics(
    G: nx.Graph,
    distance_key: str = "weight",
    similarity_key: str = "similarity_weight",
    fast: bool = True,
    n_sources: int = 32,
    betweenness_k: int = 64,
    random_state: int = 42,
) -> dict[str, int | float | str]:
    """Compute clone-level intraclonal graph and community metrics.

    Parameters
    ----------
    G : nx.Graph
        Clone graph where nodes are cells/sequences and edges encode relatedness.
    distance_key : str
        Edge attribute interpreted as distance for shortest-path and centrality computations.
    similarity_key : str
        Edge attribute interpreted as similarity for PageRank and community weighting.
    fast : bool
        If True, estimate path metrics and use approximate betweenness for speed.
    n_sources : int
        Number of sampled source nodes for shortest-path estimates in fast mode.
    betweenness_k : int
        Number of sampled pivot nodes for approximate betweenness in fast mode.
    random_state : int
        Seed for reproducible sampling in fast mode.

    Returns
    -------
    dict
        Dictionary with keys (all values are int or float unless noted):

        n_nodes : int
            Number of nodes in the clone graph.
        n_edges : int
            Number of edges in the clone graph.
        largest_community_fraction : float
            Fraction of nodes in the largest greedy-modularity community. Higher values
            indicate one dominant sub-community; lower values indicate a more even split.
        n_communities_greedy : int
            Number of communities detected by greedy modularity.
        modularity_greedy : float
            Community-separation score for the detected partition. Larger values suggest
            stronger within-community connectivity relative to between-community mixing.
        density : float
            Fraction of possible edges that are present.
        mean_degree : float
            Mean node degree; reflects average connectivity level.
        var_degree : float
            Variance of node degree; reflects connectivity heterogeneity.
        degree_assortativity : float
            Degree-degree correlation across edges. Negative values suggest hub-spoke structure.
        transitivity : float
            Transitivity (global clustering coefficient). Higher values indicate tighter triangles.
        average_clustering : float
            Average clustering coefficient. Higher values suggest tighter local neighborhoods.
        diameter_component : float
            Longest shortest path in the largest detected community.
        average_shortest_path_component : float
            Mean shortest-path distance in the largest detected community.
        mean_betweenness : float
            Mean betweenness centrality in the largest detected community.
        mean_closeness : float
            Mean closeness centrality in the largest detected community.
        max_pagerank : float
            Highest PageRank score in the largest detected community.
        mode : str
            "fast" for sampled/approximate calculations or "exact" for full calculations.
    """
    if G.number_of_nodes() == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "largest_community_fraction": np.nan,
            "n_communities_greedy": np.nan,
            "modularity_greedy": np.nan,
            "density": np.nan,
            "mean_degree": np.nan,
            "var_degree": np.nan,
            "degree_assortativity": np.nan,
            "transitivity": np.nan,
            "average_clustering": np.nan,
            "diameter_component": np.nan,
            "average_shortest_path_component": np.nan,
            "mean_betweenness": np.nan,
            "mean_closeness": np.nan,
            "max_pagerank": np.nan,
            "mode": "fast" if fast else "exact",
        }

    H = _with_similarity_weights(
        G, distance_key=distance_key, similarity_key=similarity_key
    )
    degrees = np.array([d for _, d in H.degree()], dtype=float)

    n_communities, modularity, largest_community_fraction = (
        _greedy_community_summary(H, similarity_key=similarity_key)
    )

    communities = list(
        nx.algorithms.community.greedy_modularity_communities(
            H, weight=similarity_key
        )
    )

    if communities:
        largest_community_nodes = max(communities, key=len)
        H_main = H.subgraph(largest_community_nodes).copy()
    else:
        H_main = _largest_connected_component(H)

    if fast:
        diameter, avg_shortest_path, sampled_lengths = (
            _sample_pairwise_distance_summary(
                H_main,
                distance_key=distance_key,
                n_sources=n_sources,
                random_state=random_state,
            )
        )
        k = min(max(2, betweenness_k), H_main.number_of_nodes())
        betweenness = nx.betweenness_centrality(
            H_main, weight=distance_key, k=k, seed=random_state
        )
        closeness_vals = [
            nx.closeness_centrality(H_main, u=node, distance=distance_key)
            for node in sampled_lengths.keys()
        ]
    else:
        diameter, avg_shortest_path, _ = _exact_pairwise_distance_summary(
            H_main, distance_key=distance_key
        )
        betweenness = nx.betweenness_centrality(H_main, weight=distance_key)
        closeness_vals = list(
            nx.closeness_centrality(H_main, distance=distance_key).values()
        )

    pagerank = nx.pagerank(H_main, weight=similarity_key)

    return {
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
        "largest_community_fraction": float(largest_community_fraction),
        "n_communities_greedy": int(n_communities),
        "modularity_greedy": float(modularity),
        "density": float(nx.density(H)),
        "mean_degree": float(np.mean(degrees)),
        "var_degree": float(np.var(degrees)),
        "degree_assortativity": float(nx.degree_assortativity_coefficient(H)),
        "transitivity": float(nx.transitivity(H)),
        "average_clustering": float(nx.average_clustering(H)),
        "diameter_component": float(diameter),
        "average_shortest_path_component": float(avg_shortest_path),
        "mean_betweenness": float(np.mean(list(betweenness.values()))),
        "mean_closeness": (
            float(np.mean(closeness_vals)) if closeness_vals else np.nan
        ),
        "max_pagerank": float(np.max(list(pagerank.values()))),
        "mode": "fast" if fast else "exact",
    }


def _split_clone_ids(clone_value: object) -> list[str]:
    """Split a possibly multi-assigned clone_id string into separate clone identifiers."""
    if clone_value is None or pd.isna(clone_value):
        return []
    return [
        part.strip() for part in str(clone_value).split("|") if part.strip()
    ]


def _expanded_clone_assignments(
    vdj_obj: DandelionPolars,
    clone_col: str = "clone_id",
    cell_col: str = "cell_id",
    group_by: str | None = None,
) -> pd.DataFrame:
    """Return one row per cell-clone assignment with smart splitting of pipe-delimited IDs.

    Splitting rule:
    - For labels like A|B, split into A and B only if at least one component (A or B)
      is observed in more than one distinct original clone label across the dataset.
    - If none of the components are reused elsewhere, keep the original combined label A|B.

    The ``original_parts`` column records the individual pipe-delimited parts from
    the original label (regardless of whether the label was ultimately split or kept),
    so that callers can resolve graph keys when the graph dict uses the raw parts.
    """
    data = vdj_obj.data
    required_cols = [cell_col, clone_col]
    if group_by is not None:
        required_cols.append(group_by)

    if isinstance(data, pd.DataFrame):
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(
                "Missing required columns in vdj_obj.data: "
                + ", ".join(sorted(missing_cols))
            )
        node_clone_df = (
            data[required_cols]
            .drop_duplicates(subset=[cell_col], keep="first")
            .copy()
        )
    else:
        schema_cols = (
            data.collect_schema().names()
            if isinstance(data, pl.LazyFrame)
            else data.columns
        )
        missing_cols = [c for c in required_cols if c not in schema_cols]
        if missing_cols:
            raise ValueError(
                "Missing required columns in vdj_obj.data: "
                + ", ".join(sorted(missing_cols))
            )
        frame = data.select(required_cols).unique(
            subset=[cell_col], keep="first"
        )
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        node_clone_df = frame.to_pandas()

    node_clone_df["original_clone_id"] = node_clone_df[clone_col]
    node_clone_df["parts"] = node_clone_df[clone_col].map(_split_clone_ids)

    part_to_original_ids: dict[str, set[str]] = {}
    for _, row in node_clone_df.iterrows():
        original_id = str(row["original_clone_id"])
        for part in set(row["parts"]):
            if part not in part_to_original_ids:
                part_to_original_ids[part] = set()
            part_to_original_ids[part].add(original_id)

    def _smart_assign(parts: list[str], original_id: object) -> list[str]:
        if len(parts) <= 1:
            return parts
        shared_with_others = any(
            len(part_to_original_ids.get(part, set())) > 1 for part in parts
        )
        if shared_with_others:
            return parts
        return [str(original_id)]

    # Resolve what the assigned clone_id(s) will be after smart splitting.
    node_clone_df["assigned_clone_ids"] = node_clone_df.apply(
        lambda row: _smart_assign(row["parts"], row["original_clone_id"]),
        axis=1,
    )

    # FIX: retain the raw individual parts alongside the assigned label so
    # that when a combined label like "A|B" is kept unsplit (assigned_clone_ids
    # = ["A|B"]), we can still look up individual graph keys "A" and "B" in
    # graph_obj when it is a dict keyed by the per-part clone IDs.
    node_clone_df["original_parts"] = node_clone_df["parts"]

    node_clone_df[clone_col] = node_clone_df["assigned_clone_ids"]
    node_clone_df = node_clone_df.explode(clone_col)
    node_clone_df = node_clone_df.dropna(subset=[clone_col])
    node_clone_df = node_clone_df.drop(columns=["parts", "assigned_clone_ids"])
    return node_clone_df


def _require_clone_graph(
    vdj_obj: DandelionPolars,
) -> nx.Graph | dict[str, nx.Graph]:
    """Return ``vdj_obj.graph[1]`` or raise a clear error if unavailable."""
    graph_tuple = getattr(vdj_obj, "graph", None)
    if graph_tuple is None or not isinstance(graph_tuple, tuple):
        raise ValueError(
            "intraclonal_diversity requires vdj_obj.graph[1]. "
            "Run ddl.tl.generate_network(vdj_obj) first."
        )
    if len(graph_tuple) < 2 or graph_tuple[1] is None:
        raise ValueError(
            "intraclonal_diversity requires vdj_obj.graph[1]. "
            "Run ddl.tl.generate_network(vdj_obj) first."
        )
    return graph_tuple[1]


def intraclonal_diversity(
    vdj_obj: DandelionPolars,
    clone_col: str = "clone_id",
    cell_col: str = "cell_id",
    group_by: str | None = None,
    min_cells: int = 20,
    top_n: int | None = 25,
    fast: bool = True,
    n_sources: int = 16,
    betweenness_k: int = 32,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute one intraclonal-diversity row per clonotype.

    Smart handling of combined clone labels containing '|':
    - Split and tabulate separately only when at least one component is reused
      in another original clone label (interpreted as genuine sharing).
    - Otherwise keep the original combined label unchanged.

    Parameters
    ----------
    vdj_obj : object
        Dandelion object containing sequence metadata in vdj_obj.data and graphs in vdj_obj.graph.
    clone_col : str
        Column with clone assignments.
    cell_col : str
        Column with node identifiers used in the graph.
    group_by : str | None
        Optional metadata column to compute intraclonal metrics separately per group.
    min_cells : int
        Minimum number of unique cells required to keep a clone.
    top_n : int | None
        Keep only the top N largest clones after filtering when provided.
    fast : bool
        If True, estimate path metrics and use approximate betweenness for speed.
    n_sources : int
        Number of sampled source nodes for shortest-path estimates in fast mode.
    betweenness_k : int
        Number of sampled pivot nodes for approximate betweenness in fast mode.
    random_state : int
        Seed for reproducible sampling in fast mode.

    Returns
    -------
    pd.DataFrame
        One row per reported clone with clone identifier, clone size, and all
        intraclonal diversity metrics returned by the helper:

        clone_id : str
            Reported clonotype label after smart split logic.
        clone_size : int
            Number of unique cells assigned to the reported clonotype.
        group_by column (optional) : str
            Present only when ``group_by`` is provided; identifies the group for
            each clonotype row.
        n_nodes : int
            Number of cells in the clonotype subgraph (typically equals clone_size).
        n_edges : int
            Number of observed relationships among cells in the clonotype.
        largest_community_fraction : float
            Fraction of clonotype cells in the largest subgroup. High values mean most
            cells sit in one main subgroup; low values mean the clone is split into several parts.
        n_communities_greedy : int
            Number of detected subgroups inside the clonotype. More subgroups usually
            means more internal heterogeneity.
        modularity_greedy : float
            How clearly the clonotype separates into subgroups. Higher values mean
            subgroup boundaries are stronger.
        density : float
            Fraction of all possible within-clonotype links that are present.
            Higher values indicate a more tightly interconnected clonotype.
        mean_degree : float
            Average number of within-clonotype neighbors per cell.
        var_degree : float
            How uneven connectivity is across cells. High values mean a mix of
            highly connected cells and sparsely connected cells.
        degree_assortativity : float
            Whether highly connected cells mostly link to other highly connected cells.
            Negative values mean a "core-and-branches" pattern (few central cells linked
            to many less-connected cells). Positive values mean cells tend to connect to
            others with similar connectivity.
        transitivity : float
            How often connected triplets of clonotype cells form closed triangles.
            Higher values suggest tighter local groupings.
        average_clustering : float
            Average local cohesiveness around each cell. Higher values indicate cells tend
            to form tight local neighborhoods.
        diameter_component : float
            Largest shortest-path distance within the clonotype's main subgroup.
            Higher values suggest broader spread of the lineage in sequence space.
        average_shortest_path_component : float
            Average shortest-path distance within the clonotype's main subgroup.
            Lower values indicate a more compact lineage.
        mean_betweenness : float
            Average tendency of cells to sit on paths connecting other cells.
            Higher values suggest more bridge-like cells between subgroup regions.
        mean_closeness : float
            Average proximity of cells to the rest of the main subgroup.
            Higher values indicate cells are, on average, closer to one another.
        max_pagerank : float
            Highest centrality score in the main subgroup. High values indicate one or a few
            especially influential/central cells in that clonotype structure.
        mode : str
            "fast" for sampled/approximate calculations or "exact" for full calculations.

    Raises
    ------
    ValueError
        If ``vdj_obj.graph[1]`` is missing. Run ``ddl.tl.generate_network(vdj_obj)`` first.
    """
    graph_obj = _require_clone_graph(vdj_obj)

    node_clone_df = _expanded_clone_assignments(
        vdj_obj,
        clone_col=clone_col,
        cell_col=cell_col,
        group_by=group_by,
    )
    grouping_cols = [clone_col] if group_by is None else [group_by, clone_col]

    clone_sizes_pd = (
        node_clone_df.groupby(grouping_cols)[cell_col]
        .nunique()
        .rename("clone_size")
        .reset_index()
        .sort_values("clone_size", ascending=False)
    )
    clone_sizes_pd = clone_sizes_pd[clone_sizes_pd["clone_size"] >= min_cells]
    if top_n is not None:
        clone_sizes_pd = clone_sizes_pd.head(top_n)

    if clone_sizes_pd.empty:
        return pd.DataFrame()

    clone_to_nodes = (
        node_clone_df.groupby(grouping_cols)[cell_col]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    clone_to_original_ids = (
        node_clone_df.groupby(grouping_cols)["original_clone_id"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    # FIX: build a mapping from assigned clone_id -> all raw parts that were
    # present in the original pipe-delimited labels for those cells.  This is
    # needed when graph_obj is a dict keyed by individual part IDs and the
    # smart-split logic chose to keep a combined label (e.g. "A|B") rather
    # than splitting it.  In that case clone_to_original_ids gives ["A|B"]
    # which won't be found in graph_obj; we need the parts ["A", "B"] instead.
    clone_to_all_parts = (
        node_clone_df.groupby(grouping_cols)["original_parts"]
        .apply(lambda x: sorted({part for parts in x for part in parts}))
        .to_dict()
    )

    metrics_rows: list[dict[str, int | float | str]] = []

    for _, row in clone_sizes_pd.iterrows():
        clone_id = row[clone_col]
        if group_by is None:
            group_value = None
            lookup_key: str | tuple[str, str] = clone_id
        else:
            group_value = row[group_by]
            lookup_key = (group_value, clone_id)

        clone_nodes = clone_to_nodes.get(lookup_key, [])
        if not clone_nodes:
            continue

        if isinstance(graph_obj, dict):
            original_clone_ids = clone_to_original_ids.get(lookup_key, [])

            # First attempt: look up by the assigned clone label(s) directly.
            candidate_graphs = [
                graph_obj[oid] for oid in original_clone_ids if oid in graph_obj
            ]

            # FIX: if the assigned label is a kept-combined label like "A|B"
            # that doesn't exist as a key in graph_obj, fall back to looking
            # up by the individual raw parts ("A", "B") that were extracted
            # during splitting.  This covers the case where graph_obj was
            # built with per-part keys before clone IDs were combined.
            if not candidate_graphs:
                all_parts = clone_to_all_parts.get(lookup_key, [])
                candidate_graphs = [
                    graph_obj[part] for part in all_parts if part in graph_obj
                ]

            if not candidate_graphs:
                continue
            G_source = nx.compose_all(candidate_graphs)
        else:
            G_source = graph_obj

        G_clone = G_source.subgraph(clone_nodes).copy()
        if G_clone.number_of_nodes() == 0:
            continue

        metrics = _intraclonal_diversity_metrics(
            G_clone,
            fast=fast,
            n_sources=n_sources,
            betweenness_k=betweenness_k,
            random_state=random_state,
        )
        metrics[clone_col] = str(clone_id)
        metrics["clone_size"] = int(row["clone_size"])
        if group_by is not None:
            metrics[group_by] = str(group_value)
        metrics_rows.append(metrics)

    if not metrics_rows:
        return pd.DataFrame()

    out = pd.DataFrame(metrics_rows)

    first_cols = [clone_col, "clone_size"]
    if group_by is not None:
        first_cols.append(group_by)
    remaining_cols = [c for c in out.columns if c not in set(first_cols)]
    out = out[first_cols + remaining_cols]

    sort_cols = ["clone_size", "average_shortest_path_component"]
    sort_ascending = [False, False]
    if group_by is not None:
        sort_cols = [group_by] + sort_cols
        sort_ascending = [True] + sort_ascending

    out = out.sort_values(sort_cols, ascending=sort_ascending)
    return out


def intraclonal_metrics_per_clone(
    vdj_obj: DandelionPolars,
    clone_col: str = "clone_id",
    cell_col: str = "cell_id",
    group_by: str | None = None,
    min_cells: int = 20,
    top_n: int | None = 25,
    fast: bool = True,
    n_sources: int = 16,
    betweenness_k: int = 32,
    random_state: int = 42,
) -> pd.DataFrame:
    """Backward-compatible wrapper for ``intraclonal_diversity``."""
    return intraclonal_diversity(
        vdj_obj=vdj_obj,
        clone_col=clone_col,
        cell_col=cell_col,
        group_by=group_by,
        min_cells=min_cells,
        top_n=top_n,
        fast=fast,
        n_sources=n_sources,
        betweenness_k=betweenness_k,
        random_state=random_state,
    )
