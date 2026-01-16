from __future__ import annotations

import multiprocessing
import re
import time

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from collections import defaultdict
from joblib import Parallel, delayed
from pathlib import Path
from polyleven import levenshtein
from scanpy import logging as logg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import csgraph_from_dense
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


from dandelion.tools._tools_polars import vdj_sample
from dandelion.tools._layout import generate_layout
from dandelion.utilities._polars import DandelionPolars, _sanitize_data_polars
from dandelion.utilities._core import Dandelion
from dandelion.utilities._distances import (
    Metric,
    dist_func_long_sep,
    prepare_sequences_with_separator,
    resolve_metric,
)
from dandelion.utilities._utilities import (
    flatten,
    present,
    Tree,
    FALSES,
)


def generate_network(
    vdj: DandelionPolars,
    adata: AnnData | None = None,
    key: str | None = None,
    clone_key: str | None = None,
    min_size: int = 2,
    sample: int | None = None,
    force_replace: bool = False,
    verbose: bool = True,
    compute_graph: bool = True,
    compute_layout: bool = True,
    layout_method: Literal["mod_fr", "sfdp"] = "mod_fr",
    expanded_only: bool = False,
    use_existing_graph: bool = True,
    n_cpus: int = 1,
    sequential_chain: bool = False,
    distance_mode: Literal["clone", "full"] = "clone",
    dist_func: Callable | str | None = None,
    pad_to_max: bool = False,
    lazy: bool = False,
    zarr_path: Path | str | None = None,
    chunk_size: int | None = None,
    chunk_clone_limit: int | None = None,
    memory_limit_gb: float | None = None,
    memory_safety_fraction: float = 0.3,
    compress: bool = True,
    random_state: int | np.random.RandomState | None = None,
    **kwargs,
) -> DandelionPolars | tuple[DandelionPolars, AnnData]:
    """
    Generate a Levenshtein distance network based on VDJ and VJ sequences.

    The distance matrices are then combined into a singular matrix.

    Parameters
    ----------
    vdj : DandelionPolars
        Dandelion object.
    key : str | None, optional
        column name for distance calculations. None defaults to 'sequence_alignment_aa'.
    clone_key : str | None, optional
        column name to build network on.
    min_size : int, optional
        For visualization purposes, two graphs are created where one contains all cells and a trimmed second graph.
        This value specifies the minimum number of edges required otherwise node will be trimmed in the secondary graph.
    sample : int | None, optional
        If specified, cells will be randomly sampled to the integer provided. If the integer is larger than the number of cells,
        sampling with replacement is used and the same cell may appear multiple times with different sequence and cell ids. If None,
        no resampling is performed. A new Dandelion class will be returned.
    force_replace : bool, optional
        whether or not to sample with replacement when `sample` is smaller or equal to than the number of cells.
    verbose : bool, optional
        whether or not to print the progress bars.
    compute_graph : bool, optional
        whether or not to generate the graph after distance matrix calculation.
    compute_layout : bool, optional
        whether or not to generate the layout. May be time consuming if too many cells.
    layout_method : Literal["sfdp", "mod_fr"], optional
        accepts one of 'sfdp' or 'mod_fr'. 'sfdp' refers to `sfdp_layout` from `graph_tool` (C++ implementation; fast)
        whereas 'mod_fr' refers to modified Fruchterman-Reingold layout originally implemented in dandelion (python
        implementation; slightly slower).
    expanded_only : bool, optional
        whether or not to only compute layout on expanded clonotypes.
    use_existing_graph : bool, optional
        whether or not to just compute the layout using the existing graph if it exists in the object.
    n_cpus : int, optional
        number of cores to use for parallelizable steps. -1 uses all available cores.
    sequential_chain : bool, optional
        whether or not to use the original method for distance calculation method where each chain is calculated
        separately and sequentially added to the total distance matrix. This method is slower but would be more
        precise calculation. If False, concatenated sequences with a long separator are used for distance calculation.
        Ignored if lazy=True as the lazy method always uses the long separator approach. The long separator approach
        inserts a long string of consistent characters on a per-chain basis to ensure that distances between chains are large
        and do not interfere with intra-chain distances.
    distance_mode : Literal["clone", "full"], optional
        method to compute distance matrix. 'clone' refers to the original membership-based distance calculation where
        only distances within clones are calculated. Whereas 'full' computes the full pairwise distance matrix.
    dist_func : Callable | str | None, optional
        distance function to use. If None, `polyleven.levenshtein` is used. If a string is provided, it will use Bio.Align's
        substitution matrices (e.g., 'BLOSUM62', 'PAM250'). See `Bio.Align.substitution_matrices.load` for available options.
    blosom_matrix : str | None, optional
        If provided, dist_func is ignored and this substitution matrix from Bio.Align is used instead.
    pad_to_max : bool, optional
        whether or not to pad sequences to the maximum length in the dataset before distance calculation. This will
        allow for distance calculations that need sequences of the same length (e.g., Hamming distance). Note that this
        may increase memory usage and computation time.
    lazy: bool, optional
        If True, computation will be performed lazily using Dask/Zarr arrays. True will also return a Dask array view of the
        distance matrix stored on disk instead of a numpy array stored in memory.
    zarr_path: Path | str | None, optional
        Path to store Zarr array when using lazy mode. If None, "distance_matrix.zarr" will be created in the current working directory.
    chunk_size: int | None, optional
        Chunk size for distance matrix computation when using lazy mode. If None, chunk size is automatically computed
        based on available memory and number of cores. The automatic chunk size can be further adjusted using
        `memory_limit_gb` and `memory_safety_fraction` parameters.
    chunk_clone_limit: int | None, optional
        Maximum number of clones to process per chunk when using lazy mode and distance method = "clone". If None, chunk sizes will be
        automatically determined based on available memory and number of cores.
    memory_limit_gb: float | None, optional
        Memory limit per worker in GB for Dask. None defaults to all available memory/cores.
    memory_safety_fraction: float, optional
        Fraction of available memory to use. Defaults to 0.3 (i.e., 30% of available memory will be used for chunk size calculation).
    compress: bool, optional
        Whether to compress the Zarr array using Blosc with zstd.
    rnandom_state : int | np.random.RandomState | None, optional
        Random state for reproducible sampling.
    **kwargs
        additional kwargs passed to options specified in `networkx.drawing.layout.spring_layout` or
        `graph_tool.draw.sfdp_layout`.

    Returns
    -------
    Dandelion | tuple[Dandelion, AnnData]
        Dandelionbject with `.edges`, `.layout`, `.graph` initialized.

    Raises
    ------
    ValueError
        if any errors with dandelion input.
    """
    # normalize n_cpus convention (-1 => use all CPUs)
    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count()
    n_cpus = max(1, int(n_cpus))
    clone_key = clone_key if clone_key is not None else "clone_id"
    dist_func = levenshtein if dist_func is None else dist_func
    metric = resolve_metric(dist_func)
    if not compute_graph:
        compute_layout = False

    if distance_mode == "clone" or compute_graph or compute_layout:
        if clone_key not in vdj._data:
            raise ValueError(
                "Data does not contain clone information. Please run ddl.tl.find_clones."
            )
    regenerate = True
    if vdj.graph is not None:
        if (min_size != 2) or (sample is not None):
            pass
        elif use_existing_graph:
            start = logg.info(
                "Generating network layout from pre-computed network"
            )
            if isinstance(vdj, Dandelion):
                regenerate = False
                g, g_, lyt, lyt_ = generate_layout(
                    vertices=None,
                    edges=None,
                    min_size=min_size,
                    weight=None,
                    verbose=verbose,
                    compute_layout=compute_layout,
                    layout_method=layout_method,
                    expanded_only=expanded_only,
                    graphs=(vdj.graph[0], vdj.graph[1]),
                    **kwargs,
                )

    if regenerate:
        start = logg.info("Generating network")

        key_ = key if key is not None else "sequence_alignment_aa"

        if key_ not in vdj._data:
            raise ValueError(f"key {key_} not found in data.")

        if lazy:
            from dandelion.tools._lazydistances_polars import (
                dask_safe_slice_square,
            )

        if sample is not None:
            if adata is not None:
                vdj, adata = vdj_sample(
                    vdj=vdj,
                    size=sample,
                    adata=adata,
                    force_replace=force_replace,
                    random_state=random_state,
                )
            else:
                vdj = vdj_sample(
                    vdj=vdj,
                    size=sample,
                    force_replace=force_replace,
                    random_state=random_state,
                )

        dat = vdj[
            vdj.data.locus.is_in(
                ["IGH", "TRB", "TRD", "IGK", "IGL", "TRA", "TRG"]
            )
        ]

        if "ambiguous" in dat.data.collect_schema().names():
            # Convert FALSES to strings only (remove boolean False) for Polars compatibility
            falses_strings = [str(f) for f in FALSES if f is not False]
            falses_strings.append(
                "False"
            )  # Ensure both uppercase and lowercase are included
            dat = dat[
                dat.data["ambiguous"].cast(pl.String).is_in(falses_strings)
            ]

        dat_seq = dat._split(key_, explode=True)
        dat_seq = dat_seq.rename(
            {
                col: re.sub(f"^{key_}_", "", col)
                for col in dat_seq.collect_schema().names()
                if col.startswith(f"{key_}_")
            }
        )
        # Build a mapping of cell_id to row position using explicit row indexing
        dat_seq_indexed = (
            dat_seq.with_row_index("_row_pos")
            if isinstance(dat_seq, pl.DataFrame)
            else dat_seq.lazy()
            .with_row_index("_row_pos")
            .collect(engine="streaming")
        )
        cell_id_to_pos = dict(
            zip(
                dat_seq_indexed["cell_id"].to_list(),
                dat_seq_indexed["_row_pos"].to_list(),
            )
        )

        if compute_graph or compute_layout or distance_mode == "clone":
            membership = {}
            if isinstance(dat._metadata, pl.LazyFrame):
                clone_data = dat._metadata.select(
                    ["cell_id", clone_key]
                ).collect(engine="streaming")
            elif isinstance(dat._metadata, pl.DataFrame):
                clone_data = dat._metadata.select(["cell_id", clone_key])
            else:
                clone_data = dat._metadata[["cell_id", clone_key]]

            # Keep as polars and iterate using polars operations
            if isinstance(clone_data, pl.LazyFrame):
                clone_data = clone_data.collect(engine="streaming")

            for cell_name, clone_str in zip(
                clone_data["cell_id"].to_list(), clone_data[clone_key].to_list()
            ):
                if (
                    clone_str is not None
                    and clone_str.lower() != "none"
                    and clone_str != ""
                ):
                    clone_ids = str(clone_str).split("|")
                    for clone_id in clone_ids:
                        clone_id = clone_id.strip()
                        if (
                            clone_id
                            and clone_id.lower() != "none"
                            and clone_id != ""
                        ):
                            if clone_id not in membership:
                                membership[clone_id] = []
                            if cell_name not in membership[clone_id]:
                                membership[clone_id].append(cell_name)

        # compute total_dist using chosen mode (original uses membership)
        logg.info(
            f"Calculating distance matrix {'lazily' if lazy else ''} with distance_mode = '{distance_mode}'\n"
        )
        if distance_mode == "clone":
            if lazy:
                from dandelion.tools._lazydistances_polars import (
                    calculate_distance_matrix_zarr,
                )

                # Determine Zarr destination and mark embedding intent
                if zarr_path is None:
                    import tempfile

                    zarr_tmp = tempfile.mkdtemp()
                    # Flags on object to indicate pending embed on write
                    try:
                        setattr(vdj, "_distance_zarr_path", zarr_tmp)
                        setattr(vdj, "_distance_embed_pending", True)
                    except Exception:
                        pass
                else:
                    # External Zarr mode
                    try:
                        setattr(vdj, "_distance_zarr_path", str(zarr_path))
                        setattr(vdj, "_distance_embed_pending", False)
                    except Exception:
                        pass

                _ = calculate_distance_matrix_zarr(
                    dat_seq,
                    metric=metric,
                    pad_to_max=pad_to_max,
                    membership=membership,
                    zarr_path=(zarr_tmp if zarr_path is None else zarr_path),
                    chunk_size=chunk_size,
                    n_cpus=n_cpus,
                    memory_limit_gb=memory_limit_gb,
                    compress=compress,
                    verbose=verbose,
                )
                # Create a Dask view of the on-disk distances
                import dask.array as da

                zpath = zarr_tmp if zarr_path is None else zarr_path
                try:
                    total_dist = da.from_zarr(
                        str(zpath) + "/distance_matrix.zarr/distance_matrix"
                    )
                except Exception:
                    total_dist = da.from_zarr(
                        str(zpath) + "/distance_matrix.zarr"
                    )
            else:
                if sequential_chain:
                    total_dist = calculate_distance_matrix_original(
                        dat_seq,
                        membership,
                        metric=metric,
                        pad_to_max=pad_to_max,
                        verbose=verbose,
                    )
                else:
                    total_dist = calculate_distance_matrix_long(
                        dat_seq,
                        membership=membership,
                        metric=metric,
                        pad_to_max=pad_to_max,
                        n_cpus=n_cpus,
                        verbose=verbose,
                    )
        elif distance_mode == "full":
            if lazy:
                from dandelion.tools._lazydistances_polars import (
                    calculate_distance_matrix_zarr,
                )

                # Determine Zarr destination and mark embedding intent
                if zarr_path is None:
                    import tempfile

                    zarr_tmp = tempfile.mkdtemp()
                    try:
                        setattr(vdj, "_distance_zarr_path", zarr_tmp)
                        setattr(vdj, "_distance_embed_pending", True)
                    except Exception:
                        pass
                else:
                    try:
                        setattr(vdj, "_distance_zarr_path", str(zarr_path))
                        setattr(vdj, "_distance_embed_pending", False)
                    except Exception:
                        pass

                _ = calculate_distance_matrix_zarr(
                    dat_seq,
                    metric=metric,
                    pad_to_max=pad_to_max,
                    membership=None,
                    zarr_path=(zarr_tmp if zarr_path is None else zarr_path),
                    chunk_size=chunk_size,
                    n_cpus=n_cpus,
                    memory_limit_gb=memory_limit_gb,
                    compress=compress,
                    verbose=verbose,
                )
                # Create a Dask view of the on-disk distances
                import dask.array as da

                zpath = zarr_tmp if zarr_path is None else zarr_path
                try:
                    total_dist = da.from_zarr(
                        str(zpath) + "/distance_matrix.zarr/distance_matrix"
                    )
                except Exception:
                    total_dist = da.from_zarr(
                        str(zpath) + "/distance_matrix.zarr"
                    )
            else:
                if sequential_chain:
                    total_dist = calculate_distance_matrix_original_full(
                        dat_seq,
                        metric=metric,
                        pad_to_max=pad_to_max,
                        n_cpus=n_cpus,
                        verbose=verbose,
                    )
                else:
                    total_dist = calculate_distance_matrix_long(
                        dat_seq,
                        membership=None,
                        metric=metric,
                        pad_to_max=pad_to_max,
                        n_cpus=n_cpus,
                        verbose=verbose,
                    )
        if compute_graph:
            tmp_clusterdist = Tree()
            cluster_dist = {}
            overlap = []

            # Normalize metadata to Polars DataFrame if lazy
            if isinstance(vdj._metadata, pl.LazyFrame):
                meta_df = vdj._metadata.collect(engine="streaming")
            elif isinstance(vdj._metadata, pl.DataFrame):
                meta_df = vdj._metadata
            else:
                meta_df = pl.from_pandas(vdj._metadata)

            # Add row index to preserve original positions
            meta_df = meta_df.with_row_index("_orig_idx")

            # Vectorized split and overlap detection
            meta_df_split = meta_df.with_columns(
                pl.col(str(clone_key)).str.split("|").alias("_clone_list")
            ).with_columns(
                pl.col("_clone_list").list.len().gt(1).alias("_is_overlap")
            )

            # Collect overlaps efficiently
            overlap = [
                [c for c in row["_clone_list"] if c != "None"]
                for row in meta_df_split.filter(pl.col("_is_overlap"))
                .select("_clone_list")
                .iter_rows(named=True)
            ]

            # Explode and populate tmp_clusterdist in one vectorized operation
            # Use cell_id for mapping to distance matrix positions
            meta_exploded = (
                meta_df_split.select(["cell_id", "_clone_list"])
                .explode("_clone_list")
                .filter(pl.col("_clone_list") != "None")
                .rename({"_clone_list": "clone_id"})
            )

            # Single iteration on the exploded (pre-filtered) data
            for row in meta_exploded.iter_rows(named=True):
                tmp_clusterdist[row["clone_id"]][row["cell_id"]].value = 1
            tmp_clusterdist2 = {}
            for x in tmp_clusterdist:
                tmp_clusterdist2[x] = list(tmp_clusterdist[x])

            # Get valid row count for index filtering
            n_rows = meta_df.height
            cluster_dist = {}
            # NOTE: dist_mat_ is kept as pandas DataFrame because:
            # 1. NetworkX (used in mst() and nx.to_pandas_edgelist()) requires pandas
            # 2. These are small per-clone submatrices, not the full distance matrix
            # 3. All downstream operations (MST, graph construction) use pandas
            for c_ in tqdm(
                tmp_clusterdist2,
                desc="Sorting into clusters ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):
                idxs = [i for i in tmp_clusterdist2[c_] if i in cell_id_to_pos]
                if c_ in list(flatten(overlap)):
                    for ol in overlap:
                        if c_ in ol:
                            idx = list(
                                set(
                                    flatten(
                                        [tmp_clusterdist2[c_x] for c_x in ol]
                                    )
                                )
                            )
                            if len(list(set(idx))) > 1:
                                # Mirror pandas: slice using the current clone's members (idxs)
                                # even when overlapping clones exist.
                                pos = [cell_id_to_pos[i] for i in idxs]
                                if lazy:
                                    dist_mat_ = pd.DataFrame(
                                        dask_safe_slice_square(
                                            total_dist, pos
                                        ).compute(),
                                        index=idxs,
                                        columns=idxs,
                                    )
                                else:
                                    dist_mat_ = pd.DataFrame(
                                        total_dist[np.ix_(pos, pos)],
                                        index=idxs,
                                        columns=idxs,
                                    )
                                s1, s2 = dist_mat_.shape
                                if s1 > 1 and s2 > 1:
                                    cluster_dist["|".join(ol)] = dist_mat_
                else:
                    pos = [cell_id_to_pos[i] for i in idxs]
                    if lazy:
                        dist_mat_ = pd.DataFrame(
                            dask_safe_slice_square(total_dist, pos).compute(),
                            index=idxs,
                            columns=idxs,
                        )
                    else:
                        dist_mat_ = pd.DataFrame(
                            total_dist[np.ix_(pos, pos)],
                            index=idxs,
                            columns=idxs,
                        )
                    s1, s2 = dist_mat_.shape
                    if s1 > 1 and s2 > 1:
                        cluster_dist[c_] = dist_mat_
            # Minimum spanning trees (unchanged)
            # mst_tree = mst(cluster_dist, n_cpus=n_cpus, verbose=verbose)
            mst_tree = mst(cluster_dist, n_cpus=1, verbose=verbose)
            # generate edge list
            edge_list = Tree()
            for c in tqdm(
                mst_tree,
                desc="Generating edge list ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):
                edge_list[c] = nx.to_pandas_edgelist(mst_tree[c])
                if edge_list[c].shape[0] > 0:
                    edge_list[c]["weight"] = edge_list[c]["weight"] - 1
                    edge_list[c].loc[edge_list[c]["weight"] < 0, "weight"] = 0

            # Normalize metadata to pandas for downstream operations
            meta_pd = meta_df.to_pandas()
            clone_ref = dict(zip(meta_pd["cell_id"], meta_pd[clone_key]))
            clone_ref = {k: r for k, r in clone_ref.items() if r != "None"}
            tmp_clone_tree = Tree()
            for x in meta_pd["cell_id"].tolist():
                if x in clone_ref:
                    if "|" in clone_ref[x]:
                        for x_ in clone_ref[x].split("|"):
                            if x_ != "None":
                                tmp_clone_tree[x_][x].value = 1
                    else:
                        tmp_clone_tree[clone_ref[x]][x].value = 1
            tmp_clone_tree2 = Tree()
            for x in tmp_clone_tree:
                tmp_clone_tree2[x] = list(tmp_clone_tree[x])

            tmp_clone_tree3 = Tree()
            tmp_clone_tree3_overlap = Tree()
            for x in tqdm(
                tmp_clone_tree2,
                desc="Computing overlap ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):
                # this is to catch all possible cells that may potentially match up with this clone that's joined together
                if x in list(flatten(overlap)):
                    for ol in overlap:
                        if x in ol:
                            if len(tmp_clone_tree2[x]) > 1:
                                for x_ in tmp_clone_tree2[x]:
                                    tmp_clone_tree3_overlap["|".join(ol)][
                                        "".join(x_)
                                    ].value = 1
                            else:
                                tmp_clone_tree3_overlap["|".join(ol)][
                                    "".join(tmp_clone_tree2[x])
                                ].value = 1
                else:
                    tmp_ = pd.DataFrame(
                        index=tmp_clone_tree2[x], columns=tmp_clone_tree2[x]
                    )
                    tmp_ = pd.DataFrame(
                        np.tril(tmp_) + 1,
                        index=tmp_clone_tree2[x],
                        columns=tmp_clone_tree2[x],
                    )
                    tmp_ = tmp_.astype(float).fillna(0)
                    tmp_clone_tree3[x] = tmp_

            for x in tqdm(
                tmp_clone_tree3_overlap,
                desc="Adjust overlap ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):  # repeat for the overlap clones
                tmp_ = pd.DataFrame(
                    index=tmp_clone_tree3_overlap[x],
                    columns=tmp_clone_tree3_overlap[x],
                )
                tmp_ = pd.DataFrame(
                    np.tril(tmp_) + 1,
                    index=tmp_clone_tree3_overlap[x],
                    columns=tmp_clone_tree3_overlap[x],
                )
                tmp_ = tmp_.astype(float).fillna(0)
                tmp_clone_tree3[x] = tmp_

            # free up memory
            del tmp_clone_tree2

            # convert total_dist to sparse graph
            tmp_g = csgraph_from_dense(
                total_dist.compute() + 1
                if lazy
                else total_dist + 1  # +1 to keep zeros as infinite distance
            )
            # construct edge list as a dictionary
            Gcoo = tmp_g.tocoo()
            # remove self-loops
            mask = Gcoo.row != Gcoo.col
            rows = Gcoo.row[mask]
            cols = Gcoo.col[mask]
            weights = (
                Gcoo.data[mask] - 1
            )  # -1 to revert back to original distance
            # Ensure cell ordering matches the distance matrix rows (dat_seq order)
            meta_cells = dat_seq_indexed["cell_id"].to_list()
            tmp_totaldiststack = {
                meta_cells[r] + "|" + meta_cells[c]: w
                for r, c, w in zip(rows, cols, weights)
            }
            tmp_totaldiststack = pd.DataFrame(
                pd.Series(tmp_totaldiststack, name="weight")
            )
            # convert tmp_totaldist to edge list and rename the index
            tmp_edge_list = Tree()
            for c in tqdm(
                tmp_clone_tree3,
                desc="Linking edges ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):
                if len(tmp_clone_tree3[c]) > 1:
                    G = create_networkx_graph(
                        tmp_clone_tree3[c],
                        drop_zero=True,
                    )
                    G.remove_edges_from(nx.selfloop_edges(G))
                    tmp_edge_list[c] = nx.to_pandas_edgelist(G)
                    set_edge_list_index(tmp_edge_list[c])

                    tmp_edge_list[c].update(
                        {"weight": tmp_totaldiststack["weight"]}
                    )
                    # keep only edges when there is 100% identity, to minimise crowding
                    tmp_edge_list[c] = tmp_edge_list[c][
                        tmp_edge_list[c]["weight"] == 0
                    ]
                    tmp_edge_list[c].reset_index(inplace=True)

            # try to catch situations where there's no edge (only singletons)
            try:
                edge_listx = pd.concat([edge_list[x] for x in edge_list])
                set_edge_list_index(edge_listx)
                tmp_edge_listx = pd.concat(
                    [tmp_edge_list[x] for x in tmp_edge_list]
                )
                tmp_edge_listx = tmp_edge_listx.drop("index", axis=1)
                set_edge_list_index(tmp_edge_listx)

                edge_list_final = edge_listx.combine_first(tmp_edge_listx)

                common_idx = edge_list_final.index.intersection(
                    tmp_totaldiststack.index
                )
                edge_list_final.loc[common_idx, "weight"] = (
                    tmp_totaldiststack.loc[common_idx, "weight"]
                )

                # for i, row in tmp_totaldiststack.iterrows():
                #     if i in edge_list_final.index:
                #         edge_list_final.at[i, "weight"] = row["weight"]

                # return the edge list
                edge_list_final = edge_list_final.reset_index(drop=True)
            except Exception:
                edge_list_final = None

            # final layout + graph creation (unchanged)
            g, g_, lyt, lyt_ = generate_layout(
                vertices=meta_pd["cell_id"].tolist(),
                edges=edge_list_final,
                min_size=min_size,
                weight=None,
                verbose=verbose,
                compute_layout=compute_layout,
                layout_method=layout_method,
                expanded_only=expanded_only,
                **kwargs,
            )

    logg.info(
        " finished.\n   Updated Dandelion object\n",
        time=start,
        deep=(
            "   'layout', graph layout\n"
            if compute_layout
            else (
                ""
                "   'graph', network constructed from distance matrices of VDJ- and VJ- chains\n"
                if compute_graph
                else (
                    "" "   'distances', VDJ + VJ distance matrix\n"
                    if regenerate
                    else ""
                )
            )
        ),
    )

    # return or re-initialize vdj
    germline = getattr(vdj, "germline", None)
    if regenerate:
        distances = total_dist if lazy else csr_matrix(total_dist)
        if not lazy:
            distances._index_names = vdj.metadata_names
    else:
        distances = vdj.distances
    graph = (g, g_) if compute_graph else None
    layout = (lyt, lyt_) if compute_graph and compute_layout else None
    if sample is not None:
        out = Dandelion(
            data=dat_,
            metadata=vdj._metadata,
            clone_key=clone_key,
            layout=layout,
            graph=graph,
            distances=distances,
            germline=germline,
            verbose=False,
        )
        if adata is None:
            return out
        else:
            return out, adata
    else:
        vdj.__init__(
            data=vdj._data,
            metadata=vdj._metadata,
            clone_key=clone_key,
            layout=layout,
            graph=graph,
            distances=distances,
            germline=germline,
            initialize=False,
            verbose=False,
        )


def mst(
    mat: dict,
    n_cpus: int | None = None,
    verbose: bool = True,
) -> Tree:
    """
    Construct minimum spanning tree based on supplied matrix in dictionary.

    Parameters
    ----------
    mat : dict
        Dictionary containing numpy ndarrays.
    n_cpus: int, optional
        Number of cores to run this step. Parallelise using `sklearn.metrics.pairwise_distances` if n_cpus > 1..
    verbose : bool, optional
        Whether or not to show logging information.

    Returns
    -------
    Tree
        Dandelion `Tree` object holding DataFrames of constructed minimum spanning trees.
    """
    mst_tree = Tree()

    if n_cpus == 1:
        for c in tqdm(
            mat,
            desc="Calculating minimum spanning tree ",
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            _, mst_tree[c] = process_mst_per_clonotype(mat=mat, c=c)
    else:
        results = Parallel(n_jobs=n_cpus)(
            delayed(process_mst_per_clonotype)(mat, c)
            for c in tqdm(
                mat,
                desc=f"Calculating minimum spanning tree, parallelized across {n_cpus} cores ",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        )
        for result in results:
            mst_tree[result[0]] = result[1]
    return mst_tree


def process_mst_per_clonotype(
    mat: dict[str, pd.DataFrame], c: str
) -> tuple[str, nx.Graph]:
    """
    Function to calculate minimum spanning tree.

    Parameters
    ----------
    mat : dict[str, pd.DataFrame]
        Dictionary holding distance matrices.
    c : str
        Name of clonotype.

    Returns
    -------
    tuple[str, nx.Graph]
        Graph holding minimum spanning tree.
    """
    tmp = mat[c] + 1
    tmp[np.isnan(tmp)] = 0
    G = create_networkx_graph(tmp, drop_zero=True)
    return c, nx.minimum_spanning_tree(G)


def create_networkx_graph(
    adjacency: pd.DataFrame, drop_zero: bool = True
) -> nx.Graph:
    """
    Create a networkx graph from an adjacency matrix in chunks.

    Parameters
    ----------
    adjacency : pd.DataFrame
        Adjacency matrix.
    drop_zero : bool, optional
        Whether or not to drop edges with zero weight, by default True.

    Returns
    -------
    nx.Graph
        NetworkX graph.
    """
    G = nx.Graph()
    # add nodes
    nodes = list(adjacency.index)
    G.add_nodes_from(nodes)
    # convert adjacency matrix to edge list
    edge_list = adjacency_to_edge_list(adjacency, drop_zero=drop_zero)
    G.add_weighted_edges_from(ebunch_to_add=edge_list.values)
    return G


def set_edge_list_index(edge_list: pd.DataFrame) -> None:
    """
    Set the index of the edge list in-place.

    Parameters
    ----------
    edge_list : pd.DataFrame
        Edge list.
    """
    # edge_list.index = [
    #     str(s) + "|" + str(t)
    #     for s, t in zip(edge_list["source"], edge_list["target"])
    # ]
    edge_list.index = (
        edge_list["source"].astype(str) + "|" + edge_list["target"].astype(str)
    )


def adjacency_to_edge_list(
    adjacency: pd.DataFrame, drop_zero: bool = False, rename_index: bool = False
) -> pd.DataFrame:
    """
    Convert adjacency matrix to edge list that excludes self-loops.

    Parameters
    ----------
    adjacency : pd.DataFrame
        Adjacency matrix.
    drop_zero : bool, optional
        Whether or not to drop edges with zero weight, by default False.
    rename_index : bool, optional
        Whether or not to rename the index, by default False.

    Returns
    -------
    pd.DataFrame
        Edge list.
    """
    edge_list = adjacency.stack().reset_index()
    edge_list.columns = ["source", "target", "weight"]
    if rename_index:
        set_edge_list_index(edge_list)
    if drop_zero:
        edge_list = edge_list[edge_list["weight"] != 0]
    return edge_list


def clone_degree(vdj: Dandelion, weight: str | None = None) -> Dandelion:
    """
    Calculate node degree in BCR/TCR network.

    Parameters
    ----------
    vdj : Dandelion
        Dandelionect after `tl.generate_network` has been run.
    weight : str | None, optional
        Attribute name for retrieving edge weight in graph. None defaults to ignoring this. See `networkx.Graph.degree`.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not Dandelion class.
    """
    if isinstance(vdj, Dandelion):
        if vdj.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj.graph[0]
            cd = pd.DataFrame.from_dict(G.degree(weight=weight))
            cd.set_index(0, inplace=True)
            vdj._metadata["clone_degree"] = pd.Series(cd[1])
    else:
        raise TypeError("Input object must be of {}".format(Dandelion))


def clone_centrality(vdj: Dandelion):
    """
    Calculate node closeness centrality in BCR/TCR network.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object after `tl.generate_network` has been run.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not Dandelion class.
    """
    if isinstance(vdj, Dandelion):
        if vdj.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj.graph[0]
            cc = nx.closeness_centrality(G)
            cc = pd.DataFrame.from_dict(
                cc, orient="index", columns=["clone_centrality"]
            )
            vdj._metadata["clone_centrality"] = pd.Series(
                cc["clone_centrality"]
            )
    else:
        raise TypeError("Input object must be of {}".format(Dandelion))


def calculate_distance_matrix_original(
    dat_seq: pl.DataFrame,
    membership: dict,
    metric: Metric,
    pad_to_max: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Re-implementation of original membership-based distance calculation.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame with sequence columns and 'cell_id' column.
    membership : dict
        Mapping from clone_id -> list of cell_ids (these cell_ids must be present in dat_seq['cell_id']).
    metric : Metric
        Distance metric to use.
    pad_to_max : bool, optional
        Whether to pad sequences to maximum length before distance calculation.
    verbose : bool, optional
        Whether to show progress.

    Returns
    -------
    total_dist : np.ndarray
        Aggregated distance matrix across all columns; diagonal set to NaN by caller.
    """
    # Ensure dat_seq is a DataFrame (not LazyFrame)
    if isinstance(dat_seq, pl.LazyFrame):
        dat_seq = dat_seq.collect(engine="streaming")

    n = dat_seq.height
    cell_id_list = dat_seq["cell_id"].to_list()
    cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_id_list)}

    dmat_per_column = defaultdict(list)

    for clone in tqdm(
        membership,
        disable=not verbose,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        clone_cell_ids = membership[clone]
        if len(clone_cell_ids) > 1:
            tmp = dat_seq.filter(pl.col("cell_id").is_in(clone_cell_ids))
            tmp_cell_ids = tmp["cell_id"].to_list()

            seq_cols = [
                col for col in tmp.collect_schema().names() if col != "cell_id"
            ]

            for col in seq_cols:
                seq_series = (
                    tmp[col]
                    .cast(pl.String)
                    .str.replace_all(r"\.", "")
                    .fill_null("")
                    .str.replace_all("None", "")
                )
                seqs = seq_series.to_numpy()
                tdarray = seqs.reshape(-1, 1)

                d_mat_tmp = squareform(
                    pdist(
                        tdarray,
                        lambda x, y: (
                            dist_func_long_sep(
                                x[0],
                                y[0],
                                metric=metric,
                                pad_to_max=pad_to_max,
                                sep="" if not pad_to_max else "#",
                            )
                            if (x[0] is not None and y[0] is not None)
                            else 0
                        ),
                    )
                )

                df_block = pd.DataFrame(
                    d_mat_tmp, index=tmp_cell_ids, columns=tmp_cell_ids
                )
                dmat_per_column[col].append(df_block)

    dist_matrices = []
    for col, blocks in dmat_per_column.items():
        if not blocks:
            continue
        full = pd.concat(blocks)

        if any(full.index.duplicated()):
            dup_indices = full.index[full.index.duplicated()]
            tmp1 = full.drop(dup_indices)
            tmp2 = full.loc[dup_indices]
            tmp2 = tmp2.groupby(level=0).apply(lambda df: df.sum(axis=0))
            full = pd.concat([tmp1, tmp2])

        full = full.reindex(index=cell_id_list, columns=cell_id_list).fillna(
            0.0
        )
        dist_matrices.append(full.values)

    if len(dist_matrices) == 0:
        total_dist = np.zeros((n, n))
    else:
        total_dist = np.sum(dist_matrices, axis=0)

    np.fill_diagonal(total_dist, np.nan)
    return total_dist


def calculate_distance_matrix_original_full(
    dat_seq: pl.DataFrame,
    metric: Metric,
    pad_to_max: bool = False,
    n_cpus: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Re-implementation of original membership-based distance calculation.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame with sequence columns and 'cell_id' column.
    metric : Metric
        Distance metric to use.
    pad_to_max : bool, optional
        Whether to pad sequences to maximum length before distance calculation.
    n_cpus : int, optional
        Number of cores to run this step. Parallelise using `sklearn.metrics.pairwise_distances` if n_cpus > 1.
    verbose : bool, optional
        Whether to show progress.

    Returns
    -------
    total_dist : np.ndarray
        Aggregated distance matrix across all columns; diagonal set to NaN by caller.
    """
    start_time = time.time()
    n = dat_seq.height
    total_dist = np.zeros((n, n), dtype=float)

    seq_cols = [
        col for col in dat_seq.collect_schema().names() if col != "cell_id"
    ]
    for col in seq_cols:
        seq_series = (
            dat_seq[col]
            .cast(pl.String)
            .str.replace_all(r"\.", "")
            .fill_null("")
            .str.replace_all("None", "")
        )

        # Check if we have any non-empty sequences (matching pandas logic)
        nonnull = seq_series.drop_nulls()
        if nonnull.len() <= 1:
            continue

        # Prepare sequences for single column (reshape to list of single-element lists)
        seqs_raw = [[s] for s in seq_series.to_numpy()]
        prepared_seqs = prepare_sequences_with_separator(
            seqs_raw,
            metric=metric,
            pad_to_max=pad_to_max,
            sep="" if not pad_to_max else "#",
        )

        # Compute distance matrix using vectorized metric
        results = metric.compute_vectorized(prepared_seqs)
        total_dist += results

    np.fill_diagonal(total_dist, np.nan)
    if verbose:
        end_time = time.time()
        logg.info(
            f"Distances calculated in {end_time - start_time:.2f} seconds"
        )
    return total_dist


def calculate_distance_matrix_long(
    dat_seq: pl.DataFrame,
    membership: dict | None,
    metric: Metric,
    pad_to_max: bool = False,
    n_cpus: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Re-implementation of original membership-based distance calculation but using concatenated sequences
    using a long separator.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame with sequence columns and 'cell_id' column.
    membership : dict | None
        Mapping from clone_id -> list of cell_ids (these cell_ids must be present in dat_seq['cell_id']).
        None indicates full pairwise distance calculation.
    metric : Metric
        Distance metric to use.
    pad_to_max : bool, optional
        whether or not to pad sequences to the maximum length in the dataset before distance calculation. This will
        allow for distance calculations that need sequences of the same length (e.g., Hamming distance). Note that this
        may increase memory usage and computation time.
    n_cpus : int, optional
        Number of cores to run this step. Parallelise using `sklearn.metrics.pairwise_distances` if n_cpus > 1..
    verbose : bool, optional
        Whether to show progress.

    Returns
    -------
    total_dist : np.ndarray (n x n)
        Aggregated distance matrix across all columns; diagonal set to NaN by caller.
    """
    start_time = time.time()

    # Step 1: clean sequences
    # Ensure dat_seq is a DataFrame (not LazyFrame)
    if isinstance(dat_seq, pl.LazyFrame):
        dat_seq = dat_seq.collect(engine="streaming")

    seq_cols = [
        col for col in dat_seq.collect_schema().names() if col != "cell_id"
    ]

    dat_seq_clean = dat_seq.select(
        [
            pl.col("cell_id"),
            *[
                pl.col(col)
                .cast(pl.String)
                .str.replace_all(r"\.", "")
                .fill_null("")
                .str.replace_all("None", "")
                .alias(col)
                for col in seq_cols
            ],
        ]
    )

    # Step 2: prepare sequences (concatenate with separators, apply padding)
    # This happens ONCE upfront, not per-pair
    seqs_raw = dat_seq_clean.select(seq_cols).to_numpy().tolist()
    prepared_seqs = prepare_sequences_with_separator(
        seqs_raw,
        metric=metric,
        pad_to_max=pad_to_max,
        sep="#",
    )

    # Step 3: initialize distance matrix
    n = dat_seq_clean.height
    cell_id_list = dat_seq_clean["cell_id"].to_list()
    cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_id_list)}
    total_dist = np.zeros((n, n))

    if membership is None:
        # Step 4: compute full distance matrix at once using vectorized metric
        results = metric.compute_vectorized(prepared_seqs)
        total_dist += results
    else:
        # Step 4: iterate over clone memberships
        for clone in tqdm(
            membership,
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            clone_cell_ids = membership[clone]
            if len(clone_cell_ids) > 1:
                tmp = dat_seq_clean.filter(
                    pl.col("cell_id").is_in(clone_cell_ids)
                )
                tmp_cell_ids = tmp["cell_id"].to_list()

                # Map cell_ids to indices
                indices = [cell_id_to_idx[cid] for cid in tmp_cell_ids]

                # Extract prepared sequences for this clone
                clone_seqs = [prepared_seqs[i] for i in indices]

                # Compute distance matrix using vectorized metric
                d_mat_tmp = metric.compute_vectorized(clone_seqs)

                total_dist[np.ix_(indices, indices)] += d_mat_tmp

    np.fill_diagonal(total_dist, np.nan)
    if verbose:
        end_time = time.time()
        logg.info(
            f"Distances calculated in {end_time - start_time:.2f} seconds"
        )
    return total_dist
