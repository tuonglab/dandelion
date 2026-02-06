from __future__ import annotations

import functools
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
from scipy.sparse.csgraph import (
    connected_components as scipy_cc,
    minimum_spanning_tree as scipy_mst,
)

from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

from dandelion.polars.core._core_polars import DandelionPolars
from dandelion.polars.tools._tools_polars import vdj_sample

from dandelion.tools._layout import generate_layout
from dandelion.utilities._distances import (
    Metric,
    prepare_sequences_with_separator,
    resolve_metric,
)
from dandelion.utilities._utilities import (
    Tree,
    FALSES,
)


def _merge_overlapping_clones(
    membership: pl.DataFrame, clone_col: str
) -> pl.DataFrame:
    """
    Merge overlapping clones into single membership groups.

    Uses scipy connected_components on a bipartite cell-clone graph for O(n)
    performance instead of iterative DataFrame joins.

    For cells with multiple clones ("|"-separated), this finds all cells that share
    any clone and assigns them to the same group.

    Parameters
    ----------
    membership : pl.DataFrame
        DataFrame with 'cell_id' and clone column ("|"-separated values)
    clone_col : str
        Name of the clone column

    Returns
    -------
    pl.DataFrame
        DataFrame with 'cell_id' and 'membership_id' where overlapping clones are merged
    """
    # Explode "|"-separated clones to get one row per cell-clone pair
    exploded = (
        membership.with_columns(
            pl.col(clone_col).str.split("|").alias("_clones")
        )
        .explode("_clones")
        .with_columns(pl.col("_clones").str.strip_chars().alias("_clone"))
        .filter(
            pl.col("_clone").is_not_null()
            & (pl.col("_clone") != "")
            & (pl.col("_clone").str.to_lowercase() != "none")
        )
        .select(["cell_id", "_clone"])
    )

    if exploded.height == 0:
        return pl.DataFrame(
            {
                "cell_id": membership["cell_id"],
                "membership_id": [None] * membership.height,
            }
        )

    # Encode cells and clones as integer indices
    cell_ids = exploded["cell_id"].unique().sort()
    clone_ids = exploded["_clone"].unique().sort()
    n_cells = cell_ids.len()
    n_clones = clone_ids.len()

    # Build lookup dicts: cell/clone string → integer index
    cell_to_idx = dict(zip(cell_ids.to_list(), range(n_cells)))
    clone_to_idx = dict(zip(clone_ids.to_list(), range(n_clones)))

    # Map exploded pairs to integer indices
    cell_indices = np.array(
        [cell_to_idx[c] for c in exploded["cell_id"].to_list()]
    )
    clone_indices = np.array(
        [clone_to_idx[c] + n_cells for c in exploded["_clone"].to_list()]
    )

    # Build bipartite adjacency: cells [0, n_cells) ↔ clones [n_cells, n_cells+n_clones)
    n_total = n_cells + n_clones
    ones = np.ones(len(cell_indices), dtype=np.float32)
    adj = csr_matrix(
        (ones, (cell_indices, clone_indices)), shape=(n_total, n_total)
    )
    adj = adj + adj.T  # make symmetric

    # Single-pass connected components
    _, labels = scipy_cc(adj, directed=False)

    # Extract labels for cell nodes only (first n_cells entries)
    cell_labels = labels[:n_cells]
    cell_id_list = cell_ids.to_list()

    cell_groups = pl.DataFrame(
        {
            "cell_id": cell_id_list,
            "membership_id": [str(lbl) for lbl in cell_labels],
        }
    )

    # Join back to original membership order
    result = membership.select("cell_id").join(
        cell_groups, on="cell_id", how="left"
    )

    return result


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
    layout_method: Literal[
        "mod_fr",
        "mod_fr2",
        "mod_fr2_gpu",
        "mod_fr_bh",
        "mod_fr_bh_gpu",
        "sfdp",
        "fa2",
    ] = "mod_fr2",
    singleton_mass: float = 0.5,
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
    layout_method : Literal["mod_fr", "mod_fr2", "mod_fr2_gpu", "mod_fr_bh", "mod_fr_bh_gpu", "sfdp", "fa2"], optional
        Layout algorithm. Options:
        - 'mod_fr': Original python modified FR layout
        - 'mod_fr2': Numba-accelerated modified FR (faster CPU)
        - 'mod_fr2_gpu': PyTorch GPU modified FR (auto-tiles for >100K nodes)
        - 'mod_fr_bh': Barnes-Hut O(N log N) CPU layout (scalable for large graphs)
        - 'mod_fr_bh_gpu': Barnes-Hut O(N log N) GPU layout (scalable for large graphs, requires CUDA)
        - 'sfdp': graph_tool sfdp_layout (requires graph-tool)
        - 'fa2': ForceAtlas2 (requires fa2-modified)
    singleton_mass : float, optional
        Mass assigned to singleton nodes (no edges) in Barnes-Hut layouts.
        Lower values reduce their impact on pushing connected components apart.
        Default 0.5. Only used with 'mod_fr_bh' and 'mod_fr_bh_gpu'.
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
    memory_limit_gb: float | None, optional
        Memory limit per worker in GB for Dask. None defaults to all available memory/cores.
    memory_safety_fraction: float, optional
        Fraction of available memory to use. Defaults to 0.3 (i.e., 30% of available memory will be used for chunk size calculation).
    compress: bool, optional
        Whether to compress the Zarr array using Blosc with zstd.
    random_state : int | np.random.RandomState | None, optional
        Random state for reproducible sampling.
    **kwargs
        additional kwargs passed to options specified in `networkx.drawing.layout.spring_layout` or
        `graph_tool.draw.sfdp_layout`.

    Returns
    -------
    DandelionPolars | tuple[DandelionPolars, AnnData]
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
            if isinstance(vdj, DandelionPolars):
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
                    singleton_mass=singleton_mass,
                    **kwargs,
                )

    if regenerate:
        start = logg.info("Generating network")

        key_ = key if key is not None else "sequence_alignment_aa"

        if key_ not in vdj._data:
            raise ValueError(f"key {key_} not found in data.")

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

        # Align dat_seq to vdj._metadata order to ensure indices match
        # pre-computed distances from find_clones (which are indexed by metadata position)
        meta_cell_ids = vdj._metadata.select("cell_id")
        if isinstance(meta_cell_ids, pl.LazyFrame):
            meta_cell_ids = meta_cell_ids.collect(engine="streaming")

        # Left join to preserve metadata order, missing cells get null sequences
        dat_seq = meta_cell_ids.join(dat_seq, on="cell_id", how="left")

        # Build a position lookup table (cell_id → row index)
        # Positions match metadata indices
        pos_map = (
            dat_seq.with_row_index("_row_pos")
            if isinstance(dat_seq, pl.DataFrame)
            else dat_seq.lazy()
            .with_row_index("_row_pos")
            .collect(engine="streaming")
        ).select(["cell_id", pl.col("_row_pos").cast(pl.Int64).alias("pos")])

        if compute_graph or compute_layout or distance_mode == "clone":
            # Get clone membership as DataFrame and merge overlapping clones
            clone_df = dat._merge(clone_key, unique=True)
            membership = _merge_overlapping_clones(clone_df, clone_key)

        # Check if pre-computed distances are available
        if hasattr(vdj, "distances") and vdj.distances is not None:
            logg.info("Using pre-computed distances from .distances\n")
            total_dist = vdj.distances
            if isinstance(total_dist, np.ndarray):
                total_dist = csr_matrix(total_dist)
        else:
            # compute total_dist using chosen mode (original uses membership)
            logg.info(
                f"Calculating distance matrix {'lazily ' if lazy else ' '}with distance_mode = '{distance_mode}'\n"
            )
            if distance_mode == "clone":
                if lazy:
                    from dandelion.polars.tools._lazydistances_polars import (
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
                        zarr_path=(
                            zarr_tmp if zarr_path is None else zarr_path
                        ),
                        chunk_size=chunk_size,
                        n_cpus=n_cpus,
                        memory_limit_gb=memory_limit_gb,
                        memory_safety_fraction=memory_safety_fraction,
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
                    from dandelion.polars.tools._lazydistances_polars import (
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
                        zarr_path=(
                            zarr_tmp if zarr_path is None else zarr_path
                        ),
                        chunk_size=chunk_size,
                        n_cpus=n_cpus,
                        memory_limit_gb=memory_limit_gb,
                        memory_safety_fraction=memory_safety_fraction,
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
            # Normalize metadata to Polars DataFrame if lazy
            if isinstance(vdj._metadata, pl.LazyFrame):
                meta_df = vdj._metadata.collect(engine="streaming")
            elif isinstance(vdj._metadata, pl.DataFrame):
                meta_df = vdj._metadata
            else:
                meta_df = pl.from_pandas(vdj._metadata)

            # Vectorized split and overlap detection
            meta_df_split = meta_df.with_columns(
                pl.col(str(clone_key)).str.split("|").alias("_clone_list")
            ).with_columns(
                pl.col("_clone_list").list.len().gt(1).alias("_is_overlap")
            )

            # Explode and add positions via join (avoids slow Python dict lookup)
            meta_exploded = (
                meta_df_split.select(["cell_id", "_clone_list"])
                .explode("_clone_list")
                .filter(pl.col("_clone_list") != "None")
                .rename({"_clone_list": "clone_id"})
                .join(pos_map, on="cell_id", how="left")
            )

            # Get unique overlap groups with their cells
            overlap_groups = (
                meta_df_split.filter(pl.col("_is_overlap"))
                .select(["cell_id", "_clone_list"])
                .with_columns(
                    pl.col("_clone_list")
                    .list.eval(pl.element().filter(pl.element() != "None"))
                    .alias("_overlap_group")
                )
                .group_by("_overlap_group", maintain_order=True)
                .agg(pl.col("cell_id").alias("cells_in_group"))
            )

            # For each overlap group, get the positions by joining with meta_exploded
            # Use partial to bind meta_exploded to the helper function
            get_positions_fn = functools.partial(
                _get_positions_for_group, meta_exploded=meta_exploded
            )

            # Add positions
            overlap_groups = overlap_groups.with_columns(
                [
                    pl.col("cells_in_group")
                    .map_elements(
                        get_positions_fn, return_dtype=pl.List(pl.Int64)
                    )
                    .alias("_overlap_positions"),
                    pl.col("_overlap_group").alias("_full_group"),
                ]
            )

            # Explode to get one row per clone_id
            _overlap_info = (
                overlap_groups.explode("_overlap_group")
                .rename({"_overlap_group": "clone_id"})
                .group_by("clone_id")
                .agg(
                    [
                        pl.col("_full_group").alias("_overlap_keys"),
                        pl.col("_overlap_positions").alias(
                            "_overlap_positions_list"
                        ),
                    ]
                )
            )

            # Join onto meta_exploded
            meta_exploded = meta_exploded.join(
                _overlap_info, on="clone_id", how="left"
            )

            # ===================================================================
            # CREATE TWO SEPARATE GROUPINGS: MST vs ZERO-DISTANCE
            # ===================================================================

            # 1. MST GROUPS: For overlaps, use one representative clone's cells (last one)
            meta_for_mst = meta_exploded.with_columns(
                pl.when(pl.col("_overlap_keys").is_not_null())
                .then(pl.col("_overlap_keys").list.first().list.join("|"))
                .otherwise(pl.col("clone_id"))
                .alias("mst_key")
            )

            mst_groups = (
                meta_for_mst.group_by(
                    ["mst_key", "clone_id"]
                )  # Keep clones separate initially
                .agg(
                    [
                        pl.col("cell_id").alias("cell_ids"),
                        pl.col("pos").alias("positions"),
                    ]
                )
                .filter(pl.col("positions").list.len() >= 2)
            )

            # For each mst_key, take last clone (mimics overwrite behavior)
            mst_groups = (
                mst_groups.group_by("mst_key")
                .agg(
                    [
                        pl.col("cell_ids").last().alias("cell_ids"),
                        pl.col("positions").last().alias("positions"),
                    ]
                )
                .rename({"mst_key": "group_key"})
            )

            # 2. ZERO-DISTANCE GROUPS: For overlaps, use UNION of all cells
            meta_for_zero = meta_exploded.filter(
                pl.col("_overlap_keys").is_not_null()
            )
            meta_no_overlap_zero = meta_exploded.filter(
                pl.col("_overlap_keys").is_null()
            )

            if len(meta_for_zero) > 0:
                zero_overlap_exploded = (
                    meta_for_zero.explode("_overlap_keys")
                    .with_columns(
                        pl.col("_overlap_keys").list.join("|").alias("zero_key")
                    )
                    .select(["cell_id", "pos", "zero_key"])
                )

                zero_overlap_groups = (
                    zero_overlap_exploded.group_by("zero_key")
                    .agg(
                        [
                            pl.col("cell_id").unique().alias("cell_ids"),
                            pl.col("pos").unique().alias("positions"),
                        ]
                    )
                    .filter(pl.col("positions").list.len() >= 2)
                    .rename({"zero_key": "group_key"})
                )
            else:
                zero_overlap_groups = pl.DataFrame(
                    schema={
                        "group_key": pl.String,
                        "cell_ids": pl.List(pl.String),
                        "positions": pl.List(pl.Int64),
                    }
                )

            if len(meta_no_overlap_zero) > 0:
                zero_no_overlap_groups = (
                    meta_no_overlap_zero.group_by("clone_id")
                    .agg(
                        [
                            pl.col("cell_id").alias("cell_ids"),
                            pl.col("pos").alias("positions"),
                        ]
                    )
                    .filter(pl.col("positions").list.len() >= 2)
                    .rename({"clone_id": "group_key"})
                )
            else:
                zero_no_overlap_groups = pl.DataFrame(
                    schema={
                        "group_key": pl.String,
                        "cell_ids": pl.List(pl.String),
                        "positions": pl.List(pl.Int64),
                    }
                )

            zero_groups = pl.concat(
                [zero_overlap_groups, zero_no_overlap_groups]
            )

            # ===================================================================
            # MST COMPUTATION
            # ===================================================================
            # Use partial to bind total_dist and lazy to the helper function
            create_mst_fn = functools.partial(
                _create_mst_edges, total_dist=total_dist, lazy=lazy
            )

            mst_groups = mst_groups.with_columns(
                pl.struct(["positions", "cell_ids"])
                .map_elements(
                    lambda x: create_mst_fn(
                        positions=x["positions"],
                        cell_ids=x["cell_ids"],
                    ),
                    return_dtype=pl.Object,
                )
                .alias("mst_edges")
            ).filter(pl.col("mst_edges").is_not_null())

            # ===================================================================
            # ZERO-DISTANCE EDGES
            # ===================================================================
            # Use partial to bind total_dist and lazy to the helper function
            find_zero_fn = functools.partial(
                _find_zero_dist_edges, total_dist=total_dist, lazy=lazy
            )

            zero_groups = zero_groups.with_columns(
                pl.struct(["positions", "cell_ids"])
                .map_elements(
                    lambda x: find_zero_fn(
                        positions=x["positions"],
                        cell_ids=x["cell_ids"],
                    ),
                    return_dtype=pl.Object,
                )
                .alias("zero_edges")
            )

            # ===================================================================
            # COLLECT EDGES BY KEY
            # ===================================================================
            mst_edge_dict = dict(
                zip(
                    mst_groups["group_key"].to_list(),
                    mst_groups["mst_edges"].to_list(),
                )
            )

            zero_edge_dict = {}
            for row in zero_groups.iter_rows(named=True):
                if row["zero_edges"] is not None:
                    zero_edge_dict[row["group_key"]] = row["zero_edges"]

            # ===================================================================
            # MERGE MST AND ZERO-DISTANCE EDGES
            # ===================================================================
            try:
                # Concat all MST edges
                if mst_edge_dict:
                    edge_listx = _add_sorted_index(
                        pd.concat(
                            list(mst_edge_dict.values()), ignore_index=True
                        )
                    )
                else:
                    edge_listx = pd.DataFrame(
                        columns=["source", "target", "weight"]
                    )

                # Concat all zero-distance edges
                if zero_edge_dict:
                    tmp_edge_listx = _add_sorted_index(
                        pd.concat(
                            list(zero_edge_dict.values()), ignore_index=True
                        )
                    )
                else:
                    tmp_edge_listx = pd.DataFrame(
                        columns=["source", "target", "weight"]
                    )

                # Combine: MST edges take priority
                edge_list_final = edge_listx.combine_first(tmp_edge_listx)

                # Vectorized weight lookup from total_dist
                if len(edge_list_final) > 0:
                    sources = edge_list_final["source"].tolist()
                    targets = edge_list_final["target"].tolist()

                    # Build reverse lookup: cell_id -> position
                    cell_to_pos = (
                        meta_exploded.select(["cell_id", "pos"])
                        .unique()
                        .to_dict(as_series=False)
                    )
                    cell_to_pos = dict(
                        zip(cell_to_pos["cell_id"], cell_to_pos["pos"])
                    )

                    src_pos = np.array([cell_to_pos[s] for s in sources])
                    tgt_pos = np.array([cell_to_pos[t] for t in targets])

                    if lazy:
                        # Dask: extract pairs individually
                        actual_weights = da.stack(
                            [
                                total_dist[src_pos[i], tgt_pos[i]]
                                for i in range(len(src_pos))
                            ]
                        ).compute()
                    else:
                        actual_weights = total_dist[src_pos, tgt_pos]
                        # CSR indexing returns np.matrix; flatten to 1-D array
                        if hasattr(actual_weights, "A1"):
                            actual_weights = actual_weights.A1

                    edge_list_final["weight"] = actual_weights
                    edge_list_final = edge_list_final.reset_index(drop=True)

            except Exception:
                edge_list_final = None

            # final layout + graph creation (unchanged)
            g, g_, lyt, lyt_ = generate_layout(
                vertices=meta_df["cell_id"].to_list(),
                edges=edge_list_final,
                min_size=min_size,
                weight=None,
                verbose=verbose,
                compute_layout=compute_layout,
                layout_method=layout_method,
                expanded_only=expanded_only,
                singleton_mass=singleton_mass,
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
        if lazy:
            distances = total_dist
        elif isinstance(total_dist, csr_matrix):
            distances = total_dist
        else:
            distances = csr_matrix(total_dist)
        if not lazy:
            distances._index_names = vdj.metadata_names
    else:
        distances = vdj.distances
    graph = (g, g_) if compute_graph else None
    layout = (lyt, lyt_) if compute_graph and compute_layout else None
    if sample is not None:
        out = DandelionPolars(
            data=vdj._data.collect(),
            metadata=vdj._metadata.collect(),
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
        vdj._reinitialize_attributes(
            data=vdj._data.collect(),
            metadata=vdj._metadata.collect(),
            clone_key=clone_key,
            layout=layout,
            graph=graph,
            distances=distances,
            germline=germline,
            reinitialize=False,
        )


def _get_positions_for_group(
    cells_list: list, meta_exploded: pl.DataFrame
) -> list[int]:
    """
    Get positions for a group of cells by joining with metadata.

    Parameters
    ----------
    cells_list : list
        List of cell IDs
    meta_exploded : pl.DataFrame
        Exploded metadata DataFrame with cell_id and pos columns

    Returns
    -------
    list[int]
        List of unique positions for the cells
    """
    cells_df = pl.DataFrame({"cell_id": cells_list})
    positions = (
        cells_df.join(
            meta_exploded.select(["cell_id", "pos"]),
            on="cell_id",
            how="inner",
        )
        .select("pos")
        .unique()
        .to_series()
        .to_list()
    )
    return positions


def _create_mst_edges(
    total_dist: np.ndarray,
    positions: list[int],
    cell_ids: list[str],
    lazy: bool = False,
) -> pd.DataFrame | None:
    """
    Create minimum spanning tree edges for a group.

    Parameters
    ----------
    total_dist : np.ndarray
        Distance matrix
    positions : list[int]
        Positions to slice from the distance matrix
    cell_ids : list[str]
        Cell IDs corresponding to positions
    lazy : bool
        Whether using lazy/Dask arrays

    Returns
    -------
    pd.DataFrame | None
        DataFrame with source, target, weight columns, or None if insufficient data
    """
    if len(positions) < 2:
        return None

    if lazy:
        from dandelion.polars.tools._lazydistances_polars import (
            dask_safe_slice_square,
        )

        submat = dask_safe_slice_square(total_dist, positions).compute()
    else:
        submat = total_dist[np.ix_(positions, positions)]
        if hasattr(submat, "toarray"):
            submat = submat.toarray()

    if submat.shape[0] < 2 or submat.shape[1] < 2:
        return None

    # Compute MST directly using scipy
    values = submat.astype(float)
    shifted = values + 1.0
    np.fill_diagonal(shifted, 0.0)

    mst_sparse = scipy_mst(shifted)
    coo = mst_sparse.tocoo()

    if coo.nnz == 0:
        return None

    # Undo the +1 shift, clamp at 0
    weights = np.maximum(coo.data - 1.0, 0.0)

    return pd.DataFrame(
        {
            "source": [cell_ids[i] for i in coo.row],
            "target": [cell_ids[j] for j in coo.col],
            "weight": weights,
        }
    )


def _find_zero_dist_edges(
    total_dist: np.ndarray,
    positions: list[int],
    cell_ids: list[str],
    lazy: bool = False,
) -> pd.DataFrame | None:
    """
    Find edges with zero distance between cells.

    Parameters
    ----------
    total_dist : np.ndarray
        Distance matrix
    positions : list[int]
        Positions to slice from the distance matrix
    cell_ids : list[str]
        Cell IDs corresponding to positions
    lazy : bool
        Whether using lazy/Dask arrays

    Returns
    -------
    pd.DataFrame | None
        DataFrame with source, target, weight columns (weight=0), or None if no zero distances
    """
    if len(positions) < 2:
        return None

    # Slice the distance matrix
    if lazy:
        from dandelion.polars.tools._lazydistances_polars import (
            dask_safe_slice_square,
        )

        submat = dask_safe_slice_square(total_dist, positions).compute()
    else:
        submat = total_dist[np.ix_(positions, positions)]
        if hasattr(submat, "toarray"):
            submat = submat.toarray()

    # Find all pairs in lower triangle with distance == 0
    n = len(cell_ids)
    row_idx, col_idx = np.tril_indices(n, k=-1)
    mask = submat[row_idx, col_idx] == 0

    if not mask.any():
        return None

    return pd.DataFrame(
        {
            "source": [cell_ids[i] for i in row_idx[mask]],
            "target": [cell_ids[j] for j in col_idx[mask]],
            "weight": 0.0,
        }
    )


def _add_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sorted pair index to edge DataFrame.

    Creates an index from sorted (source, target) pairs in format "a|b".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with source and target columns

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with sorted pair index
    """
    pairs = np.sort(df[["source", "target"]].values, axis=1)
    df = df.copy()
    df.index = [f"{a}|{b}" for a, b in pairs]
    return df


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


def clone_degree(
    vdj: DandelionPolars, weight: str | None = None
) -> DandelionPolars:
    """
    Calculate node degree in BCR/TCR network.

    Parameters
    ----------
    vdj : DandelionPolars
        DandelionPolars object after `tl.generate_network` has been run.
    weight : str | None, optional
        Attribute name for retrieving edge weight in graph. None defaults to ignoring this. See `networkx.Graph.degree`.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not DandelionPolars class.
    """
    if isinstance(vdj, DandelionPolars):
        if vdj.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj.graph[0]
            degree_dict = dict(G.degree(weight=weight))
            df = pl.DataFrame(
                {
                    "cell_id": list(degree_dict.keys()),
                    "clone_degree": list(degree_dict.values()),
                }
            )
            # now merge into vdj._metadata
            vdj._metadata = (
                vdj._metadata.lazy()
                .with_row_index("_orig_idx")
                .join(df.lazy(), on="cell_id", how="left")
                .sort("_orig_idx")
                .drop("_orig_idx")
                .collect(engine="streaming")
            )
    else:
        raise TypeError("Input object must be of {}".format(DandelionPolars))


def clone_centrality(vdj: DandelionPolars):
    """
    Calculate node closeness centrality in BCR/TCR network.

    Parameters
    ----------
    vdj : DandelionPolars
        DandelionPolars object after `tl.generate_network` has been run.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not DandelionPolars class.
    """
    if isinstance(vdj, DandelionPolars):
        if vdj.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj.graph[0]
            cc = nx.closeness_centrality(G)
            df = pl.DataFrame(
                {
                    "cell_id": list(cc.keys()),
                    "clone_centrality": list(cc.values()),
                }
            )

            vdj._metadata = (
                vdj._metadata.lazy()
                .with_row_index("_orig_idx")
                .join(df.lazy(), on="cell_id", how="left")
                .sort("_orig_idx")
                .drop("_orig_idx")
                .collect(engine="streaming")
            )
    else:
        raise TypeError("Input object must be of {}".format(DandelionPolars))


def calculate_distance_matrix_original(
    dat_seq: pl.DataFrame,
    membership: pl.DataFrame,
    metric: Metric,
    pad_to_max: bool = False,
    verbose: bool = True,
) -> csr_matrix:
    """
    Re-implementation of original membership-based distance calculation.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame with sequence columns and 'cell_id' column.
    membership : pl.DataFrame
        DataFrame with 'cell_id' and 'membership_id' columns mapping cells to clone groups.
    metric : Metric
        Distance metric to use.
    pad_to_max : bool, optional
        Whether to pad sequences to maximum length before distance calculation.
    verbose : bool, optional
        Whether to show progress.

    Returns
    -------
    total_dist : csr_matrix
        Sparse distance matrix; diagonal is 0 (self-distance).
    """
    # Ensure dat_seq is a DataFrame (not LazyFrame)
    if isinstance(dat_seq, pl.LazyFrame):
        dat_seq = dat_seq.collect(engine="streaming")

    n = dat_seq.height
    cell_id_list = dat_seq["cell_id"].to_list()

    dmat_per_column = defaultdict(list)

    # Join with membership and partition by membership_id
    dat_seq_with_membership = dat_seq.join(
        membership, on="cell_id", how="inner"
    )
    groups = dat_seq_with_membership.partition_by("membership_id", as_dict=True)

    for group_df in tqdm(
        groups.values(),
        disable=not verbose,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        if group_df.height > 1:
            tmp_cell_ids = group_df["cell_id"].to_list()

            seq_cols = [
                col
                for col in group_df.collect_schema().names()
                if col not in ("cell_id", "membership_id")
            ]

            for col in seq_cols:
                seq_series = (
                    group_df[col]
                    .cast(pl.String)
                    .str.replace_all(r"\.", "")
                    .fill_null("")
                    .str.replace_all("None", "")
                )
                seqs = seq_series.to_list()
                seqs_raw = [[s] for s in seqs]
                prepared_seqs = prepare_sequences_with_separator(
                    seqs_raw,
                    metric=metric,
                    pad_to_max=pad_to_max,
                    sep="" if not pad_to_max else "#",
                )

                # Deduplicate sequences for faster computation
                unique_seqs = list(set(prepared_seqs))
                seq_to_unique_idx = {
                    seq: i for i, seq in enumerate(unique_seqs)
                }

                # Compute distances only for unique sequences
                d_mat_unique = metric.compute_vectorized(unique_seqs)

                # Use vectorized indexing to expand to full matrix
                unique_indices = np.array(
                    [seq_to_unique_idx[seq] for seq in prepared_seqs]
                )
                d_mat_tmp = d_mat_unique[np.ix_(unique_indices, unique_indices)]

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
        return csr_matrix((n, n))

    total_dist = np.sum(dist_matrices, axis=0)
    np.fill_diagonal(total_dist, 0.0)
    return csr_matrix(total_dist)


def calculate_distance_matrix_original_full(
    dat_seq: pl.DataFrame,
    metric: Metric,
    pad_to_max: bool = False,
    n_cpus: int = 1,
    verbose: bool = True,
) -> csr_matrix:
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
    total_dist : csr_matrix
        Sparse distance matrix; diagonal is 0 (self-distance).
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

        # Deduplicate sequences for faster computation
        unique_seqs = list(set(prepared_seqs))
        seq_to_unique_idx = {seq: i for i, seq in enumerate(unique_seqs)}

        # Compute distances only for unique sequences
        results_unique = metric.compute_vectorized(unique_seqs, n_cpus=n_cpus)

        # Use vectorized indexing to expand to full matrix
        unique_indices = np.array(
            [seq_to_unique_idx[seq] for seq in prepared_seqs]
        )
        results = results_unique[np.ix_(unique_indices, unique_indices)]

        total_dist += results

    np.fill_diagonal(total_dist, 0.0)
    if verbose:
        end_time = time.time()
        logg.info(
            f"Distances calculated in {end_time - start_time:.2f} seconds"
        )
    return csr_matrix(total_dist)


def calculate_distance_matrix_long(
    dat_seq: pl.DataFrame,
    membership: pl.DataFrame | None,
    metric: Metric,
    pad_to_max: bool = False,
    n_cpus: int = 1,
    verbose: bool = True,
) -> csr_matrix:
    """
    Re-implementation of original membership-based distance calculation but using concatenated sequences
    using a long separator.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame with sequence columns and 'cell_id' column.
    membership : pl.DataFrame | None
        DataFrame with 'cell_id' and 'membership_id' columns mapping cells to clone groups.
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
    total_dist : csr_matrix (n x n)
        Sparse distance matrix; diagonal is 0 (self-distance).
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

    # Step 3: initialize
    n = dat_seq_clean.height
    cell_id_list = dat_seq_clean["cell_id"].to_list()
    cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_id_list)}

    if membership is None:
        # Full mode: dense computation, convert to CSR at end
        results = metric.compute_vectorized(prepared_seqs, n_cpus=n_cpus)
        np.fill_diagonal(results, 0.0)
        total_dist = csr_matrix(results)
    else:
        # Clone mode: sparse COO accumulation — no dense N×N allocation
        all_rows: list[np.ndarray] = []
        all_cols: list[np.ndarray] = []
        all_vals: list[np.ndarray] = []

        dat_seq_with_membership = dat_seq_clean.join(
            membership, on="cell_id", how="inner"
        )
        groups = dat_seq_with_membership.partition_by(
            "membership_id", as_dict=True
        )

        for group_df in tqdm(
            groups.values(),
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            if group_df.height > 1:
                tmp_cell_ids = group_df["cell_id"].to_list()

                # Map cell_ids to indices
                indices = np.array(
                    [cell_id_to_idx[cid] for cid in tmp_cell_ids]
                )

                # Extract prepared sequences for this clone
                clone_seqs = [prepared_seqs[i] for i in indices]

                # Deduplicate sequences for faster computation
                unique_seqs = list(set(clone_seqs))
                seq_to_unique_idx = {
                    seq: i for i, seq in enumerate(unique_seqs)
                }

                # Compute distances only for unique sequences
                d_mat_unique = metric.compute_vectorized(
                    unique_seqs, n_cpus=n_cpus
                )

                # Use vectorized indexing to expand to full matrix
                unique_indices = np.array(
                    [seq_to_unique_idx[seq] for seq in clone_seqs]
                )
                d_mat_tmp = d_mat_unique[np.ix_(unique_indices, unique_indices)]

                # Collect COO entries (exclude diagonal — self-distance is 0)
                k = len(indices)
                row_global = np.repeat(indices, k)
                col_global = np.tile(indices, k)
                vals_flat = d_mat_tmp.ravel()
                off_diag = row_global != col_global

                all_rows.append(row_global[off_diag])
                all_cols.append(col_global[off_diag])
                all_vals.append(vals_flat[off_diag])

        if all_rows:
            total_dist = csr_matrix(
                coo_matrix(
                    (
                        np.concatenate(all_vals),
                        (
                            np.concatenate(all_rows),
                            np.concatenate(all_cols),
                        ),
                    ),
                    shape=(n, n),
                )
            )
        else:
            total_dist = csr_matrix((n, n))

    if verbose:
        end_time = time.time()
        logg.info(
            f"Distances calculated in {end_time - start_time:.2f} seconds"
        )
    return total_dist
