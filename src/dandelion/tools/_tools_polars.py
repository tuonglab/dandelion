from __future__ import annotations
import math
import re

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc

from anndata import AnnData
from collections import defaultdict, Counter
from distance import hamming
from itertools import product
from scanpy import logging as logg
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from tqdm import tqdm
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mudata import MuData
    from awkward import Array

from dandelion.utilities._polars import DandelionPolars, TRUES_STR
from dandelion.utilities._utilities import (
    FALSES,
    VCALL,
    JCALL,
    VCALLG,
    STRIPALLELENUM,
    EMPTIES,
    flatten,
    is_categorical,
    type_check,
    present,
    check_same_celltype,
    Tree,
)
from dandelion.utilities._distances import (
    IdentityMetric,
    resolve_metric,
)


def find_clones(
    vdj: DandelionPolars,
    identity: dict[str, float] | float = 0.85,
    hard_cutoff: int | float | None = None,
    key: dict[str, str] | str | None = None,
    dist_func: (
        Literal["hamming", "levenshtein", "identity"] | Callable | str
    ) = "hamming",
    same_vj: bool = True,
    same_length: bool = True,
    by_alleles: bool = False,
    key_added: str | None = None,
    recalculate_length: bool = True,
    verbose: bool = True,
) -> DandelionPolars:
    """
    Find clones based on VDJ chain and VJ chain CDR3 junction hamming distance.

    Parameters
    ----------
    vdj : DandelionPolars
        Dandelion object.
    identity : dict[str, float] | float, optional
        Similarity parameter. Default 0.85. Distance cutoff is calculated as
        `threshold = floor(length * (1 - identity))`. If `dist_func` is 'identity', `threshold` is set to 0.
        If `dist_func` is 'levenshtein' or a substitution matrix, the threshold is calculated based on normalized
        length internally. If a single float value is provided, this will be used for all loci.
        If provided as a dictionary, please use the following keys:'ig', 'tr-ab', 'tr-gd'.
    hard_cutoff : int | float | None, optional
        Absolute distance cutoff. If supplied, `identity` is ignored. Only for use with specific distance functions
        such as levenshtein and substitution matrices. Default is `None`.
    key : dict[str, str] | str | None, optional
        column name for performing clone clustering. `None` defaults to a dictionary where:
            {'ig': 'junction_aa', 'tr-ab': 'junction', 'tr-gd': 'junction'}
        If provided as a string, this key will be used for all loci.
    dist_func : Literal["hamming", "levenshtein", "identity"] | Callable | str, optional
        Distance function to use. Can be 'hamming', 'levenshtein', 'identity', substitution matrix name, or a custom lambda function.
        `None` defaults to 'hamming'.
    same_vj : bool, optional
        whether or not to require same V and J gene assignments to be in the same clone. Default is True.
    same_length : bool, optional
        whether or not to require same junction length to be in the same clone. Default is True.
    by_alleles : bool, optional
        whether or not to collapse alleles to genes. `None` defaults to False.
    key_added : str | None, optional
        If specified, this will be the column name for clones. `None` defaults to 'clone_id'
    recalculate_length : bool, optional
        whether or not to re-calculate junction length, rather than rely on parsed assignment (which occasionally is
        wrong). Default is True
    verbose : bool, optional
        whether or not to print progress.

    Returns
    -------
    Dandelion
        Dandelion object with clone_id annotated in `.data` slot and `.metadata` initialized.

    Raises
    ------
    ValueError
        if `key` not found in Dandelion.data.
    """
    start = logg.info("Finding clonotypes")
    df = vdj._data
    # Collect lazy frame if necessary, then convert to pandas
    if isinstance(df, pl.LazyFrame):
        # we will load this to memory to enable the rest.
        df = df.collect(engine="streaming")

    # Default locus dictionary
    locus_dict = {
        "ig": (["IGH"], ["IGK", "IGL"]),
        "tr-ab": (["TRB"], ["TRA"]),
        "tr-gd": (["TRD"], ["TRG"]),
    }
    # Locus logging dictionary
    locus_log = {"ig": "B", "tr-ab": "abT", "tr-gd": "gdT"}
    # Default identity
    default_identity = {"ig": 0.85, "tr-ab": 1.0, "tr-gd": 1.0}
    default_key = {
        "ig": "junction_aa",
        "tr-ab": "junction",
        "tr-gd": "junction",
    }
    metric = resolve_metric(dist_func)
    # Default identity
    if identity is None:
        identity = default_identity
    elif isinstance(identity, dict):
        default_identity.update(identity)
        identity = default_identity
    elif not isinstance(identity, dict):
        # Single float value - use for all loci
        identity = {"ig": identity, "tr-ab": identity, "tr-gd": identity}
    # Default key (junction column)
    if key is None:
        key = default_key
    elif isinstance(key, str):
        # Single string - use for all loci
        key = {"ig": key, "tr-ab": key, "tr-gd": key}
    # Default key_added
    key_added = "clone_id" if key_added is None else key_added
    # Initialize clone column
    df = df.with_columns(pl.lit("").alias(key_added))
    # Also initialise the original order column
    df = df.with_row_index("_original_order")

    # Store results from each locus
    locus_results = {}

    # Process each locus
    for locus, (vdj_loci, vj_loci) in locus_dict.items():
        # Filter to this locus
        df_locus = df.filter(pl.col("locus").is_in(vdj_loci + vj_loci))
        if "ambiguous" in df_locus.collect_schema():
            df_locus = df_locus.filter(~pl.col("ambiguous").is_in(TRUES_STR))

        # early skip if no rows
        if df_locus.height == 0:
            continue

        # Get locus-specific parameters
        locus_identity = identity[locus]
        locus_key = key[locus]
        locus_celltype = locus_log[locus]

        # Add celltype to this locus
        df_locus = df_locus.with_columns(
            pl.lit(locus_celltype).alias("_celltype")
        )

        # Check for VDJ and VJ chains
        has_vdj, has_vj = _check_chains(df_locus, vdj_loci, vj_loci)

        # Initialize results for this locus
        df_vdj_result = None
        df_vj_result = None

        vdj_chain, vj_chain = "VDJ", "VJ"

        # process VDJ chain
        if has_vdj:
            df_vdj = df_locus.filter(pl.col("locus").is_in(vdj_loci))

            # Group sequences
            df_vdj_grp = _group_sequences(
                df=df_vdj,
                key=locus_key,
                same_vj=same_vj,
                same_length=same_length,
                recalculate_length=recalculate_length,
                by_alleles=by_alleles,
            )
            # Build aggregation list dynamically
            agg_cols = [pl.col(locus_key)]
            if same_length:
                agg_cols.append(pl.col(f"_{locus_key}_length").first())
            else:
                agg_cols.append(pl.col(f"_{locus_key}_length").max())
            grouped = (
                df_vdj_grp.group_by("_membership")
                .agg(agg_cols)
                .sort("_membership")
            )
            clones_vdj = defaultdict(dict)
            for row in tqdm(
                grouped.iter_rows(named=True),
                desc=f"Finding clones based on {locus_celltype} cell {vdj_chain} chains using {locus_key}",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                total=df_vdj_grp.select("_membership").n_unique(),
                disable=not verbose,
            ):
                seqs = row[locus_key]
                membership = row["_membership"]
                length = row[f"_{locus_key}_length"]
                if isinstance(metric, IdentityMetric):
                    threshold = 0
                else:
                    if hard_cutoff is None:
                        threshold = math.floor(
                            int(length) * (1 - locus_identity)
                        )
                    else:
                        threshold = None
                d_mat = metric.compute_vectorized(seqs)

                if d_mat.shape[0] > 1:
                    seq_tmp_dict = _clustering_scipy(
                        d_mat,
                        threshold=threshold,
                        sequences=seqs,
                        hard_threshold=hard_cutoff,
                    )
                else:
                    seq_tmp_dict = {seqs[0]: (seqs[0],)}

                # Sort by size
                clones_tmp = sorted(
                    list(set(seq_tmp_dict.values())), key=len, reverse=True
                )
                for sub_group, clone_group in enumerate(clones_tmp, 1):
                    clones_vdj[membership][sub_group] = clone_group

            # Flatten to sequence -> clone mapping
            seq_to_clone = {}
            for membership, clone_dict in clones_vdj.items():
                for clone_id, sequences in clone_dict.items():
                    for seq in sequences:
                        seq_to_clone[seq] = f"{membership}_{clone_id}"

            # Apply to dataframe
            df_vdj_grp = df_vdj_grp.with_columns(
                pl.col(locus_key)
                .replace_strict(seq_to_clone, default=None)
                .alias(f"_{key_added}_{vdj_chain}")
            )

            # Keep only what we need
            df_vdj_result = df_vdj_grp.select(
                [
                    "_original_order",
                    f"_{key_added}_VDJ",
                ]
            )
            del df_vdj, df_vdj_grp

        # process VJ chains next
        if has_vj:
            df_vj = df_locus.filter(pl.col("locus").is_in(vj_loci))

            # Group sequences
            df_vj_grp = _group_sequences(
                df=df_vj,
                key=locus_key,
                same_vj=same_vj,
                same_length=same_length,
                recalculate_length=recalculate_length,
                by_alleles=by_alleles,
            )
            # Build aggregation list dynamically
            agg_cols = [pl.col(locus_key)]
            if same_length:
                agg_cols.append(pl.col(f"_{locus_key}_length").first())
            else:
                agg_cols.append(pl.col(f"_{locus_key}_length").max())

            grouped = (
                df_vj_grp.group_by("_membership")
                .agg(agg_cols)
                .sort("_membership")
            )

            clones_vj = defaultdict(dict)

            for row in tqdm(
                grouped.iter_rows(named=True),
                desc=f"Finding clones based on {locus_celltype} cell {vj_chain} chains using {locus_key}",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                total=grouped.height,
                disable=not verbose,
            ):
                seqs = row[locus_key]
                length = row[f"_{locus_key}_length"]
                membership = row["_membership"]
                if isinstance(metric, IdentityMetric):
                    threshold = 0
                else:
                    if hard_cutoff is None:
                        threshold = math.floor(
                            int(length) * (1 - locus_identity)
                        )
                    else:
                        threshold = None
                d_mat = metric.compute_vectorized(seqs)

                if d_mat.shape[0] > 1:
                    seq_tmp_dict = _clustering_scipy(
                        d_mat,
                        threshold=threshold,
                        sequences=seqs,
                        hard_threshold=hard_cutoff,
                    )
                else:
                    seq_tmp_dict = {seqs[0]: (seqs[0],)}

                # Sort by size
                clones_tmp = sorted(
                    list(set(seq_tmp_dict.values())), key=len, reverse=True
                )
                for sub_group, clone_group in enumerate(clones_tmp, 1):
                    clones_vj[membership][sub_group] = clone_group

            # Flatten to sequence -> clone mapping
            seq_to_clone = {}
            for membership, clone_dict in clones_vj.items():
                for clone_id, sequences in clone_dict.items():
                    for seq in sequences:
                        seq_to_clone[seq] = f"{membership}_{clone_id}"

            # Apply to dataframe
            df_vj_grp = df_vj_grp.with_columns(
                pl.col(locus_key)
                .replace_strict(seq_to_clone, default=None)
                .alias(f"_{key_added}_{vj_chain}")
            )
            # Keep only what we need
            df_vj_result = df_vj_grp.select(
                [
                    "_original_order",
                    f"_{key_added}_VJ",
                ]
            )

        # Combine VDJ + VJ for this locus
        if df_vdj_result is not None and df_vj_result is not None:
            df_locus_chains = df_vdj_result.join(
                df_vj_result,
                on="_original_order",
                how="full",
                coalesce=True,
            )
        elif df_vdj_result is not None:
            df_locus_chains = df_vdj_result.with_columns(
                pl.lit(None).alias(f"_{key_added}_VJ")
            )
        elif df_vj_result is not None:
            df_locus_chains = df_vj_result.with_columns(
                pl.lit(None).alias(f"_{key_added}_VDJ")
            )
        else:
            # No chains found for this locus
            continue

        # Add celltype back
        df_locus_chains = df_locus_chains.with_columns(
            pl.lit(locus_celltype).alias("_celltype")
        )

        # Join with cell_id from original df
        df_locus_chains = df_locus_chains.join(
            df.select(["_original_order", "cell_id"]),
            on="_original_order",
            how="left",
        )

        # Combine VDJ + VJ at the cell level for this locus
        df_locus_summary = df_locus_chains.group_by("cell_id").agg(
            [
                pl.col(f"_{key_added}_VDJ")
                .drop_nulls()
                .unique()
                .alias("_vdj_set"),
                pl.col(f"_{key_added}_VJ")
                .drop_nulls()
                .unique()
                .alias("_vj_set"),
            ]
        )

        # Create locus-specific clone IDs
        df_locus_summary = df_locus_summary.with_columns(
            pl.struct(["_vdj_set", "_vj_set"])
            .map_elements(
                lambda s: _combine_single_locus(
                    s["_vdj_set"], s["_vj_set"], locus_celltype
                ),
                return_dtype=pl.List(pl.Utf8),
            )
            .alias(f"_{key_added}_{locus_celltype}")
        )

        # Store result for this locus
        locus_results[locus_celltype] = df_locus_summary.select(
            ["cell_id", f"_{key_added}_{locus_celltype}"]
        )

    # Merge all locus results back to main df
    # Start with cell_id from original df
    df_final = df.select("cell_id").unique()

    # Join each locus result
    for locus_celltype, locus_df in locus_results.items():
        df_final = df_final.join(locus_df, on="cell_id", how="left")

    # Combine all locus clone IDs with '|'
    locus_columns = [f"_{key_added}_{ct}" for ct in locus_results.keys()]

    if locus_columns:
        # Each column contains a list of clone IDs for that locus
        # We need to flatten all lists and join with '|'
        df_final = df_final.with_columns(
            pl.struct(locus_columns)
            .map_elements(
                lambda row: _flatten_and_join_loci(row, locus_columns),
                return_dtype=pl.String,
            )
            .alias(key_added)
        ).drop(locus_columns)
    else:
        # No loci processed, add empty clone_id column
        df_final = df_final.with_columns(pl.lit(None).alias(key_added))

    # overwrite the original key_added column in df if exists
    if key_added in df.collect_schema():
        df = df.drop(key_added)
    # Join back to original df
    df = df.join(df_final, on="cell_id", how="left").drop("_original_order")

    # return
    vdj._data = df
    vdj.update_metadata(clone_key=str(key_added))
    # offload memory
    vdj._cache_data()
    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig AIRR table\n"
            "   'metadata', cell observations table\n"
        ),
    )


def _check_chains(
    df: pl.DataFrame | pl.LazyFrame,
    vdj_loci: list[str],
    vj_loci: list[str],
) -> tuple[bool, bool]:
    """
    Check if VDJ and VJ chains exist for a locus using polars.

    Vectorized check using polars filtering operations.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input AIRR dataframe.
    vdj_loci : list[str]
        VDJ loci (e.g., ['IGH'], ['TRB']).
    vj_loci : list[str]
        VJ loci (e.g., ['IGK', 'IGL'], ['TRA']).

    Returns
    -------
    tuple[bool, bool]
        (has_vdj, has_vj) indicating presence of chains.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect(engine="streaming")

    # Vectorized check for VDJ chains
    has_vdj = df.filter(pl.col("locus").is_in(vdj_loci)).shape[0] > 0

    # Vectorized check for VJ chains
    has_vj = df.filter(pl.col("locus").is_in(vj_loci)).shape[0] > 0

    return has_vdj, has_vj


def _group_sequences(
    df: pl.DataFrame | pl.LazyFrame,
    key: str,
    same_vj: bool = True,
    same_length: bool = True,
    recalculate_length: bool = True,
    by_alleles: bool = False,
):
    """
    Group sequences by V/J genes and junction length using vectorized polars.

    Vectorized polars implementation that groups contigs by (V gene, J gene)
    pairs and then by junction length. Returns numerical group IDs.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input AIRR dataframe.
    key : str
        Column name for junction sequences (e.g., 'junction', 'junction_aa').
    same_vj : bool, optional
        Whether to group by same V and J genes.
    same_length : bool, optional
        Whether to group by same junction length.
    recalculate_length : bool, optional
        Whether to recalculate junction length from sequences.
    by_alleles : bool, optional
        Whether to group by alleles or genes.

    Returns
    -------
    pl.DataFrame
        DataFrame with added '_membership' column.
    """
    # Ensure LazyFrame for efficiency
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    v_col = VCALLG if VCALLG in df.collect_schema() else VCALL

    # Vectorized V/J gene stripping using polars string operations
    if same_vj:
        if not by_alleles:
            df = df.with_columns(
                [
                    pl.col(v_col)
                    .str.replace_all(STRIPALLELENUM, "")
                    .alias("_v_gene"),
                    pl.col(JCALL)
                    .str.replace_all(STRIPALLELENUM, "")
                    .alias("_j_gene"),
                ]
            )
        else:
            df = df.with_columns(
                [
                    pl.col(v_col).alias("_v_gene"),
                    pl.col(JCALL).alias("_j_gene"),
                ]
            )

    # Handle length calculation
    if same_length:
        length_col = key + "_length"
        if length_col not in df.collect_schema():
            recalculate_length = True

        # Calculate or use existing length
        if recalculate_length:
            df = df.with_columns(
                pl.when(pl.col(key).is_null())
                .then(pl.lit(0))
                .otherwise(pl.col(key).cast(pl.String).str.len_bytes())
                .alias(f"_{key}_length")
            )
        else:
            df = df.with_columns(pl.col(length_col).alias(f"_{key}_length"))

    # Filter out rows with null junction
    filter_conditions = [pl.col(key).is_not_null()]

    if same_vj:
        filter_conditions.extend(
            [pl.col("_v_gene").is_not_null(), pl.col("_j_gene").is_not_null()]
        )

    df = df.filter(pl.all_horizontal(filter_conditions))

    # Build grouping columns dynamically
    grouping_cols = []

    if same_vj:
        grouping_cols.extend(["_v_gene", "_j_gene"])

    if same_length:
        grouping_cols.append(f"_{key}_length")

    # Create membership based on selected grouping
    if grouping_cols:
        # Create a combined group ID
        if same_vj and same_length:
            # Both VJ and length grouping
            df = df.with_columns(
                pl.struct(["_v_gene", "_j_gene"])
                .rank(method="dense")
                .alias("_vj_group")
            )
            df = df.with_columns(
                pl.col(f"_{key}_length")
                .rank(method="dense")
                .over("_vj_group")
                .alias("_length_group")
            )
            df = df.with_columns(
                pl.concat_str(
                    [
                        pl.col("_vj_group").cast(pl.String),
                        pl.lit("_"),
                        pl.col("_length_group").cast(pl.String),
                    ]
                ).alias("_membership")
            )
            df = df.drop(["_v_gene", "_j_gene", "_vj_group", "_length_group"])

        elif same_vj:
            # Only VJ grouping
            df = df.with_columns(
                pl.struct(["_v_gene", "_j_gene"])
                .rank(method="dense")
                .cast(pl.String)
                .alias("_membership")
            )
            df = df.drop(["_v_gene", "_j_gene"])

        elif same_length:
            # Only length grouping
            df = df.with_columns(
                pl.col(f"_{key}_length")
                .rank(method="dense")
                .cast(pl.String)
                .alias("_membership")
            )
    else:
        # No grouping - all sequences in one group
        df = df.with_columns(pl.lit("1").alias("_membership"))

    if isinstance(df, pl.LazyFrame):
        df = df.collect(engine="streaming")

    return df


def _clustering_scipy(
    d_mat: np.ndarray,
    threshold: float,
    sequences: list[str],
    hard_threshold: int | float | None = None,
) -> dict:
    """
    Cluster sequences using scipy connected components (fastest).

    Parameters
    ----------
    d_mat : np.ndarray
        Distance matrix (n x n).
    threshold : float
        Distance threshold for clustering.
    sequences : list[str]
        List of sequences.
    hard_threshold : int | float | None, optional
        Absolute distance cutoff. If supplied, `threshold` is ignored.

    Returns
    -------
    dict
        Dictionary mapping sequences to cluster groups.
    """
    _threshold = threshold if hard_threshold is None else hard_threshold
    # Create adjacency matrix
    adjacency = (d_mat <= _threshold).astype(int)

    # Find connected components
    _, labels = connected_components(
        csgraph=csr_matrix(adjacency), directed=False
    )

    # Group sequences by cluster labels
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Build output dict
    out_dict = {}
    for cluster_indices in clusters.values():
        cluster_seqs = tuple(
            sorted([sequences[idx] for idx in cluster_indices])
        )
        for idx in cluster_indices:
            out_dict[sequences[idx]] = cluster_seqs

    return out_dict


def _combine_single_locus(
    vdj_list: list[str] | None, vj_list: list[str] | None, celltype: str
) -> list[str]:
    """Combine VDJ/VJ for a single celltype/locus.

    Args:
        vdj_list: List of VDJ clone IDs
        vj_list: List of VJ clone IDs
        celltype: Cell type identifier (e.g., 'B', 'abT', 'gdT')

    Returns:
        List of combined clone IDs for this locus
    """
    if not vdj_list and not vj_list:
        return []

    vdj_vals = vdj_list if vdj_list else [None]
    vj_vals = vj_list if vj_list else [None]
    combos: list[str] = []

    for vdj in vdj_vals:
        for vj in vj_vals:
            parts = [celltype]
            if vdj is not None:
                parts.append(f"VDJ_{vdj}")
            if vj is not None:
                parts.append(f"VJ_{vj}")
            combos.append("_".join(parts))

    return combos


def _flatten_and_join_loci(row: dict, locus_columns: list[str]) -> str | None:
    """Flatten clone IDs from all loci and join with '|'.

    Args:
        row: Dictionary containing clone ID lists for each locus
        locus_columns: List of column names containing locus-specific clone IDs

    Returns:
        Pipe-separated string of all clone IDs, or None if no clones found
    """
    all_clones: list[str] = []
    for col_name in locus_columns:
        locus_clones = row[col_name]
        if locus_clones is not None:
            all_clones.extend(locus_clones)
    return "|".join(all_clones) if all_clones else None


def _update_distance_matrix(
    d_mat: np.ndarray,
    cell_ids: list,
    cell_id_to_idx: dict,
    distance_matrix: lil_matrix,
) -> None:
    """
    Update the global distance matrix with distances from a group.

    Sums distances when multiple sequences from the same cell exist.

    Parameters
    ----------
    d_mat : np.ndarray
        Pairwise distance matrix for sequences in this group
    cell_ids : list
        Cell IDs corresponding to each sequence
    cell_id_to_idx : dict
        Mapping from cell_id to matrix index
    distance_matrix : lil_matrix
        Accumulated distance matrix (modified in-place)
    """
    n = len(cell_ids)

    for i in range(n):
        for j in range(i, n):  # Only upper triangle (symmetric)
            cell_i = cell_ids[i]
            cell_j = cell_ids[j]

            idx_i = cell_id_to_idx[cell_i]
            idx_j = cell_id_to_idx[cell_j]

            dist = d_mat[i, j]

            # Sum distance
            distance_matrix[idx_i, idx_j] += dist

            # Mirror for symmetry (unless diagonal)
            if idx_i != idx_j:
                distance_matrix[idx_j, idx_i] += dist


def transfer(
    adata: AnnData | MuData,
    vdj: DandelionPolars,
    expanded: bool = False,
    gex_key: str | None = None,
    vdj_key: str | None = None,
    clone_key: str | None = None,
    collapse_nodes: bool = False,
    overwrite: bool | list[str] | str | None = None,
    obs: bool = True,
    obsm: bool = True,
    uns: bool = True,
    obsp: bool = True,
) -> None:
    """
    Transfer data in Dandelion slots to AnnData, updating `.obs`, `.uns`, `.obsm`, and `.obsp`.
    Transfers both graphs:
      - graph[0] -> adata.obsm['X_vdj_all']
      - graph[1] -> adata.obsm['X_vdj_expanded']
    The `expanded` flag controls which graph becomes the *main* adjacency written to
    adata.obsp['connectivities'] / ['distances'] (but both graphs are stored).

    Parameters
    ----------
    adata : AnnData | MuData
        AnnData object or `MuData` object.
    dandelion : DandelionPolars
        Dandelion object.
    expanded : bool, optional
        Whether or not to transfer the embedding with all cells with BCR (False) or only for expanded clones (True).
    gex_key : str | None, optional
        prefix for stashed RNA connectivities and distances.
    vdj_key : str | None, optional
        prefix for stashed VDJ connectivities and distances.
    clone_key : str | None, optional
        column name of clone/clonotype ids. Only used for integration with scirpy.
    collapse_nodes : bool, optional
        Whether or not to transfer a cell x cell or clone x clone connectivity matrix into `.uns`. Only used for
        integration with scirpy.
    overwrite : bool | list[str] | str | None, optional
        Whether or not to overwrite existing anndata columns. Specifying a string indicating column name or
        list of column names will overwrite that specific column(s).
    """
    start = logg.info("Transferring network")

    # if the provide adata is an MuData, we need to transfer to mudata.mod['gex']
    # but we don't want to add mudata as a dependency here, so we do a duck-typing check
    if hasattr(adata, "mod"):
        if "airr" in adata.mod:
            recipient = adata.mod["airr"]
        else:
            raise ValueError(
                "Provided AnnData is a MuData object without 'airr' modality."
            )
    # we just associate recipient to adata directly
    else:
        recipient = adata
    if isinstance(vdj, DandelionPolars):
        if vdj._backend == "polars":
            vdj.to_pandas()
    # --- 1) metadata -> adata.obs (preserve original overwrite semantics) ---
    if obs:
        for x in vdj._metadata.columns:
            if x not in recipient.obs.columns:
                recipient.obs[x] = pd.Series(vdj._metadata[x])
            elif overwrite is True:
                recipient.obs[x] = pd.Series(vdj._metadata[x])
            if type_check(vdj._metadata, x):
                recipient.obs[x] = recipient.obs[x].replace(np.nan, "No_contig")
            if recipient.obs[x].dtype == "bool":
                recipient.obs[x] = recipient.obs[x].astype(str)

        # explicit overwrite list/string handling (matches original)
        if (overwrite is not None) and (overwrite is not True):
            if not isinstance(overwrite, list):
                overwrite = [overwrite]
            for ow in overwrite:
                recipient.obs[ow] = pd.Series(vdj._metadata[ow])
                if type_check(vdj._metadata, ow):
                    recipient.obs[ow] = recipient.obs[ow].replace(
                        np.nan, "No_contig"
                    )

    # also check that all the cells in dandelion are in recipient
    common_cells = recipient.obs_names.intersection(vdj._metadata.index)
    # subset to common cells only
    vdj = vdj[vdj._metadata.index.isin(common_cells)].copy()

    # If there's no graph, we're done with metadata only
    if vdj.graph is None:
        logg.info(
            " finished", time=start, deep=("updated `.obs` with `.metadata`\n")
        )
        return

    # --- 2) prepare neighbor keys and stash RNA neighbors/connectivities if present ---
    neighbors_key = "neighbors"
    skip_stash = neighbors_key not in recipient.uns
    if obsp:
        gex_key = "gex" if gex_key is None else gex_key
        g_connectivities_key = f"{gex_key}_connectivities"
        g_distances_key = f"{gex_key}_distances"
        vdj_key = "vdj" if vdj_key is None else vdj_key
        v_connectivities_key = f"{vdj_key}_connectivities"
        v_distances_key = f"{vdj_key}_distances"

        # Stash RNA connectivities/distances before we overwrite connectivities/distances
        if not skip_stash:
            # preferred stash from obsp if exist
            recipient.obsp[g_connectivities_key] = recipient.obsp[
                "connectivities"
            ].copy()
            recipient.obsp[g_distances_key] = recipient.obsp["distances"].copy()
            g_neighbors_key = f"{gex_key}_{neighbors_key}"
            recipient.uns[g_neighbors_key] = recipient.uns[neighbors_key].copy()

    # --- 3) Convert both graphs ---
    graph_connectivities, graph_distances = {}, {}
    # handle graph[0] and graph[1]
    for idx in (0, 1):
        G = None
        if vdj.graph is not None:
            try:
                G = vdj.graph[idx]
            except Exception:
                pass

        if G is not None:
            graph_connectivities[idx], graph_distances[idx] = (
                _graph_to_matrices(G, recipient, None)
            )

    # handle precomputed distances (sparse or DataFrame)
    if getattr(vdj, "distances", None) is not None:
        graph_connectivities[2], graph_distances[2] = _graph_to_matrices(
            None, recipient, vdj.distances
        )

    # Determine main graph index
    main_idx = 1 if expanded else 0
    if main_idx not in graph_connectivities:
        main_idx = next(iter(graph_connectivities.keys()))

    if obsp:
        # --- 4) Update recipient.obsp ---
        recipient.obsp["connectivities"] = graph_connectivities[main_idx].copy()
        recipient.obsp["distances"] = graph_distances[main_idx].copy()

        # store the all (graph[0]) and expanded graph (graph[1]) if available
        if 0 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_all"] = (
                graph_connectivities[0].copy()
            )
            recipient.obsp[f"{v_distances_key}_all"] = graph_distances[0].copy()
        if 1 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_expanded"] = (
                graph_connectivities[1].copy()
            )
            recipient.obsp[f"{v_distances_key}_expanded"] = graph_distances[
                1
            ].copy()
        if 2 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_full"] = (
                graph_connectivities[2].copy()
            )
            recipient.obsp[f"{v_distances_key}_full"] = graph_distances[
                2
            ].copy()
        recipient.uns[neighbors_key] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "params": {
                "n_neighbors": 1,
                "method": "custom",
                "metric": "precomputed",
            },
        }

    if uns:
        # --- 5) Clone-level mapping (scirpy compatible) ---
        clone_key = clone_key if clone_key is not None else "clone_id"

        if not collapse_nodes:
            for idx in graph_connectivities:
                graph_connectivities[idx][
                    graph_connectivities[idx].nonzero()
                ] = 1
            cell_indices = {
                str(i): np.array([k])
                for i, k in zip(
                    range(0, len(recipient.obs_names)), recipient.obs_names
                )
            }
            bin_conn = graph_connectivities[main_idx]
        else:
            invalid = [
                "",
                "unassigned",
                "NaN",
                "NA",
                "nan",
                "None",
                "none",
                None,
            ]
            cell_indices = Tree()
            for x, y in recipient.obs[clone_key].items():
                if y not in invalid:
                    cell_indices[y][x].value = 1
            cell_indices = {
                str(x): np.array(list(r))
                for x, r in zip(
                    range(0, len(cell_indices)), cell_indices.values()
                )
            }
            bin_conn = np.zeros([len(cell_indices), len(cell_indices)])
            np.fill_diagonal(bin_conn, 1)
            bin_conn = csr_matrix(bin_conn)

        recipient.uns[clone_key] = {
            # this is a symmetrical, pairwise, sparse distance matrix of clonotypes
            # the matrix is offset by 1, i.e. 0 = no connection, 1 = distance 0
            "distances": bin_conn,
            # '0' refers to the row/col index in the `distances` matrix
            # (numeric index, but needs to be strbecause of h5py)
            # np.array(["cell1", "cell2"]) points to the rows in `recipient.obs`
            "cell_indices": cell_indices,
        }

    if obsm:
        # --- 6) Layouts ---
        if vdj.layout is not None:
            stored_embeddings = {}
            for idx, obsm_name in (
                (0, "X_vdj_all"),
                (1, "X_vdj_expanded"),
            ):
                try:
                    layout = vdj.layout[idx]
                except Exception:
                    continue
                if layout is None:
                    continue
                coord = pd.DataFrame.from_dict(layout, orient="index")
                coord = coord.reindex(index=recipient.obs_names).fillna(np.nan)
                if coord.shape[1] >= 2:
                    embedding = coord.iloc[:, :2].to_numpy(dtype=np.float32)
                else:
                    col0 = (
                        coord.iloc[:, 0]
                        .to_numpy(dtype=np.float32)
                        .reshape(-1, 1)
                    )
                    col1 = np.zeros_like(col0)
                    embedding = np.hstack([col0, col1])

                recipient.obsm[obsm_name] = embedding
                stored_embeddings[idx] = obsm_name

            # Set the "active" embedding safely
            main_idx = 1 if expanded else 0
            active_obsm = stored_embeddings.get(main_idx)
            if active_obsm is not None:
                recipient.obsm["X_vdj"] = recipient.obsm[active_obsm].copy()

    # break up the message depending on which parts were executed
    message_parts = []
    if obs:
        message_parts += [f"updated `.obs` with `.metadata`\n"]
    if obsm:
        message_parts += [
            f"wrote `.obsm['X_vdj']` and `.obsm['X_vdj_expanded']`\n"
        ]
    if obsp:
        message_parts += [
            f"wrote adata.obsp['connectivities'] & ['distances'] from graph[{main_idx}]\n",
            "stored RNA matrices under rna_* keys (stashed)\n",
            f"stored vdj matrices under '{v_connectivities_key}' (+ '_expanded' and + '_full' if available)\n",
        ]
    if uns:
        message_parts += [f"added `.uns['{clone_key}']` clone-level mapping"]

    # --- 7) Done ---
    logg.info(
        " finished",
        time=start,
        deep="".join(message_parts),
    )


tf = transfer  # alias for transfer


def _graph_to_matrices(
    G: nx.Graph | None,
    adata: AnnData,
    distances: csr_matrix | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Convert a graph or provided distances into properly aligned sparse
    connectivities and distances matrices.

    Rules:
    - If G is provided, convert edges → sparse distance matrix.
    - If a CSR distance matrix is provided, must have `._index_names`.
    - If a DataFrame is provided, use its index/columns.
    - Reindex to `adata.obs_names` without dense conversion.
    - Compute connectivities as exp(-d) on non-zero entries.
    - Add tiny self-edge if matrix is entirely empty.
    """

    target_names = list(adata.obs_names)
    n = len(target_names)
    name_to_new = {name: i for i, name in enumerate(target_names)}
    # CASE A: Build distances from a NetworkX graph
    if distances is None and G is not None:
        # Build COO arrays directly
        edges = list(G.edges(data=True))
        if not edges:
            distances = csr_matrix((n, n), dtype=np.float32)
        else:
            u, v, w = zip(*[(u, v, d.get("weight", 1.0)) for u, v, d in edges])
            # Filter edges where both nodes exist in target
            mask_u = np.array([node in name_to_new for node in u])
            mask_v = np.array([node in name_to_new for node in v])
            mask = mask_u & mask_v  # vectorized AND
            u = np.array(u)[mask]
            v = np.array(v)[mask]
            w = np.array(w, dtype=np.float32)[mask]

            # Map names to target indices
            u_idx = np.array([name_to_new[x] for x in u])
            v_idx = np.array([name_to_new[x] for x in v])

            # Make symmetric
            rows = np.concatenate([u_idx, v_idx])
            cols = np.concatenate([v_idx, u_idx])
            vals = np.concatenate([w, w])
            vals += 1.0

            distances = csr_matrix(
                (vals, (rows, cols)), shape=(n, n), dtype=np.float32
            )

    # CASE B: distances provided as a csr_matrix with _index_names
    elif isinstance(distances, csr_matrix):
        old_names = np.array(distances._index_names)
        coo = distances.tocoo()

        # Map old names to target indices, missing names → -1
        old_row_names = old_names[coo.row]
        old_col_names = old_names[coo.col]

        row_idx = np.array(
            [name_to_new.get(name, -1) for name in old_row_names]
        )
        col_idx = np.array(
            [name_to_new.get(name, -1) for name in old_col_names]
        )

        # Keep only edges where both row and col exist in target
        mask = (row_idx >= 0) & (col_idx >= 0)
        rows = row_idx[mask]
        cols = col_idx[mask]
        vals = coo.data[mask]
        vals += 1.0

        distances = csr_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=np.float32
        )
    # Build connectivities = exp(-d) for non-zero entries
    connectivities = distances.copy()
    if connectivities.nnz > 0:
        connectivities.data = np.exp(-connectivities.data)
        connectivities.data = np.clip(connectivities.data, 1e-45, np.inf)

    distances.data -= 1.0

    # Ensure matrix is not completely empty
    if connectivities.nnz == 0:
        connectivities = connectivities.tolil()
        distances = distances.tolil()
        connectivities[0, 0] = 1e-10
        distances[0, 0] = 0.0
        connectivities = connectivities.tocsr()
        distances = distances.tocsr()

    return connectivities, distances


def clone_view(
    adata: AnnData,
    mode: Literal["all", "expanded", "full", "gex"] | None = "expanded",
    connectivities_key: str | None = None,
    distances_key: str | None = None,
    embedding_key: str | None = None,
):
    """
    Swap the 'active' connectivities, distances, and optionally embedding in AnnData.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    mode : Literal["all", "expanded", "full", "gex"] | None, optional
        If specified, set the active connectivities/distances/embedding to one of the preset modes.
    connectivities_key : str | None, optional
        The key in `.obsp` to set as active `.obsp["connectivities"]` if `mode` is None.
    distances_key : str | None, optional
        The key in `.obsp` to set as active `.obsp["distances"]` if `mode` is None.
    embedding_key : str | None, optional
        If specified, set `.obsm["X_vdj"]` to `.obsm[embedding_key]` if `mode` is None.
    """
    if mode is None:
        # use the other key directly
        if connectivities_key in adata.obsp:
            adata.obsp["connectivities"] = adata.obsp[connectivities_key].copy()
        else:
            raise KeyError(f"{connectivities_key} not found in adata.obsp")

        if distances_key in adata.obsp:
            adata.obsp["distances"] = adata.obsp[distances_key].copy()
        else:
            raise KeyError(f"{distances_key} not found in adata.obsp")

        if embedding_key is not None:
            if embedding_key in adata.obsm:
                adata.obsm["X_vdj"] = adata.obsm[embedding_key].copy()
            else:
                raise KeyError(f"{embedding_key} not found in adata.obsm")
    else:
        if mode == "gex":
            conn_key = f"{mode}_connectivities"
            dist_key = f"{mode}_distances"
            neighbors_key = f"{mode}_neighbors"
            emb_key = None
        else:
            conn_key = f"vdj_connectivities_{mode}"
            dist_key = f"vdj_distances_{mode}"
            neighbors_key = None
            emb_key = f"X_vdj_{mode}" if mode != "full" else None
        adata.obsp["connectivities"] = adata.obsp[conn_key].copy()
        adata.obsp["distances"] = adata.obsp[dist_key].copy()
        if emb_key is not None:
            adata.obsm["X_vdj"] = adata.obsm[emb_key].copy()
        if neighbors_key is not None:
            adata.uns["neighbors"] = adata.uns[neighbors_key].copy()
        else:
            adata.uns["neighbors"] = {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
                "params": {
                    "n_neighbors": 1,
                    "method": "custom",
                    "metric": "precomputed",
                },
            }


def tabulate_clone_sizes(
    metadata_: pd.DataFrame, clonesize_dict: dict, clonekey: str
) -> pd.Series:
    """Tabulate clone sizes."""
    return pd.Series(
        dict(
            zip(
                metadata_.index,
                [
                    str(y) if pd.notnull(y) else str(0)
                    for y in [
                        (
                            sorted(
                                list(
                                    {clonesize_dict[c_] for c_ in c.split("|")}
                                ),
                                key=lambda x: (
                                    int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x)
                                ),
                                reverse=True,
                            )[0]
                            if "|" in c
                            else clonesize_dict[c]
                        )
                        for c in metadata_[str(clonekey)]
                    ]
                ],
            )
        )
    )


def clone_size(
    vdj: DandelionPolars | AnnData | MuData,
    groupby: str | None = None,
    max_size: int | None = None,
    clone_key: str | None = None,
    key_added: str | None = None,
) -> None:
    """
    Quantify clone sizes, globally or per group.

    If `groupby` is specified, clone sizes and proportions are calculated
    within each group separately. Each cell is then annotated with the size,
    proportion, and frequency category based on sizes similar to scRepertoire.
    If a cell belongs to multiple clones (e.g., multiple chains assigned
    to different clones), the largest clone is used for annotation.

    Parameters
    ----------
    vdj : Dandelion | AnnData | MuData
        VDJ data.
    groupby : str | None, optional
        Column in metadata to group by before calculating clone sizes.
        If None, calculates global clone sizes.
    max_size : int | None, optional
        Clip clone size values at this maximum.
    clone_key : str | None, optional
        Column specifying clone identifiers. Defaults to 'clone_id'.
    key_added : str | None, optional
        Prefix for new metadata column names.
    """
    # --- Select metadata
    if hasattr(vdj, "mod"):
        metadata_ = vdj.mod["airr"].obs.copy()
    elif isinstance(vdj, AnnData):
        metadata_ = vdj.obs.copy()
    elif isinstance(vdj, DandelionPolars):
        if vdj._backend == "polars":
            vdj.to_pandas()
        metadata_ = vdj._metadata.copy()

    clone_key = "clone_id" if clone_key is None else clone_key
    if clone_key not in metadata_.columns:
        raise KeyError(f"Column '{clone_key}' not found in metadata.")

    # --- Expand multi-clone entries
    tmp = metadata_[clone_key].astype(str).str.split("|", expand=True).stack()
    # drop None/No_contig entries
    tmp = tmp[~tmp.isin(["No_contig", "unassigned"] + EMPTIES)]
    tmp = tmp.reset_index(drop=False)
    tmp.columns = ["cell_id", "tmp", clone_key]

    # --- Compute clone sizes (global or per group)
    if groupby is None:
        clonesize = tmp[clone_key].value_counts()
        prop = clonesize / metadata_.shape[0]
    else:
        # Merge with groupby column using cell_id as key
        # Reset index to make cell_id a regular column for merging
        metadata_with_index = metadata_.reset_index()
        metadata_with_index = metadata_with_index.rename(
            columns={"index": "cell_id"}
        )

        tmp = tmp.merge(
            metadata_with_index[["cell_id", groupby]], on="cell_id", how="left"
        )
        clonesize = tmp.groupby([groupby, clone_key]).size()
        group_sizes = metadata_[groupby].value_counts()

        # Calculate proportion correctly for each group
        prop_dict = {}
        for grp in clonesize.index.get_level_values(0).unique():
            group_clones = clonesize.loc[grp]
            group_total = group_sizes[grp]
            for clone_id, size in group_clones.items():
                prop_dict[(grp, clone_id)] = size / group_total

        # Create Series with MultiIndex
        prop = pd.Series(prop_dict)
        prop.index = pd.MultiIndex.from_tuples(
            prop.index, names=[groupby, clone_key]
        )

    # --- Create max_size categories if specified
    if max_size is not None:

        def categorize_size(size):
            if pd.isna(size):
                return np.nan
            if size < max_size:
                return str(int(size))
            else:
                return f">= {max_size}"

        clonesize_cat = clonesize.apply(categorize_size)
        clonesize_cat_map = clonesize_cat.to_dict()

    # --- Define clone frequency bins
    bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    labels = ["Rare", "Small", "Medium", "Large", "Hyperexpanded"]
    if groupby is None:
        prop_bins = pd.cut(prop, bins=bins, labels=labels, include_lowest=True)
    else:
        # Apply pd.cut to the entire Series at once, preserving the MultiIndex
        prop_bins = pd.cut(prop, bins=bins, labels=labels, include_lowest=True)

    # --- Build lookup maps
    size_map = clonesize.to_dict()
    prop_map = prop.to_dict()
    cat_map = prop_bins.to_dict()

    # --- Assign to each cell
    cell_sizes = []
    cell_props = []
    cell_cats = []
    cell_size_cats = [] if max_size is not None else None

    for i, row in metadata_.iterrows():
        clone_ids = str(row[clone_key])
        # Check for empty/invalid entries
        if pd.isna(clone_ids) or clone_ids in [
            "No_contig",
            "unassigned",
            "None",
            "nan",
        ]:
            cell_sizes.append(np.nan)
            cell_props.append(np.nan)
            cell_cats.append(np.nan)
            if max_size is not None:
                cell_size_cats.append(np.nan)
            continue

        clones = clone_ids.split("|")

        if groupby is None:
            # look up sizes directly
            sizes = [size_map.get(c, np.nan) for c in clones]
            props = [prop_map.get(c, np.nan) for c in clones]
            cats = [cat_map.get(c, np.nan) for c in clones]
            if max_size is not None:
                size_cats = [clonesize_cat_map.get(c, np.nan) for c in clones]
        else:
            grp = row[groupby]
            # Use tuple keys for grouped lookups
            sizes = [size_map.get((grp, c), np.nan) for c in clones]
            props = [prop_map.get((grp, c), np.nan) for c in clones]
            cats = [cat_map.get((grp, c), np.nan) for c in clones]
            if max_size is not None:
                size_cats = [
                    clonesize_cat_map.get((grp, c), np.nan) for c in clones
                ]

        # take the largest available clone (by numeric size)
        if len(sizes) == 0 or all(pd.isna(sizes)):
            cell_sizes.append(np.nan)
            cell_props.append(np.nan)
            cell_cats.append(np.nan)
            if max_size is not None:
                cell_size_cats.append(np.nan)
        else:
            max_idx = np.nanargmax(sizes)
            cell_sizes.append(sizes[max_idx])
            cell_props.append(props[max_idx])
            cell_cats.append(cats[max_idx])
            if max_size is not None:
                cell_size_cats.append(size_cats[max_idx])

    metadata_[f"{clone_key}_size"] = cell_sizes
    metadata_[f"{clone_key}_size_prop"] = cell_props
    metadata_[f"{clone_key}_size_category"] = cell_cats
    if max_size is not None:
        metadata_[f"{clone_key}_size_max_{max_size}"] = cell_size_cats

    # --- Write results back to object
    col_key = key_added if key_added is not None else clone_key

    if isinstance(vdj, DandelionPolars):
        vdj._metadata[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj._metadata[f"{col_key}_size_prop"] = metadata_[
            f"{clone_key}_size_prop"
        ]
        vdj._metadata[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj._metadata[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]
    elif isinstance(vdj, AnnData):
        vdj.obs[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj.obs[f"{col_key}_size_prop"] = metadata_[f"{clone_key}_size_prop"]
        vdj.obs[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj.obs[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]
    elif hasattr(vdj, "mod"):
        vdj.mod["airr"].obs[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj.mod["airr"].obs[f"{col_key}_size_prop"] = metadata_[
            f"{clone_key}_size_prop"
        ]
        vdj.mod["airr"].obs[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj.mod["airr"].obs[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]


def clone_overlap(
    vdj: DandelionPolars | AnnData,
    groupby: str,
    min_clone_size: int | None = None,
    weighted_overlap: bool = False,
    clone_key: str | None = None,
) -> pd.DataFrame:
    """
    A function to tabulate clonal overlap for input as a circos-style plot.

    Parameters
    ----------
    vdj : Dandelion | AnnData
        Dandelion or AnnData object.
    groupby : str
        column name in obs/metadata for collapsing to columns in the clone_id x groupby data frame.
    min_clone_size : int | None, optional
        minimum size of clone for plotting connections. Defaults to 2 if left as None.
    weighted_overlap : bool, optional
        if True, instead of collapsing to overlap to binary, overlap will be returned as the number of cells.
        In the future, there will be the option to use something like a jaccard index.
    clone_key : str | None, optional
        column name for clones. `None` defaults to 'clone_id'.

    Returns
    -------
    pd.DataFrame
        clone_id x groupby overlap :class:`pandas.core.frame.DataFrame'.

    Raises
    ------
    ValueError
        if min_clone_size is 0.
    """
    start = logg.info("Calculating clone overlap")
    if isinstance(vdj, DandelionPolars):
        if vdj._backend == "polars":
            vdj.to_pandas()
        data = vdj._metadata.copy()
    elif isinstance(vdj, AnnData):
        data = vdj.obs.copy()
    elif isinstance(vdj, MuData):
        data = vdj.mod["airr"].obs.copy()

    if min_clone_size is None:
        min_size = 2
    else:
        min_size = int(min_clone_size)

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    # get rid of problematic rows that appear because of category conversion?
    allgroups = list(data[groupby].unique())
    data = data[
        ~(
            data[clone_].isin(
                [np.nan, "nan", "NaN", "No_contig", "unassigned", "None", None]
            )
        )
    ]

    # prepare a summary table
    datc_ = data[clone_].str.split("|", expand=True).stack()
    datc_ = pd.DataFrame(datc_)
    datc_.reset_index(drop=False, inplace=True)
    datc_.columns = ["cell_id", "tmp", clone_]
    datc_.drop("tmp", inplace=True, axis=1)
    datc_ = datc_[
        ~(
            datc_[clone_].isin(
                [
                    "",
                    np.nan,
                    "nan",
                    "NaN",
                    "No_contig",
                    "unassigned",
                    "None",
                    None,
                ]
            )
        )
    ]
    dictg_ = dict(data[groupby])
    datc_[groupby] = [dictg_[l] for l in datc_["cell_id"]]

    overlap = pd.crosstab(datc_[clone_], datc_[groupby])
    for x in allgroups:
        if x not in overlap:
            overlap[x] = 0

    if min_size == 0:
        raise ValueError("min_size must be greater than 0.")
    if not weighted_overlap:
        if min_size > 2:
            overlap[overlap < min_size] = 0
            overlap[overlap >= min_size] = 1
        elif min_size == 2:
            overlap[overlap >= min_size] = 1

    overlap.index.name = None
    overlap.columns.name = None

    if isinstance(vdj, AnnData):
        vdj.uns["clone_overlap"] = overlap.copy()
        logg.info(
            " finished",
            time=start,
            deep=("Updated AnnData: \n" "   'uns', clone overlap table"),
        )
    else:
        return overlap


def clustering(
    distance_dict: dict, threshold: float, sequences_dict: dict[str, str]
) -> dict:
    """Clustering the sequences."""
    out_dict = {}
    # find out the unique indices in this subset
    i_unique = list(set(flatten(distance_dict)))
    # for every pair of i1,i2 is their dictance smaller than the thresholdeshold?
    i_pair_d = {
        (i1, i2): (
            distance_dict[(i1, i2)] <= threshold
            if (i1, i2) in distance_dict
            else False
        )
        for i1, i2 in product(i_unique, repeat=2)
    }
    i_pair_d.update(
        {
            (i2, i1): (
                distance_dict[(i2, i1)] <= threshold
                if (i2, i1) in distance_dict
                else False
            )
            for i1, i2 in product(i_unique, repeat=2)
        }
    )
    # so which indices should not be part of a clone?
    canbetogether = defaultdict(list)
    for ii1, ii2 in product(i_unique, repeat=2):
        if i_pair_d[(ii1, ii2)] or i_pair_d[(ii2, ii1)]:
            if (ii1, ii2) in distance_dict:
                canbetogether[ii1].append((ii1, ii2))
                canbetogether[ii2].append((ii1, ii2))
            elif (ii2, ii1) in distance_dict:
                canbetogether[ii2].append((ii2, ii1))
                canbetogether[ii1].append((ii2, ii1))
        else:
            if (ii1, ii2) or (ii2, ii1) in distance_dict:
                canbetogether[ii1].append(())
                canbetogether[ii2].append(())
    for x in canbetogether:
        canbetogether[x] = list({y for y in canbetogether[x] if len(y) > 0})
    # convert the indices to sequences
    for x in canbetogether:
        if len(canbetogether[x]) > 0:
            out_dict[sequences_dict[x]] = tuple(
                sorted(
                    set(
                        list(
                            [
                                sequences_dict[y]
                                for y in flatten(canbetogether[x])
                            ]
                        )
                        + [sequences_dict[x]]
                    )
                )
            )
        else:
            out_dict[sequences_dict[x]] = tuple([sequences_dict[x]])
    return out_dict


def vj_usage_pca(
    adata: AnnData,
    groupby: str,
    min_size: int = 20,
    mode: Literal["B", "abT", "gdT"] = "abT",
    use_vdj_v: bool = True,
    use_vdj_j: bool = True,
    use_vj_v: bool = True,
    use_vj_j: bool = True,
    transfer_mapping=None,
    n_comps: int = 30,
    groups: list[str] | None = None,
    allowed_chain_status: list[str] | None = [
        "Single pair",
        "Extra pair",
        "Extra pair-exception",
        "Orphan VDJ-exception",
    ],
    verbose=False,
    **kwargs,
) -> AnnData:
    """
    Extract productive V/J gene usage from single cell data and compute PCA.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata with Dandelion VDJ info transferred.
    groupby : str
        Column name in `adata.obs` to groupby as observations for PCA.
    min_size : int, optional
        Minimum cell size numbers to keep for computing the final matrix. Defaults to 20.
    mode : Literal["B", "abT", "gdT"], optional
        Mode for extract the V/J genes.
    use_vdj_v : bool, optional
        Whether to use V gene from VDJ contigs for tabulation. Defaults to True.
    use_vdj_j : bool, optional
        Whether to use J gene from VDJ contigs for tabulation. Defaults to True.
    use_vj_v : bool, optional
        Whether to use V genes from VJ contigs for tabulation. Defaults to True.
    use_vj_j : bool, optional
        Whether to use J genes from VJ contigs for tabulation. Defaults to True.
    transfer_mapping : None, optional
        If provided, the columns will be mapped to the output AnnData from the original AnnData.
    n_comps : int, optional
        Number of principal components to compute. Defaults to 30.
    groups : list[str] | None, optional
        If provided, only the following groups/categories will be used for computing the PCA.
    allowed_chain_status : list[str] | None, optional
        If provided, only the ones in this list are kept from the `chain_status` column.
        Defaults to ["Single pair", "Extra pair", "Extra pair-exception", "Orphan VDJ-exception"].
    verbose : bool, optional
        Whether to display progress
    **kwargs
        Additional keyword arguments passed to `scanpy.pp.pca`.

    Returns
    -------
    AnnData
        AnnData object with obs as groups and V/J genes as features.
    """
    start = logg.info("Computing PCA for V/J gene usage")
    # filtering
    if allowed_chain_status is not None:
        adata_ = adata[
            adata.obs["chain_status"].isin(allowed_chain_status)
        ].copy()

    if groups is not None:
        adata_ = adata_[adata_.obs[groupby].isin(groups)].copy()
    # build config
    gene_config = {
        "vdj_v": dict(
            enabled=use_vdj_v,
            main=f"v_call_{mode}_VDJ_main",
            full=f"v_call_{mode}_VDJ",
        ),
        "vdj_j": dict(
            enabled=use_vdj_j,
            main=f"j_call_{mode}_VDJ_main",
            full=f"j_call_{mode}_VDJ",
        ),
        "vj_v": dict(
            enabled=use_vj_v,
            main=f"v_call_{mode}_VJ_main",
            full=f"v_call_{mode}_VJ",
        ),
        "vj_j": dict(
            enabled=use_vj_j,
            main=f"j_call_{mode}_VJ_main",
            full=f"j_call_{mode}_VJ",
        ),
    }
    if not any(cfg["enabled"] for cfg in gene_config.values()):
        raise ValueError("At least one of the use_vj/vdj_v/j must be True.")

    # Determine which groups to keep
    cell_counts = adata_.obs[groupby].value_counts()
    keep_groups = cell_counts[cell_counts >= min_size].index

    # collect gene lists
    gene_lists = {}
    for key, cfg in gene_config.items():
        if cfg["enabled"]:
            uniq = adata_.obs[cfg["main"]].unique().tolist()
            gene_lists[key] = [
                g for g in uniq if g not in ("None", "No_contig")
            ]
        else:
            gene_lists[key] = []

    all_genes = [g for genes in gene_lists.values() for g in genes]

    # initialise results df
    vdj_df = pd.DataFrame(
        index=keep_groups, columns=all_genes, dtype=float
    ).fillna(0)

    # count genes per group
    for group in tqdm(
        vdj_df.index,
        desc="Tabulating V/J gene usage",
        disable=not verbose,
    ):
        group_mask = adata_.obs[groupby] == group
        obs_group = adata_.obs.loc[group_mask]

        for key, cfg in gene_config.items():
            if not cfg["enabled"]:
                continue

            counts = Counter(obs_group[cfg["full"]])
            for gene in gene_lists[key]:
                vdj_df.loc[group, gene] = counts.get(gene, 0)

    # normalize each chain separately
    for key, cfg in gene_config.items():
        if not cfg["enabled"]:
            continue

        cols = gene_lists[key]
        colsum = vdj_df[cols].sum(axis=1)
        vdj_df.loc[:, cols] = vdj_df[cols].div(colsum, axis=0) * 100

    # Create new AnnData + PCA
    obs_df = pd.DataFrame(index=vdj_df.index)
    obs_df["cell_type"] = vdj_df.index
    obs_df["cell_count"] = cell_counts.loc[vdj_df.index]

    vdj_adata = AnnData(
        X=vdj_df.values,
        obs=obs_df,
        var=pd.DataFrame(index=vdj_df.columns),
    )

    sc.pp.pca(vdj_adata, n_comps=n_comps, use_highly_variable=False, **kwargs)

    # Transfer old obs columns to new AnnData
    if transfer_mapping is not None:
        collapsed = adata_.obs.drop_duplicates(subset=groupby)
        for to in transfer_mapping:
            mapping = dict(zip(collapsed[groupby], collapsed[to]))
            vdj_adata.obs[to] = vdj_adata.obs.index.map(mapping)

    logg.info(
        " finished",
        time=start,
        deep=("Returned AnnData: \n" "   'obsm', X_pca for V/J gene usage"),
    )
    return vdj_adata


def group_sequences(
    input_vdj: pd.DataFrame,
    junction_key: str,
    recalculate_length: bool = True,
    by_alleles: bool = False,
    locus: Literal["ig", "tr-ab", "tr-gd"] = "ig",
):
    """
    Groups sequence IDs with the same V and J genes, and then splits the groups based on the lengths of the sequences.

    Parameters
    ----------
    input_vdj : pd.DataFrame
        The input data frame.
    junction_key : str
        The name of the column in the input data that contains the junction sequences.
    recalculate_length : bool, optional
        Whether to recalculate the lengths of the junction sequences.
    by_alleles : bool, optional
        Whether to group sequence IDs by their V and J gene alleles, rather than by only the V and J genes.
    locus : Literal["ig", "tr-ab", "tr-gd"], optional
        The locus of the input data. One of "ig", "tr-ab", or "tr-gd".

    Returns
    -------
    vj_len_grp : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence length.
    seq_grp : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence.

    Raises
    ------
    ValueError
        Raised when the sequence length column is not found in the input table.
    """
    locus_log1_dict = {"ig": "IGH", "tr-ab": "TRB", "tr-gd": "TRD"}
    # retrieve the J genes and J genes
    if not by_alleles:
        if VCALLG in input_vdj.columns:
            V = [re.sub(STRIPALLELENUM, "", str(v)) for v in input_vdj[VCALLG]]
        else:
            V = [re.sub(STRIPALLELENUM, "", str(v)) for v in input_vdj[VCALL]]
        J = [re.sub(STRIPALLELENUM, "", str(j)) for j in input_vdj[JCALL]]
    else:
        if VCALLG in input_vdj.columns:
            V = [str(v) for v in input_vdj[VCALLG]]
        else:
            V = [str(v) for v in input_vdj[VCALL]]
        J = [str(j) for j in input_vdj[JCALL]]
    # collapse the alleles to just genes
    V = [",".join(list(set(v.split(",")))) for v in V]
    J = [",".join(list(set(j.split(",")))) for j in J]
    seq = dict(zip(input_vdj.index, input_vdj[junction_key]))
    if recalculate_length:
        # Set length to 0 for null/NaN values, otherwise calculate from sequence
        seq_length = [
            len(str(l)) if pd.notnull(l) else 0 for l in input_vdj[junction_key]
        ]
    else:
        try:
            seq_length = [l for l in input_vdj[junction_key + "_length"]]
        except:
            raise ValueError(
                "{} not found in {} input table.".format(
                    junction_key + "_length", locus_log1_dict[locus]
                )
            )
    seq_length_dict = dict(zip(input_vdj.index, seq_length))
    # Create a dictionary and group sequence ids with same V and J genes
    V_J = dict(zip(input_vdj.index, zip(V, J)))
    vj_grp = defaultdict(list)
    for key, val in sorted(V_J.items()):
        vj_grp[val].append(key)
    # and now we split the groups based on lengths of the seqs
    vj_len_grp = Tree()
    seq_grp = Tree()
    for g in vj_grp:
        # first, obtain what's the unique lengths
        jlen = []
        for contig_id in vj_grp[g]:
            jlen.append(seq_length_dict[contig_id])
        setjlen = list(set(jlen))
        # then for each unique length, we add the contigs to a new tree if it matches the length
        # and also the actual seq sequences into a another one
        for s in setjlen:
            for contig_id in vj_grp[g]:
                jlen_ = seq_length_dict[contig_id]
                if jlen_ == s:
                    vj_len_grp[g][s][contig_id].value = 1
                    seq_grp[g][s][seq[contig_id]].value = 1
                    for c in [contig_id]:
                        vj_len_grp[g][s][c] = seq[c]
    return vj_len_grp, seq_grp


def group_pairwise_hamming_distance(
    clonotype_vj_len_group: Tree,
    clonotype_sequence_group: Tree,
    identity: float,
    locus: str,
    chain: Literal["VDJ", "VJ"],
    junction_key: str,
    verbose: bool = True,
) -> Tree:
    """
    Group clonotypes by pairwise hamming distance and generate a nested dictionary of clonotype groups.

    Parameters
    ----------
    clonotype_vj_len_group : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence length.
    clonotype_sequence_group : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence.
    identity : float
        The identity threshold for grouping sequences.
    locus : str
        The locus of the chain.
    chain : Literal["VDJ", "VJ"]
        The chain type.
    verbose : bool, optional
        Whether to show the progress bar, by default True.

    Returns
    -------
    Tree
        A nested dictionary containing contigs assigned to clonotype groups.
    """
    clones = Tree()

    # for each seq group, calculate the hamming distance matrix
    for g in tqdm(
        clonotype_sequence_group,
        desc=f"Finding clones based on {locus} cell {chain} chains using {junction_key}".format(),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        for l in clonotype_sequence_group[g]:
            seq_ = list(clonotype_sequence_group[g][l])
            tdarray = np.array(seq_).reshape(-1, 1)
            d_mat = squareform(pdist(tdarray, lambda x, y: hamming(x[0], y[0])))
            # then calculate what the acceptable threshold is for each length of sequence
            tr = math.floor(int(l) * (1 - identity))
            # convert diagonal and upper triangle to zeroes
            d_mat = np.tril(d_mat)
            np.fill_diagonal(d_mat, 0)
            # get the coordinates/indices of seqs to match against the threshold later
            indices_temp = []
            indices = []
            indices_temp = [list(x) for x in np.tril_indices_from(d_mat)]
            indices = list(zip(indices_temp[0], indices_temp[1]))
            # if there's more than 1 contig, remove the diagonal
            if len(indices) > 1:
                for pairs in indices:
                    # remove diagonals
                    if pairs[0] == pairs[1]:
                        indices.remove(pairs)
            indices_j = []
            # use the coordinates/indices to retrieve the seq sequences
            for p in range(0, len(indices)):
                a1, b1 = indices[p]
                indices_j.append(seq_[a1])
                indices_j.append(seq_[b1])
            # convert the distance matrix to coordinate (source) and distance (target) and create it
            # as a dictionary
            source, target = d_mat.nonzero()
            source_target = list(zip(source.tolist(), target.tolist()))
            if len(source) == 0 & len(target) == 0:
                source_target = list([(0, 0)])
            dist = {}
            for st in source_target:
                dist.update({st: d_mat[st]})
            if d_mat.shape[0] > 1:
                seq_tmp_dict = clustering(dist, tr, seq_)
            else:
                seq_tmp_dict = {seq_[0]: tuple([seq_[0]])}
            # sort the list so that clones that are larger have a smaller number
            clones_tmp = sorted(
                list(set(seq_tmp_dict.values())),
                key=len,
                reverse=True,
            )
            for x in range(0, len(clones_tmp)):
                clones[g][l][x + 1] = clones_tmp[x]
    # now to retrieve the contig ids that are grouped together
    cid = Tree()
    for g in clones:
        for l in clones[g]:
            # retrieve the clone 'numbers'
            for c in clones[g][l]:
                grp_seq = clones[g][l][c]
                for key, value in clonotype_vj_len_group[g][l].items():
                    if value in grp_seq:
                        cid[g][l][c][key].value = 1

    return cid


def rename_clonotype_ids(
    clonotype_groups: Tree,
    prefix: str = "",
) -> dict[str, str]:
    """
    Renames clonotype IDs to numerical barcode system.

    Parameters
    ----------
    clonotype_groups : Tree
        A nested dictionary that containing clonotype groups of contigs.
    prefix : str, optional
        Prefix to append to front, if necessary.
    Returns
    -------
    dict[str, str]
        A dictionary that maps sequence IDs to clonotype IDs.
    """
    clone_dict = {}
    first_key = []
    for k1 in clonotype_groups.keys():
        first_key.append(k1)
    first_key = list(set(first_key))
    first_key_dict = dict(zip(first_key, range(1, len(first_key) + 1)))
    for g in clonotype_groups:
        second_key = []
        for k2 in clonotype_groups[g].keys():
            second_key.append(k2)
        second_key = list(set(second_key))
        second_key_dict = dict(zip(second_key, range(1, len(second_key) + 1)))
        for l in clonotype_groups[g]:
            third_key = []
            for k3 in clonotype_groups[g][l].keys():
                third_key.append(k3)
            third_key = list(set(third_key))
            third_key_dict = dict(zip(third_key, range(1, len(third_key) + 1)))
            for key, value in dict(clonotype_groups[g][l]).items():
                for v in value:
                    if type(v) is int:
                        break
                    clone_dict[v] = (
                        prefix
                        + str(first_key_dict[g])
                        + "_"
                        + str(second_key_dict[l])
                        + "_"
                        + str(third_key_dict[key])
                    )

    return clone_dict


def refine_clone_assignment(
    dat: pd.DataFrame,
    clone_key: str,
    clone_dict_vj: dict,
    verbose: bool = True,
) -> None:
    """
    Refines the clone assignment of VDJ sequences based on their VJ chain pairing.

    Parameters
    ----------
    dat : pd.DataFrame
        The input data frame containing the full VDJ/VJ data.
    clone_key : str
        The name of the column in the input data that contains the clone IDs.
    clone_dict_vj : dict
        A dictionary for the VJ chains for us to map sequence IDs to clone IDs.
    verbose : bool, optional
        Whether to print progress messages.
    """
    cellclonetree = Tree()
    seqcellclonetree = Tree()
    for c, s, z in zip(dat["cell_id"], dat["sequence_id"], dat[clone_key]):
        seqcellclonetree[c][s].value = 1
        if pd.notnull(z):
            cellclonetree[c][z].value = 1

    for c in cellclonetree:
        cellclonetree[c] = list(cellclonetree[c])

    fintree = Tree()
    for c in tqdm(
        cellclonetree,
        desc="Refining clone assignment based on VJ chain pairing ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        suffix = [
            clone_dict_vj[x] for x in seqcellclonetree[c] if x in clone_dict_vj
        ]
        fintree[c] = []
        if len(suffix) > 0:
            for cl in cellclonetree[c]:
                if present(cl):
                    for s in suffix:
                        if check_same_celltype(cl, s):
                            fintree[c].append(
                                cl + "_" + "".join(s.split("_", 1)[1])
                            )
                        else:
                            fintree[c].append(cl + "|" + s)
        else:
            for cl in cellclonetree[c]:
                if present(cl):
                    fintree[c].append(cl)
        fintree[c] = "|".join(fintree[c])
    dat[clone_key] = [fintree[x] for x in dat["cell_id"]]
    for i, row in dat.iterrows():  # is this going to be slow...?
        if not present(row[clone_key]):
            if i in clone_dict_vj and i in dat.index:
                dat.at[i, clone_key] = clone_dict_vj[i]


def check_chains(dat_vdj: pd.DataFrame, dat_vj: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of whether the chain is orphan or not.

    Parameters
    ----------
    dat_vdj : pd.DataFrame
        Input dataframe containing VDJ chains of a particular locus.
    dat_vj : pd.DataFrame
        Input dataframe containing VJ chains of a particular locus.

    Returns
    -------
    pd.DataFrame
        Output dataframe containing chain status.
    """

    vj_check = pd.crosstab(dat_vj.cell_id, dat_vj.locus).apply(sum, axis=1)
    vj_check[vj_check > 1] = 1
    vdj_check = pd.crosstab(dat_vdj.cell_id, dat_vdj.locus).apply(sum, axis=1)
    vdj_check[vdj_check > 1] = 1
    chain_check = pd.concat([vdj_check, vj_check], axis=1)
    chain_check.columns = ["VDJ", "VJ"]
    chain_check["Orphan VDJ"] = pd.notnull(chain_check["VDJ"]) & (
        pd.isnull(chain_check["VJ"])
    )
    chain_check["Orphan VJ"] = pd.notnull(chain_check["VJ"]) & (
        pd.isnull(chain_check["VDJ"])
    )
    chain_check["All VDJ"] = pd.notnull(chain_check["VDJ"])
    chain_check["All VJ"] = pd.notnull(chain_check["VJ"])
    chain_check["VDJ"] = (
        (~chain_check["Orphan VDJ"])
        & (~chain_check["Orphan VJ"])
        & (pd.notnull(chain_check["VDJ"]))
    )
    chain_check["VJ"] = (
        (~chain_check["Orphan VDJ"])
        & (~chain_check["Orphan VJ"])
        & (pd.notnull(chain_check["VJ"]))
    )
    return chain_check


def vdj_sample(
    vdj: DandelionPolars,
    size: int,
    adata: AnnData | MuData | None = None,
    p: list[float] | np.ndarray[float] | None = None,
    force_replace: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[DandelionPolars, AnnData] | DandelionPolars:
    """
    Resample vdj data and corresponding AnnData to a specified size.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object containing VDJ data.
    size : int
        Desired size for resampling.
    adata : AnnData | MuData | None, optional
        AnnData or MuData object corresponding to the gene expression data.
    p : list[float] | np.ndarray[float] | None, optional
        Drawing probabilities for each cell, must sum to 1. If None, uniform probabilities are used.
    force_replace : bool, optional
        Whether to force sampling with replacement, by default False.
    random_state : int | np.random.RandomState | None, optional
        Random state for reproducibility, by default None.


    Returns
    -------
    tuple[DandelionPolars, AnnData] | DandelionPolars
        Resampled Dandelion and AnnData objects if adata is provided, otherwise only Dandelion.
    """
    logg.info("Resampling to {} cells.".format(str(size)))

    rng = np.random.default_rng(random_state)

    if adata is None:
        # Determine if we need replacement
        # Only collect metadata when needed for numpy indexing
        if isinstance(vdj._metadata, pl.LazyFrame):
            metadata = vdj._metadata.collect(engine="streaming")
        else:
            metadata = vdj._metadata

        n_cells = vdj.n_obs
        replace = True if size > n_cells else False
        if force_replace:
            replace = True

        # Use numpy for index-based sampling - more efficient and polars-native
        if p is not None:
            p_array = np.asarray(p)
            p_array = p_array / p_array.sum()  # Ensure probabilities sum to 1
        else:
            p_array = None

        # Get sampled indices using numpy
        sample_indices = rng.choice(
            n_cells, size=size, replace=replace, p=p_array
        )

        # Get the cell IDs from metadata at those indices
        if isinstance(metadata, pl.DataFrame):
            keep_cells = metadata[sample_indices]["cell_id"].to_list()
        else:
            keep_cells = metadata.iloc[sample_indices].index.tolist()
    else:
        # check if MuData and extract the gex modality
        if hasattr(adata, "mod"):
            adata = adata.mod["gex"].copy()
        else:
            adata = adata.copy()
        # ensure only cells present in both vdj and adata are sampled
        common_cells = list(
            set(vdj._metadata.index).intersection(set(adata.obs_names))
        )
        adata = adata[adata.obs_names.isin(common_cells)].copy()
        vdj = vdj[vdj._metadata.index.isin(common_cells)].copy()

        replace = True if size > vdj._metadata.shape[0] else False
        if force_replace:
            replace = True
        # use scanpy to sample
        sc.pp.sample(adata, n=size, replace=replace, rng=random_state, p=p)
        keep_cells = list(adata.obs_names)

    # Get the .data without ambiguous assignments
    if isinstance(vdj._data, pl.LazyFrame):
        cols = set(vdj._data.collect_schema().names())
    else:
        cols = set(vdj._data.columns)

    if "ambiguous" in cols:
        vdj_dat = vdj._data.filter(pl.col("ambiguous").is_in(FALSES))
    else:
        vdj_dat = vdj._data.clone()

    # Filter to keep only sampled cells
    vdj_dat = vdj_dat.filter(pl.col("cell_id").is_in(keep_cells))

    if replace:
        # For replacement logic, need to collect if lazy
        if isinstance(vdj_dat, pl.LazyFrame):
            vdj_dat = vdj_dat.collect(engine="streaming")

        # sample with replacement
        cell_counts = Counter(keep_cells)

        # Only process cells that appear more than once
        duplicated_cells = {
            cell: count for cell, count in cell_counts.items() if count > 1
        }

        if duplicated_cells:
            # Separate data for duplication
            if isinstance(vdj_dat, pl.DataFrame):
                vdj_dat_to_duplicate = vdj_dat.filter(
                    pl.col("cell_id").is_in(list(duplicated_cells.keys()))
                ).clone()
                vdj_dat_to_keep = vdj_dat.filter(
                    ~pl.col("cell_id").is_in(list(duplicated_cells.keys()))
                ).clone()
            else:
                vdj_dat_to_duplicate = vdj_dat[
                    vdj_dat["cell_id"].isin(duplicated_cells.keys())
                ].copy()
                vdj_dat_to_keep = vdj_dat[
                    ~vdj_dat["cell_id"].isin(duplicated_cells.keys())
                ].copy()

            # Create duplicates for both dat and adata in one loop
            all_duplicated_vdj = []

            for cell_id, count in duplicated_cells.items():
                # Duplicate dat rows
                if isinstance(vdj_dat, pl.DataFrame):
                    cell_rows = vdj_dat_to_duplicate.filter(
                        pl.col("cell_id") == cell_id
                    ).clone()
                else:
                    cell_rows = vdj_dat_to_duplicate[
                        vdj_dat_to_duplicate["cell_id"] == cell_id
                    ].copy()

                for i in range(count):
                    suffix = f"-{str(i)}" if i > 0 else ""

                    # Add dat rows
                    temp_rows = (
                        cell_rows.clone()
                        if isinstance(cell_rows, pl.DataFrame)
                        else cell_rows.copy()
                    )
                    if suffix:
                        if isinstance(temp_rows, pl.DataFrame):
                            temp_rows = temp_rows.with_columns(
                                [
                                    (pl.col("cell_id") + suffix).alias(
                                        "cell_id"
                                    ),
                                    (pl.col("sequence_id") + suffix).alias(
                                        "sequence_id"
                                    ),
                                ]
                            )
                        else:
                            temp_rows["cell_id"] = temp_rows["cell_id"] + suffix
                            temp_rows["sequence_id"] = (
                                temp_rows["sequence_id"] + suffix
                            )
                    all_duplicated_vdj.append(temp_rows)

            # Combine everything back together
            if isinstance(vdj_dat, pl.DataFrame):
                vdj_dat = pl.concat([vdj_dat_to_keep] + all_duplicated_vdj)
            else:
                vdj_dat = pd.concat(
                    [vdj_dat_to_keep] + all_duplicated_vdj, ignore_index=True
                )

    # reinitialise a copy of the sampled dandelion object using vdj_dat
    vdj = DandelionPolars(vdj_dat)
    if adata is not None:
        adata.obs_names_make_unique()
        if hasattr(adata, "mod"):
            # if MuData, update the gex modality
            return vdj, to_scirpy(vdj, gex_adata=adata)
        else:
            return vdj, adata
    else:
        return vdj


def to_scirpy(
    data: DandelionPolars,
    transfer: bool = False,
    to_mudata: bool = True,
    gex_adata: AnnData | None = None,
    key: tuple[str, str] = ("gex", "airr"),
    **kwargs,
) -> AnnData | MuData:
    """
    Convert Dandelion data to scirpy-compatible format.

    Parameters
    ----------
    data: DandelionPolars
        The Dandelion object containing the data to be converted.
    transfer : bool, optional
        Whether to transfer additional information from Dandelion to the converted data. Defaults to False.
    to_mudata : bool, optional
        Whether to convert the data to MuData format instead of AnnData. Defaults to True.
        If converting to AnnData, it will assert that the same cell_ids and .obs_names are present in the `gex_adata` provided.
    gex_adata : AnnData, optional
        An existing AnnData object to be used as the base for the converted data if provided.
    key : tuple[str, str], optional
        A tuple specifying the keys for the 'gex' and 'airr' fields in the converted data. Defaults to ("gex", "airr").
    **kwargs
        Additional keyword arguments passed to `scirpy.io.read_airr`.

    Returns
    -------
    AnnData | MuData
        The converted data in either AnnData or MuData format.
    """
    if data._backend == "polars":
        data.to_pandas()
    # if gex_adata is provided, make sure to only transfer cells that are present in both
    # we will only filter the data to match gex_adata
    if gex_adata is not None:
        data = data[data.metadata_names.isin(gex_adata.obs_names)].copy()
        tmp_gex = gex_adata.copy()
        if not to_mudata:
            tf(
                tmp_gex, data, obs=False, uns=True, obsp=False, obsm=False
            )  # so that the slots are properly filled
    else:
        tmp_gex = None

    if "umi_count" not in data._data and "duplicate_count" in data._data:
        data._data["umi_count"] = data._data["duplicate_count"]
    for h in [
        "sequence",
        "rev_comp",
        "sequence_alignment",
        "germline_alignment",
        "v_cigar",
        "d_cigar",
        "j_cigar",
    ]:
        if h not in data._data:
            data._data[h] = None

    airr, obs = to_ak(data._data, **kwargs)
    if to_mudata:
        airr_adata = _create_anndata(airr, obs)
        if tmp_gex is not None:
            tf(
                airr_adata,
                data,
                obs=False,
                uns=True,
                obsp=False,
                obsm=False,
            )
        mdata = _create_mudata(tmp_gex, airr_adata, key)
        if transfer:
            tf(mdata, data)
        return mdata
    else:
        adata = _create_anndata(airr, obs, tmp_gex)
        if transfer:
            tf(adata, data)
        return adata


def from_scirpy(data: AnnData | MuData) -> DandelionPolars:
    """
    Convert data from scirpy format to Dandelion format.

    Parameters
    ----------
    data : AnnData | MuData
        The input data in scirpy format.

    Returns
    -------
    DandelionPolars
        The converted data in Dandelion format.
    """
    if not isinstance(data, AnnData):
        data = data.mod["airr"]
    data = data.copy()
    data.obsm["airr"]["cell_id"] = data.obs.index
    df = from_ak(data.obsm["airr"])
    vdj = DandelionPolars(df, verbose=False)
    # Reverse transfer (recover metadata + clone graph)
    _reverse_transfer(data, vdj)
    return vdj


def _reverse_transfer(
    data: AnnData | MuData,
    dandelion: DandelionPolars,
    clone_key: str = "clone_id",
) -> None:
    """
    Reverse-transfer scirpy data (AnnData/MuData) into a Dandelion object.

    Pulls metadata, clone mappings, graphs, and embeddings from scirpy's structure.

    Parameters
    ----------
    data : AnnData | MuData
        Input scirpy object (AnnData or MuData with .mod['airr']).
    dandelion : Dandelion
        The Dandelion object to update in place.
    clone_key : str, optional
        Key under .uns containing scirpy clone-level mapping (default: 'clone_id').
    """
    # --- Handle MuData case ---
    if hasattr(data, "mod"):
        if "airr" not in data.mod:
            raise ValueError(
                "MuData object must contain an 'airr' modality for scirpy data."
            )
        adata = data.mod["airr"]
    else:
        adata = data

    # --- Copy metadata ---
    for col in adata.obs:
        if col not in dandelion._metadata.columns:
            dandelion._metadata[col] = adata.obs[col]

    # --- Extract clone-level connection info ---
    if clone_key in adata.uns:
        clone_uns = adata.uns[clone_key]
        distances = clone_uns["distances"]
        cell_indices = clone_uns["cell_indices"]
        # --- Rebuild graph ---
        G = nx.from_scipy_sparse_array(distances)
        # Relabel nodes: scirpy stores numeric keys ("0", "1", ...) mapped to arrays of cell_ids
        mapping = {}
        for k, v in cell_indices.items():
            k_int = int(k)
            if isinstance(v, (list, np.ndarray)):
                # If clone node has multiple cells, store them all in node attribute
                mapping[k_int] = str(v[0]) if len(v) > 0 else str(k)
                G.nodes[k_int]["cells"] = list(v)
            else:
                mapping[k_int] = str(v)
                G.nodes[k_int]["cells"] = [v]
        G = nx.relabel_nodes(G, mapping)

        # Store the graph
        dandelion.graph = [G, None]

    # map the obs back to data as well
    dandelion.update_data()


def from_ak(airr: Array) -> pd.DataFrame:
    """
    Convert an AIRR-formatted array to a pandas DataFrame.

    Parameters
    ----------
    airr : Array
        The AIRR-formatted array to be converted.

    Returns
    -------
    pd.DataFrame
        The converted pandas DataFrame.

    Raises
    ------
    KeyError
        If `sequence_id` not found in the data.
    """
    import awkward as ak

    df = ak.to_dataframe(airr)
    # check if 'sequence_id' column does not exist or if any value in 'sequence_id' is NaN
    if "sequence_id" not in df.columns or df["sequence_id"].isnull().any():
        df_reset = df.reset_index()

        # create a new 'sequence_id' column
        df_reset["sequence_id"] = df_reset.apply(
            lambda row: f"{row['cell_id']}_contig_{row['subentry'] + 1}", axis=1
        )

        # set 'entry' and 'subentry' back as the index
        df = df_reset.set_index(["entry", "subentry"])

    if "sequence_id" in df.columns:
        df.set_index("sequence_id", drop=False, inplace=True)
    if "cell_id" not in df.columns:
        df["cell_id"] = [c.split("_contig")[0] for c in df["sequence_id"]]

    return df


def to_ak(
    data: pd.DataFrame,
    **kwargs,
) -> tuple[Array, pd.DataFrame]:
    """
    Convert data from a DataFrame to an AnnData object with AIRR format.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    **kwargs
        Additional keyword arguments passed to `scirpy.io.read_airr`.

    Returns
    -------
    tuple[Array, pd.DataFrame]
        A tuple containing the AIRR-formatted data as an ak.Array and the cell-level attributes as a pd.DataFrame.
    """

    try:
        import scirpy as ir
    except:
        raise ImportError("Please install scirpy to use this function.")

    adata = ir.io.read_airr(data, **kwargs)

    return adata.obsm["airr"], adata.obs


def _create_anndata(
    airr: Array,
    obs: pd.DataFrame,
    adata: AnnData | None = None,
) -> AnnData:
    """
    Create an AnnData object with the given AIRR array and observation data.

    Parameters
    ----------
    airr : Array
        The AIRR array.
    obs : pd.DataFrame
        The observation data.
    adata : AnnData | None, optional
        An existing AnnData object to update. If None, a new AnnData object will be created.

    Returns
    -------
    AnnData
        The AnnData object with the AIRR array and observation data.
    """
    obsm = {"airr": airr}
    temp = AnnData(X=None, obs=obs, obsm=obsm)

    if adata is None:
        adata = temp
    else:
        cell_names = adata.obs_names.intersection(temp.obs_names)
        adata = adata[adata.obs_names.isin(cell_names)].copy()
        temp = temp[temp.obs_names.isin(cell_names)].copy()
        adata.obsm = dict() if adata.obsm is None else adata.obsm
        adata.obsm.update(temp.obsm)

    return adata


def _create_mudata(
    gex: AnnData,
    adata: AnnData,
    key: tuple[str, str] = ("gex", "airr"),
) -> MuData:
    """
    Create a MuData object from the given AnnData objects.

    Parameters
    ----------
    gex : AnnData
        The AnnData object containing gene expression data.
    adata : AnnData
        The AnnData object containing additional data.
    key : tuple[str, str], optional
        The keys to use for the gene expression and additional data in the MuData object. Defaults to ("gex", "airr").

    Returns
    -------
    MuData
        The created MuData object.

    Raises
    ------
    ImportError
        If the mudata package is not installed.
    """

    try:
        import mudata
    except ImportError:
        raise ImportError("Please install mudata. pip install mudata")
    if gex is not None:
        return mudata.MuData({key[0]: gex, key[1]: adata})
    return mudata.MuData({key[1]: adata})


def concat(
    arrays: (
        list[DandelionPolars | pl.DataFrame | pl.LazyFrame | pd.DataFrame]
        | dict[
            str, DandelionPolars | pl.DataFrame | pl.LazyFrame | pd.DataFrame
        ]
    ),
    check_unique: bool = True,
    collapse_cells: bool = True,
    sep: str = "_",
    suffixes: list[str] | None = None,
    prefixes: list[str] | None = None,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = True,
) -> DandelionPolars:
    """
    Concatenate data frames and return as Dandelion object.

    If both suffixes and prefixes are `None` and check_unique is True, then a sequential number suffix will be appended.

    Parameters
    ----------
    arrays : list[DandelionPolars | pl.DataFrame | pl.LazyFrame | pd.DataFrame] | dict[
            str, DandelionPolars | pl.DataFrame | pl.LazyFrame | pd.DataFrame
        ]
        List or dictionary of DandelionPolars objects or pandas/polars DataFrames to concatenate.
    check_unique : bool, optional
        Check the new index for duplicates. Otherwise defer the check until necessary.
        Setting to False will improve the performance of this method.
    collapse_cells : bool, optional
        whether or not to collapse multiple contigs per cell into one row in the
        metadata. By default True.
    sep : str, optional
        the separator to append suffix/prefix.
    suffixes : list[str] | None, optional
        List of suffixes to append to sequence_id and cell_id.
    prefixes : list[str] | None, optional
        List of prefixes to append to sequence_id and cell_id.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        Whether to print the messages, by default True.

    Returns
    -------
    DandelionPolars
        concatenated Dandelion object

    Raises
    ------
    ValueError
        if both prefixes and suffixes are provided.
    """
    if (suffixes is not None) and (prefixes is not None):
        raise ValueError("Please provide only prefixes or suffixes, not both.")

    if suffixes is not None:
        if len(arrays) != len(suffixes):
            raise ValueError(
                "Please provide the same number of suffixes as the number of objects to concatenate."
            )

    if prefixes is not None:
        if len(arrays) != len(prefixes):
            raise ValueError(
                "Please provide the same number of prefixes as the number of objects to concatenate."
            )

    # Convert dict to list if necessary
    if isinstance(arrays, dict):
        arrays = [
            (
                arrays[x].copy()
                if isinstance(arrays[x], DandelionPolars)
                else arrays[x]
            )
            for x in arrays
        ]

    # Convert all inputs to DandelionPolars
    vdjs_ = []
    for x in arrays:
        if isinstance(x, DandelionPolars):
            vdjs_.append(x.copy())
        elif isinstance(x, pl.LazyFrame):
            vdjs_.append(
                DandelionPolars(x.collect(engine="streaming"), verbose=False)
            )
        elif isinstance(x, pl.DataFrame):
            vdjs_.append(DandelionPolars(x, verbose=False))
        elif isinstance(x, pd.DataFrame):
            # Convert pandas to polars
            vdjs_.append(DandelionPolars(pl.from_pandas(x), verbose=False))
        else:
            raise ValueError(
                "All input arrays must be either DandelionPolars instances, "
                "Polars DataFrames/LazyFrames, or pandas DataFrames."
            )

    # Collect metadata and data names
    tmp_meta_names, tmp_data_names = [], []
    for tmp in vdjs_:
        # Get names as lists
        if isinstance(tmp._metadata, pl.LazyFrame):
            tmp_meta_names.extend(
                tmp._metadata.select("cell_id")
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
        elif isinstance(tmp._metadata, pl.DataFrame):
            tmp_meta_names.extend(
                tmp._metadata.select("cell_id").to_series().to_list()
            )

        if isinstance(tmp._data, pl.LazyFrame):
            tmp_data_names.extend(
                tmp._data.select("sequence_id")
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
        elif isinstance(tmp._data, pl.DataFrame):
            tmp_data_names.extend(
                tmp._data.select("sequence_id").to_series().to_list()
            )

    if collapse_cells:
        # Preserve order but remove duplicates
        tmp_meta_names = list(dict.fromkeys(tmp_meta_names))

    if len(tmp_meta_names) != len(set(tmp_meta_names)):
        metadata_index_order = None
    else:
        metadata_index_order = tmp_meta_names

    if len(tmp_data_names) != len(set(tmp_data_names)):
        data_index_order = None
    else:
        data_index_order = tmp_data_names

    # Handle unique indices with suffixes/prefixes
    if check_unique:
        if metadata_index_order is None and data_index_order is None:
            metadata_index_order, data_index_order = [], []
            for i in range(0, len(vdjs_)):
                if (suffixes is None) and (prefixes is None):
                    vdjs_[i].add_cell_suffix(
                        str(i),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif suffixes is not None:
                    vdjs_[i].add_cell_suffix(
                        str(suffixes[i]),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif prefixes is not None:
                    vdjs_[i].add_cell_prefix(
                        str(prefixes[i]),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                # Collect updated names
                if isinstance(vdjs_[i]._metadata, pl.LazyFrame):
                    metadata_index_order.extend(
                        vdjs_[i]
                        ._metadata.select("cell_id")
                        .collect(engine="streaming")
                        .to_series()
                        .to_list()
                    )
                else:
                    metadata_index_order.extend(
                        vdjs_[i]
                        ._metadata.select("cell_id")
                        .to_series()
                        .to_list()
                    )

                if isinstance(vdjs_[i]._data, pl.LazyFrame):
                    data_index_order.extend(
                        vdjs_[i]
                        ._data.select("sequence_id")
                        .collect(engine="streaming")
                        .to_series()
                        .to_list()
                    )
                else:
                    data_index_order.extend(
                        vdjs_[i]
                        ._data.select("sequence_id")
                        .to_series()
                        .to_list()
                    )

        elif data_index_order is None:
            data_index_order = []
            for i in range(0, len(vdjs_)):
                if (suffixes is None) and (prefixes is None):
                    vdjs_[i].add_sequence_suffix(
                        str(i),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif suffixes is not None:
                    vdjs_[i].add_sequence_suffix(
                        str(suffixes[i]),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif prefixes is not None:
                    vdjs_[i].add_sequence_prefix(
                        str(prefixes[i]),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                if isinstance(vdjs_[i]._data, pl.LazyFrame):
                    data_index_order.extend(
                        vdjs_[i]
                        ._data.select("sequence_id")
                        .collect(engine="streaming")
                        .to_series()
                        .to_list()
                    )
                else:
                    data_index_order.extend(
                        vdjs_[i]
                        ._data.select("sequence_id")
                        .to_series()
                        .to_list()
                    )
    else:
        if metadata_index_order is None or data_index_order is None:
            raise ValueError(
                "Cell/contig indices are not unique. Please set check_unique=True to append suffixes/prefixes or ensure unique indices before concatenation."
            )

    # Collect all metadata column names
    all_meta_cols = set()
    for vdj_ in vdjs_:
        if isinstance(vdj_._metadata, pl.LazyFrame):
            all_meta_cols.update(vdj_._metadata.collect_schema().names())
        else:
            all_meta_cols.update(vdj_._metadata.columns)

    # Handle v_call_genotyped consistency
    genotyped_v_call = [
        True
        for vdj in vdjs_
        if "v_call_genotyped"
        in (
            vdj._data.collect_schema().names()
            if isinstance(vdj._data, pl.LazyFrame)
            else vdj._data.columns
        )
    ]
    if len(genotyped_v_call) > 0:
        if len(genotyped_v_call) != len(vdjs_):
            if verbose:
                logg.info(
                    "For consistency, 'v_call_genotyped' will be used where available. Filling missing values from 'v_call'."
                )
            for i in range(0, len(vdjs_)):
                data_cols = (
                    vdjs_[i]._data.collect_schema().names()
                    if isinstance(vdjs_[i]._data, pl.LazyFrame)
                    else vdjs_[i]._data.columns
                )
                if "v_call_genotyped" not in data_cols:
                    vdjs_[i]._data = vdjs_[i]._data.with_columns(
                        pl.col("v_call").alias("v_call_genotyped")
                    )

    # Concatenate the data (Polars DataFrames)
    arrays_ = [vdj._data for vdj in vdjs_]
    vdj_concat = DandelionPolars(
        pl.concat(arrays_, how="diagonal"), verbose=False
    )

    # Handle missing metadata columns
    vdj_meta_cols = set(
        vdj_concat._metadata.collect_schema().names()
        if isinstance(vdj_concat._metadata, pl.LazyFrame)
        else vdj_concat._metadata.columns
    )
    missing_meta_cols = all_meta_cols - vdj_meta_cols

    if len(missing_meta_cols) > 0:
        # Collect metadata if lazy for easier manipulation
        if isinstance(vdj_concat._metadata, pl.LazyFrame):
            meta_df = vdj_concat._metadata.collect(engine="streaming")
        else:
            meta_df = vdj_concat._metadata.clone()

        # Add missing columns with nulls
        for col in missing_meta_cols:
            meta_df = meta_df.with_columns(pl.lit(None).alias(col))

        # Fill in values from original dataframes
        for vdj_ in vdjs_:
            # Collect if lazy
            if isinstance(vdj_._metadata, pl.LazyFrame):
                source_meta = vdj_._metadata.collect(engine="streaming")
            else:
                source_meta = vdj_._metadata

            for col in missing_meta_cols:
                source_cols = source_meta.columns
                if col in source_cols:
                    # Create a mapping from cell_id to values
                    mapping = source_meta.select(["cell_id", col])

                    # Join to update values
                    meta_df = (
                        meta_df.join(
                            mapping.rename({col: f"_temp_{col}"}),
                            on="cell_id",
                            how="left",
                        )
                        .with_columns(
                            pl.coalesce(
                                pl.col(f"_temp_{col}"), pl.col(col)
                            ).alias(col)
                        )
                        .drop(f"_temp_{col}")
                    )

        vdj_concat._metadata = meta_df

    # Reorder metadata according to original order
    if isinstance(vdj_concat._metadata, pl.LazyFrame):
        concat_meta = vdj_concat._metadata.collect(engine="streaming")
    else:
        concat_meta = vdj_concat._metadata

    # Create a mapping dataframe with desired order
    order_df = pl.DataFrame(
        {
            "cell_id": metadata_index_order,
            "_original_order": range(len(metadata_index_order)),
        }
    )

    # Join to add order column, then sort and drop
    reordered_meta = (
        concat_meta.join(order_df, on="cell_id", how="inner")
        .sort("_original_order")
        .drop("_original_order")
    )

    vdj_concat._metadata = (
        reordered_meta.lazy() if vdj_concat.lazy else reordered_meta
    )

    return vdj_concat


def group_pairwise_hamming_distance_polars(
    clonotype_vj_len_group: Tree,
    clonotype_sequence_group: Tree,
    identity: float,
    locus: str,
    chain: Literal["VDJ", "VJ"],
    junction_key: str,
    verbose: bool = True,
) -> Tree:
    """
    Group clonotypes by pairwise hamming distance using polars-native approach.

    Computes hamming distance matrix for each (V, J, length) group and
    clusters sequences using a threshold-based algorithm.

    Parameters
    ----------
    clonotype_vj_len_group : Tree
        Tree grouping contigs by V/J and length.
    clonotype_sequence_group : Tree
        Tree grouping sequences by V/J and length.
    identity : float
        Identity threshold (0-1).
    locus : str
        Locus name.
    chain : Literal["VDJ", "VJ"]
        Chain type.
    junction_key : str
        Junction column name.
    verbose : bool, optional
        Show progress bar.

    Returns
    -------
    Tree
        Tree of contigs assigned to clonotypes.
    """
    clones = Tree()

    for g in tqdm(
        clonotype_sequence_group,
        desc=f"Finding clones based on {locus} cell {chain} chains using {junction_key}",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        for l in clonotype_sequence_group[g]:
            seq_list = list(clonotype_sequence_group[g][l].keys())

            if len(seq_list) == 0:
                continue

            # Vectorized hamming distance: convert sequences to numpy array
            seq_array = np.array(seq_list).reshape(-1, 1)
            d_mat = squareform(
                pdist(seq_array, lambda x, y: hamming(x[0], y[0]))
            )

            # Threshold based on identity
            tr = math.floor(int(l) * (1 - identity))

            # Convert to lower triangle
            d_mat = np.tril(d_mat)
            np.fill_diagonal(d_mat, 0)

            # Get distance dict from matrix
            source, target = d_mat.nonzero()
            source_target = list(zip(source.tolist(), target.tolist()))

            if len(source_target) == 0:
                source_target = [(0, 0)]

            dist = {st: d_mat[st] for st in source_target}

            # Cluster
            if d_mat.shape[0] > 1:
                seq_tmp_dict = _clustering_polars(dist, tr, seq_list)
            else:
                seq_tmp_dict = {seq_list[0]: (seq_list[0],)}

            # Sort by size
            clones_tmp = sorted(
                list(set(seq_tmp_dict.values())), key=len, reverse=True
            )

            for x, clone_group in enumerate(clones_tmp, 1):
                clones[g][l][x] = clone_group

    # Map contigs to clonotypes
    cid = Tree()
    for g in clones:
        for l in clones[g]:
            for c in clones[g][l]:
                grp_seq = clones[g][l][c]
                for key, value in clonotype_vj_len_group[g][l].items():
                    if value in grp_seq:
                        cid[g][l][c][key] = value

    return cid


def rename_clonotype_ids_polars(
    clonotype_groups: Tree,
    prefix: str = "",
) -> dict[str, str]:
    """
    Rename clonotype IDs using vectorized polars-compatible approach.

    Maps contigs to hierarchical clone IDs of format:
    {prefix}{g}_{l}_{c} where g, l, c are sequential indices.

    Parameters
    ----------
    clonotype_groups : Tree
        Tree of clonotype groups.
    prefix : str, optional
        Prefix for clone IDs.

    Returns
    -------
    dict[str, str]
        Mapping from contig ID to clone ID.
    """
    clone_dict = {}

    # Get unique keys at each level
    first_keys = sorted(list(set(clonotype_groups.keys())))
    first_key_dict = {k: i for i, k in enumerate(first_keys, 1)}

    for g in clonotype_groups:
        second_keys = sorted(list(set(clonotype_groups[g].keys())))
        second_key_dict = {k: i for i, k in enumerate(second_keys, 1)}

        for l in clonotype_groups[g]:
            third_keys = sorted(list(set(clonotype_groups[g][l].keys())))
            third_key_dict = {k: i for i, k in enumerate(third_keys, 1)}

            for key, value in dict(clonotype_groups[g][l]).items():
                for v in value:
                    if not isinstance(v, int):
                        clone_dict[v] = (
                            f"{prefix}"
                            f"{first_key_dict[g]}_"
                            f"{second_key_dict[l]}_"
                            f"{third_key_dict[key]}"
                        )

    return clone_dict


def refine_clone_assignment_polars(
    df: pl.DataFrame | pl.LazyFrame,
    clone_key: str,
    clone_dict_vj: dict,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Refine clone assignment based on VJ chain pairing (polars-native).

    For each cell, integrates VDJ and VJ clone assignments by pairing
    chains based on cell-level logic.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input AIRR dataframe with VDJ clones assigned.
    clone_key : str
        Column name for clone IDs.
    clone_dict_vj : dict
        Mapping from sequence ID to VJ clone ID.
    verbose : bool, optional
        Show progress.

    Returns
    -------
    pl.DataFrame
        DataFrame with refined clone assignments.
    """
    # Collect if lazy
    if isinstance(df, pl.LazyFrame):
        df = df.collect(engine="streaming")

    # Add row index to maintain order
    df = df.with_row_index("_original_order")

    # Build cell->clone tree and seq->cell tree
    cellclonetree = Tree()
    seqcellclonetree = Tree()

    for row in df.iter_rows(named=True):
        cell_id = row["cell_id"]
        seq_id = row["sequence_id"]
        clone_id = row[clone_key]

        seqcellclonetree[cell_id][seq_id] = seq_id
        if clone_id and clone_id != "" and clone_id is not None:
            if clone_id not in cellclonetree[cell_id]:
                cellclonetree[cell_id][clone_id] = clone_id

    # Convert to lists
    for c in cellclonetree:
        cellclonetree[c] = list(cellclonetree[c])

    # Refine assignments
    fintree = Tree()
    for c in tqdm(
        cellclonetree,
        desc="Refining clone assignment based on VJ chain pairing ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        suffix = [
            clone_dict_vj[x] for x in seqcellclonetree[c] if x in clone_dict_vj
        ]
        fintree[c] = []

        if len(suffix) > 0:
            for cl in cellclonetree[c]:
                if present(cl):
                    # Check if cl is already a VJ-only clone that's in suffix
                    # In that case, don't combine it with itself
                    if cl in suffix:
                        fintree[c].append(cl)
                        continue

                    # Check if any suffix matches celltype - if so, use combined ID only
                    matched = False
                    for s in suffix:
                        if check_same_celltype(cl, s):
                            fintree[c].append(
                                cl + "_" + "".join(s.split("_", 1)[1])
                            )
                            matched = True
                            break  # Only add once per VDJ clone
                    if not matched:
                        # No matching suffix, add combinations
                        for s in suffix:
                            fintree[c].append(cl + "|" + s)
        else:
            for cl in cellclonetree[c]:
                if present(cl):
                    fintree[c].append(cl)

        # Join all alternatives with "|" (same as pandas version)
        if fintree[c]:
            fintree[c] = "|".join(sorted(fintree[c]))
        else:
            fintree[c] = ""

    # Create mapping from cell_id to refined clone
    cell_clone_map = {c: fintree[c] for c in fintree}

    # Update dataframe with refined clones using map_elements
    df = df.with_columns(
        pl.col("cell_id")
        .map_elements(
            lambda x: cell_clone_map.get(x, ""), return_dtype=pl.String
        )
        .alias(clone_key)
    )

    # Restore original order and drop the tracking column
    df = df.sort("_original_order").drop("_original_order")

    return df


def clone_size_polars(
    vdj,  # DandelionPolars type
    groupby: str | None = None,
    max_size: int | None = None,
    clone_key: str | None = None,
    key_added: str | None = None,
) -> None:
    """
    Quantify clone sizes, globally or per group using polars.

    If `groupby` is specified, clone sizes and proportions are calculated
    within each group separately. Each cell is then annotated with the size,
    proportion, and frequency category based on sizes similar to scRepertoire.
    If a cell belongs to multiple clones (e.g., multiple chains assigned
    to different clones), the largest clone is used for annotation.

    Parameters
    ----------
    vdj : DandelionPolars
        VDJ data.
    groupby : str | None, optional
        Column in metadata to group by before calculating clone sizes.
        If None, calculates global clone sizes.
    max_size : int | None, optional
        Clip clone size values at this maximum.
    clone_key : str | None, optional
        Column specifying clone identifiers. Defaults to 'clone_id'.
    key_added : str | None, optional
        Prefix for new metadata column names.
    """
    # Get metadata
    metadata_ = vdj._metadata
    if isinstance(metadata_, pl.LazyFrame):
        metadata_ = metadata_.collect(engine="streaming")

    clone_key = "clone_id" if clone_key is None else clone_key
    if clone_key not in metadata_.columns:
        raise KeyError(f"Column '{clone_key}' not found in metadata.")

    # Define empty values
    empty_vals = ["No_contig", "unassigned", "None", "nan", "", None]

    # Expand multi-clone entries using polars
    tmp = (
        metadata_.with_row_count("_row_idx")
        .select(
            [
                pl.col("_row_idx"),
                pl.col(clone_key).cast(pl.String).alias("_clone_orig"),
            ]
        )
        .with_columns(
            [pl.col("_clone_orig").str.split("|").alias("_clones_list")]
        )
        .explode("_clones_list")
        .select([pl.col("_row_idx"), pl.col("_clones_list").alias(clone_key)])
        .filter(
            ~pl.col(clone_key).is_in(empty_vals)
            & pl.col(clone_key).is_not_null()
        )
    )

    # Compute clone sizes (global or per group)
    if groupby is None:
        # Global clone sizes
        clonesize = (
            tmp.group_by(clone_key)
            .agg(pl.count().alias("size"))
            .sort("size", descending=True)
        )

        total_cells = metadata_.height
        clonesize = clonesize.with_columns(
            (pl.col("size") / total_cells).alias("proportion")
        )
    else:
        # Per-group clone sizes
        # Add groupby column to tmp
        groupby_col = metadata_.with_row_count("_row_idx").select(
            [pl.col("_row_idx"), pl.col(groupby)]
        )

        tmp = tmp.join(groupby_col, on="_row_idx", how="left")

        # Calculate clone sizes per group
        clonesize = tmp.group_by([groupby, clone_key]).agg(
            pl.count().alias("size")
        )

        # Calculate group totals
        group_sizes = metadata_.group_by(groupby).agg(
            pl.count().alias("group_total")
        )

        # Join and calculate proportions
        clonesize = clonesize.join(
            group_sizes, on=groupby, how="left"
        ).with_columns(
            (pl.col("size") / pl.col("group_total")).alias("proportion")
        )

    # Create max_size categories if specified
    if max_size is not None:
        clonesize = clonesize.with_columns(
            [
                pl.when(pl.col("size") < max_size)
                .then(pl.col("size").cast(pl.String))
                .otherwise(pl.lit(f">= {max_size}"))
                .alias("size_category")
            ]
        )

    # Define clone frequency bins
    bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    labels = ["Rare", "Small", "Medium", "Large", "Hyperexpanded"]

    clonesize = clonesize.with_columns(
        [
            pl.when(pl.col("proportion") <= 0.0001)
            .then(pl.lit("Rare"))
            .when(pl.col("proportion") <= 0.001)
            .then(pl.lit("Small"))
            .when(pl.col("proportion") <= 0.01)
            .then(pl.lit("Medium"))
            .when(pl.col("proportion") <= 0.1)
            .then(pl.lit("Large"))
            .otherwise(pl.lit("Hyperexpanded"))
            .alias("frequency_category")
        ]
    )

    # Now map back to cells - for each cell, find the largest clone
    if groupby is None:
        # Create lookup - just use clone_key
        size_map = dict(zip(clonesize[clone_key], clonesize["size"]))
        prop_map = dict(zip(clonesize[clone_key], clonesize["proportion"]))
        cat_map = dict(
            zip(clonesize[clone_key], clonesize["frequency_category"])
        )
        if max_size is not None:
            sizecat_map = dict(
                zip(clonesize[clone_key], clonesize["size_category"])
            )
    else:
        # Create lookup with (group, clone) tuples
        size_map = {
            (row[groupby], row[clone_key]): row["size"]
            for row in clonesize.iter_rows(named=True)
        }
        prop_map = {
            (row[groupby], row[clone_key]): row["proportion"]
            for row in clonesize.iter_rows(named=True)
        }
        cat_map = {
            (row[groupby], row[clone_key]): row["frequency_category"]
            for row in clonesize.iter_rows(named=True)
        }
        if max_size is not None:
            sizecat_map = {
                (row[groupby], row[clone_key]): row["size_category"]
                for row in clonesize.iter_rows(named=True)
            }

    # Process each cell to find largest clone
    cell_sizes = []
    cell_props = []
    cell_cats = []
    cell_size_cats = [] if max_size is not None else None

    for row in metadata_.iter_rows(named=True):
        clone_ids = (
            str(row[clone_key]) if row[clone_key] is not None else "None"
        )

        # Check for empty/invalid entries
        if clone_ids in empty_vals + ["None"]:
            cell_sizes.append(None)
            cell_props.append(None)
            cell_cats.append(None)
            if max_size is not None:
                cell_size_cats.append(None)
            continue

        clones = clone_ids.split("|")

        if groupby is None:
            sizes = [size_map.get(c, None) for c in clones]
            props = [prop_map.get(c, None) for c in clones]
            cats = [cat_map.get(c, None) for c in clones]
            if max_size is not None:
                size_cats = [sizecat_map.get(c, None) for c in clones]
        else:
            grp = row[groupby]
            sizes = [size_map.get((grp, c), None) for c in clones]
            props = [prop_map.get((grp, c), None) for c in clones]
            cats = [cat_map.get((grp, c), None) for c in clones]
            if max_size is not None:
                size_cats = [sizecat_map.get((grp, c), None) for c in clones]

        # Take the largest clone
        valid_sizes = [s for s in sizes if s is not None]
        if len(valid_sizes) == 0:
            cell_sizes.append(None)
            cell_props.append(None)
            cell_cats.append(None)
            if max_size is not None:
                cell_size_cats.append(None)
        else:
            max_idx = sizes.index(max(valid_sizes))
            cell_sizes.append(sizes[max_idx])
            cell_props.append(props[max_idx])
            cell_cats.append(cats[max_idx])
            if max_size is not None:
                cell_size_cats.append(size_cats[max_idx])

    # Add columns to metadata
    col_key = key_added if key_added is not None else clone_key

    metadata_ = metadata_.with_columns(
        [
            pl.Series(f"{col_key}_size", cell_sizes),
            pl.Series(f"{col_key}_size_prop", cell_props),
            pl.Series(f"{col_key}_size_category", cell_cats),
        ]
    )

    if max_size is not None:
        metadata_ = metadata_.with_columns(
            [pl.Series(f"{col_key}_size_max_{max_size}", cell_size_cats)]
        )

    # Update metadata
    vdj._metadata = metadata_ if not vdj.lazy else metadata_.lazy()


def clone_overlap_polars(
    vdj,  # DandelionPolars type
    groupby: str,
    min_clone_size: int | None = None,
    weighted_overlap: bool = False,
    clone_key: str | None = None,
) -> pl.DataFrame:
    """
    Tabulate clonal overlap for circos-style plots using polars.

    Parameters
    ----------
    vdj : DandelionPolars
        DandelionPolars object.
    groupby : str
        Column name in metadata for collapsing to columns in the clone_id x groupby data frame.
    min_clone_size : int | None, optional
        Minimum size of clone for plotting connections. Defaults to 2 if left as None.
    weighted_overlap : bool, optional
        If True, instead of collapsing to overlap to binary, overlap will be returned as the number of cells.
    clone_key : str | None, optional
        Column name for clones. `None` defaults to 'clone_id'.

    Returns
    -------
    pl.DataFrame
        clone_id x groupby overlap DataFrame.

    Raises
    ------
    ValueError
        If min_clone_size is 0.
    """
    # Get metadata
    metadata_ = vdj._metadata
    if isinstance(metadata_, pl.LazyFrame):
        metadata_ = metadata_.collect(engine="streaming")

    if min_clone_size is None:
        min_size = 2
    else:
        min_size = int(min_clone_size)

    if min_size == 0:
        raise ValueError("min_size must be greater than 0.")

    clone_ = "clone_id" if clone_key is None else clone_key

    # Get all groups
    allgroups = metadata_[groupby].unique().to_list()

    # Filter out invalid clones
    empty_vals = ["nan", "NaN", "No_contig", "unassigned", "None", None]
    data = metadata_.filter(
        ~pl.col(clone_).is_in(empty_vals) & pl.col(clone_).is_not_null()
    )

    # Expand multi-clone entries
    datc_ = (
        data.with_row_count("_idx")
        .select(
            [
                pl.col("_idx"),
                pl.col(clone_).cast(pl.String).alias("_clone_orig"),
                pl.col(groupby),
            ]
        )
        .with_columns(
            [pl.col("_clone_orig").str.split("|").alias("_clones_list")]
        )
        .explode("_clones_list")
        .select(
            [
                pl.col("_idx"),
                pl.col("_clones_list").alias(clone_),
                pl.col(groupby),
            ]
        )
        .filter(
            ~pl.col(clone_).is_in([""] + empty_vals)
            & pl.col(clone_).is_not_null()
        )
    )

    # Create crosstab: count occurrences of each (clone, group) pair
    overlap = (
        datc_.group_by([clone_, groupby])
        .agg(pl.count().alias("count"))
        .pivot(
            values="count",
            index=clone_,
            columns=groupby,
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    # Ensure all groups are present
    for grp in allgroups:
        if grp not in overlap.columns:
            overlap = overlap.with_columns(pl.lit(0).alias(str(grp)))

    # Apply min_size filtering
    if not weighted_overlap:
        # Convert to binary based on min_size
        for col in overlap.columns:
            if col != clone_:
                if min_size > 2:
                    overlap = overlap.with_columns(
                        [
                            pl.when(pl.col(col) < min_size)
                            .then(0)
                            .when(pl.col(col) >= min_size)
                            .then(1)
                            .otherwise(pl.col(col))
                            .alias(col)
                        ]
                    )
                elif min_size == 2:
                    overlap = overlap.with_columns(
                        [
                            pl.when(pl.col(col) >= min_size)
                            .then(1)
                            .otherwise(pl.col(col))
                            .alias(col)
                        ]
                    )

    return overlap


def productive_ratio(
    adata: AnnData,
    vdj: DandelionPolars,
    groupby: str,
    groups: list[str] | None = None,
    locus: Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"] = "TRB",
):
    """
    Compute the cell-level productive/non-productive contig ratio.

    Only the contig with the highest umi count in a cell will be used for this
    tabulation.

    Returns inplace AnnData with `.uns['productive_ratio']`.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata (`.obs`).
    vdj : DandelionPolars
        DandelionPolars object holding the repertoire data (`.data`).
    groupby : str
        Name of column in `AnnData.obs` to return the row tabulations.
    groups : list[str] | None, optional
        Optional list of categories to return.
    locus : Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"], optional
        One of the accepted locuses to perform the tabulation
    """
    start = logg.info("Tabulating productive ratio")

    # Filter to cells present in adata
    vdjx = vdj[vdj._metadata["cell_id"].is_in(list(adata.obs_names))]

    # Get data, handling lazy frames
    if isinstance(vdjx._data, pl.LazyFrame):
        data = vdjx._data.collect(engine="streaming")
    else:
        data = vdjx._data

    # Filter by locus and ambiguous status
    if "ambiguous" in data.columns:
        data_filtered = data.filter(
            (pl.col("locus") == locus) & (pl.col("ambiguous").is_in(FALSES))
        )
    else:
        data_filtered = data.filter(pl.col("locus") == locus)

    # Drop duplicates on cell_id, keeping first (highest umi due to sorting)
    df_unique = data_filtered.unique(subset=["cell_id"], keep="first")

    # Create mapping of cell_id to productive status
    dict_df = dict(zip(df_unique["cell_id"], df_unique["productive"]))

    # Determine groups if not provided
    if groups is None:
        if is_categorical(adata.obs[groupby]):
            groups = list(adata.obs[groupby].cat.categories)
        else:
            groups = list(set(adata.obs[groupby]))

    # Initialize results DataFrame
    res = pd.DataFrame(
        columns=["productive", "non-productive", "total"],
        index=groups,
    )

    # Add productive status to adata
    adata.obs[locus + "_productive"] = pd.Series(dict_df)

    # Calculate ratios per group
    for i in range(res.shape[0]):
        cell = res.index[i]
        res.loc[cell, "total"] = sum(adata.obs[groupby] == cell)
        if res.loc[cell, "total"] > 0:
            res.loc[cell, "productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(["T"])
                )
                / res.loc[cell, "total"]
                * 100
            )
            res.loc[cell, "non-productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(["F"])
                )
                / res.loc[cell, "total"]
                * 100
            )

    res[groupby] = res.index
    res["productive+non-productive"] = res["productive"] + res["non-productive"]
    out = {"results": res, "locus": locus, "groupby": groupby}
    adata.uns["productive_ratio"] = out
    logg.info(
        " finished",
        time=start,
        deep=("Updated AnnData: \n" "   'uns', productive_ratio"),
    )
