from __future__ import annotations

import functools
import math

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc

from anndata import AnnData
from collections import defaultdict, Counter
from contextlib import contextmanager
from scanpy import logging as logg
from scipy.sparse import coo_matrix, csr_matrix, eye as speye
from scipy.sparse.csgraph import connected_components

from tqdm import tqdm
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mudata import MuData
    from awkward import Array

from dandelion.polars.core._core import DandelionPolars, SCHEMA_OVERRIDES
from dandelion.utilities._utilities import (
    VCALL,
    TRUES_STR,
    FALSES_STR,
    FALSES,
    JCALL,
    VCALLG,
    STRIPALLELENUM,
    EMPTIES,
    is_categorical,
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
    store_distances: bool = True,
    group_by: str | None = None,
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
    store_distances : bool, optional
        whether or not to store the distance matrix as a sparse matrix in `vdj.distances`. Default is True.
    group_by : str | None, optional
        Column name to partition cells by before clonal clustering. When provided, clones are found
        independently within each group (e.g. per-sample), and cells/contigs in different groups can
        never be assigned to the same clone, even if their junction sequences are identical. The column
        must exist in either `vdj.data` or `vdj.metadata`; if it is only present in one, it will be
        propagated to the other internally (`vdj.update_data()` if only in `.metadata`, or
        `vdj.update_metadata(retrieve=group_by, unique=True, split=False)` if only in `.data`).
        `None` disables grouping and clones are found across all cells as before.
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

    # If grouping, make sure the group_by column is available on both
    # `.data` and `.metadata` before we do anything else, since the
    # contig-level clustering below reads from `.data`.
    if group_by is not None:
        data_schema = (
            vdj._data.collect_schema()
            if isinstance(vdj._data, pl.LazyFrame)
            else vdj._data.schema
        )
        metadata_schema = (
            vdj._metadata.collect_schema()
            if isinstance(vdj._metadata, pl.LazyFrame)
            else vdj._metadata.schema
        )
        in_data = group_by in data_schema
        in_metadata = group_by in metadata_schema
        if not in_data and not in_metadata:
            raise ValueError(
                f"`group_by` column '{group_by}' not found in either "
                "`.data` or `.metadata`."
            )
        if in_metadata and not in_data:
            # Only in metadata - propagate down into .data
            vdj.update_data()
        elif in_data and not in_metadata:
            # Only in .data - propagate up into .metadata
            vdj.update_metadata(retrieve=group_by, unique=True, split=False)

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
    locus_to_col = {
        "IGH": key["ig"],
        "IGK": key["ig"],
        "IGL": key["ig"],
        "TRB": key["tr-ab"],
        "TRA": key["tr-ab"],
        "TRD": key["tr-gd"],
        "TRG": key["tr-gd"],
    }
    # Default key_added
    key_added = "clone_id" if key_added is None else key_added
    # Initialize clone column
    df = df.with_columns(pl.lit("").alias(key_added))
    # Also initialise the original order column
    df = df.with_row_index("_original_order")

    # Store results from each locus
    locus_results = defaultdict(list)

    # Initialize distance storage based on backend choice
    distance_results = []
    if store_distances:
        # Memory-based mode (original approach)
        metadata_for_mapping = vdj._metadata
        if isinstance(metadata_for_mapping, pl.LazyFrame):
            metadata_for_mapping = metadata_for_mapping.collect(
                engine="streaming"
            )
        all_cell_ids = metadata_for_mapping["cell_id"].to_list()
        cell_id_to_meta_idx = {
            cell_id: i for i, cell_id in enumerate(all_cell_ids)
        }
        n_cells = len(all_cell_ids)

    # Determine the groups to iterate over. When `group_by` is None, we
    # process everything in a single pass (group_val stays None and no
    # filtering / prefixing happens), preserving the original behaviour.
    if group_by is not None:
        group_values = (
            df.select(pl.col(group_by)).unique().to_series().to_list()
        )
    else:
        group_values = [None]

    for group_val in group_values:
        if group_by is not None:
            df_group = df.filter(pl.col(group_by) == group_val)
            # Prefix clone ids with the group value so that clones found
            # independently within different groups never collide/merge
            # when the per-group results are combined back together.
            group_prefix = f"{group_val}_"
        else:
            df_group = df
            group_prefix = ""

        # Process each locus
        for locus, (vdj_loci, vj_loci) in locus_dict.items():
            # Filter to this locus (and, if group_by is set, to this group)
            df_locus = df_group.filter(
                pl.col("locus").is_in(vdj_loci + vj_loci)
            )
            if "ambiguous" in df_locus.collect_schema():
                df_locus = df_locus.filter(
                    ~pl.col("ambiguous").is_in(TRUES_STR)
                )

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
                # Collect both sequences and their original order for proper tracking
                agg_cols = [
                    pl.col(locus_key),
                    pl.col("_original_order"),
                    pl.col("cell_id"),
                ]
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
                # Also track which original rows belong to which clone
                row_to_clone = {}

                for row in tqdm(
                    grouped.iter_rows(named=True),
                    desc=f"Finding clones based on {locus_celltype} cell {vdj_chain} chains using {locus_key}",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    total=df_vdj_grp.select("_membership").n_unique(),
                    disable=not verbose,
                ):
                    seqs = row[locus_key]
                    orig_orders = row["_original_order"]
                    cell_ids = row["cell_id"]
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

                    # Filter out empty strings for distance calculation only
                    seqs_non_empty = [s for s in seqs if s]
                    seqs_empty = [s for s in seqs if not s]
                    # Also filter cell_ids to match seqs_non_empty
                    cell_ids_non_empty = [
                        cell_ids[i] for i, s in enumerate(seqs) if s
                    ]

                    if seqs_non_empty:
                        # Deduplicate sequences for faster computation
                        unique_seqs = list(set(seqs_non_empty))
                        seq_to_unique_idx = {
                            seq: i for i, seq in enumerate(unique_seqs)
                        }

                        # Compute distances only for unique sequences
                        d_mat_unique = metric.compute_vectorized(unique_seqs)

                        # Filter to only cells that are in metadata (ensure alignment)
                        if store_distances:
                            valid_mask = [
                                cid in cell_id_to_meta_idx
                                for cid in cell_ids_non_empty
                            ]
                            seqs_filtered = [
                                s
                                for s, valid in zip(seqs_non_empty, valid_mask)
                                if valid
                            ]
                            meta_indices = np.array(
                                [
                                    cell_id_to_meta_idx[cid]
                                    for cid, valid in zip(
                                        cell_ids_non_empty, valid_mask
                                    )
                                    if valid
                                ]
                            )

                            if len(meta_indices) > 0 and d_mat_unique.size > 0:
                                # Use vectorized indexing to expand unique distances to full matrix
                                unique_indices = np.array(
                                    [
                                        seq_to_unique_idx[seq]
                                        for seq in seqs_filtered
                                    ]
                                )
                                d_mat = d_mat_unique[
                                    np.ix_(unique_indices, unique_indices)
                                ]
                                # Collect COO components for in-memory assembly
                                n_local = len(meta_indices)
                                rows, cols = np.meshgrid(
                                    range(n_local),
                                    range(n_local),
                                    indexing="ij",
                                )
                                rows_flat = rows.ravel()
                                cols_flat = cols.ravel()
                                data_flat = d_mat.ravel()
                                global_rows = meta_indices[rows_flat]
                                global_cols = meta_indices[cols_flat]
                                distance_results.append(
                                    (global_rows, global_cols, data_flat)
                                )

                        # Cluster on unique sequences only, then map back
                        if len(unique_seqs) > 1:
                            seq_tmp_dict = _clustering_scipy(
                                d_mat_unique,
                                threshold=threshold,
                                sequences=unique_seqs,
                                hard_threshold=hard_cutoff,
                            )
                        else:
                            seq_tmp_dict = {unique_seqs[0]: (unique_seqs[0],)}
                    else:
                        # All sequences in this membership are empty - assign them together
                        if seqs_empty:
                            seq_tmp_dict = {seqs_empty[0]: tuple(seqs_empty)}
                        else:
                            seq_tmp_dict = {}

                    # Sort by size
                    clones_tmp = sorted(
                        list(set(seq_tmp_dict.values())), key=len, reverse=True
                    )
                    for sub_group, clone_group in enumerate(clones_tmp, 1):
                        clones_vdj[membership][sub_group] = clone_group
                        # Map each original row to its clone based on its sequence
                        for seq_idx, seq_val in enumerate(seqs):
                            if seq_val and seq_val in clone_group:
                                row_to_clone[orig_orders[seq_idx]] = (
                                    f"{membership}_{sub_group}"
                                )

                # Apply clone IDs using row mapping instead of sequence mapping
                df_vdj_grp = df_vdj_grp.with_columns(
                    pl.col("_original_order")
                    .replace_strict(row_to_clone, default=None)
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
                # Collect both sequences and their original order for proper tracking
                agg_cols = [
                    pl.col(locus_key),
                    pl.col("_original_order"),
                    pl.col("cell_id"),
                ]
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
                # Also track which original rows belong to which clone
                row_to_clone = {}

                for row in tqdm(
                    grouped.iter_rows(named=True),
                    desc=f"Finding clones based on {locus_celltype} cell {vj_chain} chains using {locus_key}",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    total=grouped.height,
                    disable=not verbose,
                ):
                    seqs = row[locus_key]
                    orig_orders = row["_original_order"]
                    cell_ids = row["cell_id"]
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

                    # Filter out empty strings for distance calculation only
                    seqs_non_empty = [s for s in seqs if s]
                    seqs_empty = [s for s in seqs if not s]
                    # Also filter cell_ids to match seqs_non_empty
                    cell_ids_non_empty = [
                        cell_ids[i] for i, s in enumerate(seqs) if s
                    ]

                    if seqs_non_empty:
                        # Deduplicate sequences for faster computation
                        unique_seqs = list(set(seqs_non_empty))
                        seq_to_unique_idx = {
                            seq: i for i, seq in enumerate(unique_seqs)
                        }

                        # Compute distances only for unique sequences
                        d_mat_unique = metric.compute_vectorized(unique_seqs)

                        # Filter to only cells that are in metadata (ensure alignment)
                        if store_distances:
                            valid_mask = [
                                cid in cell_id_to_meta_idx
                                for cid in cell_ids_non_empty
                            ]
                            seqs_filtered = [
                                s
                                for s, valid in zip(seqs_non_empty, valid_mask)
                                if valid
                            ]
                            meta_indices = np.array(
                                [
                                    cell_id_to_meta_idx[cid]
                                    for cid, valid in zip(
                                        cell_ids_non_empty, valid_mask
                                    )
                                    if valid
                                ]
                            )

                            if len(meta_indices) > 0 and d_mat_unique.size > 0:
                                # Use vectorized indexing to expand unique distances to full matrix
                                unique_indices = np.array(
                                    [
                                        seq_to_unique_idx[seq]
                                        for seq in seqs_filtered
                                    ]
                                )
                                d_mat = d_mat_unique[
                                    np.ix_(unique_indices, unique_indices)
                                ]

                                # Collect COO components for in-memory assembly
                                n_local = len(meta_indices)
                                rows, cols = np.meshgrid(
                                    range(n_local),
                                    range(n_local),
                                    indexing="ij",
                                )
                                rows_flat = rows.ravel()
                                cols_flat = cols.ravel()
                                data_flat = d_mat.ravel()
                                global_rows = meta_indices[rows_flat]
                                global_cols = meta_indices[cols_flat]
                                distance_results.append(
                                    (global_rows, global_cols, data_flat)
                                )

                        # Cluster on unique sequences only, then map back
                        if len(unique_seqs) > 1:
                            seq_tmp_dict = _clustering_scipy(
                                d_mat_unique,
                                threshold=threshold,
                                sequences=unique_seqs,
                                hard_threshold=hard_cutoff,
                            )
                        else:
                            seq_tmp_dict = {unique_seqs[0]: (unique_seqs[0],)}
                    else:
                        # All sequences in this membership are empty - assign them together
                        if seqs_empty:
                            seq_tmp_dict = {seqs_empty[0]: tuple(seqs_empty)}
                        else:
                            seq_tmp_dict = {}

                    # Sort by size
                    clones_tmp = sorted(
                        list(set(seq_tmp_dict.values())), key=len, reverse=True
                    )
                    for sub_group, clone_group in enumerate(clones_tmp, 1):
                        clones_vj[membership][sub_group] = clone_group
                        # Map each original row to its clone based on its sequence
                        for seq_idx, seq_val in enumerate(seqs):
                            if seq_val and seq_val in clone_group:
                                row_to_clone[orig_orders[seq_idx]] = (
                                    f"{membership}_{sub_group}"
                                )

                # Apply clone IDs using row mapping instead of sequence mapping
                df_vj_grp = df_vj_grp.with_columns(
                    pl.col("_original_order")
                    .replace_strict(row_to_clone, default=None)
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
                    return_dtype=pl.List(pl.String),
                )
                .alias(f"_{key_added}_{locus_celltype}")
            )

            # If grouping, prefix the clone ids with the group value so
            # that identically-numbered clones from different groups are
            # never conflated once results are concatenated below.
            if group_by is not None:
                df_locus_summary = df_locus_summary.with_columns(
                    pl.col(f"_{key_added}_{locus_celltype}").map_elements(
                        lambda lst, _prefix=group_prefix: (
                            [f"{_prefix}{x}" for x in lst]
                            if lst is not None
                            else lst
                        ),
                        return_dtype=pl.List(pl.String),
                    )
                )

            # Store result for this locus/group
            locus_results[locus_celltype].append(
                df_locus_summary.select(
                    ["cell_id", f"_{key_added}_{locus_celltype}"]
                )
            )

    # Concatenate the per-group results for each locus celltype. Since
    # `group_by` partitions cells disjointly, each cell_id only appears in
    # one group's frame per locus, so a simple vertical concat is safe.
    locus_results = {
        locus_celltype: pl.concat(dfs, how="vertical")
        for locus_celltype, dfs in locus_results.items()
    }

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
    df = df.join(df_final, on="cell_id", how="left")

    # After propagation, clear clone_id for contigs missing required V/J/key fields
    v_col_common = VCALLG if VCALLG in df.collect_schema() else VCALL

    # Build when/then chain dynamically
    # Get unique column names from the key dict
    possible_cols = set(key.values())

    # Build expression to check if the relevant column is null/empty
    checks = []
    for col in possible_cols:
        checks.append(
            pl.when(pl.col("_key") == col).then(
                (pl.col(col).is_null()) | (pl.col(col) == "")
            )
        )

    # Chain them
    is_empty_expr = checks[0]
    for check in checks[1:]:
        is_empty_expr = is_empty_expr.otherwise(check)

    has_ambiguous = "ambiguous" in df.collect_schema()

    condition = (
        (~pl.col("_is_key_empty"))
        & (pl.col(v_col_common).is_not_null())
        & (pl.col(v_col_common).ne(""))
        & (pl.col(JCALL).is_not_null())
        & (pl.col(JCALL).ne(""))
    )
    if has_ambiguous:
        condition = condition & (~pl.col("ambiguous").is_in(TRUES_STR))
    df = (
        df.with_columns(pl.col("locus").replace(locus_to_col).alias("_key"))
        .with_columns(is_empty_expr.alias("_is_key_empty"))
        .with_columns(
            pl.when(condition)
            .then(pl.col(key_added))
            .otherwise(None)
            .alias(key_added)
        )
        .drop(["_original_order", "_key", "_is_key_empty"])
    )

    # return
    vdj._data = df
    vdj.update_metadata(clone_key=str(key_added))
    # offload memory
    vdj._cache_data()

    # Build sparse distance matrix from collected data
    if store_distances:
        # Batched COO construction - avoids single large concatenation
        logg.info("Storing distance matrix...")
        # Get matrix dimensions
        n_cells = len(all_cell_ids)
        # Build COO matrices in batches and sum them
        batch_size = 100  # Process 100 submatrices at a time
        csr_dist = None
        for batch_start in tqdm(
            range(0, len(distance_results), batch_size),
            desc="Building distance matrix (batched)",
            disable=not verbose,
        ):
            batch_end = min(batch_start + batch_size, len(distance_results))
            batch = distance_results[batch_start:batch_end]
            # Build COO for this batch only
            batch_coo = coo_matrix(
                (
                    np.concatenate([d for r, c, d in batch]),
                    (
                        np.concatenate([r for r, c, d in batch]),
                        np.concatenate([c for r, c, d in batch]),
                    ),
                ),
                shape=(n_cells, n_cells),
            )
            # Convert to CSR and accumulate
            if csr_dist is None:
                csr_dist = batch_coo.tocsr()
            else:
                csr_dist = csr_dist + batch_coo.tocsr()
            del batch_coo
        # Clear immediately
        distance_results.clear()

        # Store in vdj.distances
        vdj.distances = csr_dist
        logg.info(
            f"Stored distances as CSR sparse matrix: {csr_dist.shape}, density={csr_dist.nnz / (n_cells**2):.2%}"
        )

    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig AIRR table\n"
            "   'metadata', cell observations table\n"
            "   'distances', sparse distance matrix\n"
            if store_distances
            else ""
        ),
    )

    return vdj


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

    # Prepare length column independently of same_vj/same_length.
    # Downstream code expects `_{key}_length` to exist for aggregation.
    length_col = key + "_length"
    if recalculate_length or (length_col not in df.collect_schema()):
        df = df.with_columns(
            pl.when(pl.col(key).is_null())
            .then(pl.lit(0))
            .otherwise(pl.col(key).cast(pl.String).str.len_bytes())
            .alias(f"_{key}_length")
        )
    else:
        df = df.with_columns(pl.col(length_col).alias(f"_{key}_length"))

    # Filter out rows with null or empty junction (empty strings can't be clustered by distance)
    filter_conditions = [
        pl.col(key).is_not_null(),
        pl.col(key).ne(""),  # Filter out empty strings as well
    ]

    if same_vj:
        filter_conditions.extend(
            [
                pl.col("_v_gene").is_not_null(),
                pl.col("_j_gene").is_not_null(),
                pl.col("_v_gene").ne(""),
                pl.col("_j_gene").ne(""),
            ]
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


def _sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> None:
    """Mutate DataFrame to be h5ad-friendly without nullable string extension dtypes."""
    df.index = pd.Index(
        np.asarray(df.index, dtype=object),
        dtype=object,
        name=df.index.name,
    )
    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.StringDtype):
            df[col] = series.astype(object).where(series.notna(), np.nan)


def transfer(
    adata: AnnData | MuData,
    vdj: DandelionPolars,
    main_view: Literal["all", "expanded", "full"] = "all",
    gex_key: str | None = None,
    vdj_key: str | None = None,
    clone_key: str | None = None,
    collapse_nodes: bool = False,
    overwrite: bool | list[str] | str | None = True,
    obs: bool = True,
    obsm: bool = True,
    uns: bool = True,
    obsp: bool = True,
) -> None:
    """
    Transfer data in Dandelion slots to AnnData, updating `.obs`, `.uns`, `.obsm`, and `.obsp`.
    Transfers both graphs:
      - graph[0] -> adata.uns['dandelion']['X_vdj_all']
      - graph[1] -> adata.uns['dandelion']['X_vdj_expanded']
    The `main_view` flag controls which graph becomes the *main* adjacency written to
    adata.obsp['connectivities'] / ['distances'] (but both graphs are stored).

    Parameters
    ----------
    adata : AnnData | MuData
        AnnData object or `MuData` object.
    vdj : DandelionPolars
        Dandelion object.
    main_view : Literal["all", "expanded", "full"], optional
        Which graph becomes the *main* adjacency written to adata.obsp['connectivities'] / ['distances'].
        If 'full', the full distance matrix from Dandelion is transferred and stored in adata.obsp and no graph is transferred to obsm.
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
    obs : bool, optional
        Whether to transfer `.metadata` columns to `adata.obs`. Defaults to True.
    obsm : bool, optional
        Whether to transfer layout embeddings to `adata.obsm`. Defaults to True.
    uns : bool, optional
        Whether to transfer graph and layout data to `adata.uns`. Defaults to True.
    obsp : bool, optional
        Whether to transfer distance and connectivity matrices to `adata.obsp`. Defaults to True.
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
    _sanitize_dataframe_for_h5ad(recipient.obs)
    _sanitize_dataframe_for_h5ad(recipient.var)
    original_backend = vdj._backend
    original_lazy = vdj._lazy
    if original_backend == "polars":
        vdj.to_pandas()
    # --- 1) metadata -> adata.obs (preserve original overwrite semantics) ---
    if obs:
        for x in vdj._metadata.columns:
            assigned = False
            if x not in recipient.obs.columns:
                recipient.obs[x] = pd.Series(vdj._metadata[x]).reindex(
                    recipient.obs_names
                )
                assigned = True
            elif overwrite is True:
                recipient.obs[x] = pd.Series(vdj._metadata[x]).reindex(
                    recipient.obs_names
                )
                assigned = True
            if assigned and recipient.obs[x].dtype == "bool":
                recipient.obs[x] = recipient.obs[x].astype(str)

        # explicit overwrite list/string handling (matches original)
        if (overwrite is not None) and (overwrite is not True):
            if not isinstance(overwrite, list):
                overwrite = [overwrite]
            for ow in overwrite:
                recipient.obs[ow] = pd.Series(vdj._metadata[ow]).reindex(
                    recipient.obs_names
                )
                if recipient.obs[ow].dtype == "bool":
                    recipient.obs[ow] = recipient.obs[ow].astype(str)

        _sanitize_dataframe_for_h5ad(recipient.obs)
        _sanitize_dataframe_for_h5ad(recipient.var)

    # also check that all the cells in dandelion are in recipient
    common_cells = recipient.obs_names.intersection(vdj._metadata.index)
    # subset to common cells only
    vdj = vdj[vdj._metadata.index.isin(common_cells)]

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
            # stash in uns["dandelion"] instead of obsp
            recipient.uns.setdefault("dandelion", {})[g_connectivities_key] = (
                recipient.obsp["connectivities"]
            )
            recipient.uns["dandelion"][g_distances_key] = recipient.obsp[
                "distances"
            ]
            g_neighbors_key = f"{gex_key}_{neighbors_key}"
            recipient.uns[g_neighbors_key] = recipient.uns[neighbors_key]

    # --- 3) Convert both graphs ---
    graph_connectivities, graph_distances = {}, {}
    if main_view == "full":
        main_idx = 2
        # explicitly only transfer full distance matrix if requested
        if getattr(vdj, "distances", None) is not None:
            graph_connectivities[2], graph_distances[2] = _graph_to_matrices(
                None, recipient, vdj.distances
            )
    else:
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
        # Determine main graph index
        main_idx = 1 if main_view == "expanded" else 0
        if main_idx not in graph_connectivities:
            main_idx = next(iter(graph_connectivities.keys()))

    if main_idx not in graph_connectivities:
        if main_view == "full":
            raise ValueError(
                "main_view='full' requested but `vdj.distances` is not available. "
                "Run clone finding with `store_distances=True` first."
            )
        raise ValueError(
            "No VDJ graph/distance information available to transfer."
        )

    if obsp:
        # --- 4) Update recipient.obsp (active view only) ---
        recipient.obsp["connectivities"] = graph_connectivities[main_idx]
        recipient.obsp["distances"] = graph_distances[main_idx]

        # store non-active views in uns["dandelion"] to keep .obsp clean
        _ddl = recipient.uns.setdefault("dandelion", {})
        if main_idx != 2:
            # store the all (graph[0]) and expanded graph (graph[1]) if available
            if 0 in graph_connectivities:
                _ddl[f"{v_connectivities_key}_all"] = graph_connectivities[0]
                _ddl[f"{v_distances_key}_all"] = graph_distances[0]
            if 1 in graph_connectivities:
                _ddl[f"{v_connectivities_key}_expanded"] = graph_connectivities[
                    1
                ]
                _ddl[f"{v_distances_key}_expanded"] = graph_distances[1]
        else:
            if 2 in graph_connectivities:
                _ddl[f"{v_connectivities_key}_full"] = graph_connectivities[2]
                _ddl[f"{v_distances_key}_full"] = graph_distances[2]
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
                graph_connectivities[idx].data[:] = 1
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
            bin_conn = speye(len(cell_indices), format="csr")

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
                if coord.shape[1] == 0:
                    # Empty layout - skip this embedding
                    continue
                elif coord.shape[1] >= 2:
                    embedding = coord.iloc[:, :2].to_numpy(dtype=np.float32)
                else:
                    col0 = (
                        coord.iloc[:, 0]
                        .to_numpy(dtype=np.float32)
                        .reshape(-1, 1)
                    )
                    col1 = np.zeros_like(col0)
                    embedding = np.hstack([col0, col1])

                _ddl = recipient.uns.setdefault("dandelion", {})
                _ddl[obsm_name] = embedding
                stored_embeddings[idx] = obsm_name

            # Set the "active" embedding safely
            active_obsm = stored_embeddings.get(main_idx)
            if active_obsm is not None:
                _ddl = recipient.uns.setdefault("dandelion", {})
                recipient.obsm["X_vdj"] = _ddl[active_obsm]

    # break up the message depending on which parts were executed
    message_parts = []
    if obs:
        message_parts += ["updated `.obs` with `.metadata`\n"]
    if obsm:
        message_parts += [
            "wrote active layout to `.obsm['X_vdj']`; stashed all views in `.uns['dandelion']` ('X_vdj_all', 'X_vdj_expanded')\n"
        ]
    if obsp:
        message_parts += [
            f"wrote `.obsp['connectivities']` & `['distances']` from graph[{main_idx}]\n",
            (
                f"stashed GEX matrices in `.uns['dandelion']` ('{g_connectivities_key}', '{g_distances_key}')\n"
                if not skip_stash
                else ""
            ),
            (
                f"stashed VDJ matrices in `.uns['dandelion']` under '{v_connectivities_key}_all' / '_expanded'\n"
                if main_idx != 2
                else f"stashed VDJ matrices in `.uns['dandelion']` under '{v_connectivities_key}_full'\n"
            ),
        ]
    if uns:
        message_parts += [f"added `.uns['{clone_key}']` clone-level mapping"]

    # convert back
    if original_backend == "polars":
        vdj.to_polars(lazy=original_lazy)
    # --- 7) Done ---
    logg.info(
        " finished",
        time=start,
        deep="".join(message_parts),
    )


tf = transfer  # alias for transfer


def _graph_to_matrices(
    graph: nx.Graph | ig.Graph | None,
    adata: AnnData,
    distances: csr_matrix | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Convert a graph or provided distances into properly aligned sparse
    connectivities and distances matrices.

    Rules:
    - If G is provided, convert edges → sparse distance matrix.
    - If a CSR distance matrix is provided, must have `._index_names`.
    - Reindex to `adata.obs_names` without dense conversion.
    - Compute connectivities as exp(-d) on non-zero entries.
    - Add tiny self-edge if matrix is entirely empty.

    Optimized to avoid memory explosion by minimizing intermediate copies.
    """

    target_names = list(adata.obs_names)
    n = len(target_names)
    name_to_new = {name: i for i, name in enumerate(target_names)}

    # CASE A: Build distances from graphs
    if distances is None and graph is not None:
        if isinstance(graph, ig.Graph):
            if graph.ecount() == 0:
                distances = csr_matrix((n, n), dtype=np.float32)
            else:
                distances = _igraph_to_scipy_sparse(
                    graph, target_names, weight_attr="weight"
                )
                if distances.nnz > 0:
                    distances.data += 1.0
        else:
            if graph.number_of_edges() == 0:
                distances = csr_matrix((n, n), dtype=np.float32)
            else:
                # nodelist ensures alignment to adata.obs_names in one shot,
                # and nodes not in target_names are simply dropped
                present_nodes = [nm for nm in target_names if nm in graph]
                adj = nx.to_scipy_sparse_array(
                    graph,
                    nodelist=present_nodes,
                    weight="weight",
                    format="coo",
                    dtype=np.float32,
                )

                if len(present_nodes) == n:
                    distances = adj.tocsr()
                else:
                    # remap into the full n x n space if G doesn't contain every adata cell
                    idx_map = np.array(
                        [name_to_new[nm] for nm in present_nodes]
                    )
                    rows = idx_map[adj.row]
                    cols = idx_map[adj.col]
                    distances = csr_matrix(
                        (adj.data, (rows, cols)), shape=(n, n), dtype=np.float32
                    )

                if distances.nnz > 0:
                    distances.data += 1.0
    # CASE B: distances provided as a csr_matrix with _index_names
    elif isinstance(distances, csr_matrix):
        old_names = np.array(distances._index_names)
        coo = distances.tocoo()

        # Vectorized mapping using searchsorted (much faster than list comprehension)
        # Build arrays for lookup
        target_arr = np.array(target_names)

        # Get unique old names and their mapping
        unique_old = np.unique(old_names)

        # Find which old names exist in target
        # Use np.isin for vectorized membership test
        old_in_target = np.isin(unique_old, target_arr)

        # Create mapping for valid names only
        valid_old_names = unique_old[old_in_target]
        if len(valid_old_names) > 0:
            # Create mapping using pandas for efficient lookup
            name_series = pd.Series(
                range(n), index=target_names, dtype=np.int32
            )

            # Map row and column names
            old_row_names = old_names[coo.row]
            old_col_names = old_names[coo.col]

            row_idx = name_series.reindex(old_row_names, fill_value=-1).values
            col_idx = name_series.reindex(old_col_names, fill_value=-1).values

            # Keep only valid mappings
            mask = (row_idx >= 0) & (col_idx >= 0)
            rows = row_idx[mask]
            cols = col_idx[mask]
            vals = coo.data[mask]
            vals += 1.0

            distances = csr_matrix(
                (vals, (rows, cols)), shape=(n, n), dtype=np.float32
            )
            # Clean up
            del row_idx, col_idx, rows, cols, vals, mask
        else:
            distances = csr_matrix((n, n), dtype=np.float32)

    # Build connectivities = exp(-d) for non-zero entries
    connectivities = distances.copy()
    if connectivities.nnz > 0:
        connectivities.data = np.exp(-connectivities.data)
        connectivities.data = np.clip(connectivities.data, 1e-45, np.inf)

    distances.data -= 1.0

    # Ensure matrix is not completely empty - avoid expensive conversion
    if connectivities.nnz == 0:
        # Create minimal non-empty matrix directly without conversion
        connectivities = csr_matrix(
            ([1e-10], ([0], [0])), shape=(n, n), dtype=np.float32
        )
        distances = csr_matrix(
            ([0.0], ([0], [0])), shape=(n, n), dtype=np.float32
        )

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

    Raises
    ------
    KeyError
        if the requested connectivities, distances, or embedding key is not found.
    """
    if mode is None:
        # use the other key directly
        _ddl = adata.uns.get("dandelion", {})
        if connectivities_key in _ddl:
            adata.obsp["connectivities"] = _ddl[connectivities_key]
        else:
            raise KeyError(
                f"{connectivities_key} not found in adata.uns['dandelion']"
            )

        if distances_key in _ddl:
            adata.obsp["distances"] = _ddl[distances_key]
        else:
            raise KeyError(
                f"{distances_key} not found in adata.uns['dandelion']"
            )

        if embedding_key is not None:
            if embedding_key in adata.obsm:
                adata.obsm["X_vdj"] = adata.obsm[embedding_key]
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
        _ddl = adata.uns.get("dandelion", {})
        if conn_key not in _ddl:
            raise KeyError(f"{conn_key} not found in adata.uns['dandelion']")
        if dist_key not in _ddl:
            raise KeyError(f"{dist_key} not found in adata.uns['dandelion']")
        adata.obsp["connectivities"] = _ddl[conn_key]
        adata.obsp["distances"] = _ddl[dist_key]
        if emb_key is not None:
            if emb_key not in _ddl:
                raise KeyError(f"{emb_key} not found in adata.uns['dandelion']")
            adata.obsm["X_vdj"] = _ddl[emb_key]
        if neighbors_key is not None:
            adata.uns["neighbors"] = adata.uns[neighbors_key]
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


def _categorize_clone_size(size: int | float, max_size: int) -> str | float:
    """
    Categorize clone size into bins or clip at maximum.

    Parameters
    ----------
    size : int | float
        Clone size value
    max_size : int
        Maximum size for categorization

    Returns
    -------
    str | float
        String category if size is valid, otherwise NaN
    """
    if pd.isna(size):
        return np.nan
    if size < max_size:
        return str(int(size))
    else:
        return f">= {max_size}"


def clone_size(
    vdj: DandelionPolars | AnnData | MuData,
    group_by: str | None = None,
    max_size: int | None = None,
    clone_key: str | None = None,
    key_added: str | None = None,
) -> None:
    """
    Quantify clone sizes, globally or per group.

    For each clone, the **proportion** is defined as the number of cells
    belonging to that clone divided by the denominator:

    - **Global** (``group_by=None``): denominator is the total number of
      cells in the metadata.
    - **Per group** (``group_by`` specified): denominator is the total number
      of cells within that group. Proportions are therefore independent
      across groups.

    Each clone proportion is then mapped to a **frequency category** using
    the following bins (matching scRepertoire conventions):

    +------------------+-------------------------+
    | Category         | Proportion range        |
    +==================+=========================+
    | Rare             | 0 – 0.0001              |
    +------------------+-------------------------+
    | Small            | 0.0001 – 0.001          |
    +------------------+-------------------------+
    | Medium           | 0.001 – 0.01            |
    +------------------+-------------------------+
    | Large            | 0.01 – 0.1              |
    +------------------+-------------------------+
    | Hyperexpanded    | 0.1 – 1                 |
    +------------------+-------------------------+

    If a cell is assigned to multiple clones (e.g. multiple chains mapped to
    different clone IDs, separated by ``|``), the clone with the largest size
    is used for all annotation columns.

    The following columns are added to the metadata:

    - ``{key_added}_size`` : number of cells in the clone.
    - ``{key_added}_size_prop`` : clone proportion (see above).
    - ``{key_added}_size_category`` : frequency category label
      (Rare / Small / Medium / Large / Hyperexpanded).
    - ``{key_added}_size_max_{max_size}`` : *(only when* ``max_size`` *is
      set)* clone size as a string, with any size ≥ ``max_size`` collapsed
      to the label ``">= {max_size}"``.

    Parameters
    ----------
    vdj : DandelionPolars | AnnData | MuData
        VDJ data.
    group_by : str | None, optional
        Column in metadata to group by before calculating clone sizes.
        If None, calculates global clone sizes across all cells.
    max_size : int | None, optional
        When provided, adds an extra column where clone sizes are represented
        as string labels; sizes strictly below ``max_size`` are kept as their
        integer value, while sizes ≥ ``max_size`` are labelled
        ``">= {max_size}"``.
    clone_key : str | None, optional
        Column specifying clone identifiers. Defaults to ``'clone_id'``.
    key_added : str | None, optional
        Prefix for the new metadata column names
        (e.g. ``{key_added}_size``, ``{key_added}_size_prop``).
        Defaults to the value of ``clone_key``.

    Raises
    ------
    KeyError
        if ``clone_key`` is not found in metadata.
    """
    # --- Select metadata
    if hasattr(vdj, "mod"):
        metadata_ = vdj.mod["airr"].obs.copy()
    elif isinstance(vdj, AnnData):
        metadata_ = vdj.obs.copy()
    elif isinstance(vdj, DandelionPolars):
        original_backend = vdj._backend
        if original_backend == "polars":
            # originally lazy or not
            original_lazy = vdj._lazy
            vdj.to_pandas()
        else:
            original_lazy = False
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
    if group_by is None:
        clonesize = tmp[clone_key].value_counts()
        prop = clonesize / metadata_.shape[0]
    else:
        # Merge with group_by column using cell_id as key
        # Reset index to make cell_id a regular column for merging
        metadata_with_index = metadata_.reset_index()
        metadata_with_index = metadata_with_index.rename(
            columns={"index": "cell_id"}
        )

        tmp = tmp.merge(
            metadata_with_index[["cell_id", group_by]], on="cell_id", how="left"
        )
        clonesize = tmp.groupby([group_by, clone_key]).size()
        group_sizes = metadata_[group_by].value_counts()

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
            prop.index, names=[group_by, clone_key]
        )

    # --- Create max_size categories if specified
    if max_size is not None:
        # Use partial to bind max_size to the helper function
        categorize_fn = functools.partial(
            _categorize_clone_size, max_size=max_size
        )

        clonesize_cat = clonesize.apply(categorize_fn)
        clonesize_cat_map = clonesize_cat.to_dict()

    # --- Define clone frequency bins
    bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    labels = ["Rare", "Small", "Medium", "Large", "Hyperexpanded"]
    if group_by is None:
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

        if group_by is None:
            # look up sizes directly
            sizes = [size_map.get(c, np.nan) for c in clones]
            props = [prop_map.get(c, np.nan) for c in clones]
            cats = [cat_map.get(c, np.nan) for c in clones]
            if max_size is not None:
                size_cats = [clonesize_cat_map.get(c, np.nan) for c in clones]
        else:
            grp = row[group_by]
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
        # check if lazy backend and sync
        if original_backend == "polars":
            vdj.to_polars(lazy=original_lazy)
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
    group_by: str,
    min_clone_size: int | None = None,
    weighted_overlap: bool = False,
    clone_key: str | None = None,
) -> pd.DataFrame:
    """
    A function to tabulate clonal overlap for input as a circos-style plot.

    Parameters
    ----------
    vdj : DandelionPolars | AnnData
        DandelionPolars or AnnData object.
    group_by : str
        column name in obs/metadata for collapsing to columns in the clone_id x group_by data frame.
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
        clone_id x group_by overlap :class:`pandas.core.frame.DataFrame'.

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
    elif hasattr(vdj, "mod"):
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
    allgroups = list(data[group_by].unique())
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
    dictg_ = dict(data[group_by])
    datc_[group_by] = [dictg_[cell] for cell in datc_["cell_id"]]

    overlap = pd.crosstab(datc_[clone_], datc_[group_by])
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


@contextmanager
def _vj_usage_context(
    adata: AnnData, vdj: DandelionPolars, mode: Literal["B", "abT", "gdT"]
):
    """
    Context manager that temporarily adds V/J gene columns to adata.obs
    and removes them upon exit.
    """
    # Get the split data using celltype mode
    # _split_first gives us the "main" (first) gene
    v_call_main = vdj._split_first(
        cols="v_call", key_added=f"v_call_{mode}", celltype=mode
    )
    j_call_main = vdj._split_first(
        cols="j_call", key_added=f"j_call_{mode}", celltype=mode
    )
    # _split with join=True gives us all genes joined with "|"
    v_call_full = vdj._split(
        cols="v_call",
        key_added=f"v_call_{mode}",
        join=True,
        unique=False,
        celltype=mode,
    )
    j_call_full = vdj._split(
        cols="j_call",
        key_added=f"j_call_{mode}",
        join=True,
        unique=False,
        celltype=mode,
    )
    # Merge all splits into one dataframe
    merged = (
        v_call_main.drop("celltype_group")
        .join(j_call_main.drop("celltype_group"), on="cell_id", how="left")
        .join(
            v_call_full.drop("celltype_group"),
            on="cell_id",
            how="left",
            suffix="_full",
        )
        .join(
            j_call_full.drop("celltype_group"),
            on="cell_id",
            how="left",
            suffix="_full",
        )
    )

    # Convert to pandas and set index
    merged_pd = merged.to_pandas().set_index("cell_id")

    # Track original obs
    original_obs = adata.obs.copy()

    try:
        # Add columns to adata.obs
        for col in merged_pd.columns:
            adata.obs[col] = merged_pd[col]

        yield adata

    finally:
        # Clean up: restore original obs
        adata.obs = original_obs


def vj_usage_pca(
    adata: AnnData,
    vdj: DandelionPolars,
    group_by: str,
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
        AnnData object holding the cell level metadata.
    vdj : DandelionPolars
        Dandelion VDJ object to extract V/J usage from.
    group_by : str
        Column name in `adata.obs` to group_by as observations for PCA.
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

    # Use context manager to temporarily add V/J columns
    with _vj_usage_context(adata, vdj, mode) as adata_:
        # filtering
        if allowed_chain_status is not None:
            adata_ = adata_[
                adata_.obs["chain_status"].isin(allowed_chain_status)
            ].copy()

        if groups is not None:
            adata_ = adata_[adata_.obs[group_by].isin(groups)].copy()

        # build config - now using the temporarily added columns
        gene_config = {
            "vdj_v": dict(
                enabled=use_vdj_v,
                main=f"v_call_{mode}_VDJ",
                full=f"v_call_{mode}_VDJ_full",
            ),
            "vdj_j": dict(
                enabled=use_vdj_j,
                main=f"j_call_{mode}_VDJ",
                full=f"j_call_{mode}_VDJ_full",
            ),
            "vj_v": dict(
                enabled=use_vj_v,
                main=f"v_call_{mode}_VJ",
                full=f"v_call_{mode}_VJ_full",
            ),
            "vj_j": dict(
                enabled=use_vj_j,
                main=f"j_call_{mode}_VJ",
                full=f"j_call_{mode}_VJ_full",
            ),
        }
        if not any(cfg["enabled"] for cfg in gene_config.values()):
            raise ValueError("At least one of the use_vj/vdj_v/j must be True.")

        # Determine which groups to keep
        cell_counts = adata_.obs[group_by].value_counts()
        keep_groups = cell_counts[cell_counts >= min_size].index

        # collect gene lists
        gene_lists = {}
        for key, cfg in gene_config.items():
            if cfg["enabled"]:
                uniq = adata_.obs[cfg["main"]].unique().tolist()
                gene_lists[key] = [
                    g
                    for g in uniq
                    if g not in ("None", "No_contig", None) and pd.notna(g)
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
            group_mask = adata_.obs[group_by] == group
            obs_group = adata_.obs.loc[group_mask]

            for key, cfg in gene_config.items():
                if not cfg["enabled"]:
                    continue

                # Handle pipe-separated values from join=True
                all_values = []
                for val in obs_group[cfg["full"]]:
                    if pd.notna(val) and val not in ("None", "No_contig"):
                        all_values.extend(str(val).split("|"))

                counts = Counter(all_values)
                for gene in gene_lists[key]:
                    vdj_df.loc[group, gene] = counts.get(gene, 0)

        # normalize each chain separately
        for key, cfg in gene_config.items():
            if not cfg["enabled"]:
                continue

            cols = gene_lists[key]
            colsum = vdj_df[cols].sum(axis=1)
            # Avoid division by zero
            colsum = colsum.replace(0, 1)
            vdj_df.loc[:, cols] = vdj_df[cols].div(colsum, axis=0) * 100

    # Create new AnnData + PCA (outside context manager)
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
        # Need to get the original adata for this
        collapsed = adata.obs.drop_duplicates(subset=group_by)
        for to in transfer_mapping:
            mapping = dict(zip(collapsed[group_by], collapsed[to]))
            vdj_adata.obs[to] = vdj_adata.obs.index.map(mapping)

    logg.info(
        " finished",
        time=start,
        deep=("Returned AnnData: \n" "   'obsm', X_pca for V/J gene usage"),
    )
    return vdj_adata


def vdj_sample(
    vdj_data: DandelionPolars,
    size: int | None = None,
    adata: AnnData | MuData | None = None,
    p: list[float] | np.ndarray[float] | None = None,
    force_replace: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[DandelionPolars, AnnData] | DandelionPolars:
    """
    Resample vdj data and corresponding AnnData to a specified size.

    Parameters
    ----------
    vdj_data : DandelionPolars
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
    if size is None:
        raise TypeError("vdj_sample requires `size` to be provided.")

    vdj = vdj_data

    logg.info("Resampling to {} cells.".format(str(size)))

    rng = np.random.default_rng(random_state)

    if adata is None:
        # Determine if we need replacement based on metadata size (one row per cell)
        n_cells = vdj.n_obs
        replace = (size > n_cells) or force_replace

        # Normalize probabilities if provided
        if p is not None:
            # Polars Series.to_numpy() may return a read-only view; use a
            # writable copy because we sanitize probabilities in-place.
            p_array = np.array(p, dtype=float, copy=True)
            if p_array.ndim != 1 or p_array.shape[0] != n_cells:
                raise ValueError(
                    "`p` must be a 1D array-like with one value per cell in metadata."
                )
            # Treat missing/non-finite probabilities as zero weight.
            p_array[~np.isfinite(p_array)] = 0.0
            if np.any(p_array < 0):
                raise ValueError("`p` must not contain negative probabilities.")
            p_sum = p_array.sum()
            if p_sum <= 0:
                raise ValueError(
                    "`p` sums to 0 after cleaning missing values; provide at least one positive probability."
                )
            p_array = p_array / p_sum
        else:
            p_array = None

        # Sample indices using numpy (faster than polars/pandas sampling)
        sample_indices = rng.choice(
            n_cells, size=size, replace=replace, p=p_array
        )

        # Get cell IDs at sampled indices - only collect what we need
        if isinstance(vdj._metadata, pl.LazyFrame):
            # Stay lazy and only get the cell_id column
            keep_cells = (
                vdj._metadata.select("cell_id")
                .collect(streaming=True)
                .to_series()
                .gather(sample_indices)
                .to_list()
            )
        elif isinstance(vdj._metadata, pl.DataFrame):
            keep_cells = (
                vdj._metadata["cell_id"].gather(sample_indices).to_list()
            )
        else:
            # pandas DataFrame
            keep_cells = vdj._metadata.iloc[sample_indices].index.tolist()
    else:
        # Check if MuData and extract the gex modality
        if hasattr(adata, "mod"):
            adata = adata.mod["gex"].copy()
        else:
            adata = adata.copy()
        adata_obs_names_full = adata.obs_names.copy()

        # Get common cells between vdj and adata - only collect cell_id column
        if isinstance(vdj._metadata, pl.LazyFrame):
            vdj_cell_ids = set(
                vdj._metadata.select("cell_id")
                .collect(streaming=True)["cell_id"]
                .to_list()
            )
        elif isinstance(vdj._metadata, pl.DataFrame):
            vdj_cell_ids = set(vdj._metadata["cell_id"].to_list())
        else:
            vdj_cell_ids = set(vdj._metadata.index)

        common_cells = list(vdj_cell_ids.intersection(set(adata.obs_names)))

        # Filter to common cells - pass the list of common_cells directly
        # The __getitem__ will treat it as cell_ids to filter by
        adata = adata[adata.obs_names.isin(common_cells)].copy()
        vdj_filtered = vdj[common_cells]

        if p is not None:
            if isinstance(p, pd.Series):
                p_aligned = p.reindex(adata.obs_names).to_numpy(dtype=float)
            else:
                p_aligned = None

            p_array = np.array(p, dtype=float, copy=True)
            if p_array.ndim != 1:
                raise ValueError("`p` must be a 1D array-like.")

            original_n_cells = vdj.n_obs
            filtered_n_cells = vdj_filtered.n_obs

            if p_aligned is None:
                if p_array.shape[0] == original_n_cells:
                    if isinstance(vdj._metadata, pl.LazyFrame):
                        full_meta = vdj._metadata.select("cell_id").collect(
                            streaming=True
                        )
                        full_cell_ids = full_meta["cell_id"].to_list()
                    elif isinstance(vdj._metadata, pl.DataFrame):
                        full_cell_ids = vdj._metadata["cell_id"].to_list()
                    else:
                        full_cell_ids = list(vdj._metadata.index)

                    p_map = dict(zip(full_cell_ids, p_array.tolist()))
                    p_aligned = np.array(
                        [p_map.get(c, 0.0) for c in adata.obs_names],
                        dtype=float,
                    )
                elif p_array.shape[0] == adata_obs_names_full.shape[0]:
                    p_map = dict(zip(adata_obs_names_full, p_array.tolist()))
                    p_aligned = np.array(
                        [p_map.get(c, 0.0) for c in adata.obs_names],
                        dtype=float,
                    )
                elif p_array.shape[0] == filtered_n_cells:
                    p_aligned = p_array
                else:
                    raise ValueError(
                        "`p` length must match one of: original vdj metadata length, original adata length, or filtered adata length."
                    )

            p_aligned[~np.isfinite(p_aligned)] = 0.0
            if np.any(p_aligned < 0):
                raise ValueError("`p` must not contain negative probabilities.")
            p_sum = p_aligned.sum()
            if p_sum <= 0:
                raise ValueError(
                    "`p` sums to 0 after cleaning missing values; provide at least one positive probability."
                )
            p = p_aligned / p_sum

        # Determine replacement based on filtered vdj
        n_cells = vdj_filtered.n_obs
        replace = (size > n_cells) or force_replace

        # Use scanpy to sample
        sc.pp.sample(adata, n=size, replace=replace, rng=random_state, p=p)
        keep_cells = list(adata.obs_names)

        # Use the filtered vdj for downstream operations
        vdj = vdj_filtered

    # Now filter the DATA (contigs) - stay lazy as long as possible
    vdj_dat = vdj._data

    # Check if ambiguous column exists
    if isinstance(vdj_dat, (pl.LazyFrame, pl.DataFrame)):
        cols = set(vdj_dat.collect_schema().names())
        has_ambiguous = "ambiguous" in cols
    else:
        has_ambiguous = "ambiguous" in vdj_dat.columns

    # Apply filters to data while staying lazy
    if isinstance(vdj_dat, pl.LazyFrame):
        # Chain filters while staying lazy
        if has_ambiguous:
            vdj_dat = vdj_dat.filter(pl.col("ambiguous").is_in(FALSES_STR))
        vdj_dat = vdj_dat.filter(pl.col("cell_id").is_in(keep_cells))

        # Only collect if we need replacement logic
        if replace:
            vdj_dat = vdj_dat.collect(streaming=True)
    elif isinstance(vdj_dat, pl.DataFrame):
        if has_ambiguous:
            vdj_dat = vdj_dat.filter(pl.col("ambiguous").is_in(FALSES_STR))
        vdj_dat = vdj_dat.filter(pl.col("cell_id").is_in(keep_cells))
    else:
        # pandas DataFrame
        if has_ambiguous:
            vdj_dat = vdj_dat[vdj_dat["ambiguous"].isin(FALSES)].copy()
        vdj_dat = vdj_dat[vdj_dat["cell_id"].isin(keep_cells)].copy()

    # Handle replacement (requires collected data)
    if replace:
        cell_counts = Counter(keep_cells)
        duplicated_cells = {
            cell: count for cell, count in cell_counts.items() if count > 1
        }

        if duplicated_cells:
            if isinstance(vdj_dat, pl.DataFrame):
                # Separate data for duplication
                duplicated_mask = pl.col("cell_id").is_in(
                    list(duplicated_cells.keys())
                )
                vdj_dat_to_duplicate = vdj_dat.filter(duplicated_mask)
                vdj_dat_to_keep = vdj_dat.filter(~duplicated_mask)

                # Create duplicates
                all_duplicated_vdj = []
                for cell_id, count in duplicated_cells.items():
                    cell_rows = vdj_dat_to_duplicate.filter(
                        pl.col("cell_id") == cell_id
                    )

                    for i in range(count):
                        temp_rows = cell_rows.clone()
                        if i > 0:
                            suffix = f"-{i}"
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
                        all_duplicated_vdj.append(temp_rows)

                # Combine everything
                vdj_dat = pl.concat([vdj_dat_to_keep] + all_duplicated_vdj)
            else:
                # pandas DataFrame
                vdj_dat_to_duplicate = vdj_dat[
                    vdj_dat["cell_id"].isin(duplicated_cells.keys())
                ].copy()
                vdj_dat_to_keep = vdj_dat[
                    ~vdj_dat["cell_id"].isin(duplicated_cells.keys())
                ].copy()

                all_duplicated_vdj = []
                for cell_id, count in duplicated_cells.items():
                    cell_rows = vdj_dat_to_duplicate[
                        vdj_dat_to_duplicate["cell_id"] == cell_id
                    ].copy()

                    for i in range(count):
                        temp_rows = cell_rows.copy()
                        if i > 0:
                            suffix = f"-{i}"
                            temp_rows["cell_id"] = temp_rows["cell_id"] + suffix
                            temp_rows["sequence_id"] = (
                                temp_rows["sequence_id"] + suffix
                            )
                        all_duplicated_vdj.append(temp_rows)

                vdj_dat = pd.concat(
                    [vdj_dat_to_keep] + all_duplicated_vdj, ignore_index=True
                )

    # Reinitialize Dandelion object
    vdj = DandelionPolars(vdj_dat)

    if adata is not None:
        adata.obs_names_make_unique()
        if hasattr(adata, "mod"):
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

    Raises
    ------
    ImportError
        if ``scirpy`` is not installed.
    """
    original_backend = data._backend
    original_lazy = data._lazy
    if original_backend == "polars":
        data.to_pandas()
    # if gex_adata is provided, make sure to only transfer cells that are present in both
    # we will only filter the data to match gex_adata
    if gex_adata is not None:
        data = data[data.metadata_names.isin(gex_adata.obs_names)].copy()
        if data._backend == "polars":
            data.to_pandas()
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

    # conver back to original backend
    if original_backend == "polars":
        data.to_polars(lazy=original_lazy)
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
    existing_cols = (
        dandelion._metadata.collect_schema().names()
        if isinstance(dandelion._metadata, pl.LazyFrame)
        else dandelion._metadata.columns
    )
    new_cols = [col for col in adata.obs if col not in existing_cols]
    if new_cols:
        obs_sub = adata.obs[new_cols].copy()
        obs_sub.index.name = "cell_id"
        obs_pl = pl.from_pandas(obs_sub.reset_index())
        if isinstance(dandelion._metadata, pl.LazyFrame):
            obs_pl = obs_pl.lazy()
        dandelion._metadata = dandelion._metadata.join(
            obs_pl, on="cell_id", how="left"
        )

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
    except ImportError:
        raise ImportError("Please install scirpy to use this function.")

    if isinstance(data, pl.LazyFrame):
        data = data.collect().to_pandas()
    elif isinstance(data, pl.DataFrame):
        data = data.to_pandas()

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


def _dtype_supertype(dt1: pl.DataType, dt2: pl.DataType) -> pl.DataType:
    """Return the lowest common numeric supertype of two Polars dtypes.

    For numeric type pairs (int/float with different widths or signedness vs
    floating-point) this follows a simple promotion ladder so that concat never
    sees mismatched types for the same column.  For non-numeric mismatches we
    fall back to ``pl.String`` so the concat at least does not crash.
    """
    if dt1 == dt2:
        return dt1
    # Promotion rank: higher rank wins.  Floats beat ints; wider beats narrower.
    _RANK: dict[pl.DataType, int] = {
        pl.Float64: 100,
        pl.Float32: 90,
        pl.Int64: 80,
        pl.UInt64: 75,
        pl.Int32: 60,
        pl.UInt32: 55,
        pl.Int16: 40,
        pl.UInt16: 35,
        pl.Int8: 20,
        pl.UInt8: 15,
    }
    r1, r2 = _RANK.get(dt1), _RANK.get(dt2)
    if r1 is not None and r2 is not None:
        return dt1 if r1 >= r2 else dt2
    # Non-numeric mismatch — cast to String as a safe fallback.
    return pl.String


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

    # Collect metadata column names from originals BEFORE deep-copying.
    # Deep-copying a parquet-backed DandelionPolars can lose columns that were
    # added to _metadata via with_columns() after construction because the
    # lazy plan is re-materialised against the original parquet on disk.
    # Reading the schema directly here avoids that round-trip.
    all_meta_cols: set[str] = set()
    for x in arrays:
        if isinstance(x, DandelionPolars) and x._metadata is not None:
            if isinstance(x._metadata, pl.LazyFrame):
                all_meta_cols.update(x._metadata.collect_schema().names())
            else:
                all_meta_cols.update(x._metadata.columns)

    # Convert all inputs to DandelionPolars
    vdjs_ = []
    for x in arrays:
        if isinstance(x, DandelionPolars):
            vdjs_.append(x.copy())
        elif isinstance(x, pl.LazyFrame):
            tmp = DandelionPolars(x.collect(engine="streaming"), verbose=False)
            vdjs_.append(tmp)
        elif isinstance(x, pl.DataFrame):
            tmp = DandelionPolars(x, verbose=False)
            vdjs_.append(tmp)
        elif isinstance(x, pd.DataFrame):
            # Convert pandas to polars
            tmp = DandelionPolars(
                pl.from_pandas(x, schema_overrides=SCHEMA_OVERRIDES),
                verbose=False,
            )
            vdjs_.append(tmp)
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

    # Concatenate the data (Polars DataFrames).
    # Manually align schemas before calling pl.concat so that frames which
    # already possess all union columns are never passed through
    # with_columns([]) — an empty node that fails when polars tries to resolve
    # it against a parquet-backed LazyFrame.  collect_schema() is a no-op
    # (it reads the plan's output schema without executing any query), so this
    # remains fully lazy for LazyFrame inputs.
    all_schema: dict[str, pl.DataType] = {}
    for vdj in vdjs_:
        frame_schema = (
            vdj._data.collect_schema()
            if isinstance(vdj._data, pl.LazyFrame)
            else vdj._data.schema
        )
        for col_name, col_dtype in frame_schema.items():
            if col_name not in all_schema:
                all_schema[col_name] = col_dtype
            elif all_schema[col_name] != col_dtype:
                # Promote to the lowest common supertype so that pl.concat
                # never sees mismatched dtypes for the same column.
                all_schema[col_name] = _dtype_supertype(
                    all_schema[col_name], col_dtype
                )

    col_order = list(all_schema.keys())
    arrays_ = []
    for vdj in vdjs_:
        frame = vdj._data
        present_schema = dict(
            frame.collect_schema()
            if isinstance(frame, pl.LazyFrame)
            else frame.schema
        )
        fix_exprs: list[pl.Expr] = []
        for n in col_order:
            target_dt = all_schema[n]
            if n not in present_schema:
                # Column absent — add a typed null.
                fix_exprs.append(pl.lit(None).cast(target_dt).alias(n))
            elif present_schema[n] != target_dt:
                # Column present but wrong dtype — cast to the resolved type.
                fix_exprs.append(pl.col(n).cast(target_dt))
        if fix_exprs:
            frame = frame.with_columns(fix_exprs)
        arrays_.append(frame.select(col_order))

    # pl.concat requires a uniform type; if any frame is lazy, lazify all
    # eager DataFrames so the list is homogeneous.
    any_lazy = any(isinstance(f, pl.LazyFrame) for f in arrays_)
    if any_lazy:
        arrays_ = [
            f if isinstance(f, pl.LazyFrame) else f.lazy() for f in arrays_
        ]

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

        # Fill in values from the original (pre-copy) inputs so that columns
        # added to _metadata after construction (e.g. via with_columns) are
        # not lost due to the parquet-backing limitation in deep copies.
        for x in arrays:
            if not isinstance(x, DandelionPolars) or x._metadata is None:
                continue
            # Collect if lazy
            if isinstance(x._metadata, pl.LazyFrame):
                source_meta = x._metadata.collect(engine="streaming")
            else:
                source_meta = x._metadata

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
        reordered_meta.lazy() if vdj_concat._lazy else reordered_meta
    )

    return vdj_concat


def productive_ratio(
    adata: AnnData,
    vdj: DandelionPolars,
    group_by: str,
    groups: list[str] | None = None,
    locus: Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"] = "TRB",
):
    """
    Compute the cell-level productive/non-productive contig ratio.

    Only the contig with the highest umi count in a cell will be used for this
    tabulation.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata (`.obs`).
    vdj : DandelionPolars
        DandelionPolars object holding the repertoire data (`.data`).
    group_by : str
        Name of column in `AnnData.obs` to return the row tabulations.
    groups : list[str] | None, optional
        Optional list of categories to return.
    locus : Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"], optional
        One of the accepted locuses to perform the tabulation

    Returns
    -------
    None
        Modifies ``adata`` in place, storing the result in ``adata.uns['productive_ratio']``.
    """
    start = logg.info("Tabulating productive ratio")

    # Filter to cells present in adata
    vdjx = vdj[vdj.metadata["cell_id"].is_in(list(adata.obs_names))]

    # Get data, handling lazy frames
    if isinstance(vdjx._data, pl.LazyFrame):
        data = vdjx._data.collect(engine="streaming")
    else:
        data = vdjx._data

    # Filter by locus and ambiguous status
    if "ambiguous" in data.columns:
        data_filtered = data.filter(
            (pl.col("locus") == locus) & (pl.col("ambiguous").is_in(FALSES_STR))
        )
    else:
        data_filtered = data.filter(pl.col("locus") == locus)

    # Drop duplicates on cell_id, keeping first (highest umi due to sorting)
    df_unique = data_filtered.unique(subset=["cell_id"], keep="first")

    # Create mapping of cell_id to productive status
    dict_df = dict(zip(df_unique["cell_id"], df_unique["productive"]))

    # Determine groups if not provided
    if groups is None:
        if is_categorical(adata.obs[group_by]):
            groups = list(adata.obs[group_by].cat.categories)
        else:
            groups = list(set(adata.obs[group_by]))

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
        res.loc[cell, "total"] = sum(adata.obs[group_by] == cell)
        if res.loc[cell, "total"] > 0:
            res.loc[cell, "productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[group_by] == cell, locus + "_productive"
                    ].isin(["T"])
                )
                / res.loc[cell, "total"]
                * 100
            )
            res.loc[cell, "non-productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[group_by] == cell, locus + "_productive"
                    ].isin(["F"])
                )
                / res.loc[cell, "total"]
                * 100
            )

    res[group_by] = res.index
    res["productive+non-productive"] = res["productive"] + res["non-productive"]
    out = {"results": res, "locus": locus, "group_by": group_by}
    adata.uns["productive_ratio"] = out
    logg.info(
        " finished",
        time=start,
        deep=(
            f"Updated AnnData: \n"
            f"   'obs', '{locus}_productive'\n"
            "   'uns', 'productive_ratio'\n"
        ),
    )


def _igraph_to_scipy_sparse(
    Gi: ig.Graph, target_names: list[str], weight_attr="weight"
):
    """Helper function to convert an igraph object to a scipy sparse matrix."""
    name_to_new = {nm: i for i, nm in enumerate(target_names)}
    edges = Gi.get_edgelist()
    weights = (
        Gi.es[weight_attr]
        if weight_attr in Gi.es.attributes()
        else [1.0] * len(edges)
    )
    names = Gi.vs["name"]

    u_idx = np.array([name_to_new.get(names[u], -1) for u, v in edges])
    v_idx = np.array([name_to_new.get(names[v], -1) for u, v in edges])
    w = np.array(weights, dtype=np.float32)

    mask = (u_idx >= 0) & (v_idx >= 0)
    u_idx, v_idx, w = u_idx[mask], v_idx[mask], w[mask]

    rows = np.concatenate([u_idx, v_idx])
    cols = np.concatenate([v_idx, u_idx])
    vals = np.concatenate([w, w]) + 1.0

    n = len(target_names)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
