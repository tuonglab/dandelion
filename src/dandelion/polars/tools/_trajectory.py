#!/usr/bin/env python
# @author: chenqu, kp9, kelvin
from __future__ import annotations

import re

from anndata import AnnData
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from contextlib import contextmanager
from scanpy import logging as logg
from typing import Literal

from dandelion.polars.core._core import DandelionPolars


@contextmanager
def _trajectory_context(
    adata: AnnData,
    vdj: DandelionPolars,
    mode: Literal["B", "abT", "gdT"] | None,
    extract_cols: list[str] | None = None,
    productive_cols: list[str] | None = None,
):
    """
    Context manager that temporarily adds VDJ/VJ gene and productive columns to adata.obs
    for trajectory analysis, then removes them upon exit.

    Parameters
    ----------
    adata : AnnData
        AnnData object to add columns to
    vdj : DandelionPolars
        Dandelion object containing VDJ data
    mode : Literal["B", "abT", "gdT"] | None
        Cell type mode
    extract_cols : list[str] | None
        Custom column names to extract instead of standard mode-based columns
    productive_cols : list[str] | None
        Custom productive column names
    """
    # Track original obs
    original_obs = adata.obs.copy()

    # change from lazy to eager to avoid issues with context manager and polars dataframes
    if isinstance(vdj._data, pl.LazyFrame) or isinstance(
        vdj._metadata, pl.LazyFrame
    ):
        vdj.to_eager()
        original_lazy = True
    else:
        original_lazy = False
    try:
        # Add chain_status if available in vdj metadata
        if vdj._metadata is not None:
            metadata_pd = vdj._metadata.to_pandas().set_index("cell_id")
            if "chain_status" in metadata_pd.columns:
                adata.obs["chain_status"] = metadata_pd["chain_status"]

        if mode is not None:
            # Extract productive status for VDJ and VJ loci
            productive_vdj = vdj._split_first(
                cols="productive", key_added="productive", celltype=mode
            )
            productive_vdj_pd = productive_vdj.to_pandas().set_index("cell_id")

            # Add productive columns
            if "productive_VDJ" in productive_vdj_pd.columns:
                adata.obs[f"productive_{mode}_VDJ"] = productive_vdj_pd[
                    "productive_VDJ"
                ]
            if "productive_VJ" in productive_vdj_pd.columns:
                adata.obs[f"productive_{mode}_VJ"] = productive_vdj_pd[
                    "productive_VJ"
                ]

            # Extract v_call, d_call, j_call for VDJ and VJ
            v_call = vdj._split_first(
                cols="v_call", key_added="v_call", celltype=mode
            )
            d_call = vdj._split_first(
                cols="d_call", key_added="d_call", celltype=mode
            )
            j_call = vdj._split_first(
                cols="j_call", key_added="j_call", celltype=mode
            )

            # Merge all splits into one dataframe
            merged = (
                v_call.drop("celltype_group", strict=False)
                .join(
                    d_call.drop("celltype_group", strict=False),
                    on="cell_id",
                    how="left",
                )
                .join(
                    j_call.drop("celltype_group", strict=False),
                    on="cell_id",
                    how="left",
                )
            )

            # Convert to pandas and set index
            merged_pd = merged.to_pandas().set_index("cell_id")

            # Add all columns to adata.obs with mode suffix
            for col in merged_pd.columns:
                # Rename columns to match expected format
                if col.endswith("_VDJ") or col.endswith("_VJ"):
                    new_col = col.replace("v_call_", f"v_call_{mode}_")
                    new_col = new_col.replace("d_call_", f"d_call_{mode}_")
                    new_col = new_col.replace("j_call_", f"j_call_{mode}_")
                    adata.obs[new_col] = merged_pd[col]
        else:
            # Mode is None - use extract_cols if provided or default columns
            if extract_cols is None:
                # Default to generic VDJ/VJ columns
                v_call = vdj._split_first(cols="v_call", key_added="v_call")
                d_call = vdj._split_first(cols="d_call", key_added="d_call")
                j_call = vdj._split_first(cols="j_call", key_added="j_call")

                merged = v_call.join(d_call, on="cell_id", how="left").join(
                    j_call, on="cell_id", how="left"
                )

                merged_pd = merged.to_pandas().set_index("cell_id")
                for col in merged_pd.columns:
                    adata.obs[col] = merged_pd[col]
            else:
                # User provided custom extract_cols
                # Extract each column from vdj
                for col in extract_cols:
                    # Remove the mode suffix if present to get the base column name
                    base_col = col
                    for prefix in ["_B_", "_abT_", "_gdT_"]:
                        if prefix in col:
                            base_col = col.split(prefix)[0]
                            break

                    # Try to extract this column from vdj
                    if base_col in ["v_call", "d_call", "j_call"]:
                        extracted = vdj._split_first(
                            cols=base_col, key_added=base_col
                        )
                        extracted_pd = extracted.to_pandas().set_index(
                            "cell_id"
                        )
                        for extracted_col in extracted_pd.columns:
                            adata.obs[extracted_col] = extracted_pd[
                                extracted_col
                            ]

            # Handle productive columns if provided
            if productive_cols is not None:
                productive = vdj._split_first(
                    cols="productive", key_added="productive"
                )
                productive_pd = productive.to_pandas().set_index("cell_id")
                for col in productive_pd.columns:
                    adata.obs[col] = productive_pd[col]

        yield adata

    finally:
        # Clean up: restore original obs
        adata.obs = original_obs
        # Restore original laziness of vdj if we changed it
        if original_lazy:
            vdj.to_lazy()


def _filter_cells(
    adata: AnnData,
    col: str,
    filter_pattern: str | None = ",|None|No_contig",
    remove_missing: bool = True,
) -> AnnData:
    """
    Helper function that identifies filter_pattern hits in `.obs[col]` of adata, and then either removes the
    offending cells or masks the matched values with a uniform value of `col+"_missing"`.
    """
    # find filter pattern hits in our column of interest
    mask = adata.obs[col].str.contains(filter_pattern).fillna(False)
    if remove_missing:
        # remove the cells
        adata = adata[~mask].copy()
    else:
        # uniformly mask the filter pattern hits
        adata.obs.loc[mask, col] = col + "_missing"
    return adata


def setup_vdj_pseudobulk(
    adata: AnnData,
    vdj: DandelionPolars,
    mode: Literal["B", "abT", "gdT"] | None = "abT",
    subsetby: str | None = None,
    groups: list[str] | None = None,
    allowed_chain_status: list[str] | None = [
        "Single pair",
        "Extra pair",
        "Extra pair-exception",
        "Orphan VDJ",
        "Orphan VDJ-exception",
    ],
    productive_vdj: bool = True,
    productive_vj: bool = True,
    extract_cols: list[str] | None = None,
    productive_cols: list[str] | None = None,
    check_vdj_mapping: list[Literal["v_call", "d_call", "j_call"]] | None = [
        "v_call",
        "j_call",
    ],
    check_vj_mapping: list[Literal["v_call", "j_call"]] | None = [
        "v_call",
        "j_call",
    ],
    check_extract_cols_mapping: list[str] | None = None,
    filter_pattern: str | None = ",|None|No_contig",
    remove_missing: bool = True,
) -> AnnData:
    """Function to prepare AnnData for computing pseudobulk vdj feature space.

    Parameters
    ----------
    adata : AnnData
        cell adata before constructing anndata.
    vdj : DandelionPolars
        Dandelion object containing VDJ data
    mode : Literal["B", "abT", "gdT"] | None, optional
        Optional mode for extractin the V(D)J genes. If set as `None`, it requires the option `extract_cols` to be
        specified with a list of column names where this will be used to retrieve the main call.
    subsetby : str | None, optional
        If provided, only the groups/categories in this column will be used for computing the VDJ feature space.
    groups : list[str] | None, optional
        If provided, only the following groups/categories will be used for computing the VDJ feature space.
    allowed_chain_status : list[str] | None, optional
        If provided, only the ones in this list are kept from the `chain_status` column.
    productive_vdj : bool, optional
        If True, cells will only be kept if the main VDJ chain is productive.
    productive_vj : bool, optional
        If True, cells will only be kept if the main VJ chain is productive.
    extract_cols : list[str] | None, optional
        Column names where VDJ/VJ information is stored so that this will be used instead of the standard columns.
    productive_cols : list[str] | None, optional
        Column names where contig productive status is stored so that this will be used instead of the standard columns.
    check_vdj_mapping : list[Literal["v_call", "d_call", "j_call"]] | None, optional
        Only columns in the argument will be checked for unclear mapping (containing comma) in VDJ calls.
        Specifying None will skip this step.
    check_vj_mapping : list[Literal["v_call", "j_call"]] | None, optional
        Only columns in the argument will be checked for unclear mapping (containing comma) in VJ calls.
        Specifying None will skip this step.
    check_extract_cols_mapping : list[str] | None, optional
        Only columns in the argument will be checked for unclear mapping (containing comma) in columns specified in extract_cols.
        Specifying None will skip this step.
    filter_pattern : str | None, optional
        pattern to filter from object. If `None`, does not filter.
    remove_missing : bool, optional
        If True, will remove cells with contigs matching the filter from the object. If False, will mask them with a uniform
        value dependent on the column name.

    Returns
    -------
    AnnData
        filtered cell adata object.
    """
    # Use context manager to temporarily add VDJ columns
    with _trajectory_context(
        adata, vdj, mode, extract_cols, productive_cols
    ) as adata_:
        # keep ony relevant cells (ones with a pair of chains) based on productive column
        if mode is not None:
            if productive_vdj:
                mask_vdj = (
                    adata_.obs["productive_" + mode + "_VDJ"]
                    .str.startswith("T", na=False)
                    .astype(bool)
                )
                adata_ = adata_[mask_vdj].copy()
            if productive_vj:
                mask_vj = (
                    adata_.obs["productive_" + mode + "_VJ"]
                    .str.startswith("T", na=False)
                    .astype(bool)
                )
                adata_ = adata_[mask_vj].copy()
        else:
            if productive_cols is not None:
                for col in productive_cols:
                    mask_col = (
                        adata_.obs[col]
                        .str.startswith("T", na=False)
                        .astype(bool)
                    )
                    adata_ = adata_[mask_col].copy()

        if any([re.search("_VDJ_main|_VJ_main", i) for i in adata_.obs]):
            if check_vdj_mapping is not None:
                if not isinstance(check_vdj_mapping, list):
                    check_vdj_mapping = [check_vdj_mapping]
            if check_vj_mapping is not None:
                if not isinstance(check_vj_mapping, list):
                    check_vj_mapping = [check_vj_mapping]

        if allowed_chain_status is not None:
            adata_ = adata_[
                adata_.obs["chain_status"].isin(allowed_chain_status)
            ].copy()

        if (groups is not None) and (subsetby is not None):
            adata_ = adata_[adata_.obs[subsetby].isin(groups)].copy()

        if extract_cols is None:
            if mode is not None:
                adata_.obs["v_call_" + mode + "_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["v_call_" + mode + "_VDJ"]
                ]
                adata_.obs["d_call_" + mode + "_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["d_call_" + mode + "_VDJ"]
                ]
                adata_.obs["j_call_" + mode + "_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["j_call_" + mode + "_VDJ"]
                ]
                adata_.obs["v_call_" + mode + "_VJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["v_call_" + mode + "_VJ"]
                ]
                adata_.obs["j_call_" + mode + "_VJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["j_call_" + mode + "_VJ"]
                ]
            else:
                adata_.obs["v_call_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["v_call_VDJ"]
                ]
                adata_.obs["d_call_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["d_call_VDJ"]
                ]
                adata_.obs["j_call_VDJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["j_call_VDJ"]
                ]
                adata_.obs["v_call_VJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["v_call_VJ"]
                ]
                adata_.obs["j_call_VJ_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs["j_call_VJ"]
                ]
        else:
            for col in extract_cols:
                adata_.obs[col + "_main"] = [
                    x.split("|")[0] if str(x) != "None" else "None"
                    for x in adata_.obs[col]
                ]

        # remove any cells if there's unclear mapping
        if filter_pattern is not None:
            if mode is not None:
                if check_vdj_mapping is not None:
                    for col in check_vdj_mapping:
                        adata_ = _filter_cells(
                            adata=adata_,
                            col=col + "_" + mode + "_VDJ_main",
                            filter_pattern=filter_pattern,
                            remove_missing=remove_missing,
                        )
                if check_vj_mapping is not None:
                    for col in check_vj_mapping:
                        adata_ = _filter_cells(
                            adata=adata_,
                            col=col + "_" + mode + "_VJ_main",
                            filter_pattern=filter_pattern,
                            remove_missing=remove_missing,
                        )
            else:
                if extract_cols is None:
                    if check_vdj_mapping is not None:
                        for col in check_vdj_mapping:
                            adata_ = _filter_cells(
                                adata=adata_,
                                col=col + "_VDJ_main",
                                filter_pattern=filter_pattern,
                                remove_missing=remove_missing,
                            )
                    if check_vj_mapping is not None:
                        for col in check_vj_mapping:
                            adata_ = _filter_cells(
                                adata=adata_,
                                col=col + "_VJ_main",
                                filter_pattern=filter_pattern,
                                remove_missing=remove_missing,
                            )
                else:
                    if check_extract_cols_mapping is not None:
                        for col in check_extract_cols_mapping:
                            adata_ = _filter_cells(
                                adata=adata_,
                                col=col + "_main",
                                filter_pattern=filter_pattern,
                                remove_missing=remove_missing,
                            )

        # Return a copy to ensure modifications persist outside context manager
        return adata_.copy()


def _get_pbs(
    pbs: np.ndarray | sp.sparse.csr_matrix | None,
    obs_to_bulk: str | None,
    adata: AnnData,
) -> np.ndarray:
    """
    Helper function to ensure we have a cells by pseudobulks matrix which we can use for
    pseudobulking. Uses the pbs and obs_to_bulk inputs to vdj_pseudobulk() and
    gex_pseudobulk().
    """
    # well, we need some way to pseudobulk
    if pbs is None and obs_to_bulk is None:
        raise ValueError(
            "You need to specify `pbs` or `obs_to_bulk` when calling the function"
        )

    # but just one
    if pbs is not None and obs_to_bulk is not None:
        raise ValueError("You need to specify `pbs` or `obs_to_bulk`, not both")

    # turn the pseubodulk matrix dense if need be
    if pbs is not None:
        if sp.sparse.issparse(pbs):
            pbs = pbs.todense()

    # get the obs-derived pseudobulk
    if obs_to_bulk is not None:
        if type(obs_to_bulk) is list:
            # this will create a single value by pasting all the columns together
            tobulk = adata.obs[obs_to_bulk].T.astype(str).agg(",".join)
        else:
            # we just have a single column
            tobulk = adata.obs[obs_to_bulk]
        # this pandas function creates the exact pseudobulk assignment we want
        # this needs to be different than the default uint8
        # as you can have more than 255 cells in a pseudobulk, it turns out
        pbs = pd.get_dummies(tobulk, dtype="uint16").values
    return pbs


def _get_pbs_obs(
    pbs: np.ndarray, obs_to_take: str | None, adata: AnnData
) -> pd.DataFrame:
    """
    Helper function to create the pseudobulk object's obs. Uses the pbs and obs_to_take
    inputs to vdj_pseudobulk() and gex_pseudobulk().
    """
    # prepare per-pseudobulk calls of specified metadata columns
    pbs_obs = pd.DataFrame(index=np.arange(pbs.shape[1]))
    if obs_to_take is not None:
        # just in case a single is passed as a string
        if type(obs_to_take) is not list:
            obs_to_take = [obs_to_take]
        # now we can iterate over this nicely
        # using the logic of milopy's annotate_nhoods()
        for anno_col in obs_to_take:
            anno_dummies = pd.get_dummies(adata.obs[anno_col])
            # this needs to be turned to a matrix so dimensions get broadcast correctly
            anno_count = np.asmatrix(pbs).T.dot(anno_dummies.values)
            anno_frac = np.array(anno_count / anno_count.sum(1))
            anno_frac = pd.DataFrame(
                anno_frac,
                index=np.arange(pbs.shape[1]),
                columns=anno_dummies.columns,
            )
            pbs_obs[anno_col] = anno_frac.idxmax(1)
            pbs_obs[anno_col + "_fraction"] = anno_frac.max(1)
    # report the number of cells for each pseudobulk
    # ensure pbs is an array so that it sums into a vector that can go in easily
    pbs_obs["cell_count"] = np.sum(np.asarray(pbs), axis=0)
    return pbs_obs


def vdj_pseudobulk(
    adata: AnnData,
    vdj: DandelionPolars | None = None,
    pbs: np.ndarray | sp.sparse.csr_matrix | None = None,
    obs_to_bulk: list[str] | str | None = None,
    obs_to_take: list[str] | str | None = None,
    normalise: bool = True,
    renormalise: bool = False,
    min_count: int = 1,
    mode: Literal["B", "abT", "gdT"] | None = "abT",
    extract_cols: list[str] | None = [
        "v_call_abT_VDJ_main",
        "j_call_abT_VDJ_main",
        "v_call_abT_VJ_main",
        "j_call_abT_VJ_main",
    ],
) -> AnnData:
    """Function for making pseudobulk vdj feature space. One of `pbs` or `obs_to_bulk`
    needs to be specified when calling.

    Parameters
    ----------
    adata : AnnData
        Cell adata, preferably after `ddl.tl.setup_vdj_pseudobulk()`
    vdj : DandelionPolars | None, optional
        Dandelion object containing VDJ data. Only needed if columns are not already in adata.obs
    pbs : np.ndarray | sp.sparse.csr_matrix | None, optional
        Optional binary matrix with cells as rows and pseudobulk groups as columns
    obs_to_bulk : list[str] | str | None, optional
        Optional obs column(s) to group pseudobulks into; if multiple are provided, they
        will be combined
    obs_to_take : list[str] | str | None, optional
        Optional obs column(s) to identify the most common value of for each pseudobulk.
    normalise : bool, optional
        If True, will scale the counts of each V(D)J gene group to 1 for each pseudobulk.
    renormalise : bool, optional
        If True, will re-scale the counts of each V(D)J gene group to 1 for each pseudobulk with any "missing" calls removed.
        Relevant with `normalise` as True, if `setup_vdj_pseudobulk()` was ran with `remove_missing` set to False.
    min_count : int, optional
        Pseudobulks with fewer than these many non-"missing" calls in a V(D)J gene group will have their non-"missing" calls
        set to 0 for that group. Relevant with `normalise` as True.
    mode : Literal["B", "abT", "gdT"] | None, optional
        Optional mode for extracting the V(D)J genes. If set as `None`, it will use e.g. `v_call_VDJ` instead of `v_call_abT_VDJ`.
        If `extract_cols` is provided, then this argument is ignored.
    extract_cols : list[str] | None, optional
        Column names where VDJ/VJ information is stored so that this will be used instead of the standard columns.

    Returns
    -------
    AnnData
        pb_adata, whereby each observation is a pseudobulk:\n
        VDJ usage frequency/counts stored in pb_adata.X\n
        VDJ genes stored in pb_adata.var\n
        pseudobulk metadata stored in pb_adata.obs\n
        pseudobulk assignment (binary matrix with input cells as columns) stored in pb_adata.obsm['pbs']\n

    Raises
    ------
    ValueError
        if neither ``pbs`` nor ``obs_to_bulk`` is specified, or if both are specified.
    ValueError
        if required VDJ columns are not in ``adata.obs`` and ``vdj`` is not provided.
    """
    # Check if we need to use context manager
    needs_context = False
    if extract_cols is not None:
        # Check if any of the extract_cols are missing from adata.obs
        needs_context = any(
            col not in adata.obs.columns for col in extract_cols
        )
    elif mode is not None:
        # Check for mode-specific columns
        expected_cols = [
            f"v_call_{mode}_VDJ_main",
            f"j_call_{mode}_VDJ_main",
            f"v_call_{mode}_VJ_main",
            f"j_call_{mode}_VJ_main",
        ]
        needs_context = any(
            col not in adata.obs.columns for col in expected_cols
        )

    if needs_context:
        if vdj is None:
            raise ValueError(
                "vdj parameter must be provided when required columns are not in adata.obs"
            )
        # Use context manager to add required columns
        with _trajectory_context(adata, vdj, mode, extract_cols) as adata_:
            return _vdj_pseudobulk_impl(
                adata_,
                pbs,
                obs_to_bulk,
                obs_to_take,
                normalise,
                renormalise,
                min_count,
                mode,
                extract_cols,
            )
    else:
        # Columns already exist, no need for context manager
        return _vdj_pseudobulk_impl(
            adata,
            pbs,
            obs_to_bulk,
            obs_to_take,
            normalise,
            renormalise,
            min_count,
            mode,
            extract_cols,
        )


def _vdj_pseudobulk_impl(
    adata: AnnData,
    pbs: np.ndarray | sp.sparse.csr_matrix | None,
    obs_to_bulk: list[str] | str | None,
    obs_to_take: list[str] | str | None,
    normalise: bool,
    renormalise: bool,
    min_count: int,
    mode: Literal["B", "abT", "gdT"] | None,
    extract_cols: list[str] | None,
) -> AnnData:
    """Implementation of vdj_pseudobulk that operates on adata with required columns."""
    # get our cells by pseudobulks matrix
    pbs = _get_pbs(pbs, obs_to_bulk, adata)

    # if not specified by the user, use the following default dandelion VJ columns
    if extract_cols is None:
        if mode is None:
            extract_cols = [
                i
                for i in adata.obs
                if re.search(
                    "|".join(
                        [
                            "_call_VDJ_main",
                            "_call_VJ_main",
                        ]
                    ),
                    i,
                )
            ]
        else:
            extract_cols = [
                i
                for i in adata.obs
                if re.search(
                    "|".join([mode + "_VDJ_main", mode + "_VJ_main"]), i
                )
            ]

    # perform matrix multiplication of pseudobulks by cells matrix by a cells by VJs matrix
    # start off by creating the cell by VJs matrix, using the pandas dummies again
    # skip the prefix stuff as the VJ genes will be unique in the columns
    vjs = pd.get_dummies(adata.obs[extract_cols], prefix="", prefix_sep="")
    # TODO: DENAN SOMEHOW? AS IN NAN GENES?
    # can now multiply transposed pseudobulk assignments by this vjs thing, turn to df
    vj_pb_count = pbs.T.dot(vjs.values)
    df = pd.DataFrame(
        vj_pb_count, index=np.arange(pbs.shape[1]), columns=vjs.columns
    )

    if normalise:
        # identify any missing calls inserted by the setup, will end with _missing
        # negate as we want to actually remove them later
        defined_mask = ~(df.columns.str.endswith("_missing"))
        # loop over V(D)J gene categories
        for col in extract_cols:
            # identify columns holding genes belonging to the category
            # and then normalise the values to 1 for each pseudobulk
            group_mask = np.isin(df.columns, np.unique(adata.obs[col]))
            # identify the defined (non-missing) calls for the group
            group_defined_mask = group_mask & defined_mask
            # compute sum of non-missing values for each pseudobulk for this category
            # and compare to the min_count
            defined_min_count = (
                df.loc[:, group_defined_mask].sum(axis=1) >= min_count
            )
            # we can now normalise for the pseudobulks, for now all the pseudobulks
            df.loc[:, group_mask] = df.loc[:, group_mask].div(
                df.loc[:, group_mask].sum(axis=1), axis=0
            )
            if renormalise:
                # repeat the normalisation for non-missing values only
                # and only use pseudobulks crossing the min_count threshold
                df.loc[defined_min_count, group_defined_mask] = df.loc[
                    defined_min_count, group_defined_mask
                ].div(
                    df.loc[defined_min_count, group_defined_mask].sum(axis=1),
                    axis=0,
                )
            # we can now mask the pseudobulks with insufficient defined counts
            df.loc[~defined_min_count, group_defined_mask] = 0

    # create obs for the new pseudobulk object
    pbs_obs = _get_pbs_obs(pbs, obs_to_take, adata)

    # store our feature space and derived metadata into an AnnData
    pb_adata = AnnData(
        np.array(df), var=pd.DataFrame(index=df.columns), obs=pbs_obs
    )
    # store the pseudobulk assignments, as a sparse for storage efficiency
    # transpose as the original matrix is cells x pseudobulks
    pb_adata.obsm["pbs"] = sp.sparse.csr_matrix(pbs.T)
    return pb_adata


def pseudotime_transfer(
    adata: AnnData, pr_res: PResults, suffix: str = ""
) -> AnnData:
    """Function to add pseudotime and branch probabilities into adata.obs in place.

    Parameters
    ----------
    adata : AnnData
        adata for which pseudotime to be transferred to
    pr_res : PResults
        palantir pseudotime inference output object
    suffix : str, optional
        suffix to be added after the added column names, default "" (none)

    Returns
    -------
    AnnData
        transferred AnnData.
    """
    adata.obs["pseudotime" + suffix] = pr_res.pseudotime.copy()

    for col in pr_res.branch_probs.columns:
        adata.obs["prob_" + col + suffix] = pr_res.branch_probs[col].copy()

    return adata


def project_pseudotime_to_cell(
    adata: AnnData, pb_adata: AnnData, term_states: list[str], suffix: str = ""
) -> AnnData:
    """Function to project pseudotime & branch probabilities from pb_adata (pseudobulk adata) to adata (cell adata).

    Parameters
    ----------
    adata : AnnData
        Cell adata, preferably after `ddl.tl.setup_vdj_pseudobulk()`
    pb_adata : AnnData
        neighbourhood/pseudobulked adata
    term_states : list[str]
        list of terminal states with branch probabilities to be transferred
    suffix : str, optional
        suffix to be added after the added column names, default "" (none)

    Returns
    -------
    AnnData
        subset of adata whereby cells that don't belong to any neighbourhood are removed
        and projected pseudotime information stored in .obs - `pseudotime+suffix`, and `'prob_'+term_state+suffix` for each terminal state
    """
    # extract out cell x pseudobulk matrix. it's stored as pseudobulk x cell so transpose
    nhoods = np.array(pb_adata.obsm["pbs"].T.todense())

    # leave out cells that don't belong to any neighbourhood
    nhoodsum = np.sum(nhoods, axis=1)
    cdata = adata[nhoodsum > 0].copy()
    logg.info(
        f"number of cells removed due to not belonging to any neighbourhood: {sum(nhoodsum == 0),}",
    )  # print how many cells removed
    # also subset the pseudobulk_assignments
    pb_assign_trim = nhoods[nhoodsum > 0]

    # for each cell pseudotime_mean is the average of the pseudotime of all pseudobulks the cell is in, weighted by 1/neighbourhood size
    nhoods_cdata = nhoods[nhoodsum > 0, :]
    nhoods_cdata_norm = nhoods_cdata / np.sum(
        nhoods_cdata, axis=0, keepdims=True
    )

    col_list = ["pseudotime" + suffix] + [
        "prob_" + state + suffix for state in term_states
    ]
    for col in col_list:
        cdata.obs[col] = (
            np.array(
                nhoods_cdata_norm.dot(pb_adata.obs[col]).T
                / np.sum(nhoods_cdata_norm, axis=1)
            )
            .flatten()
            .copy()
        )
    cdata.uns["pseudobulk_assignments"] = pb_assign_trim.copy()

    return cdata


def pseudobulk_gex(
    adata_raw: AnnData,
    pbs: np.ndarray | sp.sparse.csr_matrix | None = None,
    obs_to_bulk: list[str] | str | None = None,
    obs_to_take: list[str] | str | None = None,
) -> AnnData:
    """Function to pseudobulk gene expression (raw count).

    Parameters
    ----------
    adata_raw : AnnData
        Needs to have raw counts in .X
    pbs : np.ndarray | sp.sparse.csr_matrix | None, optional
        Optional binary matrix with cells as rows and pseudobulk groups as columns
    obs_to_bulk : list[str] | str | None, optional
        Optional obs column(s) to group pseudobulks into; if multiple are provided, they
        will be combined
    obs_to_take : list[str] | str | None, optional
        Optional obs column(s) to identify the most common value of for each pseudobulk

    Returns
    -------
    AnnData
        pb_adata whereby each observation is a cell neighbourhood\n
        pseudobulked gene expression stored in pb_adata.X\n
        genes stored in pb_adata.var\n
        pseudobulk metadata stored in pb_adata.obs\n
        pseudobulk assignment (binary matrix with input cells as columns) stored in pb_adata.obsm['pbs']\n
    """
    # get our cells by pseudobulks matrix
    pbs = _get_pbs(pbs, obs_to_bulk, adata_raw)

    # make pseudobulk matrix
    pbs_X = adata_raw.X.T.dot(pbs)

    # create obs for the new pseudobulk object
    pbs_obs = _get_pbs_obs(pbs, obs_to_take, adata_raw)

    ## Make new anndata object
    pb_adata = AnnData(pbs_X.T, obs=pbs_obs, var=adata_raw.var)
    # store the pseudobulk assignments, as a sparse for storage efficiency
    # transpose as the original matrix is cells x pseudobulks
    pb_adata.obsm["pbs"] = sp.sparse.csr_matrix(pbs.T)
    return pb_adata
