#!/usr/bin/env python
from __future__ import annotations
import bz2
import copy
import gzip
import h5py
import os
import re
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from changeo.IO import readGermlines
from collections import defaultdict
from pandas.api.types import infer_dtype
from pathlib import Path
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from textwrap import dedent
from tqdm import tqdm
from typing import Literal

from dandelion.utilities._utilities import (
    RECEPTOR_SET,
    TRUES,
    FALSES,
    EMPTIES_STR,
    BOOLEAN_LIKE_COLUMNS,
    deprecated,
    cmp_to_key,
    present,
    all_missing,
    all_missing2,
    same_call,
    sanitize_data_for_saving,
    sanitize_data,
    format_isotype1,
    format_isotype2,
    format_locus,
    lib_type,
    movecol,
    format_chain_status,
    clear_h5file,
    get_vcall_key,
    Tree,
)
from dandelion.external.anndata._compat import (
    _normalize_index,
    unpack_index,
    Index,
)

CHECK_COLS = BOOLEAN_LIKE_COLUMNS + [
    "rev_comp",
    "productive",
    "vj_in_frame",
    "stop_codon",
    "complete_vdj",
    "v_frameshift",
    "j_frameshift",
]


class Dandelion:
    """Dandelion class object."""

    def __init__(
        self,
        data: pd.DataFrame | Path | str | None = None,
        metadata: pd.DataFrame | None = None,
        germline: dict[str, str] | None = None,
        layout: tuple[dict[str, np.array], dict[str, np.array]] | None = None,
        graph: tuple[nx.Graph, nx.Graph] | None = None,
        distances: csr_matrix | None = None,
        initialize: bool = True,
        library_type: Literal["tr-ab", "tr-gd", "ig"] | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Init method for Dandelion.

        Parameters
        ----------
        data : pd.DataFrame | Path | str | None, optional
            AIRR formatted data.
        metadata : pd.DataFrame | None, optional
            AIRR data collapsed per cell.
        germline : dict[str, str] | None, optional
            dictionary of germline gene:sequence records.
        layout : tuple[dict[str, np.array], dict[str, np.array]] | None, optional
            node positions for computed graph.
        graph : tuple[nx.Graph, nx.Graph] | None, optional
            networkx graphs for clonotype networks.
        distances : csr_matrix | None, optional
            distance matrix for sequences.
        initialize : bool, optional
            whether or not to initialize `.metadata` slot.
        library_type : Literal["tr-ab", "tr-gd", "ig"] | None, optional
            One of "tr-ab", "tr-gd", "ig".
        verbose : bool, optional
            whether or not to print initialization messages.
        **kwargs
            passed to `Dandelion.update_metadata`.
        """
        self._data = data
        self._metadata = metadata
        self.layout = layout
        self.graph = graph
        self.distances = distances
        self.germline = {}
        self.querier = None
        self.library_type = library_type
        self.data = self._data
        self.metadata = self._metadata

        if germline is not None:
            self.germline.update(germline)

        if self.data is not None:
            if self.library_type is not None:
                acceptable = lib_type(self.library_type)
            else:
                acceptable = None

            if acceptable is not None:
                self._data = self._data[
                    self._data.locus.isin(acceptable)
                ].copy()

            try:
                self._data = check_travdv(self._data)
            except:
                pass
            if (
                pd.Series(["cell_id", "umi_count", "productive"])
                .isin(self._data.columns)
                .all()
            ):  # sort so that the productive contig with the largest umi is first
                self._data.sort_values(
                    by=["cell_id", "productive", "umi_count"],
                    inplace=True,
                    ascending=[True, False, False],
                )
            # self._data = sanitize_data(self._data) # this is too slow. and unnecessary at this point.
            self.n_contigs = self._data.shape[0]
            if metadata is None:
                if initialize is True:
                    self._ensure_sanitized_data(verbose=verbose)
                    self.update_metadata(**kwargs)
                try:
                    self.n_obs = self._metadata.shape[0]
                except:
                    self.n_obs = 0
            else:
                self._metadata = metadata
                self.n_obs = self._metadata.shape[0]
        else:
            self.n_contigs = 0
            self.n_obs = 0

        # self._original_data_ids = self._data.index.copy()
        # self._original_metadata_ids = self._metadata.index.copy()
        self._original_sequence_ids = self._data["sequence_id"].copy()
        self._original_cell_ids = self._data["cell_id"].copy()

    def _gen_repr(self, n_obs, n_contigs) -> str:
        """Report."""
        # inspire by AnnData's function
        descr = f"Dandelion class object with n_obs = {n_obs} and n_contigs = {n_contigs}"
        for attr in ["data", "metadata"]:
            try:
                keys = getattr(self, attr).keys()
            except AttributeError:
                keys = []
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        if self.layout is not None:
            descr += f"\n    layout: {', '.join(['layout for '+ str(len(x)) + ' vertices' for x in (self.layout[0], self.layout[1]) if x is not None])}"
        if self.graph is not None:
            descr += f"\n    graph: {', '.join(['networkx graph of '+ str(len(x)) + ' vertices' for x in (self.graph[0], self.graph[1]) if x is not None])} "
        if self.distances is not None:
            descr += f"\n    distances: distance matrix of shape {self.distances.shape}"
        return descr

    def __repr__(self) -> str:
        """Report."""
        # inspire by AnnData's function
        return self._gen_repr(self.n_obs, self.n_contigs)

    def __getitem__(self, index: Index) -> Dandelion:
        """Return a sliced Dandelion object with synchronized data and metadata."""
        # Determine index type (metadata-based or data-based)
        if isinstance(index, np.ndarray):
            if len(index) == self._metadata.shape[0]:
                idx, idxtype = self._normalize_indices(
                    self._metadata.index[index]
                )
            elif len(index) == self._data.shape[0]:
                idx, idxtype = self._normalize_indices(self._data.index[index])
            else:
                raise IndexError(
                    "Index length does not match either metadata or data dimensions."
                )
        else:
            # Expecting index to be a boolean Series or DataFrame subset
            idx, idxtype = self._normalize_indices(index[index].index)

        # Slice data and metadata based on idxtype
        if idxtype == "metadata":
            selected_cells = self._metadata.iloc[idx].index
            _metadata = self._metadata.loc[selected_cells]
            _data = self._data[self._data["cell_id"].isin(selected_cells)]
            if self.distances is not None:
                # also filter distances matrix accordingly. the distance matrix is in the same order as metadata
                _distances = self.distances[idx, :][:, idx]
                if isinstance(_distances, csr_matrix):
                    _distances._index_names = _metadata.index
            else:
                _distances = None
        elif idxtype == "data":
            selected_cells = self._data.iloc[idx]["cell_id"]
            _data = self._data.iloc[idx]
            _metadata = self._metadata.loc[
                self._metadata.index.intersection(selected_cells)
            ]
            if self.distances is not None:
                # get the indices of the selected cells in the metadata before filtering
                # using np.where to preserve duplicates
                meta_indices = np.where(
                    self._metadata.index.isin(selected_cells)
                )[0]
                _distances = self.distances[meta_indices, :][:, meta_indices]
                if isinstance(_distances, csr_matrix):
                    _distances._index_names = _metadata.index
            else:
                _distances = None

        else:
            raise TypeError(f"Unrecognized idxtype: {idxtype}")

        # --- Final synchronization step ---
        valid_cells = set(_data["cell_id"]).intersection(_metadata.index)
        _data = _data[_data["cell_id"].isin(valid_cells)].copy()
        _metadata = _metadata.loc[_metadata.index.isin(valid_cells)].copy()
        # -------------------------------------

        # Filter layout and graph if present
        _keep_cells = valid_cells
        if self.layout is not None:
            _layout0 = {
                k: r for k, r in self.layout[0].items() if k in _keep_cells
            }
            _layout1 = {
                k: r for k, r in self.layout[1].items() if k in _keep_cells
            }
            _layout = (_layout0, _layout1)
        else:
            _layout = None

        if self.graph is not None:
            _g0 = self.graph[0].subgraph(_keep_cells)
            _g1 = self.graph[1].subgraph(
                [n for n in self.graph[1].nodes if n in _keep_cells]
            )
            _graph = (_g0, _g1)
        else:
            _graph = None

        # Construct new object
        return Dandelion(
            data=_data,
            metadata=_metadata,
            layout=_layout,
            graph=_graph,
            distances=_distances,
            verbose=False,
        )

    @property
    def data(self) -> pd.DataFrame:
        """One-dimensional annotation of contig observations (`pd.DataFrame`)."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        """data setter"""
        value = load_data(value)
        self._set_dim_df(value, "data")

    @property
    def data_names(self) -> pd.Index:
        """Names of observations (alias for `.data.index`)."""
        return self._data.index

    @data_names.setter
    def data_names(self, names: list[str]):
        """data names setter"""
        names = self._prep_dim_index(names, "data")
        self._set_dim_index(names, "data")

    @property
    def metadata(self) -> pd.DataFrame:
        """One-dimensional annotation of cell observations (`pd.DataFrame`)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: pd.DataFrame):
        """metadata setter"""
        self._set_dim_df(value, "metadata")

    @property
    def metadata_names(self) -> pd.Index:
        """Names of observations (alias for `.metadata.index`)."""
        return self._metadata.index

    @metadata_names.setter
    def metadata_names(self, names: list[str]):
        """metadata names setter"""
        names = self._prep_dim_index(names, "metadata")
        self._set_dim_index(names, "metadata")

    def _ensure_sanitized_data(self, verbose: bool = False) -> None:
        """Ensure that the data is sanitized."""
        if not self._is_sanitized(self._data):
            if verbose:
                logg.info(
                    "The AIRR data needs to undergo sanitization, apologies for any delays..."
                )
            self._data = sanitize_data(self._data)

    def _is_sanitized(self, df):
        """Check if the data is sanitized."""
        check = []
        for col in CHECK_COLS:
            if col in self._data:
                # check that in these columns, all values are str 'T' or 'F'
                if not all(df[col].isin(TRUES + FALSES)):
                    check.append(False)
                else:
                    check.append(True)
        return True if all(check) else False

    def _normalize_indices(self, index: Index) -> tuple[slice, str]:
        """retrieve indices"""
        return _normalize_indices(index, self.metadata_names, self.data_names)

    def _set_dim_df(self, value: pd.DataFrame, attr: str):
        """dim df setter"""
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"Can only assign pd.DataFrame to {attr}.")
            value_idx = self._prep_dim_index(value.index, attr)
            setattr(self, f"_{attr}", value)

    def _prep_dim_index(self, value, attr: str) -> pd.Index:
        """Prepares index to be uses as metadata_names or data_names for Dandelion object.
        If a pd.Index is passed, this will use a reference, otherwise a new index object is created.
        """
        if isinstance(value, pd.Index) and not isinstance(
            value.name, (str, type(None))
        ):
            raise ValueError(
                f"Dandelion expects .{attr}.index.name to be a string or None, "
                f"but you passed a name of type {type(value.name).__name__!r}"
            )
        else:
            value = pd.Index(value)
            if not isinstance(value.name, (str, type(None))):
                value.name = None
        # fmt: off
        if (
            not isinstance(value, pd.RangeIndex)
            and not infer_dtype(value) in ("string", "bytes")
        ):
            sample = list(value[: min(len(value), 5)])
            warnings.warn(dedent(
                f"""
                Dandelion expects .{attr}.index to contain strings, but got values like:
                    {sample}
                    Inferred to be: {infer_dtype(value)}
                """
                ), # noqa
                stacklevel=2,
            )
        # fmt: on
        return value

    def _set_dim_index(self, value: pd.Index, attr: str) -> None:
        """set dim index"""
        # Assumes _prep_dim_index has been run
        getattr(self, attr).index = value
        for v in getattr(self, f"{attr}m").values():
            if isinstance(v, pd.DataFrame):
                v.index = value

    def _update_ids(
        self,
        column: str,
        operation: str,
        value: str,
        sync: bool = True,
        sep: str | None = None,
        remove_trailing_hyphen_number: bool = False,
        **kwargs,
    ) -> None:
        """
        Internal method to update IDs and optionally sync changes.

        Parameters
        ----------
        column : str
            The column to update ('sequence_id' or 'cell_id').
        operation : str
            The operation to perform ('prefix' or 'suffix').
        value : str
            The value to add as prefix or suffix.
        sync : bool, optional
            Whether to sync changes to the other column, by default True.
        sep : str, optional
            Separator to use when adding prefix or suffix, by default None, which means no separator.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers, by default False.
        **kwargs
            Additional arguments to pass to the update_metadata method
        """
        other_column = "cell_id" if column == "sequence_id" else "sequence_id"
        sep = "" if sep is None else sep

        original_values = (
            self._original_sequence_ids
            if column == "sequence_id"
            else self._original_cell_ids
        )
        clean_func = (
            self._clean_sequence_id
            if column == "sequence_id"
            else self._clean_cell_id
        )
        cleaned_values = [
            clean_func(x, remove_trailing_hyphen_number)
            for x in original_values.astype(str)
        ]
        if operation == "prefix":
            self._data[column] = [value + sep + x for x in cleaned_values]
        elif operation == "suffix":
            self._data[column] = [x + sep + value for x in cleaned_values]

        if sync:
            other_original = (
                self._original_cell_ids
                if column == "sequence_id"
                else self._original_sequence_ids
            )
            other_clean_func = (
                self._clean_cell_id
                if column == "sequence_id"
                else self._clean_sequence_id
            )

            cleaned_other = [
                other_clean_func(x, remove_trailing_hyphen_number)
                for x in other_original.astype(str)
            ]
            if operation == "prefix":
                self._data[other_column] = [
                    value + sep + x for x in cleaned_other
                ]
            elif operation == "suffix":
                self._data[other_column] = [
                    x + sep + value for x in cleaned_other
                ]
        self._data = load_data(self._data)
        if self.metadata is not None:
            self.update_metadata(**kwargs)

    def _clean_sequence_id(
        self, value: str, remove_trailing_hyphen_number: bool = False
    ) -> str:
        """
        Clean sequence_id based on specified rules.

        Parameters
        ----------
        value : str
            Original sequence_id value.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers and _contig suffix, by default False.

        Returns
        -------
        str
            Cleaned sequence_id value.
        """
        if remove_trailing_hyphen_number:
            # First remove _contig and everything after it, then remove trailing hyphen number
            return (
                value.split("_contig")[0].split("-")[0]
                + "_contig"
                + value.split("_contig")[1]
            )
        return value

    def _clean_cell_id(
        self, value: str, remove_trailing_hyphen_number: bool = False
    ) -> str:
        """
        Clean cell_id based on specified rules.

        Parameters
        ----------
        value : str
            Original cell_id value.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers, by default False.

        Returns
        -------
        str
            Cleaned cell_id value.
        """
        if remove_trailing_hyphen_number:
            # Remove the last occurrence of hyphen and everything after it
            return value.rsplit("-", 1)[0]
        return value

    def add_sequence_prefix(
        self,
        prefix: str,
        sync: bool = True,
        remove_trailing_hyphen_number: bool = False,
        **kwargs,
    ) -> None:
        """
        Add prefix to sequence_id and then apply to cell_id as well.

        Parameters
        ----------
        prefix : str
            Prefix to add to the IDs.
        sync : bool, optional
            Whether to apply the same prefix to cell_id, by default True.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers before adding prefix, by default False.
        **kwargs
            Additional arguments to pass to the update_metadata method
        """
        self._update_ids(
            column="sequence_id",
            operation="prefix",
            value=prefix,
            sync=sync,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
            **kwargs,
        )

    def add_sequence_suffix(
        self,
        suffix: str,
        sync: bool = True,
        remove_trailing_hyphen_number: bool = False,
        **kwargs,
    ) -> None:
        """
        Add suffix to sequence_id and then apply to cell_id as well.

        Parameters
        ----------
        suffix : str
            Prefix to add to the IDs.
        sync : bool, optional
            Whether to apply the same suffix to cell_id, by default True.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers before adding suffix, by default False.
        **kwargs
            Additional arguments to pass to the update_metadata method
        """
        self._update_ids(
            column="sequence_id",
            operation="suffix",
            value=suffix,
            sync=sync,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
            **kwargs,
        )

    def add_cell_prefix(
        self,
        prefix: str,
        sync: bool = True,
        remove_trailing_hyphen_number: bool = False,
        **kwargs,
    ) -> None:
        """
        Add prefix to cell_id and optionally to sequence_id.

        Parameters
        ----------
        prefix : str
            Prefix to add to the IDs.
        sync : bool, optional
            Whether to apply the same prefix to sequence_id, by default True.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers before adding prefix, by default False.
        **kwargs
            Additional arguments to pass to the update_metadata method
        """
        self._update_ids(
            column="cell_id",
            operation="prefix",
            value=prefix,
            sync=sync,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
            **kwargs,
        )

    def add_cell_suffix(
        self,
        suffix: str,
        sync: bool = True,
        remove_trailing_hyphen_number: bool = False,
        **kwargs,
    ) -> None:
        """
        Add prefix to cell_id and optionally to sequence_id.

        Parameters
        ----------
        suffix : str
            Suffix to add to the IDs.
        sync : bool, optional
            Whether to apply the same suffix to sequence_id, by default True.
        remove_trailing_hyphen_number : bool, optional
            Whether to remove trailing hyphen numbers before adding suffix, by default False.
        **kwargs
            Additional arguments to pass to the update_metadata method
        """
        self._update_ids(
            column="cell_id",
            operation="suffix",
            value=suffix,
            sync=sync,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
            **kwargs,
        )

    # def reset_ids(self) -> None:
    #     """
    #     Reset both IDs to their original values.

    #     This method restores both sequence_id and cell_id in the .data and .metadata slots to their original state when the Dandelion class was initialized.
    #     """
    #     self._data.index = self._original_data_ids
    #     self._metadata.index = self._original_metadata_ids
    #     self._data["sequence_id"] = self._original_sequence_ids
    #     self._data["cell_id"] = self._original_cell_ids

    def simplify(self, **kwargs) -> None:
        """Disambiguate VDJ and C gene calls when there's multiple calls separated by commas and strip the alleles."""
        # strip alleles from VDJ and constant gene calls
        for col in ["v_call", "v_call_genotyped", "d_call", "j_call", "c_call"]:
            if col in self._data:
                self._data[col] = self._data[col].str.replace(
                    r"\*.*", "", regex=True
                )
                # only keep the main annotation
                self._data[col] = self._data[col].str.split(",").str[0]
        self.update_metadata(**kwargs)

    def update_data(self, skip: list[str] = []) -> None:
        """Sync missing metadata columns into data via dictionary mapping."""

        new_cols_added = []

        for col in self._metadata.columns:
            # skip blacklisted columns
            if col in skip:
                continue

            # skip columns that already exist in data
            if col in self._data.columns:
                continue

            # skip if base column already exists (for _VDJ, _VJ, _B, _abT, _gdT variants, _status, _main, etc.)
            base_col = col.split("_")[0]
            if base_col in self._data.columns:
                continue

            # create a mapping dictionary and assign new column
            mapping = self._metadata[col].to_dict()
            self._data[col] = self._data["cell_id"].map(mapping)
            new_cols_added.append(col)

    def _initialize_metadata(
        self,
        cols: list[str],
        clonekey: str,
        v_call_key: str,
        collapse_alleles: bool,
        report_productive_only: bool,
        reinitialize: bool,
        custom_isotype_dict: dict[str, str] | None = None,
    ) -> None:
        """Initialize Dandelion metadata."""
        init_dict = {}
        for col in cols:
            init_dict.update(
                {
                    col: {
                        "query": col,
                        "retrieve_mode": "split and merge",
                    }
                }
            )
        if clonekey in init_dict:
            init_dict.update(
                {
                    clonekey: {
                        "query": clonekey,
                        "retrieve_mode": "merge and unique only",
                    }
                }
            )
        if "sample_id" in init_dict:
            init_dict.update(
                {
                    "sample_id": {
                        "query": "sample_id",
                        "retrieve_mode": "merge and unique only",
                    }
                }
            )
        self._update_rearrangement_status(v_call_key)

        if "ambiguous" in self._data:
            dataq = self._data[self._data["ambiguous"].isin(FALSES)]
        else:
            dataq = self._data
        if self.querier is None:
            querier = Query(dataq)
            self.querier = querier
        else:
            if self.metadata is not None:
                if reinitialize:
                    querier = Query(dataq)
                else:
                    if any(~self.metadata_names.isin(self._data.cell_id)):
                        querier = Query(dataq)
                        self.querier = querier
                    else:
                        querier = self.querier
            else:
                querier = self.querier

        meta_ = defaultdict(dict)
        for k, v in init_dict.copy().items():
            if all_missing(self._data[k]):
                init_dict.pop(k)
                continue
            meta_[k] = querier.retrieve(**v)
            if k in [
                "v_call",
                "v_call_genotyped",
                "d_call",
                "j_call",
                "c_call",
                "productive",
            ]:
                meta_[k + "_split"] = querier.retrieve_celltype(**v)
            if k in ["umi_count", "mu_count", "mu_freq"]:
                v.update({"retrieve_mode": "split and sum"})
                meta_[k] = querier.retrieve_celltype(**v)

        tmp_metadata = pd.concat(meta_.values(), axis=1, join="inner")

        reqcols1 = [
            "locus_VDJ",
        ]
        vcall = get_vcall_key(self._data, v_call_key)

        # remap v_call_genotyped_* to just v_call_* for column names in tmp_metadata if vcall == "v_call_genotyped"
        if vcall == "v_call_genotyped":
            for col in tmp_metadata.columns:
                if col.startswith("v_call_genotyped"):
                    new_col = col.replace("v_call_genotyped", "v_call")
                    tmp_metadata.rename(columns={col: new_col}, inplace=True)
        # This way, the function will only initialise as v_call regardless of whether v_call_key is v_call or v_call_genotyped

        reqcols2 = [
            "locus_VJ",
            "productive_VDJ",
            "productive_VJ",
            "v_call_VDJ",
            "d_call_VDJ",
            "j_call_VDJ",
            "v_call_VJ",
            "j_call_VJ",
            "c_call_VDJ",
            "c_call_VJ",
            "junction_VDJ",
            "junction_VJ",
            "junction_aa_VDJ",
            "junction_aa_VJ",
            "v_call_B_VDJ",
            "d_call_B_VDJ",
            "j_call_B_VDJ",
            "v_call_B_VJ",
            "j_call_B_VJ",
            "c_call_B_VDJ",
            "c_call_B_VJ",
            "productive_B_VDJ",
            "productive_B_VJ",
            "v_call_abT_VDJ",
            "d_call_abT_VDJ",
            "j_call_abT_VDJ",
            "v_call_abT_VJ",
            "j_call_abT_VJ",
            "c_call_abT_VDJ",
            "c_call_abT_VJ",
            "productive_abT_VDJ",
            "productive_abT_VJ",
            "v_call_gdT_VDJ",
            "d_call_gdT_VDJ",
            "j_call_gdT_VDJ",
            "v_call_gdT_VJ",
            "j_call_gdT_VJ",
            "c_call_gdT_VDJ",
            "c_call_gdT_VJ",
            "productive_gdT_VDJ",
            "productive_gdT_VJ",
        ]
        reqcols = reqcols1 + reqcols2
        for rc in reqcols:
            if rc not in tmp_metadata:
                tmp_metadata[rc] = ""

        for dc in [
            "d_call_VJ",
            "d_call_B_VJ",
            "d_call_abT_VJ",
            "d_call_gdT_VJ",
        ]:
            if dc in tmp_metadata:
                tmp_metadata.drop(dc, axis=1, inplace=True)

        for _call in ["v_call", "d_call", "j_call", "c_call"]:
            tmp_metadata[_call + "_VDJ_main"] = [
                return_none_call(x) for x in tmp_metadata[_call + "_VDJ"]
            ]
            if _call != "d_call":
                tmp_metadata[_call + "_VJ_main"] = [
                    return_none_call(x) for x in tmp_metadata[_call + "_VJ"]
                ]

        for mode in ["B", "abT", "gdT"]:
            tmp_metadata["v_call_" + mode + "_VDJ_main"] = [
                return_none_call(x)
                for x in tmp_metadata["v_call_" + mode + "_VDJ"]
            ]
            tmp_metadata["d_call_" + mode + "_VDJ_main"] = [
                return_none_call(x)
                for x in tmp_metadata["d_call_" + mode + "_VDJ"]
            ]
            tmp_metadata["j_call_" + mode + "_VDJ_main"] = [
                return_none_call(x)
                for x in tmp_metadata["j_call_" + mode + "_VDJ"]
            ]
            tmp_metadata["v_call_" + mode + "_VJ_main"] = [
                return_none_call(x)
                for x in tmp_metadata["v_call_" + mode + "_VJ"]
            ]
            tmp_metadata["j_call_" + mode + "_VJ_main"] = [
                return_none_call(x)
                for x in tmp_metadata["j_call_" + mode + "_VJ"]
            ]

        if "locus_VDJ" in tmp_metadata:
            suffix_vdj = "_VDJ"
            # suffix_vj = "_VJ"
        else:
            suffix_vdj = ""
            # suffix_vj = ""

        if clonekey in init_dict:
            tmp_metadata = _add_clone_info(
                tmp_metadata=tmp_metadata, clonekey=str(clonekey)
            )

        conversion_dict = {
            "IGHA": "IgA",
            "IGHD": "IgD",
            "IGHE": "IgE",
            "IGHG": "IgG",
            "IGHM": "IgM",
            "IGKC": "IgK",
            "IGLC": "IgL",
        }
        if custom_isotype_dict is not None:
            conversion_dict.update(custom_isotype_dict)
        isotype = []
        if "c_call" + suffix_vdj in tmp_metadata:
            for k, p in zip(
                tmp_metadata["c_call" + suffix_vdj],
                tmp_metadata["productive" + suffix_vdj],
            ):
                if isinstance(k, str):
                    if report_productive_only:
                        isotype.append(
                            "|".join(
                                [
                                    str(z)
                                    for z, pp in zip(
                                        [
                                            (
                                                conversion_dict[
                                                    y.split(",")[0][:4]
                                                ]
                                                if y.split(",")[0][:4]
                                                in conversion_dict
                                                else "None"
                                            )
                                            for y in k.split("|")
                                        ],
                                        p.split("|"),
                                    )
                                    if pp in TRUES + EMPTIES_STR
                                ]
                            )
                        )
                    else:
                        isotype.append(
                            "|".join(
                                [
                                    str(z)
                                    for z in [
                                        (
                                            conversion_dict[y.split(",")[0][:4]]
                                            if y.split(",")[0][:4]
                                            in conversion_dict
                                            else "None"
                                        )
                                        for y in k.split("|")
                                    ]
                                ]
                            )
                        )
                else:
                    isotype.append("None")
            tmp_metadata["isotype"] = isotype
            tmp_metadata["isotype_status"] = format_isotype1(tmp_metadata)

        vdj_gene_calls = ["v_call", "d_call", "j_call"]
        if collapse_alleles:
            for x in vdj_gene_calls:
                if x in self._data:
                    for c in tmp_metadata:
                        if x in c:
                            tmp_metadata[c] = [
                                "|".join(
                                    [
                                        ",".join(list(set(yy.split(","))))
                                        for yy in list(
                                            [
                                                re.sub("[*][0-9][0-9]", "", tx)
                                                for tx in t.split("|")
                                            ]
                                        )
                                    ]
                                )
                                for t in tmp_metadata[c]
                            ]

        tmp_metadata["locus_status"] = format_locus(
            tmp_metadata, vcall="v_call", productive_only=report_productive_only
        )
        tmp_metadata["chain_status"] = format_chain_status(
            tmp_metadata["locus_status"]
        )
        tmp_metadata["isotype_status"] = format_isotype2(tmp_metadata)

        if "isotype" in tmp_metadata:
            if all(tmp_metadata["isotype"] == "None"):
                tmp_metadata.drop(
                    ["isotype", "isotype_status"], axis=1, inplace=True
                )
        for rc in reqcols:
            tmp_metadata[rc] = tmp_metadata[rc].replace("", "None")
        if clonekey in init_dict:
            tmp_metadata[clonekey] = tmp_metadata[clonekey].replace("", "None")

        tmp_metadata = movecol(
            tmp_metadata,
            cols_to_move=[rc2 for rc2 in reqcols2 if rc2 in tmp_metadata],
            ref_col="locus_VDJ",
        )

        for tmpm in tmp_metadata:
            if all_missing2(tmp_metadata[tmpm]):
                tmp_metadata.drop(tmpm, axis=1, inplace=True)

        tmpxregstat = querier.retrieve(
            query="rearrangement_status", retrieve_mode="split and unique only"
        )

        for x in tmpxregstat:
            tmpxregstat[x] = [
                (
                    "chimeric"
                    if re.search("chimeric", y)
                    else "Multi" if "|" in y else y
                )
                for y in tmpxregstat[x]
            ]
            tmp_metadata[x] = pd.Series(tmpxregstat[x])

        tmp_metadata = movecol(
            tmp_metadata,
            cols_to_move=[
                rs
                for rs in [
                    "rearrangement_status_VDJ",
                    "rearrangement_status_VJ",
                ]
                if rs in tmp_metadata
            ],
            ref_col="chain_status",
        )
        # if metadata already exist, just overwrite the default columns?
        if self.metadata is not None:
            if any(~self.metadata_names.isin(self._data.cell_id)):
                self._metadata = tmp_metadata.copy()  # reindex and replace.
            for col in tmp_metadata:
                self._metadata[col] = pd.Series(tmp_metadata[col])
        else:
            self._metadata = tmp_metadata.copy()

    def _update_rearrangement_status(self, v_call_key: str) -> None:
        """Check rearrangement status."""
        vcall = get_vcall_key(self._data, v_call_key)
        contig_status = []
        for v, j, c in zip(
            self._data[vcall], self._data["j_call"], self._data["c_call"]
        ):
            if present(v):
                if present(j):
                    if present(c):
                        if len(list({v[:3], j[:3], c[:3]})) > 1:
                            contig_status.append("chimeric")
                        else:
                            contig_status.append("standard")
                    else:
                        if len(list({v[:3], j[:3]})) > 1:
                            contig_status.append("chimeric")
                        else:
                            contig_status.append("standard")
                else:
                    contig_status.append("unknown")
            else:
                contig_status.append("unknown")
        self._data["rearrangement_status"] = contig_status

    def compute(self):
        """Convert self.distances from a lazy array to a concrete numpy array."""
        if not isinstance(self.distances, np.ndarray):
            self.distances = csr_matrix(self.distances.compute())
            self.distances._index_names = self.metadata_names

    def copy(self) -> Dandelion:
        """
        Performs a deep copy of all slots in Dandelion class.

        Returns
        -------
        Dandelion
            a deep copy of Dandelion class.
        """
        return copy.deepcopy(self)

    def update_plus(
        self,
        option: Literal[
            "all",
            "sequence",
            "mutations",
            "cdr3 lengths",
            "mutations and cdr3 lengths",
        ] = "mutations and cdr3 lengths",
        **kwargs,
    ) -> None:
        """
        Retrieve additional data columns that are useful.

        Parameters
        ----------
        option : Literal["all", "sequence", "mutations", "cdr3 lengths", "mutations and cdr3 lengths", ], optional
            One of 'all', 'sequence', 'mutations', 'cdr3 lengths',
            'mutations and cdr3 lengths'
        **kwargs
            passed to `Dandelion.update_metadata`.
        """
        mutations_type = ["mu_count", "mu_freq"]
        mutationsdef = [
            "cdr_r",
            "cdr_s",
            "fwr_r",
            "fwr_s",
            "1_r",
            "1_s",
            "2_r",
            "2_s",
            "3_r",
            "3_s",
            "4_r",
            "4_s",
            "5_r",
            "5_s",
            "6_r",
            "6_s",
            "7_r",
            "7_s",
            "8_r",
            "8_s",
            "9_r",
            "9_s",
            "10_r",
            "10_s",
            "11_r",
            "11_s",
            "12_r",
            "12_s",
            "13_r",
            "13_s",
            "14_r",
            "14_s",
            "15_r",
            "15_s",
            "16_r",
            "16_s",
            "17_r",
            "17_s",
            "18_r",
            "18_s",
            "19_r",
            "19_s",
            "20_r",
            "20_s",
            "21_r",
            "21_s",
            "22_r",
            "22_s",
            "23_r",
            "23_s",
            "24_r",
            "24_s",
            "25_r",
            "25_s",
            "26_r",
            "26_s",
            "27_r",
            "27_s",
            "28_r",
            "28_s",
            "29_r",
            "29_s",
            "30_r",
            "30_s",
            "31_r",
            "31_s",
            "32_r",
            "32_s",
            "33_r",
            "33_s",
            "34_r",
            "34_s",
            "35_r",
            "35_s",
            "36_r",
            "36_s",
            "37_r",
            "37_s",
            "38_r",
            "38_s",
            "39_r",
            "39_s",
            "40_r",
            "40_s",
            "41_r",
            "41_s",
            "42_r",
            "42_s",
            "43_r",
            "43_s",
            "44_r",
            "44_s",
            "45_r",
            "45_s",
            "46_r",
            "46_s",
            "47_r",
            "47_s",
            "48_r",
            "48_s",
            "49_r",
            "49_s",
            "50_r",
            "50_s",
            "51_r",
            "51_s",
            "52_r",
            "52_s",
            "53_r",
            "53_s",
            "54_r",
            "54_s",
            "55_r",
            "55_s",
            "56_r",
            "56_s",
            "57_r",
            "57_s",
            "58_r",
            "58_s",
            "59_r",
            "59_s",
            "60_r",
            "60_s",
            "61_r",
            "61_s",
            "62_r",
            "62_s",
            "63_r",
            "63_s",
            "64_r",
            "64_s",
            "65_r",
            "65_s",
            "66_r",
            "66_s",
            "67_r",
            "67_s",
            "68_r",
            "68_s",
            "69_r",
            "69_s",
            "70_r",
            "70_s",
            "71_r",
            "71_s",
            "72_r",
            "72_s",
            "73_r",
            "73_s",
            "74_r",
            "74_s",
            "75_r",
            "75_s",
            "76_r",
            "76_s",
            "77_r",
            "77_s",
            "78_r",
            "78_s",
            "79_r",
            "79_s",
            "80_r",
            "80_s",
            "81_r",
            "81_s",
            "82_r",
            "82_s",
            "83_r",
            "83_s",
            "84_r",
            "84_s",
            "85_r",
            "85_s",
            "86_r",
            "86_s",
            "87_r",
            "87_s",
            "88_r",
            "88_s",
            "89_r",
            "89_s",
            "90_r",
            "90_s",
            "91_r",
            "91_s",
            "92_r",
            "92_s",
            "93_r",
            "93_s",
            "94_r",
            "94_s",
            "95_r",
            "95_s",
            "96_r",
            "96_s",
            "97_r",
            "97_s",
            "98_r",
            "98_s",
            "99_r",
            "99_s",
            "100_r",
            "100_s",
            "101_r",
            "101_s",
            "102_r",
            "102_s",
            "103_r",
            "103_s",
            "104_r",
            "104_s",
            "cdr1_r",
            "cdr1_s",
            "cdr2_r",
            "cdr2_s",
            "fwr1_r",
            "fwr1_s",
            "fwr2_r",
            "fwr2_s",
            "fwr3_r",
            "fwr3_s",
            "v_r",
            "v_s",
        ]
        mutations = [] + mutations_type
        for m in mutations_type:
            for d in mutationsdef:
                mutations.append(m + "_" + d)

        vdjlengths = [
            "junction_length",
            "junction_aa_length",
            "np1_length",
            "np2_length",
        ]
        seqinfo = [
            "sequence",
            "sequence_alignment",
            "sequence_alignment_aa",
            "junction",
            "junction_aa",
            "germline_alignment",
            "fwr1",
            "fwr1_aa",
            "fwr2",
            "fwr2_aa",
            "fwr3",
            "fwr3_aa",
            "fwr4",
            "fwr4_aa",
            "cdr1",
            "cdr1_aa",
            "cdr2",
            "cdr2_aa",
            "cdr3",
            "cdr3_aa",
            "v_sequence_alignment_aa",
            "d_sequence_alignment_aa",
            "j_sequence_alignment_aa",
        ]
        mutations = [x for x in mutations if x in self._data]
        vdjlengths = [x for x in vdjlengths if x in self._data]
        seqinfo = [x for x in seqinfo if x in self._data]

        if option == "all":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    retrieve_mode="split and sum",
                    **kwargs,
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="sum", **kwargs
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    retrieve_mode="split and average",
                    **kwargs,
                )
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo, retrieve_mode="split and merge", **kwargs
                )
        if option == "sequence":
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo, retrieve_mode="split and merge", **kwargs
                )
        if option == "mutations":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    retrieve_mode="split and sum",
                    **kwargs,
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="sum", **kwargs
                )
        if option == "cdr3 lengths":
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    retrieve_mode="split and average",
                    **kwargs,
                )
        if option == "mutations and cdr3 lengths":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    retrieve_mode="split and sum",
                    **kwargs,
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="sum", **kwargs
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    retrieve_mode="split and average",
                    **kwargs,
                )

    def store_germline_reference(
        self,
        corrected: dict[str, str] | str | None = None,
        germline: str | None = None,
        org: Literal["human", "mouse"] = "human",
        db: Literal["imgt", "ogrdb"] = "imgt",
    ) -> None:
        """
        Update germline reference with corrected sequences and store in Dandelion object.

        Parameters
        ----------
        corrected : dict[str, str] | str | None, optional
            dictionary of corrected germline sequences or file path to corrected germline sequences fasta file.
        germline : str | None, optional
            path to germline database folder. Defaults to `` environmental variable.
        org : Literal["human", "mouse"], optional
            organism of reference folder. Default is 'human'.
        db : Literal["imgt", "ogrdb"], optional
            database of reference sequences. Default is 'imgt'.
        Raises
        ------
        KeyError
            if `GERMLINE` environmental variable is not set.
        TypeError
            if incorrect germline provided.
        """
        start = logg.info("Updating germline reference")
        env = os.environ.copy()
        if germline is None:
            try:
                gml = Path(env["GERMLINE"])
            except KeyError:
                raise KeyError(
                    "Environmental variable GERMLINE must be set. Otherwise, "
                    + "please provide path to folder containing germline IGHV, IGHD, and IGHJ fasta files."
                )
            gml = gml / db / org / "vdj"
        else:
            if isinstance(germline, list):
                if len(germline) < 3:
                    raise TypeError(
                        "Input for germline is incorrect. Please provide path to folder containing germline IGHV, IGHD, "
                        + "and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ fasta "
                        + "files (with .fasta extension) as a list."
                    )
                else:
                    gml = []
                    for x in germline:
                        if not x.endswith((".fasta", ".fa")):
                            raise TypeError(
                                "Input for germline is incorrect. Please provide path to folder containing germline "
                                + "IGHV, IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ fasta "
                                + "files (with .fasta extension) as a list."
                            )
                        gml.append(x)
            elif type(germline) is not list:
                if os.path.isdir(germline):
                    germline_ = [
                        str(Path(germline, g)) for g in os.listdir(germline)
                    ]
                    if len(germline_) < 3:
                        raise TypeError(
                            "Input for germline is incorrect. Please provide path to folder containing germline IGHV, "
                            + "IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ "
                            + "fasta files (with .fasta extension) as a list."
                        )
                    else:
                        gml = []
                        for x in germline_:
                            if not x.endswith((".fasta", ".fa")):
                                raise TypeError(
                                    "Input for germline is incorrect. Please provide path to folder containing germline "
                                    + "IGHV, IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, "
                                    + "and IGHJ fasta files (with .fasta extension) as a list."
                                )
                            gml.append(x)
                elif os.path.isfile(germline) and str(germline).endswith(
                    (".fasta", ".fa")
                ):
                    gml = []
                    gml.append(germline)
                    warnings.warn(
                        "Only 1 fasta file provided to updating germline slot. Please check if this is intended.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if type(gml) is not list:
            gml = [gml]

        gml = [str(g) for g in gml]

        germline_ref = readGermlines(gml)
        if corrected is not None:
            if type(corrected) is dict:
                personalized_ref_dict = corrected
            elif os.path.isfile(str(corrected)):
                personalized_ref_dict = readGermlines([corrected])
            # update with the personalized germline database
            if "personalized_ref_dict" in locals():
                germline_ref.update(personalized_ref_dict)
            else:
                raise TypeError(
                    "Input for corrected germline fasta is incorrect. Please provide path to file containing "
                    + "corrected germline fasta sequences."
                )

        self.germline.update(germline_ref)
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'germline', updated germline reference\n"
            ),
        )

    def update_metadata(
        self,
        retrieve: list[str] | str | None = None,
        clone_key: str | None = None,
        retrieve_mode: Literal[
            "split and unique only",
            "merge and unique only",
            "split and merge",
            "split and sum",
            "split and average",
            "split",
            "merge",
            "sum",
            "average",
        ] = "split and merge",
        collapse_alleles: bool = True,
        reinitialize: bool = True,
        by_celltype: bool = False,
        report_status_productive: bool = True,
        genotyped_v_call: bool = True,
        custom_isotype_dict: dict[str, str] | None = None,
    ) -> None:
        """
        A Dandelion initialisation function to update and populate the `.metadata` slot.

        Parameters
        ----------
        retrieve : list[str] | str | None, optional
            column name in `.data` slot to retrieve and update the metadata.
        clone_key : str | None, optional
            column name of clone id. None defaults to 'clone_id'.
        v_call_key : str , optional
            column name of V gene call. Defaults to 'v_call'.
        retrieve_mode : Literal["split and unique only", "merge and unique only", "split and merge", "split and sum", "split and average", "split", "merge", "sum", "average", ], optional
            one of:
                `split and unique only`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for unique elements.
                `merge and unique only`
                    returns the retrieval merged into one column,
                    separated by `|` for unique elements.
                `split and merge`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for every elements.
                `split`
                    returns the retrieval splitted into separate columns for each contig.
                `merge`
                    returns the retrieval merged into one columns for each contig,
                    separated by `|` for unique elements.
                `split and sum`
                    returns the retrieval sum in the VDJ and VJ columns (separately).
                `split and average`
                    returns the retrieval averaged in the VDJ and VJ columns (separately).
                `sum`
                    returns the retrieval sum into one column for all contigs.
                `average`
                    returns the retrieval averaged into one column for all contigs.
        collapse_alleles : bool, optional
            returns the V(D)J genes with allelic calls if False.
        reinitialize : bool, optional
            whether or not to reinitialize the current metadata.
            useful when updating older versions of `dandelion` to newer version.
        by_celltype : bool, optional
            whether to return the query/update by celltype.
        report_status_productive : bool, optional
            whether to report the locus and chain status for only productive contigs.
        genotyped : bool, optional
            whether or not to use genotyped v_call data to initialize metadata if available.
        custom_isotype_dict : dict[str, str] | None, optional
            custom isotype dictionary to update the default isotype dictionary.

        Raises
        ------
        KeyError
            if columns provided not found in Dandelion.data.
        ValueError
            if missing columns in Dandelion.data.
        """
        clonekey = clone_key if clone_key is not None else "clone_id"
        v_call_key = "v_call"
        if genotyped_v_call:
            if f"{v_call_key}_genotyped" in self._data:
                v_call_key = f"{v_call_key}_genotyped"

        cols = [
            "sequence_id",
            "cell_id",
            "locus",
            "productive",
            v_call_key,
            "d_call",
            "j_call",
            "c_call",
            "umi_count",
            "junction",
            "junction_aa",
        ]

        if "umi_count" not in self._data:
            raise ValueError(
                "Unable to initialize metadata due to missing keys. "
                "Please ensure either 'umi_count' or 'duplicate_count' is in the input data."
            )

        if not all([c in self._data for c in cols]):
            raise ValueError(
                "Unable to initialize metadata due to missing keys. "
                "Please ensure the input data contains all the following columns: {}".format(
                    cols
                )
            )

        if "sample_id" in self._data:
            cols = ["sample_id"] + cols

        for c in ["sequence_id", "cell_id"]:
            cols.remove(c)

        if clonekey in self._data:
            if not all_missing2(self._data[clonekey]):
                cols = [clonekey] + cols

        metadata_status = self._metadata
        if (metadata_status is None) or reinitialize:
            self._initialize_metadata(
                cols,
                clonekey,
                v_call_key,
                collapse_alleles,
                report_status_productive,
                reinitialize,
                custom_isotype_dict,
            )

        tmp_metadata = self._metadata.copy()

        if retrieve is not None:
            ret_dict = {}
            if type(retrieve) is str:
                retrieve = [retrieve]
            if self.querier is None:
                querier = Query(self._data)
                self.querier = querier
            else:
                if any([r not in self.querier.data for r in retrieve]):
                    querier = Query(self._data)
                    self.querier = querier
                else:
                    querier = self.querier

            if type(retrieve_mode) is str:
                retrieve_mode = [retrieve_mode]
                if len(retrieve) > len(retrieve_mode):
                    retrieve_mode = [x for x in retrieve_mode for i in retrieve]
            for ret, mode in zip(retrieve, retrieve_mode):
                ret_dict.update(
                    {
                        ret: {
                            "query": ret,
                            "retrieve_mode": mode,
                        }
                    }
                )

            vdj_gene_ret = ["v_call", "d_call", "j_call"]

            retrieve_ = defaultdict(dict)
            for k, v in ret_dict.items():
                if k in self._data.columns:
                    if by_celltype:
                        retrieve_[k] = querier.retrieve_celltype(**v)
                    else:
                        retrieve_[k] = querier.retrieve(**v)
                else:
                    raise KeyError(
                        "Cannot retrieve '%s' : Unknown column name." % k
                    )
            ret_metadata = pd.concat(retrieve_.values(), axis=1, join="inner")
            ret_metadata.dropna(axis=1, how="all", inplace=True)
            for col in ret_metadata:
                if all_missing(ret_metadata[col]):
                    ret_metadata.drop(col, axis=1, inplace=True)

            if collapse_alleles:
                for k in ret_dict.keys():
                    if k in vdj_gene_ret:
                        for c in ret_metadata:
                            if k in c:
                                ret_metadata[c] = [
                                    "|".join(
                                        [
                                            "|".join(list(set(yy.split(","))))
                                            for yy in list(
                                                {
                                                    re.sub(
                                                        "[*][0-9][0-9]",
                                                        "",
                                                        tx,
                                                    )
                                                    for tx in t.split("|")
                                                }
                                            )
                                        ]
                                    )
                                    for t in ret_metadata[c]
                                ]

            for r in ret_metadata:
                tmp_metadata[r] = pd.Series(ret_metadata[r])

            for dcol in [
                "d_sequence_alignment_aa_VJ",
                "d_sequence_alignment_VJ",
            ]:
                if dcol in tmp_metadata:
                    tmp_metadata.drop(dcol, axis=1, inplace=True)
            self._metadata = tmp_metadata.copy()
        # move clonekey and {clonekey}_rank to the front
        if clonekey in self._metadata:
            clonekey_rank = clonekey + "_rank"
            self._metadata = self._metadata[
                [clonekey, clonekey_rank]
                + [
                    c
                    for c in self._metadata.columns
                    if c not in [clonekey, clonekey_rank]
                ]
            ]

    def write_airr(
        self, filename: str = "dandelion_airr.tsv", **kwargs
    ) -> None:
        """
        Writes a Dandelion class to AIRR formatted .tsv format.

        Parameters
        ----------
        filename : str, optional
            path to `.tsv` file.
        **kwargs
            passed to `pandas.DataFrame.to_csv`.
        """
        data = sanitize_data(self._data)
        data.to_csv(filename, sep="\t", index=False, **kwargs)

    def write_h5ddl(
        self,
        filename: str = "dandelion_data.h5ddl",
        compression: (
            Literal[
                "gzip",
                "lzf",
                "szip",
            ]
            | None
        ) = None,
        compression_level: int | None = None,
    ):
        """
        Writes a Dandelion class to .h5ddl format.

        Parameters
        ----------
        filename : str, optional
            path to `.h5ddl` file.
        compression : Literal["gzip", "lzf", "szip"], optional
            Specifies the compression algorithm to use.
        compression_level : int | None, optional
            Specifies a compression level for data. A value of 0 disables compression.
        """
        save_args = {
            "compression": compression,
            "compression_opts": (
                9 if compression_level is None else compression_level
            ),
        }
        if compression is None:
            save_args.pop("compression", None)
            save_args.pop("compression_opts", None)
        clear_h5file(filename)

        # now to actually saving
        data = self._data.copy()
        data = sanitize_data(data)
        data, data_dtypes = sanitize_data_for_saving(data)
        # Convert the DataFrame to a NumPy structured array
        structured_data_array = np.array(
            [tuple(row) for row in data.to_numpy()], dtype=data_dtypes
        )
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(
                "data",
                data=structured_data_array,
                **save_args,
            )
        if self.metadata is not None:
            metadata = self._metadata.copy()
            metadata, metadata_dtypes = sanitize_data_for_saving(metadata)
            # Convert the DataFrame to a NumPy structured array
            structured_metadata_array = np.array(
                [tuple(row) for row in metadata.to_numpy()],
                dtype=metadata_dtypes,
            )
            structured_metadata_names_array = np.array(
                [s.encode("utf-8") for s in metadata.index.to_numpy()]
            )
            with h5py.File(filename, "a") as hf:
                hf.create_dataset(
                    "metadata",
                    data=structured_metadata_array,
                    **save_args,
                )
                hf.create_dataset(
                    "metadata_names",
                    data=structured_metadata_names_array,
                    **save_args,
                )
        if self.graph is not None:
            for i, g in enumerate(self.graph):
                G_df = nx.to_pandas_adjacency(g, nonedge=np.nan)
                G_x = csr_matrix(G_df.to_numpy())
                G_column_array = np.array(
                    [s.encode("utf-8") for s in G_df.columns.to_numpy()]
                )
                G_index_array = np.array(
                    [s.encode("utf-8") for s in G_df.index.to_numpy()]
                )
                with h5py.File(filename, "a") as hf:
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/data",
                        data=G_x.data,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/indices",
                        data=G_x.indices,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/indptr",
                        data=G_x.indptr,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/shape",
                        data=G_x.shape,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/column",
                        data=G_column_array,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{str(i)}/index",
                        data=G_index_array,
                        **save_args,
                    )
        if self.distances is not None and isinstance(
            self.distances, csr_matrix
        ):
            with h5py.File(filename, "a") as hf:
                hf.create_dataset(
                    "distances/data",
                    data=self.distances.data,
                    **save_args,
                )
                hf.create_dataset(
                    "distances/indices",
                    data=self.distances.indices,
                    **save_args,
                )
                hf.create_dataset(
                    "distances/indptr",
                    data=self.distances.indptr,
                    **save_args,
                )
                hf.create_dataset(
                    "distances/shape",
                    data=self.distances.shape,
                    **save_args,
                )
        if self.layout is not None:
            for i, l in enumerate(self.layout):
                with h5py.File(filename, "a") as hf:
                    layout_group = hf.create_group("layout/layout_" + str(i))
                    # Iterate through the dictionary and create datasets in the "layout" group
                    for key, value in l.items():
                        layout_group.create_dataset(
                            key,
                            data=value,
                            **save_args,
                        )
        if len(self.germline) > 0:
            with h5py.File(filename, "a") as hf:
                hf.create_dataset(
                    "germline/keys",
                    data=np.array(list(self.germline.keys()), dtype="S"),
                    **save_args,
                )
                hf.create_dataset(
                    "germline/values",
                    data=np.array(list(self.germline.values()), dtype="S"),
                    **save_args,
                )

    write = write_h5ddl

    def write_10x(
        self,
        folder: Path | str = "dandelion_data",
        filename_prefix: str = "all",
        sequence_key: str = "sequence",
        clone_key: str = "clone_id",
    ) -> None:
        """
        Writes a Dandelion class to 10x formatted files so that it can be ingested for other tools.

        Parameters
        ----------
        folder : Path | str, optional
            path to save the 10x formatted files.
        filename_prefix : str, optional
            prefix for the 10x formatted files.
        sequence_key : str, optional
            column name in `.data` slot to retrieve and write out in fasta format.
        clone_key : str, optional
            column name in `.data` slot for clone id information.
        """
        folder = Path(folder) if isinstance(folder, str) else folder
        folder.mkdir(parents=True, exist_ok=True)
        out_fasta = folder / f"{filename_prefix}_contig.fasta"
        out_anno_path = folder / f"{filename_prefix}_contig_annotations.csv"

        seqs = self._data[sequence_key].to_dict()
        write_fasta(seqs, out_fasta=out_fasta)

        # also create the contig_annotations.csv
        column_map = {
            "barcode": "cell_id",
            "is_cell": "is_cell_10x",
            "contig_id": "sequence_id",
            "high_confidence": "high_confidence_10x",
            "length": "length",
            "chain": "locus",
            "v_gene": "v_call",
            "d_gene": "d_call",
            "j_gene": "j_call",
            "c_gene": "c_call",
            "full_length": "complete_vdj",
            "productive": "productive",
            "cdr3": "junction_aa",
            "cdr3_nt": "junction",
            "reads": "consensus_count",
            "umis": "umi_count",
            "raw_clonotype_id": clone_key,
            "raw_consensus_id": clone_key,
        }
        if "complete_vdj" not in self._data.columns:
            column_map.pop("full_length")
        if "is_cell_10x" not in self._data.columns:
            column_map.pop("is_cell")
        if "high_confidence_10x" not in self._data.columns:
            column_map.pop("high_confidence")
        anno = []
        bool_map = {
            "T": "True",
            "F": "False",
            "True": "True",
            "False": "False",
            "TRUE": "True",
            "FALSE": "False",
        }
        for _, r in self._data.iterrows():
            info = []
            for v in column_map.values():
                if v in r.index:
                    info.append(r[v])
                elif v in ["is_cell", "high_confidence"]:
                    info.append("True")
                elif v == "length":
                    info.append(len(r[sequence_key]))
            anno.append({k: r for k, r in zip(column_map.keys(), info)})
        anno = pd.DataFrame(anno)
        anno = anno.map(lambda x: bool_map[x] if x in bool_map.keys() else x)
        anno.to_csv(out_anno_path, index=False)


class Query:
    """Query class"""

    def __init__(self, data: pd.DataFrame, verbose=False) -> None:
        """
        Query class to retrieve data from the Dandelion object.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe to query.
        verbose : bool, optional
            Whether to print the process, by default False.
        """
        self.data = data.copy()
        self.Cell = Tree()
        for contig, row in tqdm(
            data.iterrows(),
            desc="Setting up data",
            disable=not verbose,
        ):
            self.Cell[row["cell_id"]][contig].update(row)

    @property
    def querydtype(self):
        """Check dtype."""
        return str(self.data[self.query].dtype)

    def retrieve(
        self,
        query: str,
        retrieve_mode: Literal[
            "split and unique only",
            "merge and unique only",
            "split and merge",
            "split and sum",
            "split and average",
            "split",
            "merge",
            "sum",
            "average",
        ],
    ) -> pd.DataFrame:
        """
        Retrieve query.

        Parameters
        ----------
        query : str
            column name in `.data` slot to retrieve and update the metadata.
        retrieve_mode : Literal["split and unique only", "merge and unique only", "split and merge", "split and sum", "split and average", "split", "merge", "sum", "average", ]
            one of:
                `split and unique only`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for unique elements.
                `merge and unique only`
                    returns the retrieval merged into one column,
                    separated by `|` for unique elements.
                `split and merge`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for every elements.
                `split`
                    returns the retrieval splitted into separate columns for each contig.
                `merge`
                    returns the retrieval merged into one columns for each contig,
                    separated by `|` for unique elements.
                `split and sum`
                    returns the retrieval sum in the VDJ and VJ columns (separately).
                `split and average`
                    returns the retrieval averaged in the VDJ and VJ columns (separately).
                `sum`
                    returns the retrieval sum into one column for all contigs.
                `average`
                    returns the retrieval averaged into one column for all contigs.
        Returns
        -------
        pd.DataFrame
            Retrieved data.
        """
        self.query = query
        ret = {}
        for cell in self.Cell:
            cols, vdj, vj = {}, [], []
            for _, contig in self.Cell[cell].items():
                if isinstance(contig, dict):
                    if contig["locus"] in ["IGH", "TRB", "TRD"]:
                        vdj.append(contig[query])
                    elif contig["locus"] in ["IGK", "IGL", "TRA", "TRG"]:
                        vj.append(contig[query])
            if retrieve_mode == "split and unique only":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(vdj))
                                if present(x)
                            )
                        }
                    )
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(vj))
                                if present(x)
                            )
                        }
                    )
            elif retrieve_mode == "split and merge":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": "|".join(
                                str(x) for x in vdj if present(x)
                            )
                        }
                    )
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": "|".join(str(x) for x in vj if present(x))
                        }
                    )
            elif retrieve_mode == "merge and unique only":
                cols.update(
                    {
                        query: "|".join(
                            str(x) for x in set(vdj + vj) if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split and sum":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": np.sum(
                                [float(x) for x in vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VDJ": np.nan})
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": np.sum(
                                [float(x) for x in vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VJ": np.nan})
            elif retrieve_mode == "split and average":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": np.mean(
                                [float(x) for x in vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VDJ": np.nan})
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": np.mean(
                                [float(x) for x in vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VJ": np.nan})
            elif retrieve_mode == "merge":
                cols.update(
                    {query: "|".join(x for x in (vdj + vj) if present(x))}
                )
            elif retrieve_mode == "split":
                if len(vdj) > 0:
                    for i in range(1, len(vdj) + 1):
                        cols.update({query + "_VDJ_" + str(i): vdj[i - 1]})
                if len(vj) > 0:
                    for i in range(1, len(vj) + 1):
                        cols.update({query + "_VJ_" + str(i): vj[i - 1]})
            elif retrieve_mode == "sum":
                cols.update(
                    {query: np.sum([float(x) for x in vdj + vj if present(x)])}
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            elif retrieve_mode == "average":
                cols.update(
                    {query: np.mean([float(x) for x in vdj + vj if present(x)])}
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            ret.update({cell: cols})
        out = pd.DataFrame.from_dict(ret, orient="index")
        if retrieve_mode not in [
            "split and sum",
            "split and average",
            "sum",
            "average",
        ]:
            if retrieve_mode == "split":
                for x in out:
                    try:
                        out[x] = pd.to_numeric(out[x])
                    except:
                        out[x] = out[x].fillna("None")
            else:
                out = out.fillna("None")
        return out

    def retrieve_celltype(
        self,
        query: str,
        retrieve_mode: Literal[
            "split and unique only",
            "merge and unique only",
            "split and merge",
            "split and sum",
            "split and average",
            "split",
            "merge",
            "sum",
            "average",
        ],
    ) -> pd.DataFrame:
        """
        Retrieve query split by celltype.

        Parameters
        ----------
        query : str
            column name in `.data` slot to retrieve and update the metadata.
        retrieve_mode : Literal["split and unique only", "merge and unique only", "split and merge", "split and sum", "split and average", "split", "merge", "sum", "average", ]
            one of:
                `split and unique only`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for unique elements.
                `merge and unique only`
                    returns the retrieval merged into one column,
                    separated by `|` for unique elements.
                `split and merge`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for every elements.
                `split`
                    returns the retrieval splitted into separate columns for each contig.
                `merge`
                    returns the retrieval merged into one columns for each contig,
                    separated by `|` for unique elements.
                `split and sum`
                    returns the retrieval sum in the VDJ and VJ columns (separately).
                `split and average`
                    returns the retrieval averaged in the VDJ and VJ columns (separately).
                `sum`
                    returns the retrieval sum into one column for all contigs.
                `average`
                    returns the retrieval averaged into one column for all contigs.

        Returns
        -------
        pd.DataFrame
            Retrieved data.
        """
        self.query = query
        ret = {}
        for cell in self.Cell:
            cols, abt_vdj, gdt_vdj, b_vdj, abt_vj, gdt_vj, b_vj = (
                {},
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for _, contig in self.Cell[cell].items():
                if isinstance(contig, dict):
                    if contig["locus"] in ["IGH"]:
                        b_vdj.append(contig[query])
                    elif contig["locus"] in ["IGK", "IGL"]:
                        b_vj.append(contig[query])
                    elif contig["locus"] in ["TRB"]:
                        abt_vdj.append(contig[query])
                    elif contig["locus"] in ["TRD"]:
                        gdt_vdj.append(contig[query])
                    elif contig["locus"] in ["TRA"]:
                        abt_vj.append(contig[query])
                    elif contig["locus"] in ["TRG"]:
                        gdt_vj.append(contig[query])
            if retrieve_mode == "split and unique only":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(b_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(b_vj))
                                if present(x)
                            )
                        }
                    )

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(abt_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(abt_vj))
                                if present(x)
                            )
                        }
                    )

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(gdt_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(gdt_vj))
                                if present(x)
                            )
                        }
                    )
            elif retrieve_mode == "split and merge":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": "|".join(
                                str(x) for x in b_vdj if present(x)
                            )
                        }
                    )
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": "|".join(
                                str(x) for x in b_vj if present(x)
                            )
                        }
                    )

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": "|".join(
                                str(x) for x in abt_vdj if present(x)
                            )
                        }
                    )
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": "|".join(
                                str(x) for x in abt_vj if present(x)
                            )
                        }
                    )

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": "|".join(
                                str(x) for x in gdt_vdj if present(x)
                            )
                        }
                    )
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": "|".join(
                                str(x) for x in gdt_vj if present(x)
                            )
                        }
                    )

            elif retrieve_mode == "merge and unique only":
                cols.update(
                    {
                        query: "|".join(
                            str(x)
                            for x in set(
                                b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                            )
                            if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split and sum":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": np.sum(
                                [float(x) for x in b_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VDJ": np.nan})
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": np.sum(
                                [float(x) for x in b_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VJ": np.nan})

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": np.sum(
                                [float(x) for x in abt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VDJ": np.nan})
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": np.sum(
                                [float(x) for x in abt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VJ": np.nan})

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": np.sum(
                                [float(x) for x in gdt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VDJ": np.nan})
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": np.sum(
                                [float(x) for x in gdt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VJ": np.nan})
            elif retrieve_mode == "split and average":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": np.mean(
                                [float(x) for x in b_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VDJ": np.nan})
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": np.mean(
                                [float(x) for x in b_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VJ": np.nan})

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": np.mean(
                                [float(x) for x in abt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VDJ": np.nan})
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": np.mean(
                                [float(x) for x in abt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VJ": np.nan})

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": np.mean(
                                [float(x) for x in gdt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VDJ": np.nan})
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": np.mean(
                                [float(x) for x in gdt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VJ": np.nan})
            elif retrieve_mode == "merge":
                cols.update(
                    {
                        query: "|".join(
                            x
                            for x in (
                                b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                            )
                            if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split":
                if len(b_vdj) > 0:
                    for i in range(1, len(b_vdj) + 1):
                        cols.update({query + "_B_VDJ_" + str(i): b_vdj[i - 1]})
                if len(b_vj) > 0:
                    for i in range(1, len(b_vj) + 1):
                        cols.update({query + "_B_VJ_" + str(i): b_vj[i - 1]})
                if len(abt_vdj) > 0:
                    for i in range(1, len(abt_vdj) + 1):
                        cols.update(
                            {query + "_abT_VDJ_" + str(i): abt_vdj[i - 1]}
                        )
                if len(abt_vj) > 0:
                    for i in range(1, len(abt_vj) + 1):
                        cols.update(
                            {query + "_abT_VJ_" + str(i): abt_vj[i - 1]}
                        )
                if len(gdt_vdj) > 0:
                    for i in range(1, len(gdt_vdj) + 1):
                        cols.update(
                            {query + "_gdT_VDJ_" + str(i): gdt_vdj[i - 1]}
                        )
                if len(gdt_vj) > 0:
                    for i in range(1, len(gdt_vj) + 1):
                        cols.update(
                            {query + "_gdT_VJ_" + str(i): gdt_vj[i - 1]}
                        )
            elif retrieve_mode == "sum":
                cols.update(
                    {
                        query: np.sum(
                            [
                                float(x)
                                for x in b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                                if present(x)
                            ]
                        )
                    }
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            elif retrieve_mode == "average":
                cols.update(
                    {
                        query: np.mean(
                            [
                                float(x)
                                for x in b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                                if present(x)
                            ]
                        )
                    }
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            ret.update({cell: cols})
        out = pd.DataFrame.from_dict(ret, orient="index")
        if retrieve_mode not in [
            "split and sum",
            "split and average",
            "sum",
            "average",
        ]:
            if retrieve_mode == "split":
                for x in out:
                    try:
                        out[x] = pd.to_numeric(out[x])
                    except:
                        out[x] = out[x].fillna("None")
            else:
                out = out.fillna("None")
        return out


def _normalize_indices(
    index: Index | None, names0: pd.Index, names1: pd.Index
) -> tuple[slice, str]:
    """Return indices"""
    # deal with tuples of length 1
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    # deal with pd.Series
    if isinstance(index, pd.Series):
        index = index.values
    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError(
                "Dandelion can only be sliced in data or metadata rows."
            )
        # deal with pd.Series
        # TODO: The series should probably be aligned first
        if isinstance(index[1], pd.Series):
            index = index[0], index[1].values
        if isinstance(index[0], pd.Series):
            index = index[0].values, index[1]
    ax0_, _ = unpack_index(index)
    if all(ax_ in names0 for ax_ in ax0_):
        ax0 = _normalize_index(ax0_, names0)
        axtype = "metadata"
    elif all(ax_ in names1 for ax_ in ax0_):
        ax0 = _normalize_index(ax0_, names1)
        axtype = "data"
    return ax0, axtype


def return_none_call(call: str) -> str:
    """Return None if not present."""
    return call.split("|")[0] if not call in ["None", ""] else "None"


def clean_clone_list(clone_series: pd.Series) -> pd.Series:
    """Replace empty clones with 'None', remove duplicates, sort consistently."""
    clone_series = clone_series.replace("", "None")
    clone_series = clone_series.str.split("|").apply(
        lambda x: [c for c in x if c != "None"] or ["None"]
    )
    clone_series = clone_series.apply(
        lambda x: "|".join(
            sorted(set(x), key=cmp_to_key(lambda a, b: (a > b) - (a < b)))
        )
    )
    return clone_series


def flatten_and_count(tmp_metadata: pd.DataFrame, clonekey: str) -> pd.Series:
    """Return a Series of clone counts for all unique clones."""
    tmp = tmp_metadata[clonekey].str.split("|", expand=True).stack()
    clone_counts = tmp.value_counts()
    clone_counts = clone_counts.drop("None", errors="ignore")
    return clone_counts


def get_receptor_prefix(clone: str) -> str:
    """Return receptor type prefix if matches RECEPTOR_SET, else None."""
    prefix = clone.split("_")[0]
    return prefix if prefix in RECEPTOR_SET else None


def assign_clone_numbers(clone_counts: pd.Series) -> dict:
    """Assign sequential numbers, possibly grouped by receptor type."""
    # Determine all receptor types present
    prefixes = {get_receptor_prefix(clone) for clone in clone_counts.index}
    prefixes.discard(None)

    size_dict = {}
    if len(prefixes) <= 1:
        # Only 1 receptor type (or none): number sequentially without prefix
        for i, clone in enumerate(clone_counts.index, start=1):
            size_dict[clone] = str(i)
    else:
        # Multiple receptor types: number sequentially per type
        receptor_to_clones = {r: [] for r in RECEPTOR_SET}
        other_clones = []
        for clone in clone_counts.index:
            prefix = get_receptor_prefix(clone)
            if prefix in RECEPTOR_SET:
                receptor_to_clones[prefix].append(clone)
            else:
                other_clones.append(clone)
        # Sort each receptor group by descending size
        for r in receptor_to_clones:
            receptor_to_clones[r].sort(key=lambda c: -clone_counts[c])
        other_clones.sort(key=lambda c: -clone_counts[c])
        # Assign numbers
        for r, clones in receptor_to_clones.items():
            for i, clone in enumerate(clones, start=1):
                size_dict[clone] = f"{r}_{i}"
        for i, clone in enumerate(other_clones, start=1):
            size_dict[clone] = f"other_{i}" if clone != "None" else "None"
    return size_dict


def _add_clone_info(tmp_metadata: pd.DataFrame, clonekey: str) -> pd.DataFrame:
    """Add a `{clonekey}_rank` column to tmp_metadata with sequential numbering per receptor type based on clone size."""
    tmp_metadata[clonekey] = clean_clone_list(tmp_metadata[clonekey])
    clone_counts = flatten_and_count(tmp_metadata, clonekey)
    size_dict = assign_clone_numbers(clone_counts)

    # Map multi-clone entries
    tmp_metadata[clonekey + "_rank"] = (
        tmp_metadata[clonekey]
        .apply(
            lambda entry: "|".join(
                size_dict.get(p, p) for p in entry.split("|")
            )
        )
        .astype("category")
    )

    # Reorder columns
    tmp_metadata = tmp_metadata[
        [clonekey, clonekey + "_rank"]
        + [
            c
            for c in tmp_metadata.columns
            if c not in [clonekey, clonekey + "_rank"]
        ]
    ]

    return tmp_metadata


@deprecated(
    deprecated_in="1.0.0",
    removed_in="1.1.0",
    details="legacy .h5ddl format will no longer be supported.",
)
def write_h5ddl_legacy(
    self: Dandelion,
    filename: Path | str = "dandelion_data.h5ddl",
    **kwargs,
) -> None:  # pragma: no cover
    """
    Writes a Dandelion class to .h5ddl format for legacy support.

    Parameters
    ----------
    self : Dandelion
        input Dandelion object.
    filename : Path | str, optional
        path to `.h5ddl` file, by default "dandelion_data.h5ddl".
    **kwargs
        Additional arguments to `pd.DataFrame.to_hdf`.
    """
    clear_h5file(filename)
    # now to actually saving
    data = self._data.copy()
    data = sanitize_data(data)
    data, _ = sanitize_data_for_saving(data)
    data.to_hdf(
        filename,
        "data",
        **kwargs,
    )
    if self.metadata is not None:
        metadata = self._metadata.copy()
        for col in metadata.columns:
            if pd.__version__ < "2.1.0":
                weird = (
                    metadata[[col]].applymap(type)
                    != metadata[[col]].iloc[0].apply(type)
                ).any(axis=1)
            else:
                weird = (
                    metadata[[col]].map(type)
                    != metadata[[col]].iloc[0].apply(type)
                ).any(axis=1)
            if len(metadata[weird]) > 0:
                metadata[col] = metadata[col].where(
                    pd.notnull(metadata[col]), ""
                )
        metadata.to_hdf(
            filename,
            "metadata",
            format="table",
            nan_rep=np.nan,
            **kwargs,
        )
    graph_counter = 0
    try:
        for g in self.graph:
            G = nx.to_pandas_adjacency(g, nonedge=np.nan)
            G.to_hdf(
                filename,
                "graph/graph_" + str(graph_counter),
                **kwargs,
            )
            graph_counter += 1
    except:
        pass
    with h5py.File(filename, "a") as hf:
        try:
            layout_counter = 0
            for l in self.layout:
                try:
                    hf.create_group("layout/layout_" + str(layout_counter))
                except:
                    pass
                for k in l.keys():
                    hf["layout/layout_" + str(layout_counter)].attrs[k] = l[k]
                layout_counter += 1
        except:
            pass
        if len(self.germline) > 0:
            try:
                hf.create_group("germline")
            except:
                pass
            for k in self.germline.keys():
                hf["germline"].attrs[k] = self.germline[k]


def load_data(obj: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    """
    Read in or copy dataframe object and set sequence_id as index without dropping.

    Parameters
    ----------
    obj : pd.DataFrame | Path | str | None
        file path to .tsv file or pandas DataFrame object.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        if input is not found.
    KeyError
        if `sequence_id` not found in input.
    """
    if obj is not None:
        if os.path.isfile(str(obj)):
            obj_ = pd.read_csv(obj, sep="\t")
        elif isinstance(obj, pd.DataFrame):
            obj_ = obj.copy()
        else:
            raise ValueError(
                "Either input is not of pandas DataFrame or AIRR file does not exist."
            )

        if "sequence_id" in obj_.columns:
            # assert that sequence_id is string
            obj_["sequence_id"] = obj_["sequence_id"].astype(str)
            obj_.set_index("sequence_id", drop=False, inplace=True)
            if "cell_id" not in obj_.columns:
                obj_["cell_id"] = [
                    c.split("_contig")[0] if "_contig" in c else c
                    for c in obj_["sequence_id"]
                ]
            # assert that cell_id is string
            obj_["cell_id"] = obj_["cell_id"].astype(str)
        else:
            raise KeyError("'sequence_id' not found in columns of input")

        if "duplicate_count" in obj_.columns:
            if "umi_count" not in obj_.columns:
                obj_.rename(
                    columns={"duplicate_count": "umi_count"}, inplace=True
                )

        return obj_


def write_fasta(
    fasta_dict: dict[str, str], out_fasta: Path | str, overwrite=True
) -> None:
    """
    Generic fasta writer using fasta_iterator

    Parameters
    ----------
    fasta_dict : dict[str, str]
        dictionary containing fasta headers and sequences as keys and records respectively.
    out_fasta : str
        path to write fasta file to.
    overwrite : bool, optional
        whether or not to overwrite the output file (out_fasta).
    """
    if overwrite:
        fh = open(out_fasta, "w")
        fh.close()
    out = ""
    for l in fasta_dict:
        out = ">" + l + "\n" + fasta_dict[l] + "\n"
        write_output(out, out_fasta)


def write_output(out: str, file: Path | str) -> None:
    """General line writer."""
    fh = open(file, "a")
    fh.write(out)
    fh.close()


def check_travdv(data: pd.DataFrame) -> pd.DataFrame:
    """Check if locus is TRA/D."""
    data = load_data(data)
    contig = [x for x in data["sequence_id"]]
    v = [x for x in data["v_call"]]
    d = [x for x in data["d_call"]]
    j = [x for x in data["j_call"]]
    c = [x for x in data["c_call"]]
    l = [x for x in data["locus"]]
    v_dict = dict(zip(contig, v))
    d_dict = dict(zip(contig, d))
    j_dict = dict(zip(contig, j))
    c_dict = dict(zip(contig, c))
    l_dict = dict(zip(contig, l))
    for co in contig:
        if re.search("TRAV.*/DV", v_dict[co]):
            if same_call(j_dict[co], c_dict[co], d_dict[co], "TRA"):
                if not re.search("TRA", l_dict[co]):
                    l_dict[co] = "TRA"
            elif same_call(j_dict[co], c_dict[co], d_dict[co], "TRD"):
                if not re.search("TRD", l_dict[co]):
                    l_dict[co] = "TRD"
    data["locus"] = pd.Series(l_dict)
    return data
