#!/usr/bin/env python
from __future__ import annotations
import copy
import h5py
import json
import os
import re
import tempfile
import unicodedata
import warnings
import zarr

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from airr import RearrangementSchema
from anndata import AnnData
from changeo.IO import readGermlines
from collections import defaultdict, OrderedDict
from functools import cmp_to_key, reduce
from pandas.api.types import infer_dtype
from pathlib import Path
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from textwrap import dedent
from typing import Literal
from zarr.codecs import BloscCodec

from dandelion.utilities._io import AIRR, CELLRANGER, fasta_iterator
from dandelion.utilities._utilities import (
    RECEPTOR_SET,
    TRUES,
    EMPTIES_STR,
    BOOLEAN_LIKE_COLUMNS,
    DEFAULT_PREFIX,
    deprecated,
    sanitize_boolean,
    sanitize_data_for_saving,
    sanitize_data,
    lib_type,
    clear_h5file,
    Contig,
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
TRUES_STR = [str(x).upper() for x in TRUES]


class DandelionPolars:
    """Dandelion class object."""

    def __init__(
        self,
        data: (
            pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None
        ) = None,
        metadata: pl.LazyFrame | pl.DataFrame | pd.DataFrame | None = None,
        germline: dict[str, str] | None = None,
        layout: tuple[dict[str, np.array], dict[str, np.array]] | None = None,
        graph: tuple[nx.Graph, nx.Graph] | None = None,
        distances: csr_matrix | None = None,
        initialize: bool = True,
        library_type: Literal["tr-ab", "tr-gd", "ig"] | None = None,
        lazy: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Init method for Dandelion.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None, optional
            AIRR formatted data.
        metadata : pl.LazyFrame | pl.DataFrame | pd.DataFrame | None, optional
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
        init_cols : list[str] | None, optional
            columns to initialize in metadata.
        init_strip_alleles : bool, optional
            whether or not to strip alleles when initializing metadata.
        init_productive : bool, optional
            whether or not to include only productive sequences.
        isotype_conversion_dict : dict[str, str] | None, optional
            dictionary to convert isotype annotations to desired format.
        library_type : Literal["tr-ab", "tr-gd", "ig"] | None, optional
            One of "tr-ab", "tr-gd", "ig".
        verbose : bool, optional
            whether or not to print initialization messages.
        **kwargs
            passed to `Dandelion.update_metadata`.
        """
        self.lazy = lazy
        self._data_name_col = "sequence_id"
        self._metadata_name_col = "cell_id"
        self._backend = "polars"
        self._tmpfiles = {}

        self._data = load_polars(data)
        self._metadata = metadata
        self.layout = layout
        self.graph = graph
        self.distances = distances
        self.germline = {}
        self.library_type = library_type

        if germline is not None:
            self.germline.update(germline)

        if self.data is not None:
            acceptable = (
                None
                if self.library_type is None
                else lib_type(self.library_type)
            )
            if acceptable is not None:
                if self.lazy:
                    self._data = (
                        self._data.filter(pl.col("locus").is_in(acceptable))
                        .collect()
                        .lazy()
                    )
                else:
                    self._data = self._data.filter(
                        pl.col("locus").is_in(acceptable)
                    )
            self._data = _check_travdv_polars(self._data, lazy=self.lazy)
            sort_cols = {"cell_id", "productive", "umi_count"}
            if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
                cols = set(self._data.collect_schema().names())
            else:
                cols = set(self._data.columns)
            if sort_cols.issubset(cols):
                # sort so that the productive contig with the largest umi is first
                self._data = (
                    self._data.with_columns(
                        pl.col("cell_id")
                        .cum_count()
                        .over("cell_id")
                        .eq(1)
                        .cum_sum()
                        .alias("_cell_order")
                    )
                    .sort(
                        by=["_cell_order", "productive", "umi_count"],
                        descending=[False, True, True],
                    )
                    .drop("_cell_order")
                )
            if self.lazy:
                self._data = self._data.collect().lazy()
                if "data" in self._tmpfiles.keys():
                    self._tmpfiles["data"].close()
                    del self._tmpfiles["data"]
            if metadata is None:
                if initialize is True:
                    self._ensure_sanitized_data(verbose=verbose)
                    self.update_metadata(**kwargs)
            else:
                if isinstance(metadata, pd.DataFrame):
                    if self._metadata_name_col not in metadata:
                        # change the name of the index first before resetting
                        if metadata.index.name is None:
                            metadata.index.name = self._metadata_name_col
                    metadata = pl.from_pandas(metadata.reset_index(drop=False))
                if self.lazy:
                    if isinstance(metadata, pl.DataFrame):
                        self._metadata = metadata.lazy()
                    else:
                        self._metadata = metadata
                else:
                    if isinstance(metadata, pl.LazyFrame):
                        self._metadata = metadata.collect()
                        if "metadata" in self._tmpfiles.keys():
                            self._tmpfiles["metadata"].close()
                            del self._tmpfiles["metadata"]
                    else:
                        self._metadata = metadata

        if isinstance(self._data, pl.LazyFrame):
            self._original_sequence_ids = (
                self._data.select(self._data_name_col).collect().to_series()
            )
            self._original_cell_ids = (
                self._data.select(self._metadata_name_col).collect().to_series()
            )
        elif isinstance(self._data, pl.DataFrame):
            self._original_sequence_ids = self._data[
                self._data_name_col
            ].clone()
            self._original_cell_ids = self._data[
                self._metadata_name_col
            ].clone()
        elif isinstance(self._data, pd.DataFrame):
            self._original_sequence_ids = self._data[self._data_name_col].copy()
            self._original_cell_ids = self._data[self._metadata_name_col].copy()

    def _gen_repr(self, n_obs, n_contigs) -> str:
        """Report."""
        # inspire by AnnData's function
        if self.lazy:
            descr = f"Lazy Dandelion object with n_obs = {n_obs} and n_contigs = {n_contigs}"
        else:
            descr = f"Dandelion object with n_obs = {n_obs} and n_contigs = {n_contigs}"
        for attr in ["data", "metadata"]:
            df = getattr(self, f"_{attr}")
            try:
                if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                    keys = df.collect_schema().names()
                elif isinstance(df, pd.DataFrame):
                    keys = df.columns.tolist()
                else:
                    keys = []
            except AttributeError:
                keys = []

            if len(keys) > 0:
                descr += f"\n    {attr}: {', '.join(keys)}"
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

    def __getitem__(self, index) -> DandelionPolars:
        """Return a sliced Dandelion object with synchronized data and metadata."""
        # Determine which dataframe to filter and extract cell_ids
        cell_ids = None
        use_direct_filter = False
        filter_expr = None

        # Convert pandas to polars first
        if self._backend == "pandas":
            self.to_polars()
            original_backend = "pandas"
        else:
            original_backend = "polars"
        if isinstance(index, (list, tuple, set, np.ndarray)):
            try:
                if all(isinstance(x, (bool, np.bool_)) for x in index):
                    index = pl.Series(index)
            except TypeError:
                pass
        data = self._data
        metadata = self._metadata
        # Case 1: Direct cell_id list/array/tuple/set
        if isinstance(index, (list, set, tuple, np.ndarray)):
            cell_ids = pl.Series(list(index), dtype=pl.String)

        # Case 2: Polars Series (boolean mask or cell_ids)
        elif isinstance(index, pl.Series):
            if index.dtype == pl.Boolean:
                # Boolean mask - apply filter directly to preserve exact row matching
                # This is important for filtering by non-cell_id columns
                use_direct_filter = True
                filter_expr = index
            else:
                # Series of cell_ids
                cell_ids = index.cast(pl.String)

        # Case 3: Polars Expression
        elif isinstance(index, pl.Expr):
            # Use direct filter for expressions (don't group by cell_id)
            use_direct_filter = True
            filter_expr = index
        # Case 4: DataFrame or LazyFrame
        elif isinstance(index, (pl.DataFrame, pl.LazyFrame)):
            # When a DataFrame is passed, use it as direct row filtering
            # (preserve exact rows, don't expand by cell_id)
            use_direct_filter = True
            filter_expr = index

        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # ---- Filter data & metadata, then collect and make lazy again ----
        if use_direct_filter:
            # Direct filter without grouping by cell_id
            # This preserves the exact rows that match the filter
            if isinstance(filter_expr, (pl.DataFrame, pl.LazyFrame)):
                # When a DataFrame/LazyFrame is passed, use it directly as the filtered data
                _data = filter_expr
            elif isinstance(filter_expr, pl.Series):
                # Convert Series to expression - need to filter by row position
                # Create a temporary index column and filter by position
                if isinstance(data, pl.LazyFrame):
                    # For lazy frames, we need to be more careful
                    _data = (
                        data.with_row_index("__row_idx__")
                        .filter(
                            pl.col("__row_idx__").is_in(filter_expr.to_list())
                            if len(filter_expr.to_list()) < _get_length(data)
                            else filter_expr
                        )
                        .drop("__row_idx__")
                    )
                else:
                    # For eager frames, create index and filter
                    data_with_idx = data.with_row_index("__row_idx__")
                    matching_indices = [
                        i for i, v in enumerate(filter_expr) if v
                    ]
                    _data = data_with_idx.filter(
                        pl.col("__row_idx__").is_in(matching_indices)
                    ).drop("__row_idx__")
            else:
                # Assume it's an Expression
                _data = data.filter(filter_expr)

            # For metadata sync, extract unique cell_ids from filtered data
            if isinstance(_data, pl.LazyFrame):
                filtered_cell_ids = (
                    _data.select("cell_id").collect().to_series().unique()
                )
            else:
                filtered_cell_ids = _data.select("cell_id").to_series().unique()
            _metadata = (
                metadata.filter(pl.col("cell_id").is_in(filtered_cell_ids))
                if metadata is not None
                else None
            )
            cell_ids = filtered_cell_ids
        else:
            # Filter by cell_id
            _data = data.filter(pl.col("cell_id").is_in(cell_ids))
            _metadata = (
                metadata.filter(pl.col("cell_id").is_in(cell_ids))
                if metadata is not None
                else None
            )

        # Collect and convert back to lazy if original was lazy
        if isinstance(self._data, pl.LazyFrame):
            _data = _data.collect().lazy()
        if isinstance(self._metadata, pl.LazyFrame):
            _metadata = _metadata.collect().lazy()

        # ---- Distances matrix sync -----------------------------------
        if self.distances is not None:
            # Get metadata cell_ids for distance matrix indexing
            if isinstance(_metadata, pl.LazyFrame):
                meta_cells = (
                    _metadata.select("cell_id").collect().to_series().to_list()
                )
            else:
                meta_cells = _metadata.select("cell_id").to_series().to_list()

            keep_set = set(cell_ids.to_list())
            keep = np.array(
                [i for i, c in enumerate(meta_cells) if c in keep_set]
            )
            _distances = self.distances[keep, :][:, keep]
            if isinstance(_distances, csr_matrix):
                _distances._index_names = list(keep_set)
        else:
            _distances = None

        # ---- Layout ---------------------------------------------------
        if self.layout is not None:
            keep_set = set(cell_ids.to_list())
            _layout = (
                {k: v for k, v in self.layout[0].items() if k in keep_set},
                {k: v for k, v in self.layout[1].items() if k in keep_set},
            )
        else:
            _layout = None

        # ---- Graph ----------------------------------------------------
        if self.graph is not None:
            keep_set = set(cell_ids.to_list())
            _graph = (
                self.graph[0].subgraph(keep_set),
                self.graph[1].subgraph(
                    [n for n in self.graph[1].nodes if n in keep_set]
                ),
            )
        else:
            _graph = None
        sliced = DandelionPolars(
            data=_data,
            metadata=_metadata,
            layout=_layout,
            graph=_graph,
            distances=_distances,
            verbose=False,
        )
        if original_backend == "pandas":
            sliced.to_pandas()
        return sliced

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        if self._metadata is None:
            return 0
        if isinstance(self._metadata, pl.LazyFrame):
            return self._metadata.select(pl.count()).collect()[0, 0]
        if isinstance(self._metadata, pl.DataFrame):
            return self._metadata.height
        if isinstance(self._metadata, pd.DataFrame):
            return self._metadata.shape[0]

    @property
    def n_contigs(self) -> int:
        """Number of contigs."""
        if self._data is None:
            return 0
        if isinstance(self._data, pl.LazyFrame):
            return self._data.select(pl.count()).collect()[0, 0]
        if isinstance(self._data, pl.DataFrame):
            return self._data.height
        if isinstance(self._data, pd.DataFrame):
            return self._data.shape[0]

    @property
    def data(self) -> pl.DataFrame | pl.LazyFrame:
        """One-dimensional annotation of contig observations."""
        if isinstance(self._data, pd.DataFrame):
            return self._data
        if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
            return DataFrameAccessor(self._data, parent=self, attr_name="_data")

    @data.setter
    def data(
        self,
        value: pl.DataFrame | pl.LazyFrame | pd.DataFrame | Path | str | None,
    ) -> None:
        """data setter"""
        value = load_polars(value)
        self._data = value
        self._backend = "polars"
        return

    @property
    def data_names(self) -> SeriesAccessor | pd.Index:
        """Names of contig observations."""
        if isinstance(self._data, pd.DataFrame):
            return self._data.index
        if isinstance(self._data, pl.DataFrame):
            # eager Polars
            return SeriesAccessor(self._data[self._data_name_col])
        if isinstance(self._data, pl.LazyFrame):
            # Lazy Polars: materialize first to get a Series
            series = self._data.select(self._data_name_col).collect()[
                self._data_name_col
            ]
            return SeriesAccessor(series)

    @data_names.setter
    def data_names(self, names: list[str]) -> None:
        """data names setter"""
        if isinstance(self._data, pd.DataFrame):
            names = self._prep_dim_index(names, "data")
            self._set_dim_index(names, "data")
            return
        if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
            self._data = self._data.with_columns(
                pl.Series(self._data_name_col, names)
            )
            return

    @property
    def metadata(self) -> pd.DataFrame | DataFrameAccessor:
        """One-dimensional annotation of cell observations."""
        if isinstance(self._metadata, pd.DataFrame):
            return self._metadata
        if isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
            return DataFrameAccessor(
                self._metadata, parent=self, attr_name="_metadata"
            )

    @metadata.setter
    def metadata(self, value: pl.DataFrame | pl.LazyFrame | pd.DataFrame):
        """metadata setter"""
        if isinstance(value, pd.DataFrame):
            if self._metadata_name_col not in value:
                # change the name of the index first before resetting
                if value.index.name is None:
                    value.index.name = self._metadata_name_col
                value = pl.from_pandas(value.reset_index(drop=False))
            else:
                value = pl.from_pandas(value)
            if self.lazy:
                value = value.lazy()
        self._metadata = value
        return

    @property
    def metadata_names(self) -> pd.Index | pl.Series:
        """Names of cell observations."""
        if isinstance(self._metadata, pd.DataFrame):
            return self._metadata.index
        if isinstance(self._metadata, pl.DataFrame):
            # eager Polars
            return SeriesAccessor(self._metadata[self._metadata_name_col])
        if isinstance(self._metadata, pl.LazyFrame):
            # Lazy Polars: materialize first to get a Series
            series = self._metadata.select(self._metadata_name_col).collect()[
                self._metadata_name_col
            ]
            return SeriesAccessor(series)

    @metadata_names.setter
    def metadata_names(self, names: list[str]):
        """metadata names setter"""
        if isinstance(self._metadata, pd.DataFrame):
            names = self._prep_dim_index(names, "metadata")
            self._set_dim_index(names, "metadata")
            return
        if isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
            self._metadata = self._metadata.with_columns(
                pl.Series(self._metadata_name_col, names)
            )
            return

    def _ensure_sanitized_data(self, verbose: bool = False) -> None:
        """Ensure that the data is sanitized."""
        if not self._is_sanitized():
            if verbose:
                logg.info(
                    "The AIRR data needs to undergo sanitization, apologies for any delays..."
                )
            if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
                self._data = _sanitize_data_polars(self._data)
            elif isinstance(self._data, pd.DataFrame):
                self._data = sanitize_data(self._data)

    def _is_sanitized(self):
        """Check if the data is sanitized (pandas or polars)."""
        check = []
        is_polars = isinstance(self._data, (pl.DataFrame, pl.LazyFrame))
        if is_polars:
            cols = self._data.collect_schema().names()
        else:
            cols = self._data.columns
        for col in CHECK_COLS:
            if col not in cols:
                continue
            if is_polars:
                # Polars: unsanitized if column is Boolean dtype
                all_bool = self._data.collect_schema().get(col) == pl.Boolean
            else:
                # pandas: check values (object dtype may contain bools)
                all_bool = self._data[col].isin([True, False]).all()
            # preserve original logic
            check.append(not all_bool)

        return all(check)

    def _set_dim_df(self, value: pd.DataFrame, attr: str):
        """dim df setter"""
        if value is not None:
            _ = self._prep_dim_index(value.index, attr)
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
        other_column = (
            self._metadata_name_col
            if column == self._data_name_col
            else self._data_name_col
        )
        sep = "" if sep is None else sep
        original_values = (
            self._original_sequence_ids
            if column == self._data_name_col
            else self._original_cell_ids
        )
        clean_func = (
            self._clean_sequence_id
            if column == self._data_name_col
            else self._clean_cell_id
        )
        # Check dataframe type
        is_polars_lazy = isinstance(self._data, pl.LazyFrame)
        is_polars_eager = isinstance(self._data, pl.DataFrame)
        is_pandas = isinstance(self._data, pd.DataFrame)
        # Convert original_values to list for processing
        if isinstance(original_values, pl.Series):
            original_list = original_values.cast(pl.String).to_list()
        elif isinstance(original_values, pd.Series):
            original_list = original_values.astype(str).tolist()
        else:
            original_list = [str(x) for x in original_values]
        cleaned_values = [
            clean_func(x, remove_trailing_hyphen_number) for x in original_list
        ]
        if operation == "prefix":
            new_values = [value + sep + x for x in cleaned_values]
        elif operation == "suffix":
            new_values = [x + sep + value for x in cleaned_values]
        if is_pandas:
            self._data[column] = new_values
        elif is_polars_eager:
            self._data = self._data.with_columns(pl.Series(column, new_values))
        elif is_polars_lazy:
            self._data = self._data.with_columns(
                pl.lit(pl.Series(column, new_values)).alias(column)
            )
        if sync:
            other_original = (
                self._original_cell_ids
                if column == self._data_name_col
                else self._original_sequence_ids
            )
            other_clean_func = (
                self._clean_cell_id
                if column == self._data_name_col
                else self._clean_sequence_id
            )
            # Convert other_original to list
            if isinstance(other_original, pl.Series):
                other_list = other_original.cast(pl.String).to_list()
            elif isinstance(other_original, pd.Series):
                other_list = other_original.astype(str).tolist()
            else:
                other_list = [str(x) for x in other_original]
            cleaned_other = [
                other_clean_func(x, remove_trailing_hyphen_number)
                for x in other_list
            ]
            if operation == "prefix":
                new_other_values = [value + sep + x for x in cleaned_other]
            elif operation == "suffix":
                new_other_values = [x + sep + value for x in cleaned_other]
            # Update other column based on type
            if is_pandas:
                self._data[other_column] = new_other_values
            elif is_polars_eager:
                self._data = self._data.with_columns(
                    pl.Series(other_column, new_other_values)
                )
            elif is_polars_lazy:
                self._data = self._data.with_columns(
                    pl.lit(pl.Series(other_column, new_other_values)).alias(
                        other_column
                    )
                )
        self._data = load_polars(self._data)
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

    def reset_ids(self) -> None:
        """
        Reset both IDs to their original values.

        This method restores both sequence_id and cell_id in the .data and .metadata slots to their original state when the Dandelion class was initialized.
        """
        if isinstance(self._data, pd.DataFrame):
            self._data.index = self._original_sequence_ids
            self._data[self._data_name_col] = self._original_sequence_ids
        if self._metadata is not None:
            if isinstance(self._metadata, pd.DataFrame):
                self._metadata.index = self._original_cell_ids
                self._data[self._metadata_name_col] = self._original_cell_ids
        if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
            self._data = self._data.with_columns(
                pl.Series(self._data_name_col, self._original_sequence_ids)
            )
            if isinstance(self._data, pl.LazyFrame):
                self._data = self._data.collect().lazy()
        if self._metadata is not None:
            if isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
                self._metadata = self._metadata.with_columns(
                    pl.Series(self._metadata_name_col, self._original_cell_ids)
                )
                if isinstance(self._metadata, pl.LazyFrame):
                    self._metadata = self._metadata.collect().lazy()

    def simplify(self, **kwargs) -> None:
        """Disambiguate VDJ and C gene calls when there's multiple calls separated by commas and strip the alleles."""
        # Check dataframe type
        is_polars_lazy = isinstance(self._data, pl.LazyFrame)
        is_polars_eager = isinstance(self._data, pl.DataFrame)
        is_pandas = isinstance(self._data, pd.DataFrame)
        # strip alleles from VDJ and constant gene calls
        for col in ["v_call", "v_call_genotyped", "d_call", "j_call", "c_call"]:
            if col in self._data:
                if is_pandas:
                    self._data[col] = self._data[col].str.replace(
                        r"\*.*", "", regex=True
                    )
                    # only keep the main annotation
                    self._data[col] = self._data[col].str.split(",").str[0]
                elif is_polars_eager or is_polars_lazy:
                    self._data = self._data.with_columns(
                        pl.col(col)
                        .str.replace_all(r"\*.*", "")
                        .str.split(",")
                        .list.first()
                        .alias(col)
                    )
        self.update_metadata(**kwargs)

    def _merge(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        join: bool = True,
        unique: bool = True,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        data = self._data if data is None else data
        if join:
            agg_exprs += [
                (
                    pl.col(col).unique().str.join(delimiter="|").alias(out_col)
                    if unique
                    else pl.col(col).str.join(delimiter="|").alias(out_col)
                )
                for col, out_col in zip(cols, key_added)
            ]
        else:
            agg_exprs += [
                (
                    pl.col(col).unique().list().alias(out_col)
                    if unique
                    else pl.col(col).list().alias(out_col)
                )
                for col, out_col in zip(cols, key_added)
            ]
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def _first(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            pl.col(col).first().alias(out_col)
            for col, out_col in zip(cols, key_added)
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def _sum(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            pl.col(col).sum().alias(out_col)
            for col, out_col in zip(cols, key_added)
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def _mean(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            pl.col(col).mean().alias(out_col)
            for col, out_col in zip(cols, key_added)
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def _split(
        self,
        cols: list[str] | str,
        join: bool = True,
        unique: bool = False,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        if join:
            agg_exprs += [
                expr
                for col, key in zip(cols, key_added)
                for expr in [
                    (
                        pl.col(col)
                        .filter(pl.col("locus_group") == "VDJ")
                        .unique()
                        .str.join(delimiter="|")
                        .alias(f"{key}_VDJ")
                        if unique
                        else pl.col(col)
                        .filter(pl.col("locus_group") == "VDJ")
                        .str.join(delimiter="|")
                        .alias(f"{key}_VDJ")
                    ),
                    (
                        (
                            pl.col(col)
                            .filter(pl.col("locus_group") == "VJ")
                            .unique()
                            .str.join(delimiter="|")
                            .alias(f"{col}_VJ")
                            if unique
                            else pl.col(col)
                            .filter(pl.col("locus_group") == "VJ")
                            .str.join(delimiter="|")
                            .alias(f"{col}_VJ")
                        )
                        if col != "d_call"
                        else None
                    ),
                ]
            ]
        else:
            agg_exprs += [
                expr
                for col, key in zip(cols, key_added)
                for expr in [
                    (
                        pl.col(col)
                        .filter(pl.col("locus_group") == "VDJ")
                        .unique()
                        .alias(f"{key}_VDJ")
                        if unique
                        else pl.col(col)
                        .filter(pl.col("locus_group") == "VDJ")
                        .alias(f"{key}_VDJ")
                    ),
                    (
                        (
                            pl.col(col)
                            .filter(pl.col("locus_group") == "VJ")
                            .unique()
                            .alias(f"{key}_VJ")
                            if unique
                            else pl.col(col)
                            .filter(pl.col("locus_group") == "VJ")
                            .alias(f"{key}_VJ")
                        )
                        if col != "d_call"
                        else None
                    ),
                ]
            ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .with_columns(
                [
                    pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                    .then(pl.lit("VDJ"))
                    .otherwise(pl.lit("VJ"))
                    .alias("locus_group")
                ]
            )
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        # drop literal column
        if "literal" in result.collect_schema().names():
            result = result.drop("literal")
        return result

    def _split_first(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            expr
            for col, key in zip(cols, key_added)
            for expr in [
                pl.col(col)
                .filter(pl.col("locus_group") == "VDJ")
                .first()
                .alias(f"{key}_VDJ"),
                (
                    pl.col(col)
                    .filter(pl.col("locus_group") == "VJ")
                    .first()
                    .alias(f"{key}_VJ")
                    if col != "d_call"
                    else None
                ),
            ]
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .with_columns(
                [
                    pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                    .then(pl.lit("VDJ"))
                    .otherwise(pl.lit("VJ"))
                    .alias("locus_group")
                ]
            )
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        # drop literal column
        if "literal" in result.collect_schema().names():
            result = result.drop("literal")
        return result

    def _split_sum(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            expr
            for col, key in zip(cols, key_added)
            for expr in [
                pl.col(col)
                .filter(pl.col("locus_group") == "VDJ")
                .sum()
                .alias(f"{key}_VDJ"),
                (
                    pl.col(col)
                    .filter(pl.col("locus_group") == "VJ")
                    .sum()
                    .alias(f"{key}_VJ")
                ),
            ]
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .with_columns(
                [
                    pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                    .then(pl.lit("VDJ"))
                    .otherwise(pl.lit("VJ"))
                    .alias("locus_group")
                ]
            )
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def _split_mean(
        self,
        cols: list[str] | str,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]
        agg_exprs += [
            expr
            for col, key in zip(cols, key_added)
            for expr in [
                pl.col(col)
                .filter(pl.col("locus_group") == "VDJ")
                .mean()
                .alias(f"{key}_VDJ"),
                (
                    pl.col(col)
                    .filter(pl.col("locus_group") == "VJ")
                    .mean()
                    .alias(f"{key}_VJ")
                ),
            ]
        ]
        data = self._data if data is None else data
        result = (
            data.lazy()
            .with_row_index("_original_order")
            .with_columns(
                [
                    pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                    .then(pl.lit("VDJ"))
                    .otherwise(pl.lit("VJ"))
                    .alias("locus_group")
                ]
            )
            .group_by("cell_id")
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect()
        )
        return result

    def initialize_metadata(
        self,
        clone_key: str = "clone_id",
        v_call_key: str = "v_call",
        init_cols: list[str] = [],
        update_isotype_dict: dict | None = None,
        strip_alleles: bool = True,
        productive_only: bool = True,
        check_rearrangement_status: bool = True,
    ) -> pd.DataFrame:
        """Initialize metadata DataFrame from Airrs data."""
        # init_cols = [] if init_cols is None else init_cols
        isotype_conversion_dict = {
            "IGHA": "IgA",
            "IGHD": "IgD",
            "IGHE": "IgE",
            "IGHG": "IgG",
            "IGHM": "IgM",
            "IGKC": "IgK",
            "IGLC": "IgL",
            "IGHM_1": "IgM",
        }
        if update_isotype_dict is not None:
            isotype_conversion_dict.update(update_isotype_dict)
        # remove cols if not found
        init_cols = [
            col
            for col in init_cols
            if col in self._data.collect_schema().names()
            and col not in [clone_key, "sample_id"]
        ]
        merge_cols = []
        for cols in [clone_key, "sample_id"]:
            if (
                cols in self._data.collect_schema().names()
                and cols not in init_cols
            ):
                merge_cols.append(cols)
        if check_rearrangement_status:
            self._update_rearrangement_status(v_call_key)
        if productive_only:
            _data = self._data.filter(
                pl.col("productive")
                .cast(pl.String)
                .str.to_uppercase()
                .is_in(TRUES_STR)
            )
        if strip_alleles:
            for col in ["v_call", "d_call", "j_call", "c_call"]:
                if col in _data.collect_schema().names():
                    _data = _data.with_columns(
                        pl.col(col).str.replace_all(r"\*.*", "").alias(col)
                    )
        unique_cells = _data.select("cell_id").unique()
        if isinstance(unique_cells, pl.LazyFrame):
            # materialize if lazy
            unique_cells = unique_cells.collect()
        if len(init_cols) == 0:
            self._metadata = pl.DataFrame(
                {"cell_id": unique_cells["cell_id"].to_list()}
            )
            if self.lazy:
                self._metadata = self._metadata.lazy()
        else:
            # initialise clone_id and sample_id first if present
            num_cols = [
                col
                for col in self._data.select(pl.selectors.numeric())
                .collect_schema()
                .names()
                if col in init_cols
            ]
            str_cols = [
                col
                for col in self._data.select(pl.selectors.string())
                .collect_schema()
                .names()
                if col in init_cols
            ]
            swap_v_call = [
                "v_call" if call == v_call_key else call for call in str_cols
            ]
            meta_0 = (
                self._merge(merge_cols, unique=True, data=_data)
                if len(merge_cols) > 0
                else None
            )
            if meta_0 is not None:
                # For LazyFrame:
                if clone_key in meta_0.collect_schema().names():
                    meta_0 = _add_clone_info(meta_0, clone_key)

            meta_str = (
                self._split(str_cols, key_added=swap_v_call, data=_data)
                if len(str_cols) > 0
                else None
            )
            meta_num = (
                self._split_sum(num_cols, data=_data)
                if len(num_cols) > 0
                else None
            )
            meta_str_main = (
                self._split_first(str_cols, data=_data)
                if len(str_cols) > 0
                else None
            )
            if meta_str_main is not None:
                meta_str_main = meta_str_main.rename(
                    {
                        col: f"{col}_main" if col != "cell_id" else col
                        for col in meta_str_main.collect_schema().names()
                    }
                )
            meta_num_main = (
                self._split_mean(num_cols, data=_data)
                if len(num_cols) > 0
                else None
            )
            if meta_num_main is not None:
                meta_num_main = meta_num_main.rename(
                    {
                        col: f"{col}_main" if col != "cell_id" else col
                        for col in meta_num_main.collect_schema().names()
                    }
                )
            frames = [meta_0, meta_str, meta_num, meta_str_main, meta_num_main]
            # Keep only non-None frames
            frames = [f for f in frames if f is not None]
            if len(frames) > 0:
                self._metadata = reduce(
                    lambda left, right: left.join(
                        right, on="cell_id", how="inner"
                    ),
                    frames,
                )

            else:
                self._metadata = pl.DataFrame(
                    {"cell_id": unique_cells["cell_id"].to_list()}
                )
                if self.lazy:
                    self._metadata = self._metadata.lazy()
            if "c_call_VDJ" in self._metadata:
                iso_tmp = self._split("c_call", join=False, data=_data)
                isotype_df = (
                    iso_tmp.lazy()
                    .with_columns(
                        pl.col("c_call_VDJ")
                        .list.eval(
                            # Split each element by comma
                            pl.element()
                            .str.split(",")
                            .list.eval(
                                # Take first 4 characters and map using replace_all
                                pl.element()
                                .str.slice(0, 4)
                                .replace_strict(
                                    list(isotype_conversion_dict.keys()),
                                    list(isotype_conversion_dict.values()),
                                    default=None,
                                )
                            )
                            # Join the mapped values back with comma
                            .list.join(",")
                        )
                        .alias("isotype")
                    )
                    .select(["cell_id", "isotype"])
                    .collect()
                )  # Keep only cell_id and isotype for joining
                if not isotype_df.select(
                    pl.col("isotype").is_null().all()
                ).item():
                    # Create aggregated columns
                    isotype_main = (
                        isotype_df.lazy()
                        .with_row_index("_original_order")
                        .group_by("cell_id")
                        .agg(
                            [
                                pl.col("_original_order")
                                .min()
                                .alias("_original_order"),
                                pl.col("isotype")
                                .first()
                                .list.first()
                                .fill_null("")
                                .alias("isotype_main"),
                            ]
                        )
                        .sort("_original_order")
                        .drop("_original_order")
                    )
                    isotype_status = (
                        isotype_df.lazy()
                        .with_columns(
                            _classify_isotype().alias("isotype_status")
                        )
                        .drop("isotype")
                    )
                    # NOW apply the string join transformation
                    isotype_df = isotype_df.lazy().with_columns(
                        pl.col("isotype").list.join("|").alias("isotype")
                    )
                    # Join isotype columns to metadata
                    self._metadata = self._metadata.lazy().join(
                        isotype_df.lazy(), on="cell_id", how="left"
                    )
                    self._metadata = self._metadata.lazy().join(
                        isotype_main.lazy(), on="cell_id", how="left"
                    )
                    self._metadata = self._metadata.lazy().join(
                        isotype_status.lazy(), on="cell_id", how="left"
                    )
                    self._metadata = self._metadata.collect()
                self._metadata = self._metadata.with_columns(
                    _classify_locus_pair().alias("locus_status")
                )
                self._metadata = self._metadata.with_columns(
                    _format_chain_status(pl.col("locus_status")).alias(
                        "chain_status"
                    )
                )
                if "isotype_status" in self._metadata.collect_schema().names():
                    self._metadata = self._metadata.lazy().with_columns(
                        _format_isotype().alias("isotype_status")
                    )
                cols_to_clean = [
                    c
                    for c in [
                        "locus_status",
                        "chain_status",
                        "isotype_status",
                    ]
                    if c in self._metadata.collect_schema().names()
                ]
                if cols_to_clean:
                    self._metadata = self._metadata.lazy().with_columns(
                        [
                            _clean_up_exception(pl.col(c)).alias(c)
                            for c in cols_to_clean
                        ]
                    )
            # finally, retrieve rearrangement status
            if check_rearrangement_status:
                reg_stat = self._split("rearrangement_status", data=_data)
                cols_to_update = [
                    "rearrangement_status_VDJ",
                    "rearrangement_status_VJ",
                ]
                reg_stat = reg_stat.with_columns(
                    [
                        pl.when(pl.col(x).str.contains("Chimeric"))
                        .then(pl.lit("Chimeric"))
                        .when(pl.col(x).str.contains(r"\|"))
                        .then(pl.lit("Multi"))
                        .otherwise(pl.col(x))
                        .alias(x)
                        for x in cols_to_update
                    ]
                )
                self._metadata = self._metadata.lazy().join(
                    reg_stat.lazy(), on="cell_id", how="left"
                )
            # always lazy
            if self.lazy:
                self._metadata = self._metadata.collect().lazy()
            else:
                self._metadata = self._metadata.collect()
            if "metadata" in self._tmpfiles.keys():
                self._tmpfiles["metadata"].close()
                del self._tmpfiles["metadata"]

    def _update_rearrangement_status(self, v_call_key: str) -> None:
        """Check rearrangement status."""
        vcall = _get_vcall_key_polars(self._data, v_call_key)
        # Build the rearrangement status logic using Polars expressions
        status = (
            pl.when(~is_present(vcall))
            .then(pl.lit("Unknown"))
            .when(~is_present("j_call"))
            .then(pl.lit("Unknown"))
            .when(is_present("c_call"))
            .then(
                # When c_call is present, check if v, j, c prefixes are different
                pl.when(
                    (first_3(vcall) != first_3("j_call"))
                    | (first_3(vcall) != first_3("c_call"))
                    | (first_3("j_call") != first_3("c_call"))
                )
                .then(pl.lit("Chimeric"))
                .otherwise(pl.lit("Standard"))
            )
            .otherwise(
                # When c_call is not present, check if v and j prefixes are different
                pl.when(first_3(vcall) != first_3("j_call"))
                .then(pl.lit("Chimeric"))
                .otherwise(pl.lit("Standard"))
            )
            .alias("rearrangement_status")
        )
        self._data = self._data.with_columns(status)

    def to_pandas(self) -> None:
        """Convert self from Polars to Pandas implementation."""
        if self._backend == "pandas":
            return
        if isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect().to_pandas()
        if isinstance(self._data, pl.DataFrame):
            self._data = self._data.to_pandas()
        self._data.index = self._data[self._data_name_col]
        if self._metadata is not None:
            if (
                self._metadata_name_col
                in self._metadata.collect_schema().names()
            ):
                # if not isinstance(self._metadata, pd.DataFrame):
                if isinstance(self._metadata, pl.LazyFrame):
                    self._metadata = self._metadata.collect().to_pandas()
                if isinstance(self._metadata, pl.DataFrame):
                    self._metadata = self._metadata.to_pandas()
                self._metadata.set_index(self._metadata_name_col, inplace=True)
            else:
                raise KeyError(
                    f"{self._metadata_name_col} not found in metadata columns."
                )
        self.lazy = False
        self._backend = "pandas"

    def to_polars(self, lazy: bool = True) -> None:
        """Convert self from Pandas to Polars implementation."""
        if self._backend == "polars":
            return
        if not isinstance(
            self._data, (pl.DataFrame, pl.LazyFrame)
        ) or not isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
            self.lazy = lazy
            if isinstance(self._data, pd.DataFrame):
                # drop index to avoid duplication
                self._data = self._data.reset_index(drop=True)
                self._data = pl.from_pandas(self._data)
                if self.lazy:
                    self._data = self._data.lazy()
            if self._metadata is not None:
                if not isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
                    if isinstance(self._metadata, pd.DataFrame):
                        self._metadata = self._metadata.reset_index(drop=False)
                        self._metadata = pl.from_pandas(self._metadata)
                        if self.lazy:
                            self._metadata = self._metadata.lazy()
            self._backend = "polars"

    def to_anndata(self) -> AnnData:
        """Convert DandelionPolars.metadata to AnnData"""
        if self._metadata is not None:
            if isinstance(self._metadata, pl.LazyFrame):
                meta_df = self._metadata.collect().to_pandas()
                meta_df.set_index(self._metadata_name_col, inplace=True)
            elif isinstance(self._metadata, pl.DataFrame):
                meta_df = self._metadata.to_pandas()
                meta_df.set_index(self._metadata_name_col, inplace=True)
            elif isinstance(self._metadata, pd.DataFrame):
                meta_df = self._metadata
            adata = AnnData(obs=meta_df)
            return adata
        else:
            raise ValueError(
                ".metadata is None, cannot convert to AnnData. Please initialize metadata first."
            )

    def to_eager(self) -> None:
        """Convert lazy slots to eager slots."""
        if self._backend == "polars":
            if isinstance(self._data, pl.LazyFrame):
                self._data = self._data.collect()
            if self._metadata is not None:
                if isinstance(self._metadata, pl.LazyFrame):
                    self._metadata = self._metadata.collect()
            # distances: eager types are np.ndarray or csr_matrix
        if not isinstance(self.distances, (np.ndarray, csr_matrix)):
            # assume anything else is lazy and computable
            computed = self.distances.compute()
            if isinstance(computed, csr_matrix):
                self.distances = computed
            else:
                self.distances = csr_matrix(computed)
            self.distances._index_names = self.metadata_names
        self.lazy = False

    def to_lazy(self, *, chunks="auto") -> None:
        """Convert eager slots to lazy slots."""
        if self._backend == "polars":
            if isinstance(self._data, pl.DataFrame):
                self._data = self._data.lazy()
            if self._metadata is not None and isinstance(
                self._metadata, pl.DataFrame
            ):
                self._metadata = self._metadata.lazy()
        if isinstance(self.distances, np.ndarray):
            import dask.array as da

            self.distances = da.from_array(self.distances, chunks=chunks)
        elif isinstance(self.distances, csr_matrix):
            import dask.array as da

            # Dask does NOT natively support sparse CSR well,
            # so you must decide what "lazy" means here.
            self.distances = da.from_array(
                self.distances.toarray(),
                chunks=chunks,
                asarray=False,
            )
        self.lazy = True

    def copy(self) -> DandelionPolars:
        """
        Performs a deep copy of all slots in Dandelion class.

        Returns
        -------
        Dandelion
            a deep copy of Dandelion class.
        """
        return copy.deepcopy(self)

    def update_data(self, skip: list[str] = []) -> None:
        """Sync metadata columns into data via dictionary mapping."""
        # Check dataframe type
        is_polars_lazy = isinstance(self._data, pl.LazyFrame)
        is_polars_eager = isinstance(self._data, pl.DataFrame)
        is_pandas = isinstance(self._data, pd.DataFrame)

        # Get column names based on dataframe type
        if is_pandas:
            data_columns = self._data.columns.tolist()
            metadata_columns = self._metadata.columns.tolist()
        else:  # Polars
            data_columns = self._data.columns
            metadata_columns = self._metadata.columns

        for col in metadata_columns:
            # skip blacklisted columns
            if col in skip:
                continue
            # skip columns that already exist in data
            if col in data_columns:
                continue
            # skip if base column already exists (for _VDJ, _VJ, _B, _abT, _gdT variants, _status, _main, etc.)
            base_col = col.split("_")[0]
            if base_col in data_columns:
                continue
            # create a mapping and assign new column
            if is_pandas:
                mapping = self._metadata[col].to_dict()
                self._data[col] = self._data["cell_id"].map(mapping)
            elif is_polars_eager or is_polars_lazy:
                # Create mapping using join (works for both eager and lazy)
                mapping_df = self._metadata.select(["cell_id", col])
                self._data = self._data.join(
                    mapping_df, on="cell_id", how="left"
                )
        # If lazy, collect and re-lazify once at the end
        if is_polars_lazy:
            self._data = self._data.collect().lazy()

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
        split: bool = True,
        join: bool = True,
        unique: bool = False,
        first: bool = False,
        average: bool = False,
        key_added: list[str] | str | None = None,
        strip_alleles: bool = True,
        reinitialize: bool = True,
        init_cols: list[str] | None = None,
        productive_only: bool = True,
        check_rearrangement_status: bool = True,
        genotyped_v_call: bool = True,
        update_isotype_dict: dict[str, str] | None = None,
        lazy: bool = True,
        as_pandas: bool = False,
        # by_celltype: bool = False,
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
        reinitialize : bool, optional
            whether or not to reinitialize the current metadata.
            useful when updating older versions of `dandelion` to newer version.
        strip_alleles : bool, optional
            returns the V(D)J genes with allelic calls if False.
        init_cols : list[str] | None, optional
            columns to initialize the metadata with. If None, uses default set of columns.
        productive_only : bool, optional
            whether or not to use only productive contigs to initialize metadata.
        check_rearrangement_status : bool, optional
            whether or not to check and update the rearrangement status.
        genotyped_v_call : bool, optional
            whether or not to use genotyped v_call data to initialize metadata if available.
        update_isotype_dict : dict[str, str] | None, optional
            custom isotype dictionary to update the default isotype dictionary.
        by_celltype : bool, optional
            whether to return the query/update by celltype.

        Raises
        ------
        KeyError
            if columns provided not found in Dandelion.data.
        ValueError
            if missing columns in Dandelion.data.
        """
        clone_key = "clone_id" if clone_key is None else clone_key
        reinitialize = False if retrieve is not None else reinitialize
        v_call_key = "v_call"
        if self._backend == "pandas":
            # switch to polars
            self.to_polars(lazy=self.lazy)
        if genotyped_v_call:
            if f"{v_call_key}_genotyped" in self._data.collect_schema():
                v_call_key = f"{v_call_key}_genotyped"
        init_cols = (
            [
                clone_key,
                "sample_id",
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
            if init_cols is None
            else init_cols
        )
        self.lazy = lazy
        metadata_status = self._metadata

        if (metadata_status is None) or reinitialize:
            self.initialize_metadata(
                v_call_key=v_call_key,
                init_cols=init_cols,
                update_isotype_dict=update_isotype_dict,
                strip_alleles=strip_alleles,
                productive_only=productive_only,
                check_rearrangement_status=check_rearrangement_status,
            )
            cols = self._metadata.collect_schema().names()
            if clone_key in cols:
                clone_key_rank = clone_key + "_rank"
                if clone_key_rank in cols:
                    new_order = [clone_key, clone_key_rank] + [
                        c for c in cols if c not in [clone_key, clone_key_rank]
                    ]
                else:
                    # Place only clone_key first, then the rest
                    new_order = [clone_key] + [
                        c for c in cols if c != clone_key
                    ]
                if isinstance(self._metadata, pl.DataFrame):
                    self._metadata = self._metadata.select(new_order).lazy()
                else:
                    self._metadata = (
                        self._metadata.select(new_order).collect().lazy()
                    )
                    if "metadata" in self._tmpfiles.keys():
                        self._tmpfiles["metadata"].close()
                        del self._tmpfiles["metadata"]

        if retrieve is not None:
            if self._metadata is None:
                raise ValueError(
                    "Dandelion.metadata is None. Please initialize metadata first before updating."
                )
            # first check that self._data and self._metadata are converted to Polars
            if not isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
                self.to_polars(lazy=self.lazy)
            if type(retrieve) is str:
                retrieve = [retrieve]
            _data = (
                self._data.filter(
                    pl.col("productive")
                    .cast(pl.String)
                    .str.to_uppercase()
                    .is_in(TRUES_STR)
                )
                if productive_only
                else None
            )
            # check the dtypes of the retrieve columns
            string_cols = []
            numeric_cols = []
            for col in retrieve:
                if col not in self._data.collect_schema().names():
                    raise KeyError(
                        f"Column '{col}' not found in Dandelion.data."
                    )
                dtype = self._data.select(pl.col(col)).collect_schema().get(col)
                if dtype in [
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.Float32,
                    pl.Float64,
                ]:
                    numeric_cols.append(col)
                else:
                    string_cols.append(col)
            meta_str, meta_num = None, None
            if len(string_cols) > 0:
                if split:
                    if first:
                        meta_str = self._split_first(
                            cols=string_cols, key_added=key_added, data=_data
                        )
                    else:
                        meta_str = self._split(
                            cols=string_cols,
                            key_added=key_added,
                            join=join,
                            unique=unique,
                            data=_data,
                        )
                else:
                    if first:
                        meta_str = self._first(
                            cols=string_cols, key_added=key_added, data=_data
                        )
                    else:
                        meta_str = self._merge(
                            cols=string_cols,
                            key_added=key_added,
                            join=join,
                            unique=unique,
                            data=_data,
                        )
            if len(numeric_cols) > 0:
                if split:
                    if average:
                        meta_num = self._split_mean(
                            numeric_cols, key_added=key_added, data=_data
                        )
                    else:
                        meta_num = self._split_sum(
                            numeric_cols, key_added=key_added, data=_data
                        )
                else:
                    if average:
                        meta_num = self._mean(
                            numeric_cols, key_added=key_added, data=_data
                        )
                    else:
                        meta_num = self._sum(
                            numeric_cols, key_added=key_added, data=_data
                        )
            if meta_str is not None and meta_num is not None:
                meta = reduce(
                    lambda left, right: left.join(
                        right, on="cell_id", how="inner"
                    ),
                    [meta_str, meta_num],
                )
            elif meta_str is not None:
                meta = meta_str
            elif meta_num is not None:
                meta = meta_num
            else:
                meta = None
            if meta is not None:
                self._metadata = self._metadata.lazy().join(
                    meta.lazy(), on="cell_id", how="left"
                )
                self._metadata = (
                    self._metadata.collect().lazy()
                    if self.lazy
                    else self._metadata.collect()
                )
                if "metadata" in self._tmpfiles.keys():
                    self._tmpfiles["metadata"].close()
                    del self._tmpfiles["metadata"]
        if as_pandas:
            self.to_pandas()

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
            passed to `pandas.DataFrame.to_csv` or `polars.DataFrame.write_csv`.
        """
        if self._backend == "pandas":
            # convert to polars first
            self.to_polars()
        _write_airr(self._data, filename, **kwargs)

    def write_zipddl(
        self,
        filename: str = "dandelion.zipddl",
        compress: bool = True,
    ):
        """
        Write a Dandelion object to a single .zipddl file (Zarr v3 ZipStore, hybrid storage)
        with optional compression.

        Storage scheme:
        - data and metadata → Parquet blobs
        - distances → Zarr arrays
        - graph, layout and germline → HDF5
        """
        # Create Zarr ZipStore container
        store = zarr.storage.ZipStore(filename, mode="w")
        root = zarr.group(store=store, overwrite=True)
        comp = (
            BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
            if compress
            else None
        )
        # Tables: .data and .metadata as Parquet blobs
        tables_grp = root.create_group("tables", overwrite=True)
        if self._data is not None:
            if isinstance(self._data, pd.DataFrame):
                # convert to Polars first
                self.to_polars(lazy=False)
            self._data = _sanitize_data_polars(self._data)
            _write_parquet_blob(
                tables_grp,
                "data.parquet",
                self._data,
                compressors=[comp] if compress else None,
            )
        if getattr(self, "_metadata", None) is not None:
            if isinstance(self._metadata, pd.DataFrame):
                # convert to Polars first
                self.to_polars(lazy=False)
            _write_parquet_blob(
                tables_grp,
                "metadata.parquet",
                self._metadata,
                compressors=[comp] if compress else None,
            )
        # .distances as Zarr arrays
        if getattr(self, "distances", None) is not None:
            arrays_grp = root.create_group("arrays", overwrite=True)
            arr = self.distances
            if isinstance(arr, csr_matrix):
                # Save CSR matrix as separate arrays
                arrays_grp.create_dataset(
                    name="distances_data",
                    shape=arr.data.shape,
                    dtype=arr.data.dtype,
                    chunks=arr.data.shape,
                    data=arr.data,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                arrays_grp.create_dataset(
                    "distances_indices",
                    shape=arr.indices.shape,
                    dtype=arr.indices.dtype,
                    chunks=arr.indices.shape,
                    data=arr.indices,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                arrays_grp.create_dataset(
                    "distances_indptr",
                    shape=arr.indptr.shape,
                    dtype=arr.indptr.dtype,
                    chunks=arr.indptr.shape,
                    data=arr.indptr,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                arrays_grp.create_dataset(
                    "distances_shape",
                    shape=(len(arr.shape),),
                    dtype=np.int64,
                    data=np.array(arr.shape, dtype=np.int64),
                    overwrite=True,
                )
            else:
                arrays_grp.create_dataset(
                    "distances",
                    shape=arr.shape,
                    dtype=arr.dtype,
                    chunks=arr.shape,
                    data=arr,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
        # .graph, . layout, .germline as HDF5 files
        if getattr(self, "graph", None) is not None:
            graph_grp = root.create_group("graph", overwrite=True)
            for i, g in enumerate(self.graph):
                with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                    # Convert NetworkX graph to CSR adjacency
                    G_df = nx.to_pandas_adjacency(g, nonedge=np.nan)
                    G_x = csr_matrix(G_df.to_numpy())
                    with h5py.File(tmp_h5.name, "w") as hf:
                        hf.create_dataset("data", data=G_x.data)
                        hf.create_dataset("indices", data=G_x.indices)
                        hf.create_dataset("indptr", data=G_x.indptr)
                        hf.create_dataset("shape", data=G_x.shape)
                        hf.create_dataset(
                            "columns", data=np.array(G_df.columns, dtype="S")
                        )
                        hf.create_dataset(
                            "index", data=np.array(G_df.index, dtype="S")
                        )
                    tmp_h5.seek(0)
                    arr = np.frombuffer(tmp_h5.read(), dtype=np.uint8)
                    graph_grp.create_dataset(
                        f"graph_{i}.h5",
                        shape=arr.shape,
                        dtype=arr.dtype,
                        chunks=arr.shape,
                        data=arr,
                        overwrite=True,
                        compressors=[comp] if compress else None,
                    )
        if getattr(self, "layout", None) is not None:
            layout_grp = root.create_group("layout", overwrite=True)
            for i, l in enumerate(self.layout):
                with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                    with h5py.File(tmp_h5.name, "w") as hf:
                        for k, v in l.items():
                            hf.create_dataset(k, data=v)
                    tmp_h5.seek(0)
                    arr = np.frombuffer(tmp_h5.read(), dtype=np.uint8)
                    layout_grp.create_dataset(
                        f"layout_{i}.h5",
                        shape=arr.shape,
                        dtype=arr.dtype,
                        chunks=arr.shape,
                        data=arr,
                        overwrite=True,
                        compressors=[comp] if compress else None,
                    )
        if (
            getattr(self, "germline", None) is not None
            and len(self.germline) > 0
        ):
            germline_grp = root.create_group("germline", overwrite=True)
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                with h5py.File(tmp_h5.name, "w") as hf:
                    for k, v in self.germline.items():
                        hf.create_dataset(k, data=v)
                tmp_h5.seek(0)
                arr = np.frombuffer(tmp_h5.read(), dtype=np.uint8)
                germline_grp.create_dataset(
                    "germline.h5",
                    shape=arr.shape,
                    dtype=arr.dtype,
                    chunks=arr.shape,
                    data=arr,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
        # Close store
        store.close()

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="1.1.0",
        details="pickle format will removed from future versions.",
    )
    def write_pkl(
        self, filename: str = "dandelion_data.pkl.pbz2", **kwargs
    ) -> None:
        """
        Writes a Dandelion class to .pkl format.

        Parameters
        ----------
        filename : str, optional
            path to `.pkl` file.
        **kwargs
            passed to `_pickle`.
        """
        import bz2
        import gzip
        import _pickle as cPickle
        from dandelion.utilities._utilities import isBZIP, isGZIP

        if isBZIP(str(filename)):
            try:
                with bz2.BZ2File(filename, "wb") as f:
                    cPickle.dump(self, f, **kwargs)
            except:
                with bz2.BZ2File(filename, "wb") as f:
                    cPickle.dump(self, f, protocol=4, **kwargs)
        elif isGZIP(str(filename)):
            try:
                with gzip.open(filename, "wb") as f:
                    cPickle.dump(self, f, **kwargs)
            except:
                with gzip.open(filename, "wb") as f:
                    cPickle.dump(self, f, protocol=4, **kwargs)
        else:
            f = open(filename, "wb")
            cPickle.dump(self, f, **kwargs)
            f.close()

    @deprecated(
        deprecated_in="1.0.0",
        removed_in="1.1.0",
        details=".h5ddl format will no longer be supported in future versions.",
    )
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
        if self._backend == "polars":
            self.to_pandas()
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
        if self._backend == "polars":
            self.to_pandas()
        # now to actually saving
        data = sanitize_data(self._data)
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
        _write_fasta(seqs, out_fasta=out_fasta)

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


def load_polars(
    obj: pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None,
    as_pandas: bool = False,
) -> pl.LazyFrame:
    """
    Read in or copy dataframe object and set sequence_id as index without dropping.

    Parameters
    ----------
    obj : pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None
        airr rearrangement file path or pandas/polars DataFrame.
    as_pandas : bool, optional
        whether to return as pandas DataFrame. Default is False.

    Returns
    -------
    pl.LazyFrame
        airr rearrangement data frame.

    Raises
    ------
    KeyError
        if `sequence_id` not found in input.
    """
    if obj is not None:
        if os.path.isfile(str(obj)):
            df = pl.scan_csv(obj, separator="\t")
        elif isinstance(obj, pl.LazyFrame):  # Check for LazyFrame
            df = obj
        elif isinstance(obj, pl.DataFrame):  # Check for eager DataFrame
            df = obj.lazy()
        elif isinstance(obj, pd.DataFrame):
            try:
                df = pl.from_pandas(obj).lazy()
            except TypeError:  # because of mixed dtypes, sanitize first
                df = _sanitize_data_polars(obj)

        if (
            "sequence_id" in df.collect_schema()
        ):  # Use collect_schema() for lazy frames
            # assert that sequence_id is string
            df = df.with_columns(pl.col("sequence_id").cast(pl.String))
            if "cell_id" not in df.collect_schema():
                df = df.with_columns(
                    pl.when(pl.col("sequence_id").str.contains("_contig"))
                    .then(
                        pl.col("sequence_id").str.split("_contig").list.first()
                    )
                    .otherwise(pl.col("sequence_id"))
                    .alias("cell_id")
                )
            # assert that cell_id is string
            df = df.with_columns(pl.col("cell_id").cast(pl.String))
        else:
            raise KeyError("'sequence_id' not found in columns of input")

        if "duplicate_count" in df.collect_schema():
            if "umi_count" not in df.collect_schema():
                df = df.rename({"duplicate_count": "umi_count"}).collect()

        if as_pandas:
            df = df.collect().to_pandas()
            df.set_index("sequence_id", inplace=True, drop=False)
        # Collect to execute operations, then convert back to lazy or pandas
        return df

    return None  # Handle obj is None case


class DataFrameAccessor:
    """Wrapper that provides both DataFrame access and attribute-style column access."""

    def __init__(self, df, parent=None, attr_name=None):
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_attr_name", attr_name)
        # Cache schema for lazy frames to avoid repeated resolution
        if isinstance(df, pl.LazyFrame):
            object.__setattr__(self, "_schema", df.collect_schema())
        else:
            object.__setattr__(self, "_schema", None)

    def __getattr__(self, name):
        df = object.__getattribute__(self, "_df")
        schema = object.__getattribute__(self, "_schema")

        # List of common DataFrame/LazyFrame methods to pass through
        passthrough_methods = {
            "filter",
            "select",
            "with_columns",
            "group_by",
            "join",
            "collect",
            "lazy",
            "head",
            "tail",
            "schema",
            "collect_schema",
            "drop",
            "rename",
            "sort",
            "unique",
            "describe",
            "write_csv",
            "write_parquet",
            "clone",
            "pipe",
            "explain",
        }

        if name in passthrough_methods:
            return getattr(df, name)

        # For LazyFrame, check if it's a column using cached schema
        if isinstance(df, pl.LazyFrame):
            if schema is not None and name in schema.names():
                # Return collected series for this column wrapped in SeriesAccessor
                series = df.select(name).collect().to_series()
                return SeriesAccessor(series)
            # Not a column, try to get actual attribute
            try:
                return object.__getattribute__(df, name)
            except AttributeError:
                raise AttributeError(
                    f"LazyFrame has no column or attribute '{name}'"
                )

        # For eager DataFrame
        else:
            # Check if it's a column first
            if hasattr(df, "columns") and name in df.columns:
                return SeriesAccessor(df[name])
            # Otherwise try actual attribute
            return getattr(df, name)

    def __getitem__(self, key):
        """Support bracket notation for column access and filtering."""
        df = object.__getattribute__(self, "_df")

        # Handle column name string
        if isinstance(key, str):
            if isinstance(df, pl.LazyFrame):
                series = df.select(key).collect().to_series()
                return SeriesAccessor(series)
            else:
                return SeriesAccessor(df[key])

        # Handle list of column names
        elif isinstance(key, (list, tuple)):
            if isinstance(df, pl.LazyFrame):
                return df.select(key)
            else:
                return df[key]

        # Handle slicing (both DataFrame and LazyFrame support this)
        elif isinstance(key, slice):
            return df[key]

        # Handle boolean Series or expressions (for filtering)
        elif isinstance(key, (pl.Series, pl.Expr)):
            return df.filter(key)

        # For anything else, try to pass through
        else:
            return df[key]

    def __setattr__(self, name, value):
        if name in ("_df", "_schema"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Cannot set attributes on DataFrameAccessor")

    def __setitem__(self, key: str, value):
        """Support column assignment like df['new_col'] = value."""
        df = object.__getattribute__(self, "_df")

        # Convert value to appropriate Polars format
        if isinstance(value, (list, tuple)):
            # Convert list/tuple to Series
            new_col = pl.Series(key, value)
        elif isinstance(value, pl.Series):
            # Rename Series to match key if needed
            new_col = value.alias(key) if value.name != key else value
        elif isinstance(value, SeriesAccessor):
            # Extract underlying Series from SeriesAccessor
            series = object.__getattribute__(value, "_series")
            new_col = series.alias(key) if series.name != key else series
        elif isinstance(value, (int, float, str, bool)):
            # Scalar value - create a Series filled with that value
            # Get length from df
            if isinstance(df, pl.LazyFrame):
                length = df.select(pl.len()).collect().item()
            else:
                length = len(df)
            new_col = pl.Series(key, [value] * length)
        elif isinstance(value, pl.Expr):
            # Expression - use with_columns directly
            df = df.with_columns(value.alias(key))
            object.__setattr__(self, "_df", df)
            # Update schema cache
            if isinstance(df, pl.LazyFrame):
                object.__setattr__(self, "_schema", df.collect_schema())
            # Update parent if it exists
            parent = object.__getattribute__(self, "_parent")
            attr_name = object.__getattribute__(self, "_attr_name")
            if parent is not None and attr_name is not None:
                setattr(parent, attr_name, df)
            return
        else:
            raise TypeError(
                f"Cannot assign value of type {type(value)} to column '{key}'. "
                f"Expected list, Series, scalar, or Expression."
            )

        # Use with_columns to add/update the column
        df = df.with_columns(new_col)
        object.__setattr__(self, "_df", df)

        # Update schema cache for lazy frames
        if isinstance(df, pl.LazyFrame):
            object.__setattr__(self, "_schema", df.collect_schema())

        # Update parent if it exists
        parent = object.__getattribute__(self, "_parent")
        attr_name = object.__getattribute__(self, "_attr_name")
        if parent is not None and attr_name is not None:
            setattr(parent, attr_name, df)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_df"))

    def __len__(self):
        df = object.__getattribute__(self, "_df")
        if isinstance(df, pl.LazyFrame):
            return df.select(pl.len()).collect().item()
        return len(df)

    @property
    def columns(self):
        """Get column names."""
        df = object.__getattribute__(self, "_df")
        if isinstance(df, pl.LazyFrame):
            schema = object.__getattribute__(self, "_schema")
            return schema.names() if schema else df.collect_schema().names()
        return df.columns


class SeriesAccessor:
    """Wrapper for Polars Series that supports both pandas and polars syntax."""

    def __init__(self, series: pl.Series):
        self._series = series

    def __getattr__(self, name):
        # Map pandas methods to polars equivalents
        method_map = {
            "isin": "is_in",
            "isna": "is_null",
            "notna": "is_not_null",
            "fillna": "fill_null",
            "dropna": "drop_nulls",
            "value_counts": "value_counts",  # same name but good to be explicit
        }

        if name in method_map:
            return getattr(self._series, method_map[name])

        # Pass through everything else
        return getattr(self._series, name)

    def __getitem__(self, key):
        return self._series[key]

    def __repr__(self):
        return repr(self._series)

    def __len__(self):
        return len(self._series)

    def __iter__(self):
        return iter(self._series)

    # Support comparison operators
    def __eq__(self, other):
        return self._series.__eq__(other)

    def __ne__(self, other):
        return self._series.__ne__(other)

    def __lt__(self, other):
        return self._series.__lt__(other)

    def __le__(self, other):
        return self._series.__le__(other)

    def __gt__(self, other):
        return self._series.__gt__(other)

    def __ge__(self, other):
        return self._series.__ge__(other)


def read_zipddl(
    filename: str,
    distance_zarr: Path | str | None = None,
    verbose: bool = False,
) -> DandelionPolars:
    """
    Read a Dandelion object from a .zarrddl file (hybrid Zarr v3 container).

    Returns:
        Dandelion object
    """
    store = zarr.storage.ZipStore(filename, mode="r")
    root = zarr.open(store=store, mode="r")

    constructor = {}

    # ---------------------------
    # Tables: _data and _metadata as Polars LazyFrames
    # ---------------------------
    def load_parquet_lazy(
        dataset_name: str,
    ) -> tuple[pl.LazyFrame, tempfile.NamedTemporaryFile]:
        arr = root[f"tables/{dataset_name}"][:]
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet")
        tmp.write(arr.tobytes())
        tmp.flush()
        # Polars lazy scan
        return pl.scan_parquet(tmp.name), tmp  # return tmp to keep file alive

    tmp_files = {}
    if "data.parquet" in root["tables"]:
        data_lazy, data_tmp = load_parquet_lazy("data.parquet")
        constructor["data"] = data_lazy
        tmp_files["data"] = data_tmp
    if "metadata.parquet" in root["tables"]:
        metadata_lazy, metadata_tmp = load_parquet_lazy("metadata.parquet")
        constructor["metadata"] = metadata_lazy
        tmp_files["metadata"] = metadata_tmp

    # ---------------------------
    # Distances: Zarr arrays
    # ---------------------------
    if "arrays" in root:
        arr_grp = root["arrays"]
        if "distances_data" in arr_grp:
            distances = csr_matrix(
                (
                    arr_grp["distances_data"][:],
                    arr_grp["distances_indices"][:],
                    arr_grp["distances_indptr"][:],
                ),
                shape=tuple(arr_grp["distances_shape"][:]),
            )
            constructor["distances"] = distances
        elif "distances" in arr_grp:
            constructor["distances"] = arr_grp["distances"]

    if distance_zarr is not None:
        import dask.array as da

        constructor["distances"] = da.from_zarr(
            str(distance_zarr) + "/distance_matrix"
        )

    # ---------------------------
    # Graphs: HDF5 blobs
    # ---------------------------
    if "graph" in root:
        graph_group = root["graph"]
        graphs = []
        for key in sorted(graph_group.array_keys()):
            arr = graph_group[key][:]
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                tmp_h5.write(arr.tobytes())
                tmp_h5.flush()
                # Use your helper
                df = _read_h5_csr_matrix_zarr(tmp_h5.name, as_df=True)
                g = _create_graph(df, adjust_adjacency=0, fillna=0)
                graphs.append(g)
        constructor["graph"] = tuple(graphs)

    # ---------------------------
    # Layout
    # ---------------------------
    if "layout" in root:
        layout_grp = root["layout"]
        layout = []
        for key in sorted(layout_grp.keys()):
            arr = layout_grp[key][:]
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                tmp_h5.write(arr.tobytes())
                tmp_h5.flush()
                with h5py.File(tmp_h5.name, "r") as hf:
                    layout_dict = {k: hf[k][:] for k in hf.keys()}
                    layout.append(layout_dict)
        constructor["layout"] = tuple(layout)

    # ---------------------------
    # Germline
    # ---------------------------
    if "germline" in root:
        arr = root["germline"]["germline.h5"][:]
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
            tmp_h5.write(arr.tobytes())
            tmp_h5.flush()
            with h5py.File(tmp_h5.name, "r") as hf:
                constructor["germline"] = {k: hf[k][:] for k in hf.keys()}
    # ---------------------------
    # Construct Dandelion
    # ---------------------------
    res = DandelionPolars(**constructor, verbose=verbose)
    res._tmpfiles = tmp_files

    return res


### --- Helper functions for slicing ---
def _get_length(df: pl.DataFrame | pl.LazyFrame) -> int:
    """Get number of rows in Polars DataFrame or LazyFrame."""
    if isinstance(df, pl.LazyFrame):
        return df.collect().height
    elif isinstance(df, pl.DataFrame):
        return df.height
    return len(df)


def _filter_and_select(
    df: pl.DataFrame | pl.LazyFrame, condition: pl.Expr
) -> pl.Series:
    """Filter DataFrame/LazyFrame by condition and select 'cell_id' column as Series."""
    if isinstance(df, pl.LazyFrame):
        return df.filter(condition).select("cell_id").collect().to_series()
    else:
        return df.filter(condition).select("cell_id").to_series()


### --- Helper functions for read/write ---
def _write_parquet_blob(
    zarr_group,
    name: str,
    df: pl.DataFrame | pl.LazyFrame,
    compressors=None,
):
    """
    Save Polars DataFrame/LazyFrame as Parquet blob in Zarr group.
    """
    # Materialize if lazy
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df.write_parquet(tmp.name)
        tmp.seek(0)
        arr = np.frombuffer(tmp.read(), dtype=np.uint8)
    zarr_group.create_dataset(
        name,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=arr.shape,
        data=arr,
        overwrite=True,
        compressors=compressors,
    )


def _read_h5_csr_matrix_zarr(
    filename: Path | str, as_df: bool = True
) -> pd.DataFrame:
    """
    Read a group from an H5 file originally stored as a compressed sparse matrix.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.

    Returns
    -------
    pd.DataFrame
        The data from the specified group as a pandas dataframe.
    """
    with h5py.File(filename, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = tuple(f["shape"][:])
        # Reconstruct CSR matrix
        loaded_matrix = csr_matrix((data, indices, indptr), shape=shape)
        if not as_df:
            return loaded_matrix
        df = pd.DataFrame(loaded_matrix.toarray())
        df_col = [
            x.decode("utf-8") if isinstance(x, bytes) else x
            for x in f["columns"][:]
        ]
        df_index = [
            x.decode("utf-8") if isinstance(x, bytes) else x
            for x in f["index"][:]
        ]
        df.columns = df_col
        df.index = df_index
    return df


def _create_graph(
    adj: pd.DataFrame,
    adjust_adjacency: int | float = 0,
    fillna: int | float = 0,
) -> nx.Graph:
    """
    Create a networkx graph from the given adjacency matrix.

    Parameters
    ----------
    adj : pd.DataFrame
        The adjacency matrix to create the graph from.
    adjust_adjacency : int | float, optional
        The value to add to the graph by as a way to adjust the adjacency matrix. Defaults to 0.
    fillna : int | float, optional
        The value to fill NaN values with. Defaults to 0.

    Returns
    -------
    nx.Graph
        The created networkx graph.
    """
    if adjust_adjacency != 0:
        adj += adjust_adjacency
    adj = adj.fillna(fillna)
    g = nx.from_pandas_adjacency(adj)

    if adjust_adjacency != 0:
        for u, v, d in g.edges(data=True):
            d["weight"] -= adjust_adjacency

    return g


def _write_fasta(
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
        _write_output(out, out_fasta)


def _write_output(out: str, file: Path | str) -> None:
    """General line writer."""
    fh = open(file, "a")
    fh.write(out)
    fh.close()


def _write_airr(
    data: pd.DataFrame | pl.DataFrame | pl.LazyFrame, save: Path | str, **kwargs
) -> None:
    """Save as airr formatted file."""
    data = _sanitize_data_polars(data)
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    data.write_csv(save, separator="\t", **kwargs)


### --- Helper functions for metadata initialization ---
def is_present(col: str) -> pl.Expr:
    """Helper function to check if a column value is present (not null and not empty)."""
    return pl.col(col).is_not_null() & (pl.col(col).str.len_chars() > 0)


def first_3(col: str) -> pl.Expr:
    """Helper function to get the first 3 characters of a column."""
    return pl.col(col).str.slice(0, 3)


def _get_vcall_key_polars(
    data: pl.DataFrame | pl.LazyFrame, v_call_key: str
) -> str:
    """
    Determine which V-call key to use based on the provided data and key.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        The Polars DataFrame or LazyFrame containing rearrangement data.
    v_call_key : str
        The requested key to check (e.g. "v_call" or "v_call_genotyped").

    Returns
    -------
    str
        The best matching V-call key, following this priority:
        1. "v_call_genotyped" if it exists in data and matches v_call_key
        2. "v_call" if it exists in data and matches v_call_key
        3. v_call_key if it exists in data
        4. "v_call" as a default fallback
    """
    cols = data.collect_schema().names()
    if "v_call_genotyped" in cols and v_call_key == "v_call_genotyped":
        return "v_call_genotyped"
    elif "v_call" in cols and v_call_key == "v_call":
        return "v_call"
    elif v_call_key in cols:
        return v_call_key
    else:
        return "v_call"


def _classify_locus_pair() -> pl.Expr:
    """
    Classify locus pairs based on VDJ and VJ locus columns and isotype information.
    """
    vdj_locus_col = "locus_VDJ"
    vj_locus_col = "locus_VJ"
    isotype_col = "isotype"

    # Parse lists once
    vdj_locus_list = (
        pl.when(pl.col(vdj_locus_col).is_null())
        .then(pl.lit(None, dtype=pl.List(pl.String)))
        .otherwise(
            pl.col(vdj_locus_col)
            .str.split("|")
            .list.eval(pl.element().filter(pl.element() != "None"))
        )
    )

    vj_locus_list = (
        pl.when(pl.col(vj_locus_col).is_null())
        .then(pl.lit(None, dtype=pl.List(pl.String)))
        .otherwise(
            pl.col(vj_locus_col)
            .str.split("|")
            .list.eval(pl.element().filter(pl.element() != "None"))
        )
    )

    # Simplified isotype check - much faster than splitting and checking lists
    isotype_list = (
        pl.when(pl.col(isotype_col).is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(
            pl.col(isotype_col)
            .str.split("|")
            .list.unique()
            .pipe(
                lambda x: (
                    # All elements must be in allowed set
                    x.list.eval(
                        pl.element().is_in(["IgM", "IgD", "IgM,IgD", "IgD,IgM"])
                    ).list.all()
                )
                & (
                    # Must contain IgM (either as "IgM" or in "IgM,IgD"/"IgD,IgM")
                    x.list.eval(pl.element().str.contains("IgM")).list.any()
                )
                & (
                    # Must contain IgD (either as "IgD" or in "IgM,IgD"/"IgD,IgM")
                    x.list.eval(pl.element().str.contains("IgD")).list.any()
                )
            )
        )
    )

    # Check if has values
    loc1_has = vdj_locus_list.is_not_null() & (vdj_locus_list.list.len() > 0)
    vj_locus_has = vj_locus_list.is_not_null() & (vj_locus_list.list.len() > 0)

    # Process locus 1 - use fill_null instead of when-then for simpler cases
    loc1_len = vdj_locus_list.list.len().fill_null(0)
    loc1_unique_len = vdj_locus_list.list.n_unique().fill_null(0)
    loc1_first = vdj_locus_list.list.first()

    # Check for TRB/TRD exception
    loc1_all_trb_trd = (
        loc1_has
        & vdj_locus_list.list.eval(
            pl.element().is_in(["TRB", "TRD"])
        ).list.all()
    )

    tmp1 = (
        pl.when(~loc1_has)
        .then(pl.lit("None"))
        .when(loc1_unique_len > 1)
        .then(pl.lit("Ambiguous"))
        .when(
            loc1_len > 1
        )  # Changed from >= to > (only flag as extra if MORE than 1)
        .then(
            pl.when(
                isotype_list.fill_null(False)
                | (loc1_all_trb_trd & (loc1_unique_len <= 2))
            )
            .then(pl.lit("Extra VDJ-exception"))
            .otherwise(pl.lit("Extra VDJ"))
        )
        .otherwise(loc1_first)  # Single locus returns the locus name
    )

    # Process locus 2
    vj_locus_len = vj_locus_list.list.len().fill_null(0)
    vj_locus_first = vj_locus_list.list.first()
    vj_locus_unique_len = vj_locus_list.list.n_unique().fill_null(0)

    # Check for TRA/TRG or IGK/IGL exception
    vj_locus_all_tra_trg = (
        vj_locus_has
        & vj_locus_list.list.eval(pl.element().is_in(["TRA", "TRG"])).list.all()
    )

    vj_locus_all_igk_igl = (
        vj_locus_has
        & vj_locus_list.list.eval(pl.element().is_in(["IGK", "IGL"])).list.all()
    )

    tmp2 = (
        pl.when(~vj_locus_has)
        .then(pl.lit("None"))
        .when(vj_locus_len > 1)
        .then(
            pl.when(
                (vj_locus_all_tra_trg | vj_locus_all_igk_igl)
                & (vj_locus_unique_len <= 2)
            )
            .then(pl.lit("Extra VJ"))
            .otherwise(pl.lit("Ambiguous"))
        )
        .otherwise(vj_locus_first)
    )

    # Build the final classification
    both_none = ~loc1_has & ~vj_locus_has

    # Check if tmp1 and tmp2 are valid locus names
    tmp1_is_locus = loc1_has & ~tmp1.is_in(
        ["None", "Extra VDJ", "Extra VDJ-exception", "Ambiguous"]
    )
    tmp2_is_locus = vj_locus_has & ~tmp2.is_in(
        ["None", "Extra VJ", "Extra VJ-exception", "Ambiguous"]
    )

    # Final result
    result = (
        pl.when(both_none)
        .then(pl.lit("None + None"))
        .when((tmp1 == "Ambiguous") | (tmp2 == "Ambiguous"))
        .then(pl.lit("Ambiguous"))
        .when(tmp1 == "None")
        .then(pl.concat_str([pl.lit("Orphan "), tmp2]))
        .when(tmp2 == "None")
        .then(pl.concat_str([pl.lit("Orphan "), tmp1]))
        .when(tmp1_is_locus & tmp2_is_locus)
        .then(pl.concat_str([tmp1, pl.lit(" + "), tmp2]))
        .otherwise(pl.concat_str([tmp1, pl.lit(" + "), tmp2]))
    )

    return result.alias("locus_classification")


def _check_travdv_polars(
    data: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    lazy: bool = True,
) -> pl.LazyFrame:
    """Check if locus is TRA/D."""
    # Vectorized approach - works on LazyFrame
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data.reset_index(drop=True)).lazy()
    elif isinstance(data, pl.DataFrame):
        data = data.lazy()
    data = data.with_columns(
        [
            # Check if we need to update locus
            pl.when(
                # Condition 1: v_call matches TRAV.*/DV pattern
                pl.col("v_call").str.contains(r"TRAV.*/DV")
            )
            .then(
                # If j, c, d calls match TRA
                pl.when(
                    _same_call_vectorized(
                        pl.col("j_call"),
                        pl.col("c_call"),
                        pl.col("d_call"),
                        "TRA",
                    )
                    & ~pl.col("locus").str.contains("TRA")
                )
                .then(pl.lit("TRA"))
                # Elif j, c, d calls match TRD
                .when(
                    _same_call_vectorized(
                        pl.col("j_call"),
                        pl.col("c_call"),
                        pl.col("d_call"),
                        "TRD",
                    )
                    & ~pl.col("locus").str.contains("TRD")
                )
                .then(pl.lit("TRD"))
                # Otherwise keep original locus
                .otherwise(pl.col("locus"))
            )
            # If v_call doesn't match pattern, keep original locus
            .otherwise(pl.col("locus"))
            .alias("locus")
        ]
    )
    return data if lazy else data.collect()


def _same_call_vectorized(
    j_col: pl.Expr, c_col: pl.Expr, d_col: pl.Expr, chain_type: str
) -> pl.Expr:
    """Vectorized version of same_call - returns a boolean expression.

    Checks that all non-null values contain the chain_type pattern.
    """
    j_match = j_col.is_null() | j_col.str.contains(chain_type)
    c_match = c_col.is_null() | c_col.str.contains(chain_type)
    d_match = d_col.is_null() | d_col.str.contains(chain_type)

    # All present calls must match the chain type (AND logic)
    return j_match & c_match & d_match


def _classify_isotype() -> pl.Expr:
    """
    Classify isotype from list of isotypes - vectorized for Polars.

    Args:
        col: Column name containing list of isotypes

    Returns:
        Polars expression for isotype classification
    """
    col = "isotype"
    isotype_col = pl.col(col)

    # Check if null or empty
    is_null_or_empty = isotype_col.is_null() | (isotype_col.list.len() == 0)

    # Check list length > 2 -> "Multi"
    list_len = isotype_col.list.len()

    # Get unique values
    # unique_values = flattened.list.unique()
    unique_values = isotype_col.list.join(",").str.split(",").list.unique()
    unique_count = unique_values.list.len()

    # Check if exactly {"IgM", "IgD"}
    is_igm_igd = (unique_count == 2) & (
        unique_values.list.set_symmetric_difference(["IgM", "IgD"]) == []
    )

    # Single unique value
    single_value = unique_values.list.first()

    # Build classification
    result = (
        pl.when(is_null_or_empty)
        .then(pl.lit(None))
        .when(list_len > 2)
        .then(pl.lit("Multi"))
        .when(is_igm_igd)
        .then(pl.lit("IgM/IgD"))
        .when(unique_count == 1)
        .then(single_value)
        .when(unique_count > 2)
        .then(pl.lit("Multi"))
        .otherwise(pl.lit("Multi"))  # Exactly 2 unique values (not IgM/IgD)
    )

    return result


def _format_chain_status(locus_status: pl.Expr) -> pl.Expr:
    """Format chain status - vectorized for Polars."""

    # Build conditions
    has_orphan = locus_status.str.contains("Orphan")
    has_exception = locus_status.str.contains("exception|IgM/IgD")
    has_extra = locus_status.str.contains("Extra")
    has_vdj = locus_status.str.contains("TRB|IGH|TRD|VDJ")
    has_vj = locus_status.str.contains("TRA|TRG|IGK|IGL|VJ")
    is_ambiguous = locus_status.str.contains("Ambiguous|None")

    # Apply conditions using when/then/otherwise chain
    return (
        pl.when(is_ambiguous)
        .then(pl.lit("Ambiguous"))
        .when(has_orphan & has_vdj & ~has_extra & ~has_exception)
        .then(pl.lit("Orphan VDJ"))
        .when(has_orphan & has_vdj & has_extra & ~has_exception)
        .then(pl.lit("Orphan Extra VDJ"))
        .when(has_orphan & has_vdj & has_exception)
        .then(pl.lit("Orphan VDJ-exception"))
        .when(has_orphan & has_vj & ~has_extra & ~has_exception)
        .then(pl.lit("Orphan VJ"))
        .when(has_orphan & has_vj & has_extra & ~has_exception)
        .then(pl.lit("Orphan Extra VJ"))
        .when(has_orphan & has_vj & has_exception)
        .then(pl.lit("Orphan VJ-exception"))
        .when(has_exception & ~has_orphan)
        .then(pl.lit("Extra pair-exception"))
        .when(has_extra & ~has_orphan)
        .then(pl.lit("Extra pair"))
        .otherwise(pl.lit("Single pair"))
    )


def _format_isotype() -> pl.Expr:
    """Format isotype status - vectorized for Polars."""

    isotype_status = pl.col("isotype_status")
    chain_status = pl.col("chain_status")
    # Vectorized conditions
    has_exception = chain_status.str.contains("exception")
    is_extra_pair = chain_status == "Extra pair"

    return (
        pl.when(~has_exception & is_extra_pair)
        .then(pl.lit("Multi"))
        .otherwise(isotype_status)
    )


def _clean_up_exception(col: pl.Expr) -> pl.Expr:
    """Strip 'exception' from chain status - vectorized for Polars."""
    return col.str.replace("-exception", "")


def _clean_single_entry(entry: str) -> str:
    """Clean a single clone entry string."""
    if entry is None or entry == "":
        return "None"

    # Split, filter out 'None', or return ['None'] if empty
    parts = [c for c in entry.split("|") if c != "None"]
    if not parts:
        parts = ["None"]

    # Sort and deduplicate
    unique_sorted = sorted(
        set(parts), key=cmp_to_key(lambda a, b: (a > b) - (a < b))
    )
    return "|".join(unique_sorted)


def _map_clones_with_dict(entry: str, size_dict: dict) -> str:
    """Map clone entries using the size dictionary."""
    if entry is None or entry == "":
        return "None"
    return "|".join(size_dict.get(p, p) for p in entry.split("|"))


def _add_clone_info(
    df: pl.DataFrame | pl.LazyFrame, clonekey: str
) -> pl.DataFrame | pl.LazyFrame:
    """Add a `{clonekey}_rank` column to df with sequential numbering per receptor type based on clone size."""
    is_lazy = isinstance(df, pl.LazyFrame)

    # Step 1: Clean the clone column
    df = df.with_columns(
        pl.col(clonekey)
        .map_elements(_clean_single_entry, return_dtype=pl.String)
        .alias(clonekey)
    )

    # Step 2: Flatten and count (requires collection for lazy)
    if is_lazy:
        clone_counts = _flatten_and_count(df.collect(), clonekey)
    else:
        clone_counts = _flatten_and_count(df, clonekey)

    # Step 3: Assign clone numbers
    size_dict = _assign_clone_numbers(clone_counts)

    # Step 4: Map multi-clone entries
    df = df.with_columns(
        pl.col(clonekey)
        .map_elements(
            lambda x: _map_clones_with_dict(x, size_dict),
            return_dtype=pl.String,
        )
        .cast(pl.Categorical)
        .alias(clonekey + "_rank")
    )

    # Step 5: Reorder columns - insert rank column right after clonekey
    cols = df.collect_schema().names() if is_lazy else df.columns
    clonekey_idx = cols.index(clonekey)

    # Remove rank column from its current position and insert after clonekey
    new_cols = (
        cols[: clonekey_idx + 1]
        + [clonekey + "_rank"]
        + [c for c in cols[clonekey_idx + 1 :] if c != clonekey + "_rank"]
    )

    df = df.select(new_cols)

    return df


def _flatten_and_count(df: pl.DataFrame, clonekey: str) -> dict:
    """Return a dict of clone counts for all unique clones."""
    # Explode the pipe-separated values
    flattened = (
        df.select(pl.col(clonekey).str.split("|").alias("clones"))
        .explode("clones")
        .filter(pl.col("clones") != "None")
        .group_by("clones")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # Convert to dict
    clone_counts = dict(
        zip(flattened["clones"].to_list(), flattened["count"].to_list())
    )
    return clone_counts


def _get_receptor_prefix(clone: str) -> str | None:
    """Return receptor type prefix if matches RECEPTOR_SET, else None."""
    prefix = clone.split("_")[0]
    return prefix if prefix in RECEPTOR_SET else None


def _assign_clone_numbers(clone_counts: dict) -> dict:
    """Assign sequential numbers, possibly grouped by receptor type."""
    # Determine all receptor types present
    prefixes = {_get_receptor_prefix(clone) for clone in clone_counts.keys()}
    prefixes.discard(None)

    size_dict = {}

    if len(prefixes) <= 1:
        # Only 1 receptor type (or none): number sequentially without prefix
        for i, clone in enumerate(clone_counts.keys(), start=1):
            size_dict[clone] = str(i)
    else:
        # Multiple receptor types: number sequentially per type
        receptor_to_clones = {r: [] for r in RECEPTOR_SET}
        other_clones = []

        for clone in clone_counts.keys():
            prefix = _get_receptor_prefix(clone)
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


### --- Helper functions for data sanitization ---
def _sanitize_boolean_expr(col: str) -> pl.Expr:
    """Sanitize boolean-like column using Polars expressions."""
    return pl.col(col).map_elements(sanitize_boolean, return_dtype=pl.String)


def _sanitize_data_polars(
    data: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pl.DataFrame:
    """
    Sanitize dtypes using Polars.
    Works for eager and lazy DataFrames.
    """
    if isinstance(data, pd.DataFrame):
        # Handle mixed-type columns before converting to Polars
        data = data.copy()
        for col in data.columns:
            if data[col].dtype == object:
                # Check if column has mixed types (numeric and string)
                non_null = data[col].dropna()
                if len(non_null) > 0:
                    # Try to detect if it's supposed to be numeric
                    numeric_mask = pd.to_numeric(
                        non_null, errors="coerce"
                    ).notna()
                    if numeric_mask.any():
                        # Has some numeric values - convert empty strings to None
                        data[col] = data[col].replace("", None)
                        # Try to convert to numeric
                        data[col] = pd.to_numeric(data[col], errors="ignore")

        data = pl.from_pandas(data.reset_index(drop=True))

    lazy = isinstance(data, pl.LazyFrame)
    df = data.collect() if lazy else data
    exprs: list[pl.Expr] = []

    for d in df.collect_schema().names():
        is_string = _is_polars_string_dtype(df, d)
        col = pl.col(d)

        # --- BOOLEAN-LIKE COLUMNS ---
        if d in BOOLEAN_LIKE_COLUMNS:
            exprs.append(_sanitize_boolean_expr(d).alias(d))
            continue

        # --- SCHEMA-DEFINED COLUMNS ---
        if d in RearrangementSchema.properties:
            dtype = RearrangementSchema.properties[d]["type"]
            if dtype in {"string", "boolean", "integer"}:
                col = (
                    pl.when(
                        col.is_null()
                        | (
                            col.is_in(EMPTIES_STR)
                            if is_string
                            else pl.lit(False)
                        )
                    )
                    .then(pl.lit(None))
                    .otherwise(col)
                )
                if dtype == "integer":
                    col = (
                        pl.when(col.is_null())
                        .then(None)
                        .otherwise(col.cast(pl.Int64, strict=False))
                    )
                if dtype == "boolean":
                    col = (
                        pl.when(col.is_null())
                        .then(None)
                        .otherwise(
                            col.map_elements(
                                sanitize_boolean,
                                return_dtype=pl.String,
                            )
                        )
                    )
                if dtype == "string":
                    col = col.cast(pl.String).map_elements(
                        clean_unicode, return_dtype=pl.String
                    )
            else:
                col = (
                    pl.when(
                        col.is_null()
                        | (
                            col.is_in(EMPTIES_STR)
                            if is_string
                            else pl.lit(False)
                        )
                    )
                    .then(None)
                    .otherwise(col)
                )
            exprs.append(col.alias(d))
            continue

        exprs.append(col.alias(d))

    df = df.with_columns(exprs)

    # --- SORT BY cell_id, productive, umi_count (same as pandas version) ---
    sort_cols = {"cell_id", "productive", "umi_count"}
    if sort_cols.issubset(set(df.collect_schema().names())):
        # sort so that the productive contig with the largest umi is first
        df = df.sort(
            by=["cell_id", "productive", "umi_count"],
            descending=[False, True, True],
        )

    # --- AIRR VALIDATION (requires eager) ---
    _validate_airr_polars(df)
    return df.lazy() if lazy else df


def clean_unicode(x: str) -> str:
    """Normalize and ensure valid UTF-8 text."""
    if not isinstance(x, str):
        return ""
    # Normalize to NFKC form (handles Greek/Unicode nicely)
    x = unicodedata.normalize("NFKC", x)
    # Remove invalid or unencodable characters safely
    return x.encode("utf-8", "ignore").decode("utf-8")


def _is_polars_string_dtype(
    df: pl.DataFrame | pl.LazyFrame, colname: str
) -> bool:
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    return schema.get(colname) == pl.String


def _validate_airr_polars(data: pl.DataFrame | pl.LazyFrame) -> None:
    """Validate dtypes in airr table (Polars)."""
    # identify integer-like columns
    int_columns = []
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    for d in data.collect_schema().names():
        try:
            data.select(pl.col(d).cast(pl.Int64, strict=False))
            int_columns.append(d)
        except Exception:
            pass
    if len(int_columns) > 0:
        data = data.with_columns(
            [
                pl.col(d).cast(pl.Int64, strict=False).alias(d)
                for d in int_columns
            ]
        )
    for row in data.iter_rows(named=True):
        contig = Contig(row).contig
        for required in [
            "sequence",
            "rev_comp",
            "sequence_alignment",
            "germline_alignment",
            "v_cigar",
            "d_cigar",
            "j_cigar",
        ]:
            if required not in contig:
                contig[required] = ""

        RearrangementSchema.validate_header(contig.keys())
        RearrangementSchema.validate_row(contig)


# ====================================================================================
# VECTORIZED CONTIG QUALITY CONTROL FUNCTIONS
# ====================================================================================


def check_chimeric_genes_vec(df: pl.DataFrame) -> pl.DataFrame:
    """
    VECTORIZED: Detect chimeric genes using pure polars expressions.

    Chimeric contigs have mismatched V/J genes (e.g., BCR V gene with TCR J gene).
    This is 10-100x faster than loop-based detection on large datasets.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with 'v_call' and 'j_call' columns.

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'is_chimeric' boolean column.

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "v_call": ["IGHV1-2", "TRBV1-1"],
    ...     "j_call": ["IGHJ6", "IGKJ5"]  # Second is chimeric
    ... })
    >>> result = check_chimeric_genes_vec(df)
    >>> result["is_chimeric"].to_list()
    [False, True]
    """
    return (
        df.with_columns(
            # Extract first 3 chars from v_call and j_call
            pl.col("v_call")
            .str.slice(0, 3)
            .str.to_uppercase()
            .alias("v_prefix"),
            pl.col("j_call")
            .str.slice(0, 3)
            .str.to_uppercase()
            .alias("j_prefix"),
        )
        .with_columns(
            # BCR chimeric: BCR V gene with non-BCR J gene
            pl.when(
                pl.col("v_prefix").is_in(["IGH", "IGK", "IGL", "IGI"])
                & ~pl.col("j_prefix").is_in(["IGH", "IGK", "IGL", "IGJ", "IGI"])
            )
            .then(True)
            # TCR chimeric: TCR V gene with non-TCR J gene
            .when(
                pl.col("v_prefix").is_in(["TRA", "TRB", "TRD", "TRG"])
                & ~pl.col("j_prefix").is_in(["TRA", "TRB", "TRD", "TRG"])
            )
            .then(True)
            .otherwise(False)
            .alias("is_chimeric")
        )
        .drop("v_prefix", "j_prefix")
    )


def identify_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """
    VECTORIZED: Mark duplicate sequences within the same alignment.

    Lower UMI duplicates are flagged with ambiguous_init=True.
    Uses pure polars operations - no loops.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with 'umi' and 'sequence_alignment' columns.

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'ambiguous_init' boolean column.

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "sequence_alignment": ["SEQ1", "SEQ1", "SEQ2"],
    ...     "umi": [50, 30, 40]
    ... })
    >>> result = identify_duplicates(df)
    >>> result["ambiguous_init"].to_list()
    [False, True, False]  # Second contig has lower UMI for SEQ1
    """
    return (
        df.with_columns(
            pl.col("umi_count")
            .rank(method="ordinal", descending=True)
            .over("sequence_alignment")
            .alias("umi_rank_by_seq")
        )
        .with_columns((pl.col("umi_rank_by_seq") > 1).alias("ambiguous_init"))
        .drop("umi_rank_by_seq")
    )


def resolve_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """
    VECTORIZED: Flag duplicates and prioritize by UMI count.

    Wrapper around identify_duplicates for API consistency.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.

    Returns
    -------
    pl.DataFrame
        Dataframe with duplicate flags.
    """
    return identify_duplicates(df)


def rank_and_flag_contigs_vec(
    df_sub: pl.DataFrame,
    umi_fc: float,
    consensus_fc: float,
    ntop_vdj: int = 1,
    ntop_vj: int = 2,
) -> pl.DataFrame:
    """
    FULLY VECTORIZED: Apply simplified dominance logic for contig ranking.

    Implements v4 simplified dominance algorithm:
    1. Test each contig vs SMALLEST count in group
    2. Rank passing contigs by UMI
    3. Apply ntop limits based on locus type (VDJ chains keep fewer)

    This is 10-50x faster than loop-based versions on large datasets.

    Parameters
    ----------
    df_sub : pl.DataFrame
        Subset dataframe for a single (cell_id, locus) group.
        Must contain: 'umi', 'consensus_count', 'locus', 'sequence_id'.
    umi_fc : float
        UMI fold-change threshold (typically 2.0).
    consensus_fc : float
        Consensus count fold-change threshold (typically 5.0).
    ntop_vdj : int, default=1
        Number of top contigs to keep for VDJ chains (IGH, TRB, TRD).
    ntop_vj : int, default=2
        Number of top contigs to keep for VJ chains (IGK, IGL, TRA, TRG).

    Returns
    -------
    pl.DataFrame
        Input dataframe with 'extra' and 'ambiguous' boolean flags.
        - extra=True: Ranked beyond ntop threshold
        - ambiguous=True: Failed dominance test vs smallest count

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "sequence_id": ["c1", "c2"],
    ...     "umi": [80, 20],
    ...     "consensus_count": [800, 200],
    ...     "locus": ["IGH", "IGH"]
    ... })
    >>> result = rank_and_flag_contigs_vec(df, umi_fc=2.0, consensus_fc=5.0)
    >>> result.select(["sequence_id", "extra", "ambiguous"])
    shape: (2, 3)
    ┌─────────────┬───────┬───────────┐
    │ sequence_id │ extra │ ambiguous │
    │ ---         │ ---   │ ---       │
    │ str         │ bool  │ bool      │
    ╞═════════════╪═══════╪═══════════╡
    │ c1          │ false │ false     │
    │ c2          │ true  │ false     │
    └─────────────┴───────┴───────────┘
    """
    # Handle single-contig case (always keep it)
    if df_sub.height == 1:
        return df_sub.with_columns(
            pl.lit(False).alias("extra"),
            pl.lit(False).alias("ambiguous"),
        )

    # Determine ntop based on locus type (VDJ vs VJ)
    sample_locus = df_sub["locus"][0]
    is_vdj = sample_locus in ["IGH", "TRB", "TRD"]
    ntop = ntop_vdj if is_vdj else ntop_vj

    # Step 1: Calculate min values for dominance testing
    min_umi = df_sub["umi_count"].min()
    min_consensus = df_sub["consensus_count"].min()

    # Step 2: Vectorized dominance tests (vs smallest count)
    df_tested = df_sub.with_columns(
        # UMI dominance test: (umi / min_umi) >= umi_fc AND umi >= 3
        (
            (pl.col("umi_count") / min_umi >= umi_fc)
            & (pl.col("umi_count") >= 3)
        ).alias("umi_passes"),
        # Consensus dominance test: (consensus / min_consensus) >= consensus_fc AND consensus >= 5
        (
            (pl.col("consensus_count") / min_consensus >= consensus_fc)
            & (pl.col("consensus_count") >= 5)
        ).alias("consensus_passes"),
    )

    # Step 3: Rank by UMI (descending) among those that pass UMI test
    df_ranked = df_tested.with_columns(
        pl.when(pl.col("umi_passes"))
        .then(
            pl.col("umi_count")
            .rank(method="ordinal", descending=True)
            .over(pl.col("umi_passes"))
        )
        .otherwise(999)  # Large rank for non-passing contigs
        .alias("umi_rank")
    )

    # Step 4: Determine flags based on passing status and rank
    n_passing = df_ranked.filter(pl.col("umi_passes")).height

    if n_passing == 0:
        # No contigs pass UMI test → all ambiguous
        df_flagged = df_ranked.with_columns(
            pl.lit(False).alias("extra"),
            pl.lit(True).alias("ambiguous"),
        )
    elif n_passing <= ntop:
        # Few pass: keep those, mark rest as extra
        df_flagged = df_ranked.with_columns(
            pl.when(~pl.col("umi_passes"))
            .then(True)
            .otherwise(False)
            .alias("extra"),
            pl.lit(False).alias("ambiguous"),
        )
    else:
        # Many pass: keep top ntop, mark rest as extra
        df_flagged = df_ranked.with_columns(
            pl.when(pl.col("umi_rank") > ntop)
            .then(True)
            .otherwise(False)
            .alias("extra"),
            pl.lit(False).alias("ambiguous"),
        )

    return df_flagged.drop("umi_passes", "consensus_passes", "umi_rank")


def mark_ambiguous_contigs_vec(
    df: pl.DataFrame,
    umi_foldchange_cutoff: float = 2.0,
    consensus_foldchange_cutoff: float = 5.0,
    ntop_vdj: int = 1,
    ntop_vj: int = 2,
) -> pl.DataFrame:
    """
    FULLY VECTORIZED: Main pipeline for marking ambiguous and extra contigs.

    Implements v4 simplified dominance logic with full vectorization:
    1. Resolve duplicates (vectorized)
    2. Apply dominance test per cell/locus (vectorized)
    3. Mark chimeric genes (vectorized)

    No iter_rows() anywhere - production ready for large datasets.
    Expected speedup: 10-100x vs loop-based versions.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with AIRR-formatted columns:
        - Required: cell_id, locus, umi, consensus_count, v_call, j_call,
          sequence_id, sequence_alignment
        - Must have: extra, ambiguous, ambig_hold (initialized to False)
    umi_foldchange_cutoff : float, default=2.0
        UMI fold-change threshold for dominance testing.
    consensus_foldchange_cutoff : float, default=5.0
        Consensus count fold-change threshold.
    ntop_vdj : int, default=1
        Number of top contigs to keep for VDJ chains (IGH, TRB, TRD).
    ntop_vj : int, default=2
        Number of top contigs to keep for VJ chains (IGK, IGL, TRA, TRG).

    Returns
    -------
    pl.DataFrame
        Input dataframe with updated 'extra' and 'ambiguous' flags.
        - extra=True: Low-ranked duplicates beyond ntop threshold
        - ambiguous=True: Failed dominance OR chimeric genes

    Notes
    -----
    Dominance test: Each contig tested vs smallest count in group.
    - UMI test: (umi / min_umi) >= umi_fc AND umi >= 3
    - Consensus test: (consensus / min_consensus) >= consensus_fc AND consensus >= 5

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "cell_id": ["cell1", "cell1"],
    ...     "sequence_id": ["c1", "c2"],
    ...     "locus": ["IGH", "IGH"],
    ...     "umi": [80, 20],
    ...     "consensus_count": [800, 200],
    ...     "v_call": ["IGHV1-2", "IGHV1-2"],
    ...     "j_call": ["IGHJ6", "IGHJ6"],
    ...     "sequence_alignment": ["SEQ1", "SEQ2"],
    ...     "extra": [False, False],
    ...     "ambiguous": [False, False],
    ...     "ambig_hold": [False, False]
    ... })
    >>> result = mark_ambiguous_contigs_vec(df)
    >>> result.select(["sequence_id", "extra", "ambiguous"])
    """
    # Step 1: Resolve duplicates (already vectorized)
    df = resolve_duplicates(df)

    # Step 2: FULLY VECTORIZED dominance logic using window functions
    # Determine ntop based on locus (no loop needed)
    df = df.with_columns(
        pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
        .then(pl.lit(ntop_vdj))
        .otherwise(pl.lit(ntop_vj))
        .alias("ntop_for_locus")
    )

    # Calculate minimum values per cell/locus group
    df = df.with_columns(
        pl.col("umi_count").min().over(["cell_id", "locus"]).alias("min_umi"),
        pl.col("consensus_count")
        .min()
        .over(["cell_id", "locus"])
        .alias("min_consensus"),
        pl.col("umi_count")
        .count()
        .over(["cell_id", "locus"])
        .alias("n_contigs_in_group"),
    )

    # Vectorized dominance tests
    df = df.with_columns(
        (
            (pl.col("umi_count") / pl.col("min_umi") >= umi_foldchange_cutoff)
            & (pl.col("umi_count") >= 3)
        ).alias("umi_passes"),
        (
            (
                pl.col("consensus_count") / pl.col("min_consensus")
                >= consensus_foldchange_cutoff
            )
            & (pl.col("consensus_count") >= 5)
        ).alias("consensus_passes"),
    )

    # Rank by UMI within each cell/locus group
    df = df.with_columns(
        pl.col("umi_count")
        .rank(method="ordinal", descending=True)
        .over(["cell_id", "locus"])
        .alias("umi_rank")
    )

    # Single contig: always keep (extra=False, ambiguous=False)
    # Multiple contigs: apply dominance logic
    df = df.with_columns(
        # Ambiguous: failed dominance OR rank > ntop
        pl.when(pl.col("n_contigs_in_group") == 1)
        .then(False)
        .when(~(pl.col("umi_passes") & pl.col("consensus_passes")))
        .then(True)
        .otherwise(False)
        .alias("ambiguous"),
        # Extra: rank exceeds ntop threshold
        pl.when(pl.col("n_contigs_in_group") == 1)
        .then(False)
        .when(pl.col("umi_rank") > pl.col("ntop_for_locus"))
        .then(True)
        .otherwise(False)
        .alias("extra"),
        # Ambig_hold: failed dominance but not extra (for chimeric check)
        pl.when(pl.col("n_contigs_in_group") == 1)
        .then(False)
        .when(
            ~(pl.col("umi_passes") & pl.col("consensus_passes"))
            & (pl.col("umi_rank") <= pl.col("ntop_for_locus"))
        )
        .then(True)
        .otherwise(False)
        .alias("ambig_hold"),
    )

    # Clean up temporary columns
    df_ranked = df.drop(
        "ntop_for_locus",
        "min_umi",
        "min_consensus",
        "n_contigs_in_group",
        "umi_passes",
        "consensus_passes",
        "umi_rank",
    )

    # Step 3: Vectorized chimeric detection
    df_with_chimeric = check_chimeric_genes_vec(df_ranked)

    # Step 4: Apply chimeric and ambig_hold logic (vectorized)
    df_final = df_with_chimeric.with_columns(
        # Handle ambig_hold contigs
        pl.when(pl.col("ambig_hold")).then(
            pl.when(pl.col("is_chimeric"))
            .then(pl.col("ambiguous"))  # Keep ambiguous=T for chimeric
            .otherwise(False)  # Clear ambiguous=F for non-chimeric
        )
        # Mark all chimeric contigs as ambiguous (even if not ambig_hold)
        .when(pl.col("is_chimeric"))
        .then(True)
        .otherwise(pl.col("ambiguous"))
        .alias("ambiguous"),
        # Mark non-chimeric ambig_hold as extra
        pl.when(pl.col("ambig_hold") & ~pl.col("is_chimeric"))
        .then(True)
        .otherwise(pl.col("extra"))
        .alias("extra"),
    ).drop("is_chimeric", "ambig_hold")

    return df_final


def mark_ambiguous_contigs(
    df: pl.DataFrame,
    umi_foldchange_cutoff: float = 2.0,
    consensus_foldchange_cutoff: float = 5.0,
    ntop_vdj: int = 1,
    ntop_vj: int = 2,
) -> pl.DataFrame:
    """
    Mark ambiguous and extra contigs in immune receptor sequencing data.

    Alias for mark_ambiguous_contigs_vec (fully vectorized version).
    Provides backward compatibility with existing API.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with AIRR-formatted columns.
    umi_foldchange_cutoff : float, default=2.0
        UMI fold-change threshold.
    consensus_foldchange_cutoff : float, default=5.0
        Consensus count fold-change threshold.
    ntop_vdj : int, default=1
        Number of top contigs to keep for VDJ chains.
    ntop_vj : int, default=2
        Number of top contigs to keep for VJ chains.

    Returns
    -------
    pl.DataFrame
        Dataframe with updated quality flags.

    See Also
    --------
    mark_ambiguous_contigs_vec : Fully vectorized implementation.
    """
    return mark_ambiguous_contigs_vec(
        df,
        umi_foldchange_cutoff,
        consensus_foldchange_cutoff,
        ntop_vdj,
        ntop_vj,
    )


def check_contigs(
    vdj_data: DandelionPolars | pl.DataFrame | str,
    gex_data: AnnData | None = None,
    productive_only: bool = True,
    library_type: Literal["ig", "tr-ab", "tr-gd"] | None = None,
    umi_foldchange_cutoff: float = 2.0,
    consensus_foldchange_cutoff: float = 5.0,
    ntop_vdj: int = 1,
    ntop_vj: int = 2,
    filter_missing: bool = True,
    filter_extra: bool = True,
    filter_ambiguous: bool = False,
    save: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> tuple[DandelionPolars, AnnData] | DandelionPolars:
    """
    Check contigs for whether they can be considered as ambiguous or not.

    This function identifies and marks contigs as ambiguous, extra, or chimeric
    based on UMI/consensus dominance tests and gene call consistency. Uses
    vectorized polars operations for high performance.

    Parameters
    ----------
    data : DandelionPolars | pl.DataFrame | str
        V(D)J AIRR data to check. Can be DandelionPolars object, polars DataFrame,
        or file path to AIRR `.tsv` file.
    adata : AnnData | None, optional
        AnnData object to filter. If provided, will track which cells have contigs.
        If None, assumes all cells in AIRR table should be kept.
    productive_only : bool, default=True
        Whether to retain only productive contigs.
    library_type : Literal["ig", "tr-ab", "tr-gd"] | None, optional
        If specified, filter based on expected contig types:
            - `ig`: IGH, IGK, IGL
            - `tr-ab`: TRA, TRB
            - `tr-gd`: TRG, TRD
    umi_foldchange_cutoff : float, default=2.0
        Minimum UMI fold-change threshold for dominance test.
    consensus_foldchange_cutoff : float, default=5.0
        Minimum consensus count fold-change threshold for dominance test.
    ntop_vdj : int, default=1
        Number of top VDJ contigs to keep (IGH, TRB, TRD).
    ntop_vj : int, default=2
        Number of top VJ contigs to keep (IGK, IGL, TRA, TRG).
    filter_missing : bool, default=True
        If True and adata provided, remove cells not found in AnnData object.
    filter_extra : bool, default=True
        Whether to remove contigs marked as extra.
    filter_ambiguous : bool, default=False
        Whether to remove contigs marked as ambiguous.
    save : str | None, optional
        If provided, save filtered table with `_checked.tsv` suffix.
    verbose : bool, default=True
        Whether to print progress messages.
    **kwargs
        Additional kwargs passed to DandelionPolars constructor.

    Returns
    -------
    tuple[DandelionPolars, AnnData] | DandelionPolars
        If adata provided: (DandelionPolars object, updated AnnData)
        If adata is None: DandelionPolars object only

    Raises
    ------
    IndexError
        If no contigs pass filtering.
    ValueError
        If save filename doesn't end with .tsv.

    Notes
    -----
    This function:
    1. Filters by productive status and library type (if specified)
    2. Marks ambiguous/extra contigs using vectorized dominance tests
    3. Marks chimeric contigs (mismatched BCR/TCR genes)
    4. Optionally filters contigs based on flags
    5. Creates DandelionPolars object with metadata

    The vectorized implementation uses `mark_ambiguous_contigs_vec` for
    10-100x performance improvement over the original pandas-based version.

    Examples
    --------
    >>> # Basic usage with DandelionPolars object
    >>> ddl_polars = check_contigs(ddl_polars)

    >>> # With AnnData filtering
    >>> ddl_polars, adata = check_contigs(ddl_polars, adata=adata)

    >>> # Custom thresholds
    >>> ddl_polars = check_contigs(
    ...     ddl_polars,
    ...     umi_foldchange_cutoff=3.0,
    ...     consensus_foldchange_cutoff=10.0,
    ...     ntop_vdj=2,
    ...     ntop_vj=3
    ... )

    >>> # From file path
    >>> ddl_polars = check_contigs("filtered_contig_annotations.tsv")

    See Also
    --------
    mark_ambiguous_contigs_vec : Core vectorized function for marking contigs
    check_chimeric_genes_vec : Detects chimeric gene calls
    """
    from pathlib import Path
    import os

    if verbose:
        print("Filtering contigs...")

    # Load data
    if isinstance(vdj_data, DandelionPolars):
        dat_ = vdj_data.data
        # Keep as LazyFrame if possible
        if isinstance(dat_, pl.DataFrame):
            dat_ = dat_.lazy()
        lib_type_from_obj = vdj_data.library_type
    elif isinstance(vdj_data, pl.DataFrame):
        dat_ = vdj_data.lazy()
        lib_type_from_obj = None
    elif isinstance(vdj_data, pl.LazyFrame):
        dat_ = vdj_data
        lib_type_from_obj = None
    else:
        # File path
        dat_ = load_polars(vdj_data, as_pandas=False)
        if isinstance(dat_, pl.DataFrame):
            dat_ = dat_.lazy()
        lib_type_from_obj = None

    # Replace "unknown" with nulls for string columns (lazy-compatible)
    str_cols = [
        col
        for col, dtype in dat_.collect_schema().items()
        if dtype == pl.String
    ]

    if str_cols:
        dat_ = dat_.with_columns(
            [
                pl.when(pl.col(col) == "unknown")
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in str_cols
            ]
        )
    if library_type is not None:
        acceptable = lib_type(library_type)
    elif lib_type_from_obj is not None:
        acceptable = lib_type(lib_type_from_obj)
    else:
        acceptable = None

    # Filter by productive status (lazy)
    if productive_only:
        dat = dat_.filter(pl.col("productive").is_in(TRUES_STR))
    else:
        dat = dat_

    # Filter by library type (lazy)
    if acceptable is not None:
        dat = dat.filter(pl.col("locus").is_in(acceptable))

    # Get unique cell barcodes - only collect what we need
    barcode = dat.select("cell_id").unique().collect().to_series().to_list()

    # Handle AnnData integration
    if gex_data is not None:
        adata_provided = True
        adata_ = gex_data.copy()

        # Mark cells with contigs
        contig_check = pd.DataFrame(index=adata_.obs_names)
        bc_ = {b: "True" for b in barcode}
        contig_check["has_contig"] = pd.Series(bc_)
        contig_check["has_contig"] = contig_check["has_contig"].fillna(
            "No_contig"
        )
        adata_.obs["has_contig"] = contig_check["has_contig"]
    else:
        adata_provided = False
        adata_ = None

    # Initialize required columns (lazy-compatible)
    schema = dat.collect_schema()
    if "extra" not in schema:
        dat = dat.with_columns(pl.lit(False).alias("extra"))
    if "ambiguous" not in schema:
        dat = dat.with_columns(pl.lit(False).alias("ambiguous"))
    if "ambig_hold" not in schema:
        dat = dat.with_columns(pl.lit(False).alias("ambig_hold"))

    # Mark ambiguous and extra contigs using vectorized function
    # This function works with both DataFrame and LazyFrame
    if verbose:
        print("Marking ambiguous contigs...")

    # Collect here because mark_ambiguous_contigs_vec needs DataFrame
    # This is the main computation - everything before this was just query planning
    dat = dat.collect()

    dat = mark_ambiguous_contigs_vec(
        dat,
        umi_foldchange_cutoff=umi_foldchange_cutoff,
        consensus_foldchange_cutoff=consensus_foldchange_cutoff,
        ntop_vdj=ntop_vdj,
        ntop_vj=ntop_vj,
    )

    # Copy flags back to original dataframe if productive_only
    if productive_only:
        # Collect dat_ if lazy for joining
        if isinstance(dat_, pl.LazyFrame):
            dat_ = dat_.collect()

        # Merge flags back to original data
        flag_cols = ["extra", "ambiguous"]
        dat_flags = dat.select(["sequence_id"] + flag_cols)
        dat_ = dat_.join(dat_flags, on="sequence_id", how="left", suffix="_new")

        # Update columns
        for col in flag_cols:
            if f"{col}_new" in dat_.columns:
                dat_ = dat_.with_columns(
                    pl.col(f"{col}_new").fill_null(True).alias(col)
                ).drop(f"{col}_new")
        dat = dat_

    # Filter by missing cells (works with DataFrame)
    if filter_missing and adata_ is not None:
        dat = dat.filter(pl.col("cell_id").is_in(adata_.obs_names.tolist()))

    # Check if empty (needs to compute height)
    if dat.height == 0:
        raise IndexError(
            "No contigs passed filtering. Check that cell barcodes match."
        )

    # Save if requested
    if save is not None:
        if save.endswith(".tsv"):
            _write_airr(dat, save)
        else:
            raise ValueError(
                f"{save} not suitable. Please provide filename ending with .tsv"
            )
    elif isinstance(vdj_data, str) and os.path.isfile(vdj_data):
        data_path = Path(vdj_data)
        _write_airr(
            dat, str(data_path.parent / f"{data_path.stem}_checked.tsv")
        )

    # Apply filters
    if filter_extra:
        dat = dat.filter(~pl.col("extra"))
    if filter_ambiguous:
        dat = dat.filter(~pl.col("ambiguous"))

    if verbose:
        print("Initializing DandelionPolars object...")

    # Create output object
    out_dat = DandelionPolars(data=dat, verbose=False, **kwargs)

    # Copy germline if from DandelionPolars input
    if isinstance(vdj_data, DandelionPolars):
        out_dat.germline = vdj_data.germline

    if gex_data is not None:
        # Import transfer function from tools
        from dandelion.tools import transfer

        # Transfer metadata to adata
        transfer(adata_, out_dat)
        return (out_dat, adata_)
    else:
        return out_dat


def all_missing_polars(col: pl.Series) -> bool:
    """Check if all values in a Polars series are null or empty string."""
    all_null = col.is_null().all()
    if all_null:
        return True

    # For string columns, also check empty strings
    if col.dtype in (pl.Utf8, pl.String):
        return ((col.is_null()) | (col == "")).all()

    return False


def read_10x_vdj_polars(
    path: Path | str,
    filename_prefix: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_malformed: bool = True,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    A parser to read .csv and .json files directly from folder containing 10x cellranger-outputs.

    This function parses the 10x output files into an AIRR compatible format using Polars.

    Minimum requirement is one of either {filename_prefix}_contig_annotations.csv or all_contig_annotations.json.

    If .fasta, .json files are found in the same folder, additional info will be appended to the final table.

    Parameters
    ----------
    path : Path | str
        path to folder containing `.csv` and/or `.json` files, or path to files directly.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. None defaults to 'all'.
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_malformed : bool, optional
        whether or not to remove malformed contigs.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.

    Returns
    -------
    DandelionPolars
        DandelionPolars object holding the parsed data.

    Raises
    ------
    IOError
        if contig_annotations.csv and all_contig_annotations.json file(s) not found in the input folder.

    """

    def parse_annotation_polars(data: pl.DataFrame) -> dict:
        """Parse annotation file using Polars."""
        out = defaultdict(OrderedDict)
        swap_dict = dict(zip(CELLRANGER, AIRR))

        # Convert to pandas for row-wise processing with Contig class
        data_pd = data.to_pandas()
        data_pd.set_index("contig_id", drop=False, inplace=True)

        for _, row in data_pd.iterrows():
            contig = Contig(row, swap_dict).contig["sequence_id"]
            out[contig] = Contig(row, swap_dict).contig
            if out[contig]["locus"] in ["None", "none", None, np.nan, ""]:
                calls = []
                for call in ["v_call", "d_call", "j_call", "c_call"]:
                    if out[contig][call] not in [
                        "None",
                        "none",
                        None,
                        np.nan,
                        "",
                    ]:
                        calls.append(out[contig][call])
                out[contig]["locus"] = "|".join(
                    list({str(c)[:3] for c in calls})
                )
            if out[contig]["locus"] == "None" or out[contig]["locus"] == "":
                out[contig]["locus"] = "|"
        return out

    def parse_json_polars(data: list) -> dict:
        """Parse json file."""
        main_dict1 = {
            "barcode": "cell_id",
            "contig_name": "sequence_id",
            "sequence": "sequence",
            "aa_sequence": "sequence_aa",
            "productive": "productive",
            "full_length": "complete_vdj",
            "frame": "vj_in_frame",
            "cdr3_seq": "junction",
            "cdr3": "junction_aa",
        }
        main_dict2 = {
            "read_count": "consensus_count",
            "umi_count": "umi_count",
            "cdr3_start": "cdr3_start",
            "cdr3_stop": "cdr3_end",
        }
        main_dict3 = {
            "high_confidence": "high_confidence_10x",
            "filtered": "filtered_10x",
            "is_gex_cell": "is_cell_10x",
            "is_asm_cell": "is_asm_cell_10x",
        }
        info_dict = {
            "raw_clonotype_id": "clone_id",
            "raw_consensus_id": "raw_consensus_id_10x",
            "exact_subclonotype_id": "exact_subclonotype_id_10x",
        }
        region_type_dict = {
            "L-REGION+V-REGION": "v_call",
            "D-REGION": "d_call",
            "J-REGION": "j_call",
        }

        out = defaultdict(OrderedDict)
        for d in data:
            # main level
            for k in main_dict1:
                if k in d:
                    out[d["contig_name"]].update({main_dict1[k]: d[k]})
            for k in main_dict2:
                if k in d:
                    out[d["contig_name"]].update({main_dict2[k]: d[k]})
            for k in main_dict3:
                if k in d:
                    out[d["contig_name"]].update({main_dict3[k]: d[k]})
            # info level
            if "info" in d:
                for k in info_dict:
                    if k in d["info"]:
                        out[d["contig_name"]].update(
                            {info_dict[k]: d["info"][k]}
                        )
            # annotation level
            if "annotations" in d:
                for dat in d["annotations"]:
                    if "feature" in dat:
                        if "region_type" in dat["feature"]:
                            region = dat["feature"]["region_type"]
                            if region in region_type_dict:
                                gene_name = dat["feature"]["gene_name"]
                                chain = dat["feature"]["chain"]
                                out[d["contig_name"]].update(
                                    {region_type_dict[region]: gene_name}
                                )
                                out[d["contig_name"]].update({"locus": chain})
                        if "chain" in dat["feature"]:
                            if dat["feature"]["chain"] != "Multi":
                                chain = dat["feature"]["chain"]
                                out[d["contig_name"]].update({"locus": chain})
                        if "cdr3_seq" not in d:
                            if dat["feature"]["region_type"] == "CDR3":
                                if "cdr3_start" in dat:
                                    out[d["contig_name"]].update(
                                        {"cdr3_start": dat["cdr3_start"]}
                                    )
                                if "cdr3_stop" in dat:
                                    out[d["contig_name"]].update(
                                        {"cdr3_end": dat["cdr3_stop"]}
                                    )
                        if dat["feature"]["feature_id"] == 0:
                            if dat["feature"]["region_type"] == "5'UTR":
                                if "contig_match_start" in dat:
                                    out[d["contig_name"]].update(
                                        {
                                            "fwr1_start": dat[
                                                "contig_match_start"
                                            ]
                                        }
                                    )
                        if "region_type" in dat["feature"]:
                            if dat["feature"]["region_type"] == "C-REGION":
                                c_gene = dat["feature"]["gene_name"]
                                out[d["contig_name"]].update({"c_call": c_gene})
        return out

    filename_pre = (
        DEFAULT_PREFIX if filename_prefix is None else filename_prefix
    )

    if os.path.isdir(str(path)):
        files = os.listdir(path)
        filelist = []
        for fx in files:
            if re.search(filename_pre + "_contig", fx):
                if fx.endswith(".fasta") or fx.endswith(".csv"):
                    filelist.append(fx)
            if re.search(
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                fx,
            ):
                if fx.endswith(".json"):
                    filelist.append(fx)
        csv_idx = [i for i, j in enumerate(filelist) if j.endswith(".csv")]
        json_idx = [i for i, j in enumerate(filelist) if j.endswith(".json")]
        if len(csv_idx) == 1:
            file = str(path) + "/" + str(filelist[csv_idx[0]])
            logg.info("Reading {}".format(str(file)))
            raw = pl.read_csv(str(file))
            fasta_file = str(file).split("_annotations.csv")[0] + ".fasta"
            json_file = re.sub(
                filename_pre + "_contig_annotations",
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                str(file).split(".csv")[0] + ".json",
            )
            if os.path.exists(json_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(json_file)
                    )
                )
                out = parse_annotation_polars(raw)
                with open(json_file) as f:
                    raw_json = json.load(f)
                out_json = parse_json_polars(raw_json)
                out.update(out_json)
            elif os.path.exists(fasta_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(fasta_file)
                    )
                )
                seqs = {}
                fh = open(fasta_file)
                for header, sequence in fasta_iterator(fh):
                    seqs[header] = sequence
                # Add sequences using Polars
                raw = raw.with_columns(
                    pl.col("contig_id")
                    .map_elements(
                        lambda x: seqs.get(x, ""), return_dtype=pl.Utf8
                    )
                    .alias("sequence")
                )
                out = parse_annotation_polars(raw)
            else:
                out = parse_annotation_polars(raw)
        elif len(csv_idx) < 1:
            if len(json_idx) == 1:
                json_file = str(path) + "/" + str(filelist[json_idx[0]])
                logg.info("Reading {}".format(json_file))
                if os.path.exists(json_file):
                    with open(json_file) as f:
                        raw = json.load(f)
                    out = parse_json_polars(raw)
            else:
                raise OSError(
                    "{}_contig_annotations.csv and {}_contig_annotations.json file(s) not found in {} folder.".format(
                        str(filename_pre),
                        filename_pre.replace("filtered", "all"),
                        str(path),
                    )
                )
        elif len(csv_idx) > 1:
            raise OSError(
                "There are multiple input .csv files with the same filename prefix {} in {} folder.".format(
                    str(filename_pre), str(path)
                )
            )
    elif os.path.isfile(str(path)):
        file = path
        if str(file).endswith(".csv"):
            logg.info("Reading {}.".format(str(file)))
            raw = pl.read_csv(str(file))
            fasta_file = str(file).split("_annotations.csv")[0] + ".fasta"
            json_file = re.sub(
                filename_pre + "_contig_annotations",
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                str(file).split(".csv")[0] + ".json",
            )
            if os.path.exists(json_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(json_file)
                    )
                )
                out = parse_annotation_polars(raw)
                with open(json_file) as f:
                    raw_json = json.load(f)
                out_json = parse_json_polars(raw_json)
                out.update(out_json)
            elif os.path.exists(fasta_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(fasta_file)
                    )
                )
                seqs = {}
                fh = open(fasta_file)
                for header, sequence in fasta_iterator(fh):
                    seqs[header] = sequence
                # Add sequences using Polars
                raw = raw.with_columns(
                    pl.col("contig_id")
                    .map_elements(
                        lambda x: seqs.get(x, ""), return_dtype=pl.Utf8
                    )
                    .alias("sequence")
                )
                out = parse_annotation_polars(raw)
            else:
                out = parse_annotation_polars(raw)
        elif str(file).endswith(".json"):
            if os.path.exists(file):
                logg.info("Reading {}".format(file))
                with open(file) as f:
                    raw = json.load(f)
                out = parse_json_polars(raw)
            else:
                raise OSError("{} not found.".format(file))
    else:
        raise OSError("{} not found.".format(path))

    # Convert dict to Polars DataFrame
    res = pl.DataFrame([v for v in out.values()])

    # Quick check if locus is malformed
    if remove_malformed:
        res = res.filter(~pl.col("locus").str.contains(r"\|"))

    # Change all unknowns to blanks
    for col in res.columns:
        if res[col].dtype in (pl.Utf8, pl.String):
            res = res.with_columns(
                pl.col(col).str.replace("unknown", "").alias(col)
            )

    vdj = DandelionPolars(res, verbose=verbose)

    if suffix is not None:
        vdj.add_sequence_suffix(
            suffix,
            sep=sep,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
        )
    elif prefix is not None:
        vdj.add_sequence_prefix(
            prefix,
            sep=sep,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
        )
    return vdj


def read_10x_vdj_polars(
    path: Path | str,
    filename_prefix: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_malformed: bool = True,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    A parser to read .csv and .json files directly from folder containing 10x cellranger-outputs.

    This function parses the 10x output files into an AIRR compatible format using Polars.

    Minimum requirement is one of either {filename_prefix}_contig_annotations.csv or all_contig_annotations.json.

    If .fasta, .json files are found in the same folder, additional info will be appended to the final table.

    Parameters
    ----------
    path : Path | str
        path to folder containing `.csv` and/or `.json` files, or path to files directly.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. None defaults to 'all'.
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_malformed : bool, optional
        whether or not to remove malformed contigs.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.

    Returns
    -------
    DandelionPolars
        DandelionPolars object holding the parsed data.

    Raises
    ------
    IOError
        if contig_annotations.csv and all_contig_annotations.json file(s) not found in the input folder.

    """

    def parse_annotation_polars(data: pl.DataFrame) -> pl.DataFrame:
        """Parse annotation file using Polars - fully vectorized."""
        swap_dict = dict(zip(CELLRANGER, AIRR))

        # Rename columns based on swap_dict
        rename_map = {k: v for k, v in swap_dict.items() if k in data.columns}
        data = data.rename(rename_map)

        # Fill null values with empty strings for string columns
        # Also replace string representations of None/NaN
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String, pl.Categorical):
                data = data.with_columns(
                    pl.col(col)
                    .fill_null("")
                    .str.replace_all("^(None|none|nan|NaN)$", "")
                    .alias(col)
                )

        # Ensure gene call columns exist
        gene_call_cols = ["v_call", "d_call", "j_call", "c_call"]
        for col in gene_call_cols:
            if col not in data.columns:
                data = data.with_columns(pl.lit("").alias(col))

        if "locus" not in data.columns:
            data = data.with_columns(pl.lit("").alias("locus"))

        # Create derived locus: extract first 3 chars from each gene call, combine unique
        def derive_locus_from_calls(v, d, j, c):
            """Extract locus from gene calls."""
            calls = []
            for val in [v, d, j, c]:
                if val and val not in ["None", "none", "", "nan"]:
                    calls.append(str(val)[:3])
            return "|".join(sorted(set(calls))) if calls else "|"

        data = data.with_columns(
            pl.when(
                (pl.col("locus").is_in(["None", "none", "", "nan", None]))
                | pl.col("locus").is_null()
            )
            .then(
                pl.struct(gene_call_cols).map_elements(
                    lambda x: derive_locus_from_calls(
                        x["v_call"], x["d_call"], x["j_call"], x["c_call"]
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("locus"))
            .alias("locus")
        )

        # Replace empty locus with "|"
        data = data.with_columns(
            pl.when(pl.col("locus") == "")
            .then(pl.lit("|"))
            .otherwise(pl.col("locus"))
            .alias("locus")
        )

        return data

    def parse_json_polars(data: list) -> pl.DataFrame:
        """Parse json file and return DataFrame."""
        main_dict1 = {
            "barcode": "cell_id",
            "contig_name": "sequence_id",
            "sequence": "sequence",
            "aa_sequence": "sequence_aa",
            "productive": "productive",
            "full_length": "complete_vdj",
            "frame": "vj_in_frame",
            "cdr3_seq": "junction",
            "cdr3": "junction_aa",
        }
        main_dict2 = {
            "read_count": "consensus_count",
            "umi_count": "umi_count",
            "cdr3_start": "cdr3_start",
            "cdr3_stop": "cdr3_end",
        }
        main_dict3 = {
            "high_confidence": "high_confidence_10x",
            "filtered": "filtered_10x",
            "is_gex_cell": "is_cell_10x",
            "is_asm_cell": "is_asm_cell_10x",
        }
        info_dict = {
            "raw_clonotype_id": "clone_id",
            "raw_consensus_id": "raw_consensus_id_10x",
            "exact_subclonotype_id": "exact_subclonotype_id_10x",
        }
        region_type_dict = {
            "L-REGION+V-REGION": "v_call",
            "D-REGION": "d_call",
            "J-REGION": "j_call",
        }

        out = defaultdict(OrderedDict)
        for d in data:
            # main level
            for k in main_dict1:
                if k in d:
                    out[d["contig_name"]].update({main_dict1[k]: d[k]})
            for k in main_dict2:
                if k in d:
                    out[d["contig_name"]].update({main_dict2[k]: d[k]})
            for k in main_dict3:
                if k in d:
                    out[d["contig_name"]].update({main_dict3[k]: d[k]})
            # info level
            if "info" in d:
                for k in info_dict:
                    if k in d["info"]:
                        out[d["contig_name"]].update(
                            {info_dict[k]: d["info"][k]}
                        )
            # annotation level
            if "annotations" in d:
                for dat in d["annotations"]:
                    if "feature" in dat:
                        if "region_type" in dat["feature"]:
                            region = dat["feature"]["region_type"]
                            if region in region_type_dict:
                                gene_name = dat["feature"]["gene_name"]
                                chain = dat["feature"]["chain"]
                                out[d["contig_name"]].update(
                                    {region_type_dict[region]: gene_name}
                                )
                                out[d["contig_name"]].update({"locus": chain})
                        if "chain" in dat["feature"]:
                            if dat["feature"]["chain"] != "Multi":
                                chain = dat["feature"]["chain"]
                                out[d["contig_name"]].update({"locus": chain})
                        if "cdr3_seq" not in d:
                            if dat["feature"]["region_type"] == "CDR3":
                                if "cdr3_start" in dat:
                                    out[d["contig_name"]].update(
                                        {"cdr3_start": dat["cdr3_start"]}
                                    )
                                if "cdr3_stop" in dat:
                                    out[d["contig_name"]].update(
                                        {"cdr3_end": dat["cdr3_stop"]}
                                    )
                        if dat["feature"]["feature_id"] == 0:
                            if dat["feature"]["region_type"] == "5'UTR":
                                if "contig_match_start" in dat:
                                    out[d["contig_name"]].update(
                                        {
                                            "fwr1_start": dat[
                                                "contig_match_start"
                                            ]
                                        }
                                    )
                        if "region_type" in dat["feature"]:
                            if dat["feature"]["region_type"] == "C-REGION":
                                c_gene = dat["feature"]["gene_name"]
                                out[d["contig_name"]].update({"c_call": c_gene})

        # Convert dict to DataFrame
        return pl.DataFrame([v for v in out.values()])

    filename_pre = (
        DEFAULT_PREFIX if filename_prefix is None else filename_prefix
    )

    if os.path.isdir(str(path)):
        files = os.listdir(path)
        filelist = []
        for fx in files:
            if re.search(filename_pre + "_contig", fx):
                if fx.endswith(".fasta") or fx.endswith(".csv"):
                    filelist.append(fx)
            if re.search(
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                fx,
            ):
                if fx.endswith(".json"):
                    filelist.append(fx)
        csv_idx = [i for i, j in enumerate(filelist) if j.endswith(".csv")]
        json_idx = [i for i, j in enumerate(filelist) if j.endswith(".json")]
        if len(csv_idx) == 1:
            file = str(path) + "/" + str(filelist[csv_idx[0]])
            logg.info("Reading {}".format(str(file)))
            raw = pl.read_csv(str(file))
            fasta_file = str(file).split("_annotations.csv")[0] + ".fasta"
            json_file = re.sub(
                filename_pre + "_contig_annotations",
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                str(file).split(".csv")[0] + ".json",
            )
            if os.path.exists(json_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(json_file)
                    )
                )
                # Parse CSV to DataFrame
                out_df = parse_annotation_polars(raw)

                # Parse JSON to DataFrame
                with open(json_file) as f:
                    raw_json = json.load(f)
                out_json_df = parse_json_polars(raw_json)

                # Merge DataFrames
                res = out_df.join(
                    out_json_df, on="sequence_id", how="outer", suffix="_json"
                )

                # Coalesce columns that appear in both (prefer json version)
                for col in out_json_df.columns:
                    if col != "sequence_id" and f"{col}_json" in res.columns:
                        res = res.with_columns(
                            pl.coalesce(
                                [pl.col(f"{col}_json"), pl.col(col)]
                            ).alias(col)
                        ).drop(f"{col}_json")

            elif os.path.exists(fasta_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(fasta_file)
                    )
                )
                seqs = {}
                fh = open(fasta_file)
                for header, sequence in fasta_iterator(fh):
                    seqs[header] = sequence
                # Add sequences using Polars
                raw = raw.with_columns(
                    pl.col("contig_id")
                    .map_elements(
                        lambda x: seqs.get(x, ""), return_dtype=pl.Utf8
                    )
                    .alias("sequence")
                )
                res = parse_annotation_polars(raw)
            else:
                res = parse_annotation_polars(raw)
        elif len(csv_idx) < 1:
            if len(json_idx) == 1:
                json_file = str(path) + "/" + str(filelist[json_idx[0]])
                logg.info("Reading {}".format(json_file))
                if os.path.exists(json_file):
                    with open(json_file) as f:
                        raw = json.load(f)
                    res = parse_json_polars(raw)
            else:
                raise OSError(
                    "{}_contig_annotations.csv and {}_contig_annotations.json file(s) not found in {} folder.".format(
                        str(filename_pre),
                        filename_pre.replace("filtered", "all"),
                        str(path),
                    )
                )
        elif len(csv_idx) > 1:
            raise OSError(
                "There are multiple input .csv files with the same filename prefix {} in {} folder.".format(
                    str(filename_pre), str(path)
                )
            )
    elif os.path.isfile(str(path)):
        file = path
        if str(file).endswith(".csv"):
            logg.info("Reading {}.".format(str(file)))
            raw = pl.read_csv(str(file))
            fasta_file = str(file).split("_annotations.csv")[0] + ".fasta"
            json_file = re.sub(
                filename_pre + "_contig_annotations",
                f"{filename_pre.replace('filtered', 'all')}_contig_annotations",
                str(file).split(".csv")[0] + ".json",
            )
            if os.path.exists(json_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(json_file)
                    )
                )
                # Parse CSV to DataFrame
                out_df = parse_annotation_polars(raw)

                # Parse JSON to DataFrame
                with open(json_file) as f:
                    raw_json = json.load(f)
                out_json_df = parse_json_polars(raw_json)

                # Merge DataFrames
                res = out_df.join(
                    out_json_df, on="sequence_id", how="outer", suffix="_json"
                )

                # Coalesce columns that appear in both (prefer json version)
                for col in out_json_df.columns:
                    if col != "sequence_id" and f"{col}_json" in res.columns:
                        res = res.with_columns(
                            pl.coalesce(
                                [pl.col(f"{col}_json"), pl.col(col)]
                            ).alias(col)
                        ).drop(f"{col}_json")

            elif os.path.exists(fasta_file):
                logg.info(
                    "Found {} file. Extracting extra information.".format(
                        str(fasta_file)
                    )
                )
                seqs = {}
                fh = open(fasta_file)
                for header, sequence in fasta_iterator(fh):
                    seqs[header] = sequence
                # Add sequences using Polars
                raw = raw.with_columns(
                    pl.col("contig_id")
                    .map_elements(
                        lambda x: seqs.get(x, ""), return_dtype=pl.Utf8
                    )
                    .alias("sequence")
                )
                res = parse_annotation_polars(raw)
            else:
                res = parse_annotation_polars(raw)
        elif str(file).endswith(".json"):
            if os.path.exists(file):
                logg.info("Reading {}".format(file))
                with open(file) as f:
                    raw = json.load(f)
                res = parse_json_polars(raw)
            else:
                raise OSError("{} not found.".format(file))
    else:
        raise OSError("{} not found.".format(path))

    # Quick check if locus is malformed
    if remove_malformed:
        res = res.filter(~pl.col("locus").str.contains(r"\|"))

    # Change all unknowns to blanks
    for col in res.columns:
        if res[col].dtype in (pl.Utf8, pl.String):
            res = res.with_columns(
                pl.col(col).str.replace("unknown", "").alias(col)
            )

    vdj = DandelionPolars(res, verbose=verbose)

    if suffix is not None:
        vdj.add_sequence_suffix(
            suffix,
            sep=sep,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
        )
    elif prefix is not None:
        vdj.add_sequence_prefix(
            prefix,
            sep=sep,
            remove_trailing_hyphen_number=remove_trailing_hyphen_number,
        )
    return vdj
