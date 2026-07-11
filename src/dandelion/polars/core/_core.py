#!/usr/bin/env python
from __future__ import annotations
import copy
import h5py
import os
import shutil
import tempfile
import unicodedata
import warnings
import zarr

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from airr import RearrangementSchema
from anndata import AnnData
from changeo.IO import readGermlines
from functools import cmp_to_key, reduce
from pandas.api.types import infer_dtype
from pathlib import Path
from polars import ColumnNotFoundError
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from textwrap import dedent
from typing import Literal

from dandelion.utilities._utilities import (
    RECEPTOR_SET,
    EMPTIES_STR,
    BOOLEAN_LIKE_COLUMNS,
    CHECK_COLS,
    MUTATIONS,
    VDJLENGTHS,
    SEQINFO,
    sanitize_boolean,
    sanitize_data,
    sanitize_data_for_saving,
    clear_h5file,
    lib_type,
    Contig,
)
from dandelion.utilities._utilities import (
    TRUES_STR,
    write_fasta,
    LocalStore,
    ZipStore,
    BloscCodec,
    open_zarr_group,
    create_zarr_dataset,
)

# Enable string cache for Polars to optimize repeated string operations
pl.enable_string_cache()

_FLOAT_SUFFIXES = [
    "identity",
    "alignment_length",
    "number_of_mismatches",
    "number_of_gap_openings",
    "sequence_start",
    "sequence_end",
    "germline_start",
    "germline_end",
    "support",
    "score",
]

# Suffixes that only exist without _blastn
_STRING_SUFFIXES = ["source"]

_GENES = ["v", "d", "j", "c"]

SCHEMA_OVERRIDES = {
    # All v/d/j/c float columns, both with and without _blastn suffix
    **{
        f"{gene}_{suffix}{ext}": pl.Float64
        for gene in _GENES
        for suffix in _FLOAT_SUFFIXES
        for ext in ["", "_blastn"]
    },
    # All v/d/j/c string columns (no _blastn variant)
    **{
        f"{gene}_{suffix}": pl.String
        for gene in _GENES
        for suffix in _STRING_SUFFIXES
    },
    # J-only multiplicity/multimapper columns
    "j_call_multimappers": pl.String,
    "j_call_multiplicity": pl.Float64,
    "j_call_sequence_start_multimappers": pl.String,
    "j_call_sequence_end_multimappers": pl.String,
    "j_call_support_multimappers": pl.String,
}


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
        graph: (
            tuple[nx.Graph, nx.Graph] | tuple[ig.Graph, ig.Graph] | None
        ) = None,
        distances: csr_matrix | None = None,
        initialize: bool = True,
        library_type: Literal["tr-ab", "tr-gd", "ig"] | None = None,
        lazy: bool = True,
        verbose: bool = True,
        cache_handles: dict[str, tempfile.NamedTemporaryFile] | None = None,
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
        graph : tuple[nx.Graph, nx.Graph] | tuple[ig.Graph, ig.Graph] | None, optional
            networkx or igraph graphs for clonotype networks.
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
        self._lazy = lazy
        self._data_name_col = "sequence_id"
        self._metadata_name_col = "cell_id"
        self._backend = "polars"
        # Preserve existing cache_handles if re-initializing or accept provided ones
        if cache_handles is not None:
            self._cache_handles = cache_handles
        elif not hasattr(self, "_cache_handles"):
            self._cache_handles = {}

        self._data = load_polars(data, lazy=self._lazy)
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
                self._data = self._data.filter(
                    pl.col("locus").is_in(acceptable)
                )
                if self._lazy:
                    if isinstance(self._data, pl.LazyFrame):
                        self._data = self._data.collect(
                            engine="streaming"
                        ).lazy()
                    else:
                        self._data = self._data.lazy()
            self._data = _check_travdv_polars(self._data, lazy=self._lazy)
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
            if self._lazy:
                self._data = self._data.collect(engine="streaming").lazy()
                # Keep temp files alive - don't close them yet
            if metadata is None:
                if initialize is True:
                    self._ensure_sanitized_data(verbose=verbose)
                    self.update_metadata(lazy=self._lazy, **kwargs)
            else:
                if isinstance(metadata, pd.DataFrame):
                    if self._metadata_name_col not in metadata:
                        # change the name of the index first before resetting
                        if metadata.index.name is None:
                            metadata.index.name = self._metadata_name_col
                    metadata = pl.from_pandas(metadata.reset_index(drop=False))
                if self._lazy:
                    if isinstance(metadata, pl.DataFrame):
                        self._metadata = metadata.lazy()
                    else:
                        self._metadata = metadata
                else:
                    if isinstance(metadata, pl.LazyFrame):
                        self._metadata = metadata.collect(engine="streaming")
                        # Keep temp files alive
                    else:
                        self._metadata = metadata

        if isinstance(self._data, pl.LazyFrame):
            self._original_sequence_ids = (
                self._data.select(self._data_name_col)
                .collect(engine="streaming")
                .to_series()
            )
            self._original_cell_ids = (
                self._data.select(self._metadata_name_col)
                .collect(engine="streaming")
                .to_series()
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
        if self._lazy:
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
            if isinstance(self.graph[0], ig.Graph):
                descr += f"\n    graph: {', '.join(['igraph graph of '+ str(x.vcount()) + ' vertices' for x in (self.graph[0], self.graph[1]) if x is not None])} "
            else:
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

        # Convert pandas index types to polars equivalents if needed
        if isinstance(index, pd.Series):
            index = pl.from_pandas(index)
        elif isinstance(index, pd.DataFrame):
            index = pl.from_pandas(index)
        elif isinstance(index, pd.Index):
            index = pl.Series(index.tolist())

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
                _metadata = None  # Will be synced below
            elif isinstance(filter_expr, pl.Series):
                # Boolean mask provided
                # Check if the mask length matches metadata (cell-level) or data (contig-level)
                if metadata is not None:
                    if isinstance(metadata, pl.LazyFrame):
                        metadata_len = metadata.collect(
                            engine="streaming"
                        ).height
                    else:
                        metadata_len = metadata.height
                else:
                    metadata_len = 0

                mask_len = len(filter_expr)

                # If mask length matches metadata, it's a cell-level mask
                # Apply to metadata first, then filter data by resulting cell_ids
                if metadata is not None and mask_len == metadata_len:
                    # Apply mask to metadata to get cell_ids
                    if isinstance(metadata, pl.LazyFrame):
                        _metadata = (
                            metadata.with_row_index("__row_idx__")
                            .filter(
                                pl.col("__row_idx__").is_in(
                                    [i for i, v in enumerate(filter_expr) if v]
                                )
                            )
                            .drop("__row_idx__")
                        )
                        filtered_cell_ids = (
                            _metadata.select("cell_id")
                            .collect(engine="streaming")
                            .to_series()
                            .unique()
                        )
                    else:
                        _metadata = metadata.filter(filter_expr)
                        filtered_cell_ids = (
                            _metadata.select("cell_id").to_series().unique()
                        )

                    # Now filter data by those cell_ids
                    _data = data.filter(
                        pl.col("cell_id").is_in(filtered_cell_ids)
                    )
                    cell_ids = filtered_cell_ids
                else:
                    # Mask length matches data (contig-level), apply directly
                    if isinstance(data, pl.LazyFrame):
                        # LazyFrame cannot directly use an eager boolean Series; fallback to index-based filter
                        _data = (
                            data.with_row_index("__row_idx__")
                            .filter(
                                pl.col("__row_idx__").is_in(
                                    [i for i, v in enumerate(filter_expr) if v]
                                )
                            )
                            .drop("__row_idx__")
                        )
                    else:
                        # Eager DataFrame: filter directly with the boolean Series to avoid large Python lists
                        _data = data.filter(filter_expr)
                    _metadata = None  # Will be synced below
            else:
                # Assume it's an Expression
                # For now, create the filter and we'll handle metadata-only columns
                # when we try to collect
                _data = data.filter(filter_expr)
                _metadata = None  # Will be synced below

            # For metadata sync, extract unique cell_ids from filtered data
            # (skip if already extracted from metadata above)
            if cell_ids is None:
                if isinstance(_data, pl.LazyFrame):
                    try:
                        filtered_cell_ids = (
                            _data.select("cell_id")
                            .collect(engine="streaming")
                            .to_series()
                            .unique()
                        )
                    except ColumnNotFoundError:
                        # The filter expression references metadata-only columns
                        # Filter metadata instead and extract cell_ids
                        if metadata is not None:
                            filtered_metadata = metadata.filter(filter_expr)
                            if isinstance(filtered_metadata, pl.LazyFrame):
                                filtered_cell_ids = (
                                    filtered_metadata.select("cell_id")
                                    .collect(engine="streaming")
                                    .to_series()
                                    .unique()
                                )
                            else:
                                filtered_cell_ids = (
                                    filtered_metadata.select("cell_id")
                                    .to_series()
                                    .unique()
                                )
                            # Now filter data by the cell_ids from metadata
                            _data = data.filter(
                                pl.col("cell_id").is_in(filtered_cell_ids)
                            )
                            _metadata = filtered_metadata
                        else:
                            raise
                else:
                    try:
                        filtered_cell_ids = (
                            _data.select("cell_id").to_series().unique()
                        )
                    except ColumnNotFoundError:
                        # The filter expression references metadata-only columns
                        if metadata is not None:
                            filtered_metadata = metadata.filter(filter_expr)
                            if isinstance(filtered_metadata, pl.DataFrame):
                                filtered_cell_ids = (
                                    filtered_metadata.select("cell_id")
                                    .to_series()
                                    .unique()
                                )
                            else:
                                filtered_cell_ids = (
                                    filtered_metadata.select("cell_id")
                                    .collect(engine="streaming")
                                    .to_series()
                                    .unique()
                                )
                            # Now filter data by the cell_ids from metadata
                            _data = data.filter(
                                pl.col("cell_id").is_in(filtered_cell_ids)
                            )
                            _metadata = filtered_metadata
                        else:
                            raise
                # Sync metadata if not already set
                if _metadata is None:
                    _metadata = (
                        metadata.filter(
                            pl.col("cell_id").is_in(filtered_cell_ids)
                        )
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

        # Keep lazy if original was lazy; but always collect first to materialize
        # the data in memory. This ensures the result doesn't reference temp files
        # that may be deleted (e.g., from check_contigs).
        if isinstance(self._data, pl.LazyFrame):
            if isinstance(_data, pl.LazyFrame):
                # Collect to materialize, then re-lazy backed by in-memory data
                _data = _data.collect(engine="streaming").lazy()
            elif isinstance(_data, pl.DataFrame):
                _data = _data.lazy()
        if isinstance(self._metadata, pl.LazyFrame):
            if isinstance(_metadata, pl.LazyFrame):
                # Collect to materialize, then re-lazy backed by in-memory data
                _metadata = _metadata.collect(engine="streaming").lazy()
            elif isinstance(_metadata, pl.DataFrame):
                _metadata = _metadata.lazy()

        # If eager, compact memory so the slice does not retain large original buffers
        if isinstance(_data, pl.DataFrame):
            _data = _data.rechunk()
        if isinstance(_metadata, pl.DataFrame):
            _metadata = _metadata.rechunk()

        # ---- Distances matrix sync -----------------------------------
        if self.distances is not None:
            # Get ORIGINAL metadata cell_ids for distance matrix indexing
            # Use matrix shape as source of truth since _index_names may be outdated
            dist_size = self.distances.shape[0]

            # Get the first dist_size cells from original metadata
            if isinstance(self._metadata, pl.LazyFrame):
                original_cells = (
                    self._metadata.select("cell_id")
                    .head(dist_size)
                    .collect(engine="streaming")
                    .to_series()
                    .to_list()
                )
            elif isinstance(self._metadata, pl.DataFrame):
                original_cells = (
                    self._metadata.select("cell_id")
                    .head(dist_size)
                    .to_series()
                    .to_list()
                )
            else:
                # pandas DataFrame
                original_cells = self._metadata.head(dist_size).index.to_list()

            keep_set = set(cell_ids.to_list())
            keep = np.array(
                [i for i, c in enumerate(original_cells) if c in keep_set]
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
            if isinstance(self.graph[0], ig.Graph):
                _graph = (
                    self.graph[0].subgraph(
                        self.graph[0].vs.select(name_in=keep_set)
                    ),
                    self.graph[1].subgraph(
                        self.graph[1].vs.select(name_in=keep_set)
                    ),
                )
            else:
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
        # Preserve lazy distance embedding flags if present
        try:
            setattr(
                sliced,
                "_distance_zarr_path",
                getattr(self, "_distance_zarr_path", None),
            )
            setattr(
                sliced,
                "_distance_embed_pending",
                getattr(self, "_distance_embed_pending", False),
            )
        except Exception:
            pass
        if original_backend == "pandas":
            sliced.to_pandas()
        return sliced

    @property
    def n_obs(self) -> int:
        """Number of observations.

        Returns
        -------
        int
            Number of unique cells in `.metadata`.
        """
        if self._metadata is None:
            return 0
        if isinstance(self._metadata, pl.LazyFrame):
            return self._metadata.select(pl.count()).collect(
                engine="streaming"
            )[0, 0]
        if isinstance(self._metadata, pl.DataFrame):
            return self._metadata.height
        if isinstance(self._metadata, pd.DataFrame):
            return self._metadata.shape[0]

    @property
    def n_contigs(self) -> int:
        """Number of contigs.

        Returns
        -------
        int
            Number of contig rows in `.data`.
        """
        if self._data is None:
            return 0
        if isinstance(self._data, pl.LazyFrame):
            return self._data.select(pl.count()).collect(engine="streaming")[
                0, 0
            ]
        if isinstance(self._data, pl.DataFrame):
            return self._data.height
        if isinstance(self._data, pd.DataFrame):
            return self._data.shape[0]

    @property
    def data(self) -> pl.DataFrame | pl.LazyFrame:
        """One-dimensional annotation of contig observations.

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            The underlying contig-level data frame.
        """
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
        """Names of contig observations.

        Returns
        -------
        SeriesAccessor | pd.Index
            The sequence_id values indexing the contig data.
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data.index
        if isinstance(self._data, pl.DataFrame):
            # eager Polars
            return SeriesAccessor(self._data[self._data_name_col])
        if isinstance(self._data, pl.LazyFrame):
            # Lazy Polars: materialize first to get a Series
            series = self._data.select(self._data_name_col).collect(
                engine="streaming"
            )[self._data_name_col]
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
        """One-dimensional annotation of cell observations.

        Returns
        -------
        pd.DataFrame | DataFrameAccessor
            The underlying cell-level metadata frame.
        """
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
            if self._lazy:
                value = value.lazy()
        self._metadata = value
        return

    @property
    def metadata_names(self) -> pd.Index | pl.Series:
        """Names of cell observations.

        Returns
        -------
        pd.Index | pl.Series
            The cell_id values indexing the metadata.
        """
        if isinstance(self._metadata, pd.DataFrame):
            return self._metadata.index
        if isinstance(self._metadata, pl.DataFrame):
            # eager Polars
            return SeriesAccessor(self._metadata[self._metadata_name_col])
        if isinstance(self._metadata, pl.LazyFrame):
            # Lazy Polars: materialize first to get a Series
            series = self._metadata.select(self._metadata_name_col).collect(
                engine="streaming"
            )[self._metadata_name_col]
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
                self._data = _sanitize_data_polars(self._data)

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
            and infer_dtype(value) not in ("string", "bytes")
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
        for v in getattr(self, f"{attr}m", {}).values():
            if isinstance(v, pd.DataFrame):
                v.index = value

    def _cache_data(self) -> None:
        """Cache _data and _metadata into temp parquet files when lazy."""
        if not self._lazy:
            return
        self._data = self._cache_lazyframe(self._data, "data")
        self._metadata = self._cache_lazyframe(self._metadata, "metadata")

    def _cache_lazyframe(
        self, obj: pl.LazyFrame | pl.DataFrame | None, slot_name: str
    ) -> pl.LazyFrame | None:
        if obj is None:
            # Nothing to cache but make sure stale handles are cleaned up
            if slot_name in self._cache_handles:
                self._cache_handles[slot_name].close()
                del self._cache_handles[slot_name]
            return obj

        # Materialize first to avoid closing backing files too early
        df = (
            obj.collect(engine="streaming")
            if isinstance(obj, pl.LazyFrame)
            else obj
        )

        # Close and drop any stale temp file for this slot
        if slot_name in self._cache_handles:
            self._cache_handles[slot_name].close()
            del self._cache_handles[slot_name]

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete=True,
        )

        df.write_parquet(temp_file.name)
        temp_file.flush()

        lf = pl.scan_parquet(temp_file.name)

        # Store in the unified dict
        self._cache_handles[slot_name] = temp_file

        return lf

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
            Suffix to add to the IDs.
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
        Add suffix to cell_id and optionally to sequence_id.

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
                # _original_cell_ids has n_contigs entries; deduplicate for metadata index
                if isinstance(self._original_cell_ids, pd.Series):
                    unique_cell_ids_pd = (
                        self._original_cell_ids.drop_duplicates()
                    )
                else:
                    seen_pd: set = set()
                    unique_cell_ids_pd = [
                        x
                        for x in self._original_cell_ids
                        if not (x in seen_pd or seen_pd.add(x))  # type: ignore[func-returns-value]
                    ]
                self._metadata.index = unique_cell_ids_pd
                self._data[self._metadata_name_col] = self._original_cell_ids
        if isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
            self._data = self._data.with_columns(
                pl.Series(self._data_name_col, self._original_sequence_ids)
            )
            if isinstance(self._data, pl.LazyFrame):
                self._data = self._data.collect(engine="streaming").lazy()
        if self._metadata is not None:
            if isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
                # _original_cell_ids has one entry per contig (n_contigs, with dups);
                # metadata has one row per unique cell, so deduplicate while preserving order.
                if isinstance(self._original_cell_ids, pl.Series):
                    unique_cell_ids = self._original_cell_ids.unique(
                        maintain_order=True
                    )
                else:
                    seen: set = set()
                    unique_cell_ids = [
                        x
                        for x in self._original_cell_ids
                        if not (x in seen or seen.add(x))  # type: ignore[func-returns-value]
                    ]
                self._metadata = self._metadata.with_columns(
                    pl.Series(self._metadata_name_col, unique_cell_ids)
                )
                if isinstance(self._metadata, pl.LazyFrame):
                    self._metadata = self._metadata.collect(
                        engine="streaming"
                    ).lazy()
        # Ensure data is backed after potentially removing backing with .lazy()
        if self._lazy and isinstance(self._data, pl.LazyFrame):
            self._cache_data()

    def simplify(self, **kwargs) -> None:
        """Disambiguate VDJ and C gene calls when there's multiple calls separated by commas and strip the alleles.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to `update_metadata`.
        """
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
                    pl.col(col).unique().alias(out_col)
                    if unique
                    else pl.col(col).alias(out_col)
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
            .collect(engine="streaming")
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
            .collect(engine="streaming")
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
            .collect(engine="streaming")
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
            .collect(engine="streaming")
        )
        return result

    def _split(
        self,
        cols: list[str] | str,
        join: bool = True,
        explode: bool = False,
        unique: bool = False,
        key_added: list[str] | str | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        celltype: Literal["B", "abT", "gdT"] | None = None,
    ) -> pl.DataFrame:
        key_added = cols if key_added is None else key_added
        cols = [cols] if isinstance(cols, str) else cols
        key_added = [key_added] if isinstance(key_added, str) else key_added
        agg_exprs = [pl.col("_original_order").min().alias("_original_order")]

        data = self._data if data is None else data
        # Add row index once at the start
        data = data.lazy().with_row_index("_original_order")

        if celltype is not None:
            # Add a celltype_group column based on locus/isotype logic
            data = data.with_columns(
                pl.when(pl.col("locus").is_in(["IGH", "IGK", "IGL"]))
                .then(pl.lit("B"))
                .when(pl.col("locus").is_in(["TRB", "TRA"]))
                .then(pl.lit("abT"))
                .when(pl.col("locus").is_in(["TRD", "TRG"]))
                .then(pl.lit("gdT"))
                .otherwise(pl.lit("Unknown"))
                .alias("celltype_group")
            )
            # Filter for the requested celltype
            data = data.filter(pl.col("celltype_group") == celltype)
            group_keys = ["cell_id", "celltype_group"]
        else:
            group_keys = ["cell_id"]

        if explode:
            # Create separate numbered columns for each contig (like retrieve_mode="split")
            # First group by cell_id and get lists of values
            temp_result = (
                data.with_columns(
                    [
                        pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                        .then(pl.lit("VDJ"))
                        .otherwise(pl.lit("VJ"))
                        .alias("locus_group")
                    ]
                )
                .group_by(group_keys)
                .agg(
                    [pl.col("_original_order").min().alias("_original_order")]
                    + [
                        expr
                        for col in cols
                        for expr in [
                            pl.col(col)
                            .filter(
                                (pl.col("locus_group") == "VDJ")
                                & (pl.col(col).is_not_null())
                                & (pl.col(col) != "")
                            )
                            .alias(f"_{col}_VDJ_list"),
                            (
                                pl.col(col)
                                .filter(
                                    (pl.col("locus_group") == "VJ")
                                    & (pl.col(col).is_not_null())
                                    & (pl.col(col) != "")
                                )
                                .alias(f"_{col}_VJ_list")
                                if col != "d_call"
                                else None
                            ),
                        ]
                    ]
                )
                .sort("_original_order")
                .drop("_original_order")
                .collect(engine="streaming")
            )

            # Now explode into numbered columns using a more efficient approach
            result_cols = {"cell_id": temp_result["cell_id"]}
            if celltype is not None:
                result_cols["celltype_group"] = temp_result["celltype_group"]

            for col, key in zip(cols, key_added):
                # Handle VDJ
                vdj_col = f"_{col}_VDJ_list"
                if vdj_col in temp_result.columns:
                    vdj_series = temp_result[vdj_col]
                    max_vdj = vdj_series.list.len().max() or 0
                    if max_vdj > 0:
                        for i in range(max_vdj):
                            col_name = f"{key}_VDJ_{i+1}"
                            result_cols[col_name] = vdj_series.list.get(
                                i, null_on_oob=True
                            )

                # Handle VJ (skip for d_call)
                if col != "d_call":
                    vj_col = f"_{col}_VJ_list"
                    if vj_col in temp_result.columns:
                        vj_series = temp_result[vj_col]
                        max_vj = vj_series.list.len().max() or 0
                        if max_vj > 0:
                            for i in range(max_vj):
                                col_name = f"{key}_VJ_{i+1}"
                                result_cols[col_name] = vj_series.list.get(
                                    i, null_on_oob=True
                                )

            result = pl.DataFrame(result_cols)

        elif join:
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
                            .alias(f"{key}_VJ")
                            if unique
                            else pl.col(col)
                            .filter(pl.col("locus_group") == "VJ")
                            .str.join(delimiter="|")
                            .alias(f"{key}_VJ")
                        )
                        if col != "d_call"
                        else None
                    ),
                ]
            ]
            result = (
                data.with_columns(
                    [
                        pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                        .then(pl.lit("VDJ"))
                        .otherwise(pl.lit("VJ"))
                        .alias("locus_group")
                    ]
                )
                .group_by(group_keys)
                .agg(agg_exprs)
                .sort("_original_order")
                .drop("_original_order")
                .collect(engine="streaming")
            )
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
            result = (
                data.with_columns(
                    [
                        pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                        .then(pl.lit("VDJ"))
                        .otherwise(pl.lit("VJ"))
                        .alias("locus_group")
                    ]
                )
                .group_by(group_keys)
                .agg(agg_exprs)
                .sort("_original_order")
                .drop("_original_order")
                .collect(engine="streaming")
            )

        # If celltype filtering was used, rejoin with metadata to maintain all cells
        if celltype is not None:
            ref = self._metadata.with_row_index("_original_order").select(
                pl.col(["cell_id", "_original_order"])
            )
            result = (
                ref.lazy()
                .join(result.lazy(), on="cell_id", how="left")
                .sort("_original_order")
                .drop("_original_order")
                .collect(engine="streaming")
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
        celltype: Literal["B", "abT", "gdT"] | None = None,
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
        # Add row index once at the start
        data = data.lazy().with_row_index("_original_order")
        if celltype is not None:
            # Add a celltype_group column based on locus/isotype logic
            data = data.lazy().with_columns(
                pl.when(pl.col("locus").is_in(["IGH", "IGK", "IGL"]))
                .then(pl.lit("B"))
                .when(pl.col("locus").is_in(["TRB", "TRA"]))
                .then(pl.lit("abT"))
                .when(pl.col("locus").is_in(["TRD", "TRG"]))
                .then(pl.lit("gdT"))
                .otherwise(pl.lit("Unknown"))
                .alias("celltype_group")
            )
            # Filter for the requested celltype
            data = data.filter(pl.col("celltype_group") == celltype)
            group_keys = ["cell_id", "celltype_group"]
        else:
            group_keys = ["cell_id"]
        # Compute aggregation, keep _original_order
        result = (
            data.lazy()
            .with_columns(
                [
                    pl.when(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
                    .then(pl.lit("VDJ"))
                    .otherwise(pl.lit("VJ"))
                    .alias("locus_group")
                ]
            )
            .group_by(group_keys)
            .agg(agg_exprs)
            .sort("_original_order")
            .drop("_original_order")
            .collect(engine="streaming")
        )
        # Build reference with _original_order
        if celltype is not None:
            ref = self._metadata.with_row_index("_original_order").select(
                pl.col(["cell_id", "_original_order"])
            )
            # Now sort and drop _original_order
            result = (
                ref.lazy()
                .join(result.lazy(), on="cell_id", how="left")
                .sort("_original_order")
                .drop("_original_order")
                .lazy()
                .collect(engine="streaming")
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
            .collect(engine="streaming")
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
            .collect(engine="streaming")
        )
        return result

    def _reinitialize_attributes(
        self,
        data: pl.DataFrame,
        metadata: pl.DataFrame | None = None,
        clone_key: str | None = None,
        layout: (
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None
        ) = None,
        graph: (
            tuple[nx.Graph, nx.Graph] | tuple[ig.Graph, ig.Graph] | None
        ) = None,
        distances: csr_matrix | None = None,
        germline: dict[str, str] | None = None,
        reinitialize: bool = True,
        **kwargs,
    ) -> None:
        """
        Update instance attributes with collected data after processing.

        This method updates an existing DandelionPolars instance with eager
        (collected) dataframes and newly computed attributes. Unlike __init__,
        this assumes the data has already been validated and processed.

        Parameters
        ----------
        data : pl.DataFrame
            Collected VDJ data as an eager Polars DataFrame.
        metadata : pl.DataFrame | None, optional
            Optional collected metadata as an eager Polars DataFrame.
        clone_key : str | None, optional
            Column name to use as the clone identifier. Only used if reinitialize is True.
        layout : tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None, optional
            Node positions for computed graph.
        graph : tuple[nx.Graph, nx.Graph] | tuple[ig.Graph, ig.Graph] |None = None, optional
            NetworkX or igraph graphs for clonotype networks.
        distances : csr_matrix | None, optional
            Distance matrix for sequences.
        germline : dict[str, str] | None, optional
            Dictionary of germline gene:sequence records.
        reinitialize : bool, optional
            Whether to reinitialize metadata by calling update_metadata. Default is True.
        **kwargs
            Additional keyword arguments passed to update_metadata.

        Returns
        -------
        None
            Modifies the instance in place.
        """
        # Update to lazy or eager dataframes based on instance configuration
        if self._lazy:
            self._data = data.lazy()
            if metadata is not None:
                self._metadata = metadata.lazy()
        else:
            self._data = data
            if metadata is not None:
                self._metadata = metadata

        # Update optional attributes
        if layout is not None:
            self.layout = layout
        if graph is not None:
            self.graph = graph
        if distances is not None:
            self.distances = distances
        if germline is not None:
            self.germline.update(germline)  # Update dict instead of replacing

        # Reinitialize metadata if requested
        if reinitialize:
            self.update_metadata(clone_key=clone_key, **kwargs)

    def _initialize_metadata(
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
                .is_in(TRUES_STR + EMPTIES_STR)
            )
        if strip_alleles:
            for col in ["v_call", "d_call", "j_call", "c_call"]:
                if col in _data.collect_schema().names():
                    _data = _data.with_columns(
                        pl.col(col).str.replace_all(r"\*.*", "").alias(col)
                    )
        unique_cells = (
            _data.lazy()
            .with_row_index("_original_order")
            .group_by("cell_id")
            .agg(pl.col("_original_order").min())
            .sort("_original_order")
            .drop("_original_order")
            .select("cell_id")
            .collect(engine="streaming")
        )
        if len(init_cols) == 0:
            self._metadata = pl.DataFrame(
                {"cell_id": unique_cells["cell_id"].to_list()}
            )
            if self._lazy:
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
                # current_cells = meta_0["cell_id"].to_list()
                # assert current_cells == unique_cells["cell_id"].to_list()
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
                if self._lazy:
                    self._metadata = self._metadata.lazy()
            # Preserve order before join
            self._metadata = self._metadata.lazy().with_row_index("_join_order")
            if "c_call_VDJ" in self._metadata.collect_schema():
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
                    .collect(engine="streaming")
                )  # Keep only cell_id and isotype for joining
                if not isotype_df.select(
                    (
                        pl.col("isotype").is_null()
                        | (pl.col("isotype").list.len() == 0)
                        | pl.col("isotype")
                        .list.eval(pl.element().is_in(["", "None"]))
                        .list.all()
                    ).all()
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
                    self._metadata = self._metadata.with_columns(
                        _classify_locus_pair().alias("locus_status")
                    )
                else:
                    self._metadata = self._metadata.with_columns(
                        _classify_locus_pair_noiso().alias("locus_status")
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
            # sort back to original order
            self._metadata = self._metadata.sort("_join_order").drop(
                "_join_order"
            )
            # convert empty strings to nulls
            self._metadata = self._metadata.with_columns(
                [
                    pl.col(col).replace("", None)
                    for col in self._metadata.collect_schema()
                    if self._metadata.collect_schema()[col] == pl.String
                ]
            )
            # always lazy
            if self._lazy:
                self._metadata = self._metadata.collect(
                    engine="streaming"
                ).lazy()
            else:
                self._metadata = self._metadata.collect(engine="streaming")
            if "metadata" in self._cache_handles.keys():
                self._cache_handles["metadata"].close()
                del self._cache_handles["metadata"]
            if self._lazy:
                # back to tmpfile on disk when working lazily
                self._cache_data()

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
            self._data = self._data.collect(engine="streaming").to_pandas()
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
                    self._metadata = self._metadata.collect(
                        engine="streaming"
                    ).to_pandas()
                if isinstance(self._metadata, pl.DataFrame):
                    self._metadata = self._metadata.to_pandas()
                self._metadata.set_index(self._metadata_name_col, inplace=True)
            else:
                raise KeyError(
                    f"{self._metadata_name_col} not found in metadata columns."
                )
        self._lazy = False
        self._backend = "pandas"

    def to_polars(self, lazy: bool = True) -> None:
        """Convert self from Pandas to Polars implementation.

        Parameters
        ----------
        lazy : bool, optional
            Whether to use lazy (LazyFrame) mode for the converted Polars data.
            Defaults to True.
        """
        if self._backend == "polars":
            return
        if not isinstance(
            self._data, (pl.DataFrame, pl.LazyFrame)
        ) or not isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
            self._lazy = lazy
            if isinstance(self._data, pd.DataFrame):
                # drop index to avoid duplication
                self._data = self._data.reset_index(drop=True)
                self._data = pl.from_pandas(
                    self._data, schema_overrides=SCHEMA_OVERRIDES
                )
                if self._lazy:
                    self._data = self._data.lazy()
            if self._metadata is not None:
                if not isinstance(self._metadata, (pl.DataFrame, pl.LazyFrame)):
                    if isinstance(self._metadata, pd.DataFrame):
                        self._metadata = self._metadata.reset_index(drop=False)
                        self._metadata = pl.from_pandas(self._metadata)
                        if self._lazy:
                            self._metadata = self._metadata.lazy()
            self._backend = "polars"
        if self._lazy:
            self._cache_data()

    def to_anndata(self) -> AnnData:
        """Convert DandelionPolars.metadata to AnnData.

        Returns
        -------
        AnnData
            An AnnData object with `.obs` populated from `.metadata`.

        Raises
        ------
        ValueError
            If `.metadata` is None.
        """
        if self._metadata is not None:
            if isinstance(self._metadata, pl.LazyFrame):
                meta_df = self._metadata.collect(engine="streaming").to_pandas()
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
                self._data = self._data.collect(engine="streaming")
            if self._metadata is not None:
                if isinstance(self._metadata, pl.LazyFrame):
                    self._metadata = self._metadata.collect(engine="streaming")
            # distances: eager types are np.ndarray or csr_matrix
        if self.distances is not None and not isinstance(
            self.distances, (np.ndarray, csr_matrix)
        ):
            # assume anything else is lazy and computable
            computed = self.distances.compute()
            if isinstance(computed, csr_matrix):
                self.distances = computed
            else:
                self.distances = csr_matrix(computed)
            self.distances._index_names = self.metadata_names
        self._lazy = False

    def to_lazy(self, *, chunks="auto") -> None:
        """Convert eager slots to lazy slots.

        Parameters
        ----------
        chunks : str or int or tuple, optional
            Chunk sizes for converting distance arrays to dask arrays.
            Passed to `dask.array.from_array`. Defaults to ``"auto"``.
        """
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
        self._lazy = True
        self._cache_data()

    def compute(self):
        """Convert self.distances to a concrete csr matrix."""
        if not isinstance(self.distances, csr_matrix):
            try:
                self.distances = csr_matrix(self.distances.compute())
            except Exception:
                self.distances = csr_matrix(self.distances)
            self.distances._index_names = self.metadata_names

    def copy(self) -> DandelionPolars:
        """
        Performs a deep copy of all slots in Dandelion class.

        Returns
        -------
        DandelionPolars
            a deep copy of DandelionPolars class.
        """
        return copy.deepcopy(self)

    def clone(self) -> DandelionPolars:
        """Polars-style clone: duplicate frames and state without sharing cache handles.

        Returns
        -------
        DandelionPolars
            A new DandelionPolars instance with cloned frames and a fresh cache handle map.
        """

        def _clone_frame(obj):
            if isinstance(obj, pl.LazyFrame) or isinstance(obj, pl.DataFrame):
                return obj.clone()
            if isinstance(obj, pd.DataFrame):
                return obj.copy(deep=True)
            return None

        new = DandelionPolars.__new__(DandelionPolars)

        # Copy non-frame attributes; deep copy for structures that can alias
        for k, v in self.__dict__.items():
            if k in {"_data", "_metadata", "_cache_handles"}:
                continue
            if k in {"layout", "graph", "germline"}:
                setattr(new, k, copy.deepcopy(v))
            elif k == "distances":
                try:
                    setattr(new, k, v.copy())
                except Exception:
                    setattr(new, k, copy.deepcopy(v))
            else:
                # shallow for primitives, deep for common containers
                if isinstance(v, (dict, list, set, tuple)):
                    setattr(new, k, copy.deepcopy(v))
                else:
                    setattr(new, k, v)

        # Fresh cache handle map
        new._cache_handles = {}

        # Clone frames
        new._data = _clone_frame(self._data)
        new._metadata = _clone_frame(self._metadata)

        # Rebuild parquet backing if operating lazily
        if getattr(new, "lazy", False):
            new._cache_data()

        return new

    def __getstate__(self):
        """Provide a deepcopy/pickle-friendly state without open cache handles."""
        state = self.__dict__.copy()
        # Ensure cache_handles map exists but is not shared
        state["_cache_handles"] = {}

        # Materialize lazy frames to break references to temp parquet handles
        if isinstance(state.get("_data"), pl.LazyFrame):
            state["_data"] = state["_data"].collect(engine="streaming")
        if isinstance(state.get("_metadata"), pl.LazyFrame):
            state["_metadata"] = state["_metadata"].collect(engine="streaming")
        return state

    def __setstate__(self, state):
        """Restore state and rebuild cache backing when needed."""
        # Restore dict first
        self.__dict__.update(state)

        # Recreate cache handle container
        if not hasattr(self, "_cache_handles") or self._cache_handles is None:
            self._cache_handles = {}

        # Re-lazify if the object was lazy and frames are eager
        if self._lazy:
            if isinstance(self._data, pl.DataFrame):
                self._data = self._data.lazy()
            if isinstance(self._metadata, pl.DataFrame):
                self._metadata = self._metadata.lazy()
            # Rebuild parquet backing for lazy mode
            self._cache_data()

    def update_data(self, skip: list[str] = []) -> None:
        """Sync metadata columns into data via dictionary mapping.

        Parameters
        ----------
        skip : list[str], optional
            List of column names to skip when syncing metadata to data. Defaults to an empty list.
        """
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
            self._data = self._data.collect(engine="streaming").lazy()
            # Ensure data is backed after removing backing with .lazy()
            self._cache_data()

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
                personalized_ref_dict = readGermlines([str(corrected)])
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
    ) -> None:
        """
        A Dandelion initialisation function to update and populate the `.metadata` slot.

        Parameters
        ----------
        retrieve : list[str] | str | None, optional
            column name(s) in `.data` to retrieve and update the metadata.
        clone_key : str | None, optional
            column name of clone id. None defaults to 'clone_id'.
        split : bool, optional
            whether to split the retrieved values into separate VDJ and VJ
            columns. Defaults to True.
        join : bool, optional
            whether to join multiple values per cell with ``|``. Defaults to True.
        unique : bool, optional
            whether to keep only unique values when joining. Defaults to False.
        first : bool, optional
            whether to return only the first value per cell rather than
            joining all values. Defaults to False.
        average : bool, optional
            whether to average numeric columns instead of summing them.
            Defaults to False.
        key_added : list[str] | str | None, optional
            custom output column name(s) for the retrieved values. If None,
            the original column name(s) from `retrieve` are used.
        strip_alleles : bool, optional
            returns the V(D)J genes without allelic calls if True. Defaults to True.
        reinitialize : bool, optional
            whether or not to reinitialize the current metadata. Useful when
            updating older versions of `dandelion` to newer version.
        init_cols : list[str] | None, optional
            columns to initialize the metadata with. If None, uses the
            default set of columns.
        productive_only : bool, optional
            whether or not to use only productive contigs to initialize
            metadata. Defaults to True.
        check_rearrangement_status : bool, optional
            whether or not to check and update the rearrangement status.
            Defaults to True.
        genotyped_v_call : bool, optional
            whether or not to use genotyped v_call data to initialize
            metadata if available. Defaults to True.
        update_isotype_dict : dict[str, str] | None, optional
            custom isotype dictionary to update the default isotype dictionary.
        lazy : bool, optional
            whether to keep the metadata as a Polars LazyFrame after updating.
            Defaults to True.
        as_pandas : bool, optional
            whether to convert the Dandelion object back to the pandas backend
            after updating. Defaults to False.

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
            self.to_polars(lazy=self._lazy)
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
        self._lazy = lazy
        metadata_status = self._metadata

        if (metadata_status is None) or reinitialize:
            self._initialize_metadata(
                clone_key=clone_key,
                v_call_key=v_call_key,
                init_cols=init_cols,
                update_isotype_dict=update_isotype_dict,
                strip_alleles=strip_alleles,
                productive_only=productive_only,
                check_rearrangement_status=check_rearrangement_status,
            )
        if retrieve is not None:
            if self._metadata is None:
                raise ValueError(
                    "Dandelion.metadata is None. Please initialize metadata first before updating."
                )
            # first check that self._data and self._metadata are converted to Polars
            if not isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
                self.to_polars(lazy=self._lazy)
            if type(retrieve) is str:
                retrieve = [retrieve]
            _data = (
                self._data.filter(
                    pl.col("productive")
                    .cast(pl.String)
                    .str.to_uppercase()
                    .is_in(TRUES_STR + EMPTIES_STR)
                )
                if productive_only
                else self._data
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
                # Get columns that would be duplicated (present in both dataframes, excluding join key)
                meta_cols = set(meta.collect_schema().names())
                metadata_cols = set(self._metadata.collect_schema().names())
                duplicate_cols = meta_cols.intersection(metadata_cols) - {
                    "cell_id"
                }

                # Drop duplicate columns from metadata before joining
                if duplicate_cols:
                    self._metadata = self._metadata.lazy().drop(*duplicate_cols)

                self._metadata = self._metadata.lazy().join(
                    meta.lazy(), on="cell_id", how="left"
                )
                # remove empty strings
                self._metadata = self._metadata.with_columns(
                    [
                        pl.col(col).replace("", None)
                        for col in self._metadata.collect_schema()
                        if self._metadata.collect_schema()[col] == pl.String
                    ]
                )
                self._metadata = (
                    self._metadata.collect(engine="streaming").lazy()
                    if self._lazy
                    else self._metadata.collect(engine="streaming")
                )
                if "metadata" in self._cache_handles.keys():
                    self._cache_handles["metadata"].close()
                    del self._cache_handles["metadata"]
            # clean up self._data
            self._cache_data()
        if as_pandas:
            self.to_pandas()

    def update_plus(
        self,
        option: Literal[
            "all",
            "sequence",
            "mutations",
            "cdr3 lengths",
            "mutations and cdr3 lengths",
        ] = "mutations and cdr3 lengths",
    ) -> None:
        """Retrieve additional data columns that are useful.

        Parameters
        ----------
        option : Literal["all", "sequence", "mutations", "cdr3 lengths", "mutations and cdr3 lengths", ], optional
            One of 'all', 'sequence', 'mutations', 'cdr3 lengths', 'mutations and cdr3 lengths'.
        """
        if self._backend == "pandas":
            self.to_polars()
        mutations = [x for x in MUTATIONS if x in self._data.collect_schema()]
        vdjlengths = [x for x in VDJLENGTHS if x in self._data.collect_schema()]
        seqinfo = [x for x in SEQINFO if x in self._data.collect_schema()]
        if option == "all":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    split=True,
                    average=False,
                )
                self.update_metadata(
                    retrieve=mutations,
                    average=False,
                    split=False,
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    split=True,
                    average=True,
                )
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo,
                    split=True,
                    join=True,
                )
        if option == "sequence":
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo,
                    split=True,
                    join=True,
                )
        if option == "mutations":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    split=True,
                    average=False,
                )
                self.update_metadata(
                    retrieve=mutations,
                    split=False,
                    average=False,
                )
        if option == "cdr3 lengths":
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    split=True,
                    average=True,
                )
        if option == "mutations and cdr3 lengths":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations,
                    split=True,
                    average=False,
                )
                self.update_metadata(
                    retrieve=mutations,
                    split=False,
                    average=False,
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths,
                    split=True,
                    average=True,
                )

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
        write_airr(self._data, filename, **kwargs)

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
        Write a Dandelion object to .h5ddl format.

        Mirrors the base Dandelion.write_h5ddl interface. If distances is a
        dask array it cannot be stored inline in HDF5; it will instead be
        written to a Zarr store placed alongside the .h5ddl file (same stem,
        .zarr extension) and a warning is raised. The companion .zarr is
        detected automatically by read_h5ddl.

        Parameters
        ----------
        filename : str, optional
            path to `.h5ddl` file.
        compression : Literal["gzip", "lzf", "szip"], optional
            Specifies the compression algorithm to use.
        compression_level : int | None, optional
            Specifies a compression level for data. A value of 0 disables
            compression.
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

        # -- data ---------------------------------------------------------
        data = self._data
        if isinstance(data, pl.LazyFrame):
            data = data.collect(engine="streaming")
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise TypeError(
                f"Unsupported data type for write_h5ddl: {type(data)}"
            )
        data = sanitize_data(data)
        data, data_dtypes = sanitize_data_for_saving(data)
        structured_data_array = np.array(
            [tuple(row) for row in data.to_numpy()], dtype=data_dtypes
        )
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(
                "data",
                data=structured_data_array,
                **save_args,
            )

        # -- metadata -----------------------------------------------------
        if getattr(self, "_metadata", None) is not None:
            metadata = self._metadata
            if isinstance(metadata, pl.LazyFrame):
                metadata = metadata.collect(engine="streaming")
            if isinstance(metadata, pl.DataFrame):
                metadata_pd = metadata.to_pandas()
            elif isinstance(metadata, pd.DataFrame):
                metadata_pd = metadata.copy()
            else:
                raise TypeError(
                    "Unsupported metadata type for write_h5ddl: "
                    f"{type(metadata)}"
                )
            # polars stores cell_id as a column; h5ddl expects it as the index
            if "cell_id" in metadata_pd.columns:
                metadata_pd = metadata_pd.set_index("cell_id")
            metadata_pd, metadata_dtypes = sanitize_data_for_saving(metadata_pd)
            structured_metadata_array = np.array(
                [tuple(row) for row in metadata_pd.to_numpy()],
                dtype=metadata_dtypes,
            )
            structured_metadata_names_array = np.array(
                [s.encode("utf-8") for s in metadata_pd.index.to_numpy()]
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

        # -- graph --------------------------------------------------------
        if getattr(self, "graph", None) is not None:
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
                        f"graph/graph_{i}/data",
                        data=G_x.data,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{i}/indices",
                        data=G_x.indices,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{i}/indptr",
                        data=G_x.indptr,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{i}/shape",
                        data=G_x.shape,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{i}/column",
                        data=G_column_array,
                        **save_args,
                    )
                    hf.create_dataset(
                        f"graph/graph_{i}/index",
                        data=G_index_array,
                        **save_args,
                    )

        # -- distances ----------------------------------------------------
        if getattr(self, "distances", None) is not None:
            if isinstance(self.distances, csr_matrix):
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
            else:
                try:
                    import dask.array as da

                    if isinstance(self.distances, da.Array):
                        zarr_path = Path(filename).with_suffix(".zarr")
                        da.to_zarr(
                            self.distances,
                            str(zarr_path / "distance_matrix"),
                            overwrite=True,
                        )
                        logg.warning(
                            f"Distances are a dask array and cannot be stored "
                            f"inline in .h5ddl. Written to {zarr_path}. Pass "
                            f"`distance_zarr='{zarr_path}'` when reading, or "
                            f"it will be detected automatically."
                        )
                except ImportError:
                    pass

        # -- layout -------------------------------------------------------
        if getattr(self, "layout", None) is not None:
            for i, l in enumerate(self.layout):
                with h5py.File(filename, "a") as hf:
                    layout_group = hf.create_group(f"layout/layout_{i}")
                    for key, value in l.items():
                        layout_group.create_dataset(
                            key,
                            data=value,
                            **save_args,
                        )

        # -- germline -----------------------------------------------------
        if (
            getattr(self, "germline", None) is not None
            and len(self.germline) > 0
        ):
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

        Parameters
        ----------
        filename : str, optional
            path to output `.zipddl` file.
        compress : bool, optional
            whether to compress stored data using Blosc/Zstd with bitshuffle.
            Defaults to True.
        """
        # Create Zarr ZipStore container
        store = ZipStore(filename, mode="w")
        root = open_zarr_group(store, mode="w")
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
            # Collect and clean temp file before writing (sanitize may return lazy)
            if isinstance(self._data, pl.LazyFrame):
                self._data = self._data.collect(engine="streaming")
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
            if isinstance(self._metadata, pl.LazyFrame):
                self._metadata = self._metadata.collect(engine="streaming")
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
            # If pending embed from a temporary zarr, stream-copy into ZipStore
            try:
                pending = getattr(self, "_distance_embed_pending", False)
                zarr_src = getattr(self, "_distance_zarr_path", None)
            except Exception:
                pending, zarr_src = False, None
            if pending and zarr_src is not None:
                # Open source zarr and target dataset inside ZipStore
                try:
                    src_root = open_zarr_group(
                        LocalStore(str(zarr_src) + "/distance_matrix.zarr"),
                        mode="r",
                    )
                    src_arr = src_root["distance_matrix"]
                except Exception:
                    # Fallback to direct array path
                    src_arr = zarr.open_array(
                        LocalStore(str(zarr_src)), mode="r"
                    )

                # Create destination dataset and copy
                dst = create_zarr_dataset(
                    arrays_grp,
                    "distances",
                    shape=src_arr.shape,
                    dtype=src_arr.dtype,
                    chunks=src_arr.chunks,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                dst[:] = src_arr[:]

                # Clear pending flag and clean up temporary zarr
                try:
                    setattr(self, "_distance_embed_pending", False)
                except Exception:
                    pass
                try:
                    if os.path.isdir(str(zarr_src)):
                        shutil.rmtree(str(zarr_src), ignore_errors=True)
                except Exception:
                    pass
            elif isinstance(arr, csr_matrix):
                # Save CSR matrix as separate arrays
                create_zarr_dataset(
                    arrays_grp,
                    name="distances_data",
                    shape=arr.data.shape,
                    dtype=arr.data.dtype,
                    chunks=arr.data.shape,
                    data=arr.data,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                create_zarr_dataset(
                    arrays_grp,
                    "distances_indices",
                    shape=arr.indices.shape,
                    dtype=arr.indices.dtype,
                    chunks=arr.indices.shape,
                    data=arr.indices,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                create_zarr_dataset(
                    arrays_grp,
                    "distances_indptr",
                    shape=arr.indptr.shape,
                    dtype=arr.indptr.dtype,
                    chunks=arr.indptr.shape,
                    data=arr.indptr,
                    overwrite=True,
                    compressors=[comp] if compress else None,
                )
                create_zarr_dataset(
                    arrays_grp,
                    "distances_shape",
                    shape=(len(arr.shape),),
                    dtype=np.int64,
                    data=np.array(arr.shape, dtype=np.int64),
                    overwrite=True,
                )
            else:
                # Only embed eager arrays; skip lazy external dask unless pending embed
                try:
                    import dask.array as da

                    is_dask = isinstance(arr, da.Array)
                except Exception:
                    is_dask = False
                if is_dask:
                    # Skip writing for external zarr mode
                    pass
                else:
                    create_zarr_dataset(
                        arrays_grp,
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
                    # Add a tiny weight to zero edges to avoid issues with sparse representation
                    tiny_weight = 1e-12
                    if not isinstance(g, ig.Graph):
                        gi = ig.Graph.from_networkx(g)
                        if (
                            "name" not in gi.vs.attributes()
                            and "_nx_name" in gi.vs.attributes()
                        ):
                            gi.vs["name"] = gi.vs["_nx_name"]
                    else:
                        gi = g

                    names = np.array(gi.vs["name"], dtype="S")
                    n = gi.vcount()
                    edges = gi.get_edgelist()

                    if not edges:
                        G_x = csr_matrix((n, n), dtype=np.float64)
                    else:
                        edges_arr = np.array(edges, dtype=np.int64)
                        u_idx = edges_arr[:, 0]
                        v_idx = edges_arr[:, 1]
                        weights = (
                            gi.es["weight"]
                            if "weight" in gi.es.attributes()
                            else [1.0] * len(edges)
                        )
                        # u_idx = np.array([u for u, v in edges], dtype=np.int64)
                        # v_idx = np.array([v for u, v in edges], dtype=np.int64)
                        w = np.array(weights, dtype=np.float64)
                        w = np.where(w == 0, tiny_weight, w)

                        rows = np.concatenate([u_idx, v_idx])
                        cols = np.concatenate([v_idx, u_idx])
                        vals = np.concatenate([w, w])

                        G_x = csr_matrix(
                            (vals, (rows, cols)), shape=(n, n), dtype=np.float64
                        )
                    with h5py.File(tmp_h5.name, "w") as hf:
                        hf.create_dataset("data", data=G_x.data)
                        hf.create_dataset("indices", data=G_x.indices)
                        hf.create_dataset("indptr", data=G_x.indptr)
                        hf.create_dataset("shape", data=G_x.shape)
                        hf.create_dataset("columns", data=names)
                        hf.create_dataset("index", data=names)
                    tmp_h5.seek(0)
                    arr = np.frombuffer(tmp_h5.read(), dtype=np.uint8)
                    create_zarr_dataset(
                        graph_grp,
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
            for i, layout in enumerate(self.layout):
                with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_h5:
                    with h5py.File(tmp_h5.name, "w") as hf:
                        for k, v in layout.items():
                            hf.create_dataset(k, data=v)
                    tmp_h5.seek(0)
                    arr = np.frombuffer(tmp_h5.read(), dtype=np.uint8)
                    create_zarr_dataset(
                        layout_grp,
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
                create_zarr_dataset(
                    germline_grp,
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

    write = write_ddl = write_zipddl  # aliases

    def write_vdj(
        self,
        folder: Path | str = "dandelion_data",
        filename_prefix: str = "all",
        sequence_key: str = "sequence",
        clone_key: str = "clone_id",
    ) -> None:
        """
        Writes a DandelionPolars object to contig-annotation formatted files compatible with
        multiple platforms (10x Genomics, SeekGene, etc.) so that it can be ingested by
        other tools.

        Produces:

        - ``{filename_prefix}_contig.fasta`` : sequences in FASTA format.
        - ``{filename_prefix}_contig_annotations.csv`` : contig annotation table with
          columns matching the 10x / SeekGene contig annotation schema.

        Parameters
        ----------
        folder : Path | str, optional
            path to save the output files.
        filename_prefix : str, optional
            prefix for the output files.
        sequence_key : str, optional
            column name in `.data` slot to retrieve and write out in fasta format.
        clone_key : str, optional
            column name in `.data` slot for clone id information.
        """
        folder = Path(folder) if isinstance(folder, str) else folder
        folder.mkdir(parents=True, exist_ok=True)
        out_fasta = folder / f"{filename_prefix}_contig.fasta"
        out_anno_path = folder / f"{filename_prefix}_contig_annotations.csv"

        # convert to polars
        if self._backend == "pandas":
            _backend = "pandas"
            self.to_polars()
        else:
            _backend = "polars"
        # Handle both Polars LazyFrame and DataFrame, as well as pandas DataFrame
        if isinstance(self._data, pl.LazyFrame):
            data_df = self._data.collect()
        elif isinstance(self._data, pl.DataFrame):
            data_df = self._data
        else:
            # pandas DataFrame - convert to Polars
            data_df = pl.from_pandas(self._data)
        seqs = dict(
            zip(
                data_df["sequence_id"].to_list(),
                data_df[sequence_key].to_list(),
            )
        )
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
        if "complete_vdj" not in self._data.collect_schema():
            column_map.pop("full_length")
        # Support both _10x-suffixed (10x CellRanger) and plain (SeekGene) column names.
        is_cell_col = next(
            (
                c
                for c in ["is_cell_10x", "is_cell"]
                if c in self._data.collect_schema()
            ),
            None,
        )
        if is_cell_col:
            column_map["is_cell"] = is_cell_col
        else:
            column_map.pop("is_cell")
        high_confidence_col = next(
            (
                c
                for c in ["high_confidence_10x", "high_confidence"]
                if c in self._data.collect_schema()
            ),
            None,
        )
        if high_confidence_col:
            column_map["high_confidence"] = high_confidence_col
        else:
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
        for r in data_df.iter_rows(named=True):
            info = []
            for v in column_map.values():
                if v in r:
                    info.append(r[v])
                elif v in ["is_cell", "high_confidence"]:
                    info.append("True")
                elif v == "length":
                    info.append(len(r[sequence_key]))
            anno.append({k: r for k, r in zip(column_map.keys(), info)})
        anno = pd.DataFrame(anno)
        anno = anno.map(lambda x: bool_map[x] if x in bool_map.keys() else x)
        anno.to_csv(out_anno_path, index=False)

        if (
            _backend == "pandas"
        ):  # convert back to pandas if the original backend was pandas
            self.to_pandas()

    def write_10x(
        self,
        folder: Path | str = "dandelion_data",
        filename_prefix: str = "all",
        sequence_key: str = "sequence",
        clone_key: str = "clone_id",
    ) -> None:
        """
        Alias for :meth:`write_vdj` kept for backwards compatibility.

        Parameters
        ----------
        folder : Path | str, optional
            path to save the output files.
        filename_prefix : str, optional
            prefix for the output files.
        sequence_key : str, optional
            column name in `.data` slot to retrieve and write out in fasta format.
        clone_key : str, optional
            column name in `.data` slot for clone id information.
        """
        self.write_vdj(
            folder=folder,
            filename_prefix=filename_prefix,
            sequence_key=sequence_key,
            clone_key=clone_key,
        )


def load_polars(
    obj: pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None,
    lazy: bool = True,
    as_pandas: bool = False,
) -> pl.LazyFrame:
    """
    Read in or copy dataframe object and set sequence_id as index without dropping.

    Parameters
    ----------
    obj : pl.LazyFrame | pl.DataFrame | pd.DataFrame | Path | str | None
        airr rearrangement file path or pandas/polars DataFrame.
    lazy : bool, optional
        whether to read in as polars LazyFrame. Default is True.
    as_pandas : bool, optional
        whether to return as pandas DataFrame. Default is False.

    Returns
    -------
    pl.LazyFrame
        airr rearrangement data frame.

    Raises
    ------
    TypeError
        if `obj` is not a valid file path, pandas DataFrame, polars DataFrame or LazyFrame.
    KeyError
        if `sequence_id` not found in input.
    """
    if obj is not None:
        if os.path.isfile(str(obj)):
            df = pl.scan_csv(
                obj, separator="\t", schema_overrides=SCHEMA_OVERRIDES
            )
        elif isinstance(obj, pl.LazyFrame):  # Check for LazyFrame
            df = obj.with_columns(
                [
                    pl.col(c).cast(t)
                    for c, t in SCHEMA_OVERRIDES.items()
                    if c in obj.collect_schema()
                ]
            )
        elif isinstance(obj, pl.DataFrame):  # Check for eager DataFrame
            df = obj.with_columns(
                [
                    pl.col(c).cast(t)
                    for c, t in SCHEMA_OVERRIDES.items()
                    if c in obj.columns
                ]
            ).lazy()
        elif isinstance(obj, pd.DataFrame):
            try:
                df = pl.from_pandas(
                    obj, schema_overrides=SCHEMA_OVERRIDES
                ).lazy()
            except Exception:  # because of mixed dtypes, sanitize first
                df = _sanitize_data_polars(obj)
        else:
            raise TypeError(
                "Input obj must be a file path, pandas DataFrame, polars DataFrame, or polars LazyFrame."
            )

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
                df = (
                    df.rename({"duplicate_count": "umi_count"})
                    .collect(engine="streaming")
                    .lazy()
                )

        if as_pandas:
            df = df.collect(engine="streaming").to_pandas()
            df.set_index("sequence_id", inplace=True, drop=False)
        # Collect to execute operations, then convert back to lazy or pandas
        return df if lazy else df.collect(engine="streaming")

    return None  # Handle obj is None case


class LazyColumnExpr:
    """Lazy column expression that preserves source frame context."""

    def __init__(self, source_df: pl.LazyFrame, column_name: str):
        self._source_df = source_df
        self._column_name = column_name

    @property
    def expr(self) -> pl.Expr:
        return pl.col(self._column_name)

    def is_in(self, other):
        """Handle cross-frame membership checks by using concrete RHS values."""
        if isinstance(other, LazyColumnExpr):
            if other._source_df is self._source_df:
                return self.expr.is_in(pl.col(other._column_name))
            rhs = (
                other._source_df.select(other._column_name)
                .unique()
                .collect(engine="streaming")
                .to_series()
            )
            return self.expr.is_in(rhs)
        return self.expr.is_in(other)

    def __getattr__(self, name):
        attr = getattr(self.expr, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                converted_args = [
                    a.expr if isinstance(a, LazyColumnExpr) else a
                    for a in args
                ]
                converted_kwargs = {
                    k: (v.expr if isinstance(v, LazyColumnExpr) else v)
                    for k, v in kwargs.items()
                }
                return attr(*converted_args, **converted_kwargs)

            return _wrapped
        return attr

    def __repr__(self):
        return repr(self.expr)

    # Support comparison operators
    def __eq__(self, other):
        return self.expr.__eq__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )

    def __ne__(self, other):
        return self.expr.__ne__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )

    def __lt__(self, other):
        return self.expr.__lt__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )

    def __le__(self, other):
        return self.expr.__le__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )

    def __gt__(self, other):
        return self.expr.__gt__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )

    def __ge__(self, other):
        return self.expr.__ge__(
            other.expr if isinstance(other, LazyColumnExpr) else other
        )


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
                # Return a lazy expression for the column to avoid materialization
                return LazyColumnExpr(df, name)
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
                # Return an expression for lazy frames to keep pipeline lazy
                return LazyColumnExpr(df, key)
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
        elif isinstance(key, (pl.Series, pl.Expr, LazyColumnExpr)):
            if isinstance(key, LazyColumnExpr):
                key = key.expr
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
                length = df.select(pl.len()).collect(engine="streaming").item()
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
            return df.select(pl.len()).collect(engine="streaming").item()
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
        df = df.collect(engine="streaming")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df.write_parquet(tmp.name)
        tmp.seek(0)
        arr = np.frombuffer(tmp.read(), dtype=np.uint8)
    create_zarr_dataset(
        zarr_group,
        name,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=arr.shape,
        data=arr,
        overwrite=True,
        compressors=compressors,
    )


def write_airr(
    data: pd.DataFrame | pl.DataFrame | pl.LazyFrame, save: Path | str, **kwargs
) -> None:
    """Save as airr formatted file."""
    data = _sanitize_data_polars(data)
    if isinstance(data, pl.LazyFrame):
        data = data.collect(engine="streaming")
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

    # Check for IGK/IGL exception
    vj_locus_all_igk_igl = (
        vj_locus_has
        & vj_locus_list.list.eval(pl.element().is_in(["IGK", "IGL"])).list.all()
    )

    tmp2 = (
        pl.when(~vj_locus_has)
        .then(pl.lit("None"))
        .when(vj_locus_len > 1)
        .then(
            pl.when(vj_locus_all_igk_igl & (vj_locus_unique_len <= 2))
            .then(pl.lit("Extra VJ"))
            .otherwise(pl.lit("Ambiguous"))
        )
        .otherwise(vj_locus_first)
    )

    # Build the final classification
    both_none = ~loc1_has & ~vj_locus_has

    # Check if tmp1 and tmp2 are valid locus names (not None, empty string, or special labels)
    tmp1_is_locus = (
        loc1_has
        & ~tmp1.is_in(["None", "Extra VDJ", "Extra VDJ-exception", "Ambiguous"])
        & (tmp1 != "")
        & tmp1.is_not_null()
    )

    tmp2_is_locus = (
        vj_locus_has
        & ~tmp2.is_in(["None", "Extra VJ", "Extra VJ-exception", "Ambiguous"])
        & (tmp2 != "")
        & tmp2.is_not_null()
    )

    # Check for None or empty values
    tmp1_is_none = (tmp1 == "None") | (tmp1 == "") | tmp1.is_null()
    tmp2_is_none = (tmp2 == "None") | (tmp2 == "") | tmp2.is_null()

    # Final result - fixed orphan logic
    result = (
        pl.when(both_none)
        .then(pl.lit("None + None"))
        .when((tmp1 == "Ambiguous") | (tmp2 == "Ambiguous"))
        .then(pl.lit("Ambiguous"))
        .when(
            tmp1_is_none & tmp2_is_locus
        )  # Only orphan if tmp2 is a real locus
        .then(pl.concat_str([pl.lit("Orphan "), tmp2]))
        .when(
            tmp2_is_none & tmp1_is_locus
        )  # Only orphan if tmp1 is a real locus
        .then(pl.concat_str([pl.lit("Orphan "), tmp1]))
        .when(tmp1_is_locus & tmp2_is_locus)
        .then(pl.concat_str([tmp1, pl.lit(" + "), tmp2]))
        .otherwise(pl.concat_str([tmp1, pl.lit(" + "), tmp2]))
    )

    return result.alias("locus_classification")


def _classify_locus_pair_noiso() -> pl.Expr:
    """
    Classify locus pairs based on VDJ and VJ locus columns.
    """
    vdj_locus_col = "locus_VDJ"
    vj_locus_col = "locus_VJ"

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
            pl.when(loc1_all_trb_trd & (loc1_unique_len <= 2))
            .then(pl.lit("Extra VDJ-exception"))
            .otherwise(pl.lit("Extra VDJ"))
        )
        .otherwise(loc1_first)  # Single locus returns the locus name
    )

    # Process locus 2
    vj_locus_len = vj_locus_list.list.len().fill_null(0)
    vj_locus_first = vj_locus_list.list.first()
    vj_locus_unique_len = vj_locus_list.list.n_unique().fill_null(0)

    # Check for IGK/IGL exception
    vj_locus_all_igk_igl = (
        vj_locus_has
        & vj_locus_list.list.eval(pl.element().is_in(["IGK", "IGL"])).list.all()
    )

    tmp2 = (
        pl.when(~vj_locus_has)
        .then(pl.lit("None"))
        .when(vj_locus_len > 1)
        .then(
            pl.when(vj_locus_all_igk_igl & (vj_locus_unique_len <= 2))
            .then(pl.lit("Extra VJ"))
            .otherwise(pl.lit("Ambiguous"))
        )
        .otherwise(vj_locus_first)
    )

    # Build the final classification
    both_none = ~loc1_has & ~vj_locus_has

    # Check if tmp1 and tmp2 are valid locus names (not None, empty string, or special labels)
    tmp1_is_locus = (
        loc1_has
        & ~tmp1.is_in(["None", "Extra VDJ", "Extra VDJ-exception", "Ambiguous"])
        & (tmp1 != "")
        & tmp1.is_not_null()
    )

    tmp2_is_locus = (
        vj_locus_has
        & ~tmp2.is_in(["None", "Extra VJ", "Extra VJ-exception", "Ambiguous"])
        & (tmp2 != "")
        & tmp2.is_not_null()
    )

    # Check for None or empty values
    tmp1_is_none = (tmp1 == "None") | (tmp1 == "") | tmp1.is_null()
    tmp2_is_none = (tmp2 == "None") | (tmp2 == "") | tmp2.is_null()

    # Final result - fixed orphan logic
    result = (
        pl.when(both_none)
        .then(pl.lit("None + None"))
        .when((tmp1 == "Ambiguous") | (tmp2 == "Ambiguous"))
        .then(pl.lit("Ambiguous"))
        .when(
            tmp1_is_none & tmp2_is_locus
        )  # Only orphan if tmp2 is a real locus
        .then(pl.concat_str([pl.lit("Orphan "), tmp2]))
        .when(
            tmp2_is_none & tmp1_is_locus
        )  # Only orphan if tmp1 is a real locus
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
        data = pl.from_pandas(
            data.reset_index(drop=True), schema_overrides=SCHEMA_OVERRIDES
        ).lazy()
    elif isinstance(data, pl.DataFrame):
        data = data.lazy()
    # convert empty strings to nulls for consistency
    data = data.with_columns(
        [
            pl.col(col).replace("", None)
            for col in data.collect_schema()
            if data.collect_schema()[col] == pl.String
        ]
    )
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
    return data if lazy else data.collect(engine="streaming")


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

    # Deduplicate semantically similar clones
    # e.g., "B_VDJ_X_VJ_Y" and "B_VJ_Y_VJ_Y" should keep only the VDJ version
    # Strategy: Keep the one that has VDJ in it, as it's more informative
    deduplicated = {}
    for part in parts:
        # Create a normalized key based on the clone structure
        # Remove the prefix to compare the actual clone assignment
        if part.startswith("B_VDJ_"):
            # Extract everything after B_VDJ_
            key = part[6:]  # len("B_VDJ_") = 6
            category = "VDJ"
        elif part.startswith("B_VJ_"):
            # Extract everything after B_VJ_
            key = part[5:]  # len("B_VJ_") = 5
            category = "VJ"
        elif part.startswith("abT_VDJ_"):
            key = part[8:]
            category = "VDJ"
        elif part.startswith("abT_VJ_"):
            key = part[7:]
            category = "VJ"
        elif part.startswith("gdT_VDJ_"):
            key = part[8:]
            category = "VDJ"
        elif part.startswith("gdT_VJ_"):
            key = part[7:]
            category = "VJ"
        else:
            # Unknown format, keep as-is
            key = part
            category = "OTHER"

        if key in deduplicated:
            # Already have this clone - prefer VDJ over VJ
            existing_part, existing_cat = deduplicated[key]
            if category == "VDJ" and existing_cat == "VJ":
                deduplicated[key] = (part, category)  # Replace VJ with VDJ
            # Otherwise keep existing
        else:
            deduplicated[key] = (part, category)

    parts = [v[0] for v in deduplicated.values()]

    # Sort and deduplicate by string equality as well
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
        clone_counts = _flatten_and_count(
            df.collect(engine="streaming"), clonekey
        )
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

        data = pl.from_pandas(
            data.reset_index(drop=True), schema_overrides=SCHEMA_OVERRIDES
        )

    lazy = isinstance(data, pl.LazyFrame)
    df = data.collect(engine="streaming") if lazy else data
    exprs: list[pl.Expr] = []

    for d in df.collect_schema().names():
        is_string = _is_polars_string_dtype(df, d)
        col = pl.col(d)

        # --- BOOLEAN-LIKE COLUMNS ---
        if (d in BOOLEAN_LIKE_COLUMNS) or (_is_polars_boolean_dtype(df, d)):
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
    # clean empty strings in string columns to null
    df = df.with_columns(
        [
            pl.col(col).replace("", None)
            for col in df.collect_schema()
            if df.collect_schema()[col] == pl.String
        ]
    )

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


def _is_polars_boolean_dtype(
    df: pl.DataFrame | pl.LazyFrame, colname: str
) -> bool:
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    return schema.get(colname) == pl.Boolean


def _validate_airr_polars(data: pl.DataFrame | pl.LazyFrame) -> None:
    """Validate dtypes in airr table (Polars)."""
    # identify integer-like columns
    int_columns = []
    if isinstance(data, pl.LazyFrame):
        data = data.collect(engine="streaming")
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
