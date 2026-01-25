from __future__ import annotations

import h5py
import json
import os
import re
import tempfile
import zarr

import networkx as nx
from packaging import version
import pandas as pd
import polars as pl

from collections import defaultdict, OrderedDict
from pathlib import Path
from scanpy import logging as logg
from scipy.sparse import csr_matrix

from dandelion.base.io._io import read_h5ddl as _read_h5ddl
from dandelion.polars.core._core_polars import DandelionPolars, load_polars
from dandelion.utilities._utilities import (
    DEFAULT_PREFIX,
    CELLRANGER,
    AIRR,
    fasta_iterator,
)

ZARR_V3 = version.parse(zarr.__version__) >= version.parse("3.0.0")
if ZARR_V3:
    from zarr.storage import LocalStore, ZipStore

    def open_zarr_group(store, mode="a"):
        return zarr.open_group(store=store, mode=mode)

else:
    from zarr import DirectoryStore, ZipStore

    def LocalStore(path):
        return DirectoryStore(path)

    def open_zarr_group(store, mode="a"):
        import zarr

        if mode == "w":
            return zarr.group(store=store, overwrite=True)
        else:
            return zarr.group(store=store)


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
    store = ZipStore(filename, mode="r")
    root = open_zarr_group(store, mode="r")

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

    cache_handles = {}
    if "data.parquet" in root["tables"]:
        data_lazy, data_tmp = load_parquet_lazy("data.parquet")
        constructor["data"] = data_lazy
        cache_handles["data"] = data_tmp
    if "metadata.parquet" in root["tables"]:
        metadata_lazy, metadata_tmp = load_parquet_lazy("metadata.parquet")
        constructor["metadata"] = metadata_lazy
        cache_handles["metadata"] = metadata_tmp

    # ---------------------------
    # Distances: Zarr arrays
    # ---------------------------
    embedded_distances_loaded = False
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
            embedded_distances_loaded = True
        elif "distances" in arr_grp:
            # Wrap zarr array in dask for lazy access
            import dask.array as da

            zarr_arr = arr_grp["distances"]
            constructor["distances"] = da.from_zarr(zarr_arr)
            embedded_distances_loaded = True

    if distance_zarr is not None:
        import dask.array as da

        if embedded_distances_loaded:
            logg.warning(
                f"Embedded distances found (in {filename}) and external Zarr "
                f"path (distance_zarr={distance_zarr}) provided. "
                f"Using external Zarr path to override embedded distances."
            )
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
                constructor["germline"] = _collect_datasets(hf)

    # ---------------------------
    # Construct Dandelion
    # ---------------------------
    res = DandelionPolars(
        **constructor, verbose=verbose, cache_handles=cache_handles
    )

    return res


read = read_ddl = read_zipddl  # alias


# helper function to read germline correctly
def _collect_datasets(hf: h5py.File, path=""):
    datasets = {}
    for key in hf.keys():
        item = hf[key]
        full_path = f"{path}/{key}" if path else key
        if isinstance(item, h5py.Group):
            datasets.update(_collect_datasets(item, full_path))
        elif isinstance(item, h5py.Dataset):
            if item.shape == ():
                datasets[full_path] = item[()]
            else:
                datasets[full_path] = item[:]
    return datasets


def read_h5ddl(
    filename: Path | str = "dandelion_data.h5ddl",
    distance_zarr: Path | str | None = None,
    verbose: bool = False,
) -> DandelionPolars:
    """
    Read in and returns a Dandelion class from .h5ddl format.

    Legacy function that wraps base read_h5ddl and converts to DandelionPolars.

    Parameters
    ----------
    filename : Path | str, optional
        path to `.h5ddl` file
    distance_zarr : Path | str | None, optional
        path to Zarr array for distances if computed lazily.
    verbose : bool, optional
        whether or not to print messages during creation of the Dandelion object.

    Returns
    -------
    Dandelion
        Dandelion object.

    Raises
    ------
    AttributeError
        if `data` not found in `.h5ddl` file.
    """
    tmp = _read_h5ddl(filename, distance_zarr, verbose)
    tmp = DandelionPolars(tmp.data)

    return tmp


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


def read_10x_vdj(
    data: Path | str | pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
    filename_prefix: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_malformed: bool = True,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    A parser to read .csv and .json files directly from folder containing 10x cellranger-outputs,
    or parse an existing pandas/polars DataFrame.

    This function parses the 10x output files into an AIRR compatible format using Polars.

    Minimum requirement is one of either {filename_prefix}_contig_annotations.csv or all_contig_annotations.json
    when reading from file path.

    If .fasta, .json files are found in the same folder, additional info will be appended to the final table.

    Parameters
    ----------
    data : Path | str | pandas.DataFrame | polars.DataFrame | polars.LazyFrame | None
        path to folder containing `.csv` and/or `.json` files, path to files directly, or a pandas/polars
        DataFrame containing the contig annotations data.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. None defaults to 'all'. Only used when data is a file/folder.
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
    TypeError
        if data is not a valid type (Path, str, DataFrame, or LazyFrame).

    """

    def parse_annotation_polars(data: pl.DataFrame) -> pl.DataFrame:
        """Parse annotation file using Polars - fully vectorized."""
        swap_dict = dict(zip(CELLRANGER, AIRR))

        # Rename columns based on swap_dict
        rename_map = {
            k: v for k, v in swap_dict.items() if k in data.collect_schema()
        }
        data = data.rename(rename_map)

        # Fill null values with empty strings for string columns
        # Also replace string representations of None/NaN
        for col in data.collect_schema():
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
            if col not in data.collect_schema():
                data = data.with_columns(pl.lit("").alias(col))

        if "locus" not in data.collect_schema():
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
                    return_dtype=pl.String,
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

    # Handle DataFrame inputs (pandas or polars)
    if isinstance(data, pd.DataFrame):
        logg.info("Converting pandas DataFrame to polars DataFrame")
        res = pl.from_pandas(data)
        res = parse_annotation_polars(res)
    elif isinstance(data, pl.LazyFrame):
        logg.info("Converting polars LazyFrame to polars DataFrame")
        res = data.collect(engine="streaming")
        res = parse_annotation_polars(res)
    elif isinstance(data, pl.DataFrame):
        logg.info("Parsing polars DataFrame")
        res = parse_annotation_polars(data)
    elif isinstance(data, (str, Path)):
        # Handle file path inputs
        filename_pre = (
            DEFAULT_PREFIX if filename_prefix is None else filename_prefix
        )

        if os.path.isdir(str(data)):
            files = os.listdir(data)
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
            json_idx = [
                i for i, j in enumerate(filelist) if j.endswith(".json")
            ]
            if len(csv_idx) == 1:
                file = str(data) + "/" + str(filelist[csv_idx[0]])
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
                        out_json_df,
                        on="sequence_id",
                        how="outer",
                        suffix="_json",
                    )

                    # Coalesce columns that appear in both (prefer json version)
                    for col in out_json_df.columns:
                        if (
                            col != "sequence_id"
                            and f"{col}_json" in res.columns
                        ):
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
                            lambda x: seqs.get(x, ""), return_dtype=pl.String
                        )
                        .alias("sequence")
                    )
                    res = parse_annotation_polars(raw)
                else:
                    res = parse_annotation_polars(raw)
            elif len(csv_idx) < 1:
                if len(json_idx) == 1:
                    json_file = str(data) + "/" + str(filelist[json_idx[0]])
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
                            str(data),
                        )
                    )
            elif len(csv_idx) > 1:
                raise OSError(
                    "There are multiple input .csv files with the same filename prefix {} in {} folder.".format(
                        str(filename_pre), str(data)
                    )
                )
        elif os.path.isfile(str(data)):
            file = data
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
                        out_json_df,
                        on="sequence_id",
                        how="outer",
                        suffix="_json",
                    )

                    # Coalesce columns that appear in both (prefer json version)
                    for col in out_json_df.columns:
                        if (
                            col != "sequence_id"
                            and f"{col}_json" in res.columns
                        ):
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
                            lambda x: seqs.get(x, ""), return_dtype=pl.String
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
            raise OSError("{} not found.".format(data))
    else:
        raise TypeError(
            f"data must be a Path, str, pandas.DataFrame, polars.DataFrame, or polars.LazyFrame, got {type(data)}"
        )

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


def read_airr(
    file: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    Reads a standard single-cell AIRR rearrangement file.

    If you have non-single-cell data, use `.load_polars` first to load the data and then pass it to `DandelionPolars`.
    That will tell you what columns are missing and you can fill it out accordingly e.g. make up a `cell_id`.

    Parameters
    ----------
    file : Path | str
        path to AIRR rearrangement .tsv file.
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        whether or not to print messages during creation of the Dandelion object.

    Returns
    -------
    DandelionPolars
        DandelionPolars object from AIRR file.
    """
    vdj = DandelionPolars(file, verbose=verbose)
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


def read_bd_airr(
    file: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    Read the TCR or BCR `_AIRR.tsv` produced from BD Rhapsody technology.

    Parameters
    ----------
    file : Path | str
        path to `_AIRR.tsv`
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        whether or not to print messages during creation of the DandelionPolars object.

    Returns
    -------
    DandelionPolars
        DandelionPolars object from BD AIRR file.
    """
    vdj = DandelionPolars(file, verbose=verbose)
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


def read_parse_airr(
    file: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
    **kwargs,
) -> DandelionPolars:
    """
    Read the TCR or BCR `_annotation_airr.tsv` produced from Parse Biosciences Evercode technology.

    This is not to be used for any airr rearrangement file, but specifically for the one produced by Parse Biosciences.
    For standard airr rearrangement files e.g. `all_contig_dandelion.tsv`, use `ddl.Dandelion` or `ddl.read_airr` directly.

    Parameters
    ----------
    file : Path | str
        path to `_annotation_airr.tsv`
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        whether or not to print messages during creation of the Dandelion object.
    **kwargs
        additional keyword arguments passed to Dandelion.

    Returns
    -------
    Dandelion
        DandelionPolars object from Parse AIRR file.
    """
    data = load_polars(file)  # should return LazyFrame
    # Drop the wrong cell_id column if present
    if "cell_id" in data.collect_schema():
        data = data.drop(["cell_id"])
    # Rename columns using polars expressions
    rename_dict = {
        "cell_barcode": "cell_id",
        "read_count": "consensus_count",
        "transcript_count": "umi_count",
        "cdr3": "junction",
        "cdr3_aa": "junction_aa",
    }
    for old, new in rename_dict.items():
        if old in data.collect_schema():
            data = data.with_columns(pl.col(old).alias(new)).drop([old])
    data = data.collect()
    vdj = DandelionPolars(data, verbose=verbose, **kwargs)
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


def read_10x_airr(
    file: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> DandelionPolars:
    """
    Read the `airr_rearrangement.tsv` produced from Cell Ranger directly and returns a DandelionPolars object.

    This is not to be used for any airr rearrangement file, but specifically for the one produced by 10x Genomics.
    For standard airr rearrangement files e.g. `all_contig_dandelion.tsv`, use `Dandelion/DandelionPolars` or `read_airr` directly.

    Parameters
    ----------
    file : Path | str
        path to `airr_rearrangement.tsv`
    prefix : str | None, optional
        Prefix to append to sequence_id and cell_id.
    suffix : str | None, optional
        Suffix to append to sequence_id and cell_id.
    sep : str, optional
        the separator to append suffix/prefix.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        whether or not to print messages during creation of the DandelionPolars object.

    Returns
    -------
    DandelionPolars
        DandelionPolars object from 10x AIRR file.
    """
    data = load_polars(file)  # should return LazyFrame
    # If locus column is missing, derive it using polars vectorized expressions
    if "locus" not in data.columns:
        # Extract first 3 chars of each gene call, ignore nulls/empties, get unique, join with '|'
        data = data.with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("v_call")
                        .str.slice(0, 3)
                        .fill_null("")
                        .str.strip_chars(),
                        pl.col("d_call")
                        .str.slice(0, 3)
                        .fill_null("")
                        .str.strip_chars(),
                        pl.col("j_call")
                        .str.slice(0, 3)
                        .fill_null("")
                        .str.strip_chars(),
                        pl.col("c_call")
                        .str.slice(0, 3)
                        .fill_null("")
                        .str.strip_chars(),
                    ],
                    sep="|",
                )
                .str.split("|")
                .arr.unique()
                .arr.eval(pl.element().filter(pl.element() != ""))
                .arr.join("|")
                .alias("locus")
            ]
        )
    # Drop columns that are all missing
    cols_to_drop = [
        col
        for col in data.collect_schema()
        if data.select(pl.col(col).is_null().all()).collect().item()
    ]
    if cols_to_drop:
        data = data.drop(cols_to_drop)
    data = data.collect()
    vdj = DandelionPolars(data, verbose=verbose)
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
