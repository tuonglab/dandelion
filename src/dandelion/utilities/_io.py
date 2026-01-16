import h5py
import json
import os
import re

import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict, OrderedDict
from pathlib import Path
from scanpy import logging as logg
from scipy.sparse import csr_matrix

from dandelion.utilities._core import Dandelion, load_data
from dandelion.utilities._utilities import (
    DEFAULT_PREFIX,
    deprecated,
    all_missing,
    sanitize_blastn,
    sanitize_data,
    Contig,
)


AIRR = [
    "cell_id",
    "sequence_id",
    "sequence",
    "sequence_aa",
    "productive",
    "complete_vdj",
    "vj_in_frame",
    "locus",
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    "junction",
    "junction_aa",
    "consensus_count",
    "umi_count",
    "cdr3_start",
    "cdr3_end",
    "sequence_length_10x",
    "high_confidence_10x",
    "is_cell_10x",
    "fwr1_aa",
    "fwr1",
    "cdr1_aa",
    "cdr1",
    "fwr2_aa",
    "fwr2",
    "cdr2_aa",
    "cdr2",
    "fwr3_aa",
    "fwr3",
    "fwr4_aa",
    "fwr4",
    "clone_id",
    "raw_consensus_id_10x",
    "exact_subclonotype_id_10x",
]
CELLRANGER = [
    "barcode",
    "contig_id",
    "sequence",
    "aa_sequence",
    "productive",
    "full_length",
    "frame",
    "chain",
    "v_gene",
    "d_gene",
    "j_gene",
    "c_gene",
    "cdr3_nt",
    "cdr3",
    "reads",
    "umis",
    "cdr3_start",
    "cdr3_stop",
    "length",
    "high_confidence",
    "is_cell",
    "fwr1",
    "fwr1_nt",
    "cdr1",
    "cdr1_nt",
    "fwr2",
    "fwr2_nt",
    "cdr2",
    "cdr2_nt",
    "fwr3",
    "fwr3_nt",
    "fwr4",
    "fwr4_nt",
    "raw_clonotype_id",
    "raw_consensus_id",
    "exact_subclonotype_id",
]


def fasta_iterator(fh: str) -> tuple[str, str]:
    """Read in a fasta file as an iterator."""
    while True:
        line = fh.readline()
        if line.startswith(">"):
            break
    while True:
        header = line[1:-1].rstrip()
        sequence = fh.readline().rstrip()
        while True:
            line = fh.readline()
            if not line:
                break
            if line.startswith(">"):
                break
            sequence += line.rstrip()
        yield (header, sequence)
        if not line:
            return


def decode(df):
    """Decode byte strings in a DataFrame."""
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df


def read_h5ddl(
    filename: Path | str = "dandelion_data.h5ddl",
    distance_zarr: Path | str | None = None,
    verbose: bool = False,
) -> Dandelion:
    """
    Read in and returns a Dandelion class from .h5ddl format.

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
    data = decode(load_data(_read_h5_group(filename, group="data")))
    # final decode to ensure all byte strings are converted to str
    metadata = _read_h5_group(filename, group="metadata")
    metadata_names = _read_h5_group(filename, group="metadata_names")
    metadata.index = metadata_names

    try:
        g_0 = _read_h5_csr_matrix(filename, group="graph/graph_0", as_df=True)
        g_1 = _read_h5_csr_matrix(filename, group="graph/graph_1", as_df=True)
        graph0 = _create_graph(g_0, adjust_adjacency=1, fillna=0)
        graph1 = _create_graph(g_1, adjust_adjacency=1, fillna=0)
        graph = (graph0, graph1)
    except KeyError:
        pass

    try:
        distances = _read_h5_csr_matrix(
            filename, group="distances", as_df=False
        )
        distances._index_names = metadata.index
    except KeyError:
        if distance_zarr is not None:
            import dask.array as da

            distances = da.from_zarr(str(distance_zarr) + "/distance_matrix")

    try:
        layout0 = _read_h5_dict(filename, group="layout/layout_0")
        layout1 = _read_h5_dict(filename, group="layout/layout_1")
        layout = (layout0, layout1)
    except KeyError:
        pass

    try:
        germline = _read_h5_zip(
            filename, group="germline", key_group="keys", value_group="values"
        )
    except KeyError:
        pass

    constructor = {}
    constructor["data"] = data
    if "metadata" in locals():
        constructor["metadata"] = metadata
    if "germline" in locals():
        constructor["germline"] = germline
    if "layout" in locals():
        constructor["layout"] = layout
    if "graph" in locals():
        constructor["graph"] = graph
    if "distances" in locals():
        constructor["distances"] = distances

    res = Dandelion(**constructor, verbose=verbose)
    # ensure that the metadata is decoded
    res._metadata = decode(res._metadata)

    return res


def read_airr(
    file: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> Dandelion:
    """
    Reads a standard single-cell AIRR rearrangement file.

    If you have non-single-cell data, use `.load_data` first to load the data and then pass it to `ddl.Dandelion`.
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
    Dandelion
        Dandelion object from AIRR file.
    """
    vdj = Dandelion(file, verbose=verbose)
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
) -> Dandelion:
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
        whether or not to print messages during creation of the Dandelion object.

    Returns
    -------
    Dandelion
        Dandelion object from BD AIRR file.
    """
    vdj = Dandelion(file, verbose=verbose)
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
) -> Dandelion:
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
        Dandelion object from Parse AIRR file.
    """
    data = load_data(file)
    data.drop("cell_id", axis=1, inplace=True)  # it's the wrong cell_id
    data = data.rename(
        columns={
            "cell_barcode": "cell_id",
            "read_count": "consensus_count",
            "transcript_count": "umi_count",
            "cdr3": "junction",
            "cdr3_aa": "junction_aa",
        }
    )
    vdj = Dandelion(data, verbose=verbose, **kwargs)
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
) -> Dandelion:
    """
    Read the `airr_rearrangement.tsv` produced from Cell Ranger directly and returns a Dandelion object.

    This is not to be used for any airr rearrangement file, but specifically for the one produced by 10x Genomics.
    For standard airr rearrangement files e.g. `all_contig_dandelion.tsv`, use `ddl.Dandelion` or `ddl.read_airr` directly.

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
        whether or not to print messages during creation of the Dandelion object.

    Returns
    -------
    Dandelion
        Dandelion object from 10x AIRR file.
    """
    dat = load_data(file)
    # get all the v,d,j,c calls
    if "locus" not in dat:
        tmp = [
            (v, d, j, c)
            for v, d, j, c in zip(
                dat["v_call"], dat["d_call"], dat["j_call"], dat["c_call"]
            )
        ]
        locus = []
        for t in tmp:
            if all("IGH" in x for x in t if pd.notnull(x)):
                locus.append("IGH")
            elif all("IGK" in x for x in t if pd.notnull(x)):
                locus.append("IGK")
            elif all("IGL" in x for x in t if pd.notnull(x)):
                locus.append("IGL")
            elif all("TRA" in x for x in t if pd.notnull(x)):
                locus.append("TRA")
            elif all("TRB" in x for x in t if pd.notnull(x)):
                locus.append("TRB")
            elif all("TRD" in x for x in t if pd.notnull(x)):
                locus.append("TRD")
            elif all("TRG" in x for x in t if pd.notnull(x)):
                locus.append("TRG")
            else:
                locus.append(np.nan)
        dat["locus"] = locus
    null_columns = [col for col in dat.columns if all_missing(dat[col])]
    if len(null_columns) > 0:
        dat.drop(null_columns, inplace=True, axis=1)
    vdj = Dandelion(dat, verbose=verbose)
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


def read_10x_vdj(
    data: Path | str | pd.DataFrame,
    filename_prefix: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str = "_",
    remove_malformed: bool = True,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = False,
) -> Dandelion:
    """
    A parser to read .csv and .json files directly from folder containing 10x cellranger-outputs,
    or parse an existing pandas DataFrame.

    This function parses the 10x output files into an AIRR compatible format.

    Minimum requirement is one of either {filename_prefix}_contig_annotations.csv or all_contig_annotations.json
    when reading from file path.

    If .fasta, .json files are found in the same folder, additional info will be appended to the final table.

    Parameters
    ----------
    data : Path | str | pandas.DataFrame
        path to folder containing `.csv` and/or `.json` files, path to files directly, or a pandas
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
    Dandelion
        Dandelion object holding the parsed data.

    Raises
    ------
    IOError
        if contig_annotations.csv and all_contig_annotations.json file(s) not found in the input folder.
    TypeError
        if data is not a valid type (Path, str, or pandas.DataFrame).

    """
    filename_pre = (
        DEFAULT_PREFIX if filename_prefix is None else filename_prefix
    )

    # Handle DataFrame input (pandas)
    if isinstance(data, pd.DataFrame):
        logg.info("Parsing pandas DataFrame")
        raw = data.copy()
        raw.set_index("contig_id", drop=False, inplace=True)
        out = parse_annotation(raw)
    elif isinstance(data, (str, Path)):
        # Handle file path inputs
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
                raw = pd.read_csv(str(file))
                raw.set_index("contig_id", drop=False, inplace=True)
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
                    out = parse_annotation(raw)
                    with open(json_file) as f:
                        raw_json = json.load(f)
                    out_json = parse_json(raw_json)
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
                    raw["sequence"] = pd.Series(seqs)
                    out = parse_annotation(raw)
                else:
                    out = parse_annotation(raw)
            elif len(csv_idx) < 1:
                if len(json_idx) == 1:
                    json_file = str(data) + "/" + str(filelist[json_idx[0]])
                    logg.info("Reading {}".format(json_file))
                    if os.path.exists(json_file):
                        with open(json_file) as f:
                            raw = json.load(f)
                        out = parse_json(raw)
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
                raw = pd.read_csv(str(file))
                raw.set_index("contig_id", drop=False, inplace=True)
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
                    out = parse_annotation(raw)
                    with open(json_file) as f:
                        raw_json = json.load(f)
                    out_json = parse_json(raw_json)
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
                    raw["sequence"] = pd.Series(seqs)
                    out = parse_annotation(raw)
                else:
                    out = parse_annotation(raw)
            elif str(file).endswith(".json"):
                if os.path.exists(file):
                    logg.info("Reading {}".format(file))
                    with open(file) as f:
                        raw = json.load(f)
                    out = parse_json(raw)
                else:
                    raise OSError("{} not found.".format(file))
        else:
            raise OSError("{} not found.".format(data))
    else:
        raise TypeError(
            f"data must be a Path, str, or pandas.DataFrame, got {type(data)}"
        )
    res = pd.DataFrame.from_dict(out, orient="index")
    # quick check if locus is malformed
    if remove_malformed:
        res = res[~res["locus"].str.contains("[|]")]
    # change all unknowns to blanks
    res.replace("unknown", "", inplace=True)
    vdj = Dandelion(res, verbose=verbose)
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


def parse_json(data: list) -> defaultdict:
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
        # "duplicate_count": "umi_count",
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
        "C-REGION": "c_call",
    }
    required_calls = ["v_call", "d_call", "j_call", "c_call"]
    region_keys = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "fwr4"]
    out = defaultdict(OrderedDict)
    for i in range(len(data)):
        if data[i]["contig_name"] is not None:
            key = data[i]["contig_name"]
        else:
            continue
        for k in main_dict1.keys():
            if k in data[i]:
                if data[i][k] is not None:
                    out[key].update({main_dict1[k]: data[i][k]})
                else:
                    out[key].update({main_dict1[k]: ""})
            else:
                out[key].update({main_dict1[k]: ""})
        if data[i]["annotations"] is not None:
            chains = []
            for j in range(len(data[i]["annotations"])):
                chains.append(data[i]["annotations"][j]["feature"]["chain"])
            out[key].update({"locus": "|".join(list(set(chains)))})
            for j in range(len(data[i]["annotations"])):
                rtype = data[i]["annotations"][j]["feature"]["region_type"]
                if rtype in region_type_dict.keys():
                    call = region_type_dict[rtype]
                else:
                    continue
                if (
                    data[i]["annotations"][j]["feature"]["gene_name"]
                    is not None
                ):
                    out[key].update(
                        {
                            call: data[i]["annotations"][j]["feature"][
                                "gene_name"
                            ]
                        }
                    )
                else:
                    out[key].update({call: ""})
        for rc in required_calls:
            if rc not in out[key]:
                out[key].update({rc: ""})
        for k in main_dict2.keys():
            if k in data[i]:
                if data[i][k] is not None:
                    out[key].update({main_dict2[k]: data[i][k]})
                else:
                    out[key].update({main_dict2[k]: np.nan})
            else:
                out[key].update({main_dict2[k]: np.nan})
        for rk in region_keys:
            if rk in data[i]:
                if data[i][rk] is not None:
                    for k in data[i][rk]:
                        if k == "start":
                            ka = rk + "_start"
                        elif k == "stop":
                            ka = rk + "_end"
                        elif k == "nt_seq":
                            ka = rk + ""
                        elif k == "aa_seq":
                            ka = rk + "_aa"
                        else:
                            continue
                        out[key].update({ka: data[i][rk][k]})
                else:
                    for k in region_keys:
                        out[key].update({k + "_start": np.nan})
                        out[key].update({k + "_end": np.nan})
                        out[key].update({k + "": ""})
                        out[key].update({k + "_aa": ""})
            else:
                for k in region_keys:
                    out[key].update({k + "_start": np.nan})
                    out[key].update({k + "_end": np.nan})
                    out[key].update({k + "": ""})
                    out[key].update({k + "_aa": ""})
        if data[i]["info"] is not None:
            for info in data[i]["info"]:
                if data[i]["info"][info] is not None:
                    out[key].update({info_dict[info]: data[i]["info"][info]})
                else:
                    out[key].update({info_dict[info]: ""})
        for k in main_dict3.keys():
            if k in data[i]:
                if data[i][k] is not None:
                    out[key].update({main_dict3[k]: data[i][k]})
                else:
                    out[key].update({main_dict3[k]: ""})
            else:
                out[key].update({main_dict3[k]: ""})
    return out


def parse_annotation(data: pd.DataFrame) -> defaultdict:
    """Parse annotation file."""
    out = defaultdict(OrderedDict)
    swap_dict = dict(zip(CELLRANGER, AIRR))
    for _, row in data.iterrows():
        contig = Contig(row, swap_dict).contig["sequence_id"]
        out[contig] = Contig(row, swap_dict).contig
        if out[contig]["locus"] in ["None", "none", None, np.nan, ""]:
            calls = []
            for call in ["v_call", "d_call", "j_call", "c_call"]:
                if out[contig][call] not in ["None", "none", None, np.nan, ""]:
                    calls.append(out[contig][call])
            out[contig]["locus"] = "|".join(list({str(c)[:3] for c in calls}))
        if out[contig]["locus"] == "None" or out[contig]["locus"] == "":
            out[contig]["locus"] = "|"
    return out


def _read_h5_group(filename: Path | str, group: str) -> pd.DataFrame:
    """
    Read a specific group from an H5 file.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.

    Returns
    -------
    pd.DataFrame
        The data from the specified group as a pandas dataframe.

    Raises
    ------
    KeyError
        If the specified group is not found in the H5 file.
    """
    with h5py.File(filename, "r") as hf:
        data_group = hf[group]
        structured_data_array = data_group[:]
        decoded = {}
        if structured_data_array.dtype.names is not None:
            for col in structured_data_array.dtype.names:
                dtype = structured_data_array.dtype[col]
                if dtype.char == "S":  # Check if dtype is byte strings
                    # Decode byte strings
                    decoded.update(
                        {
                            col: np.array(
                                [
                                    (
                                        x.astype(str)
                                        if isinstance(x, bytes)
                                        else x
                                    )
                                    for x in structured_data_array[col]
                                ]
                            )
                        }
                    )
                else:
                    decoded.update({col: structured_data_array[col]})
            # Create a DataFrame from the structured array
            data = pd.DataFrame(decoded)
        else:
            data = np.array(
                [
                    x.astype(str) if isinstance(x, bytes) else x
                    for x in structured_data_array
                ]
            )
    return data


def _read_h5_csr_matrix(
    filename: Path | str, group: str, as_df: bool = True
) -> pd.DataFrame:
    """
    Read a group from an H5 file originally stored as a compressed sparse matrix.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.

    Returns
    -------
    pd.DataFrame
        The data from the specified group as a pandas dataframe.
    """
    with h5py.File(filename, "r") as f:
        data = f[f"{group}/data"][:]
        indices = f[f"{group}/indices"][:]
        indptr = f[f"{group}/indptr"][:]
        shape = tuple(f[f"{group}/shape"][:])
        # Reconstruct CSR matrix
        loaded_matrix = csr_matrix((data, indices, indptr), shape=shape)
        if not as_df:
            return loaded_matrix
        df = pd.DataFrame(loaded_matrix.toarray())
        df_col = _read_h5_group(filename, f"{group}/column")
        df_index = _read_h5_group(filename, f"{group}/index")
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


def _read_h5_dict(filename: Path | str, group: str) -> dict:
    """
    Read a dictionary from an H5 file.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.

    Returns
    -------
    dict
        The dictionary data from the specified group.
    """
    out_dict = {}
    with h5py.File(filename, "r") as hf:
        for k in hf[group].keys():
            out_dict[k] = hf[group][k][:]

    return out_dict


def _read_h5_zip(
    filename: Path | str, group: str, key_group: str, value_group: str
) -> dict:
    """
    Read two groups from an H5 file and return them as a dictionary.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.
    key_group : str
        The name of the group containing the keys.
    value_group : str
        The name of the group containing the values.

    Returns
    -------
    dict
        The dictionary data from the specified groups.
    """
    with h5py.File(filename, "r") as hf:
        keys = [key.decode("utf-8") for key in hf[f"{group}/{key_group}"][:]]
        values = [
            value.decode("utf-8") for value in hf[f"{group}/{value_group}"][:]
        ]
        # Reconstruct the dictionary
        out_dict = dict(zip(keys, values))
    return out_dict


@deprecated(
    deprecated_in="1.0.0",
    removed_in="1.1.0",
    details="legacy .h5ddl format will no longer be supported.",
)
def read_h5ddl_legacy(
    filename: Path | str = "dandelion_data.h5ddl",
) -> Dandelion:
    """
    Read in and returns a Dandelion class from .h5ddl format saved in legacy (version 3) format.

    Parameters
    ----------
    filename : Path | str, optional
        path to `.h5ddl` file

    Returns
    -------
    Dandelion
        Dandelion object.

    Raises
    ------
    AttributeError
        if `data` not found in `.h5ddl` file.
    """
    data = decode(load_data(_read_h5_group_legacy(filename, group="data")))
    # final decode to ensure all byte strings are converted to str
    metadata = _read_h5_group_legacy(filename, group="metadata")
    try:
        g_0 = _read_h5_group_legacy(filename, group="graph/graph_0")
        g_1 = _read_h5_group_legacy(filename, group="graph/graph_1")
        graph0 = _create_graph(g_0, adjust_adjacency=1, fillna=0)
        graph1 = _create_graph(g_1, adjust_adjacency=1, fillna=0)
        graph = (graph0, graph1)
    except KeyError:
        pass

    try:
        layout0 = _read_h5_dict(filename, group="layout/layout_0")
        layout1 = _read_h5_dict(filename, group="layout/layout_1")
        layout = (layout0, layout1)
    except KeyError:
        pass

    try:
        germline = _read_h5_zip_legacy(filename, group="germline")
    except KeyError:
        pass

    constructor = {}
    constructor["data"] = data
    if "metadata" in locals():
        constructor["metadata"] = metadata
    if "germline" in locals():
        constructor["germline"] = germline
    if "layout" in locals():
        constructor["layout"] = layout
    if "graph" in locals():
        constructor["graph"] = graph

    res = Dandelion(**constructor, verbose=False)
    # ensure that the metadata is decoded
    res._metadata = decode(res._metadata)

    return res


def _read_h5_group_legacy(filename: Path | str, group: str) -> pd.DataFrame:
    """
    Read a specific group from an H5 file in legacy format.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.

    Returns
    -------
    pd.DataFrame
        The data from the specified group as a pandas dataframe.

    Raises
    ------
    KeyError
        If the specified group is not found in the H5 file.
    """
    data = pd.read_hdf(filename, group)

    return data


def _read_h5_zip_legacy(filename: Path | str, group: str) -> dict:
    """
    Read two groups from an H5 file and return them as a dictionary.

    Parameters
    ----------
    filename : Path | str
        The path to the H5 file.
    group : str
        The group to read from the H5 file.
    key_group : str
        The name of the group containing the keys.
    value_group : str
        The name of the group containing the values.

    Returns
    -------
    dict
        The dictionary data from the specified groups.
    """
    with h5py.File(filename, "r") as hf:
        out_dict = {}
        for g in hf[group].attrs:
            out_dict.update({g: hf[group].attrs[g]})
    return out_dict


def write_airr(data: pd.DataFrame, save: Path | str) -> None:
    """Save as airr formatted file."""
    data = sanitize_data(data)
    data.to_csv(save, sep="\t", index=False)


def write_blastn(data: pd.DataFrame, save: Path | str) -> None:
    """Write blast output."""
    data = sanitize_blastn(data)
    data.to_csv(save, sep="\t", index=False)
