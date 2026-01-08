import os
import json
import re
import shutil
import tempfile
import warnings

import anndata as ad
import numpy as np
import pandas as pd

# import polars as pl

from anndata import AnnData
from Bio import Align
from operator import countOf
from pathlib import Path
from plotnine import (
    ggplot,
    geom_bar,
    geom_col,
    ggtitle,
    scale_fill_manual,
    coord_flip,
    options,
    element_blank,
    aes,
    xlab,
    ylab,
    facet_grid,
    theme_classic,
    theme,
    save_as_pdf_pages,
)
from scanpy import logging as logg
from subprocess import run
from time import sleep
from tqdm import tqdm
from typing import Literal

from dandelion.external.immcantation.changeo import (
    assigngenes_igblast,
    makedb_igblast,
    parsedb_heavy,
    parsedb_light,
    creategermlines,
)
from dandelion.external.immcantation.tigger import tigger_genotype

from dandelion.utilities._core import (
    Dandelion,
    load_data,
    write_fasta,
    check_travdv,
)

# from dandelion.utilities._polars import (
# DandelionPolars,
# load_polars,
# _check_travdv_polars,
# _sanitize_data_polars,
# _write_airr,
# )
from dandelion.utilities._io import (
    fasta_iterator,
    read_10x_vdj,
    write_airr,
    write_blastn,
)
from dandelion.utilities._utilities import (
    TRUES,
    FALSES,
    HEAVYLONG,
    LIGHTSHORT,
    NO_DS,
    DEFAULT_PREFIX,
    check_filepath,
    not_same_call,
    present,
    all_missing,
    sanitize_data,
    sanitize_column,
    lib_type,
    set_igblast_env,
    set_blast_env,
    check_data,
    Tree,
)
from dandelion.tools._tools import transfer


def format_fasta(
    fasta: Path | str,
    prefix: str | None = None,
    suffix: str | None = None,
    sep: str | None = None,
    remove_trailing_hyphen_number: bool = True,
    high_confidence_filtering: bool = False,
    out_dir: Path | str | None = None,
    filename_prefix: str | None = None,
):
    """
    Add prefix to the headers/contig ids in input fasta and annotation file.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file.
    prefix : str | None, optional
        prefix to append to the headers/contig ids.
    suffix : str | None, optional
        suffix to append to the headers/contig ids.
    sep : str | None, optional
        separator after prefix or before suffix to append to the headers/contig ids.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    high_confidence_filtering : bool, optional
        whether ot not to filter to only `high confidence` contigs.
    out_dir : str | None, optional
        path to output location. `None` defaults to 'dandelion'.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. `None` defaults to 'all'.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    filename_pre = (
        DEFAULT_PREFIX if filename_prefix is None else filename_prefix
    )

    file_path = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with=".fasta",
        within_dandelion=False,
    )

    if file_path is None:
        raise FileNotFoundError(
            "Path to fasta file is unknown. Please "
            + "specify path to fasta file or folder containing fasta file. "
            + "Starting folder should only contain 1 fasta file."
        )
    # before continuing, check if the file is not empty
    if os.stat(file_path).st_size == 0:
        raise ValueError(
            f"{str(file_path)} is empty. Please check the file and try again or remove if necessary."
        )
    fh = open(file_path)
    seqs = {}
    if sep is None:
        separator = "_"
    else:
        separator = str(sep)
    for header, sequence in fasta_iterator(fh):
        if prefix is None and suffix is None:
            seqs[header] = sequence
        elif prefix is not None:
            if suffix is not None:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
            else:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0].split("-")[0]
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = str(prefix) + separator + str(header)
            seqs[newheader] = sequence
        else:
            if suffix is not None:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(header).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = (
                        str(header).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
            else:
                newheader = str(header)
            seqs[newheader] = sequence
    fh.close()
    base_dir = file_path.parent if file_path.is_file() else Path.cwd()
    out_dir = base_dir / "dandelion" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # format the barcode and contig_id in the corresponding annotation file too
    anno = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with="_annotations.csv",
        within_dandelion=False,
    )
    data = pd.read_csv(anno, dtype="object")
    if prefix is not None:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0].split("-")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix)
                    + separator
                    + str(b).split("-")[0]
                    + separator
                    + str(suffix)
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b) + separator + str(suffix)
                    for b in data["barcode"]
                ]
        else:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0].split("-")[0]
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b).split("-")[0]
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(prefix) + separator + str(c) for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b) for b in data["barcode"]
                ]
    else:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(c).split("_contig")[0].split("-")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(b).split("-")[0] + separator + str(suffix)
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(c).split("_contig")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(b) + separator + str(suffix) for b in data["barcode"]
                ]
        else:
            data["contig_id"] = [str(c) for c in data["contig_id"]]
            data["barcode"] = [str(b) for b in data["barcode"]]
    anno = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with="_annotations.csv",
        within_dandelion=False,
    )
    out_anno = out_dir / (file_path.stem + "_annotations.csv")
    out_fasta = out_dir / file_path.name
    fh1 = open(out_fasta, "w")
    fh1.close()
    if high_confidence_filtering:
        hiconf_contigs = [
            x
            for x, y in zip(data["contig_id"], data["high_confidence"])
            if y in TRUES
        ]
        seqs = {hiconf: seqs[hiconf] for hiconf in hiconf_contigs}
        data = data[data["contig_id"].isin(hiconf_contigs)]
    write_fasta(fasta_dict=seqs, out_fasta=out_fasta)
    data.to_csv(out_anno, index=False)


def format_fastas(
    fastas: list[Path | str] | Path | str,
    prefix: list[str] | None = None,
    suffix: list[str] | None = None,
    sep: str | None = None,
    remove_trailing_hyphen_number: bool = True,
    high_confidence_filtering: bool = False,
    out_dir: Path | str | None = None,
    filename_prefix: list[str] | str | None = None,
):
    """
    Add prefix to the headers/contig ids in input fasta and annotation file.

    Parameters
    ----------
    fastas : list[Path | str]
        list of paths to fasta files.
    prefix : list[str] | None, optional
        list of prefixes to append to headers/contig ids in each fasta file.
    suffix : list[str] | None, optional
        list of suffixes to append to headers/contig ids in each fasta file.
    sep : str | None, optional
        separator after prefix or before suffix to append to the headers/contig
        ids.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    high_confidence_filtering : bool, optional
        whether ot not to filter to only `high confidence` contigs.
    out_dir : Path | str | None, optional
        path to out put location.
    filename_prefix : list[str] | str | None, optional
        list of prefixes of file names preceding '_contig'. `None` defaults to
        'all'.
    """
    fastas, filename_prefix = check_data(fastas, filename_prefix)
    if prefix is not None:
        if not isinstance(prefix, list):
            prefix = [prefix]
        prefix_dict = dict(zip(fastas, prefix))
    if suffix is not None:
        if not isinstance(suffix, list):
            suffix = [suffix]
        suffix_dict = dict(zip(fastas, suffix))

    for i in tqdm(
        range(0, len(fastas)),
        desc="Formatting fasta(s) ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        if prefix is None and suffix is None:
            format_fasta(
                fastas[i],
                prefix=None,
                suffix=None,
                sep=None,
                remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                high_confidence_filtering=high_confidence_filtering,
                out_dir=out_dir,
                filename_prefix=filename_prefix[i],
            )
        elif prefix is not None:
            if suffix is not None:
                format_fasta(
                    fastas[i],
                    prefix=prefix_dict[fastas[i]],
                    suffix=suffix_dict[fastas[i]],
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
            else:
                format_fasta(
                    fastas[i],
                    prefix=prefix_dict[fastas[i]],
                    suffix=None,
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
        else:
            if suffix is not None:
                format_fasta(
                    fastas[i],
                    prefix=None,
                    suffix=suffix_dict[fastas[i]],
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
            else:
                format_fasta(
                    fastas[i],
                    prefix=None,
                    suffix=None,
                    sep=None,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )


def assign_isotype(
    fasta: Path | str,
    org: Literal["human", "mouse"] = "human",
    evalue: float = 1e-4,
    correct_c_call: bool = True,
    correction_dict: dict[str, dict[str, str]] | None = None,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: tuple[float, float] = (4, 4),
    blastdb: Path | str | None = None,
    filename_prefix: str | None = None,
    additional_args: list[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    correct_c_call : bool, optional
        whether or not to adjust the c_calls after blast based on provided
        primers specified in `primer_dict` option.
    correction_dict : dict[str, dict[str, str]] | None, optional
        a nested dictionary contain isotype/c_genes as keys and primer
        sequences as records to use for correcting annotated c_calls. Defaults
        to a curated dictionary for human sequences if left as none.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plot.
    show_plot : bool, optional
        whether or not to show plot.
    figsize : tuple[float, float], optional
        size of figure.
    blastdb : Path | str | None, optional
        path to blast database. Defaults to `$BLASTDB` environmental variable.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. `None` defaults to 'all'.
    additional_args : list[str], optional
        additional arguments to pass to `blastn`.
    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    aligner = Align.PairwiseAligner()

    def gene_correction(df: pd.DataFrame, i: str, dictionary: dict[str, str]):
        """Generalized pairwise alignment for multiple genes."""
        seq = df.loc[i, "c_sequence_alignment"].replace("-", "")
        scores = {
            key: aligner.align(seq, dictionary[key]).score for key in dictionary
        }
        max_score = max(scores.values())
        df.at[i, "c_call"] = ",".join(
            key for key, score in scores.items() if score == max_score
        )

    def _correct_c_call(
        data: pd.DataFrame,
        org: Literal["human", "mouse"] = "human",
        primers_dict: dict[str, dict[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Pairwise alignment for c genes.

        Parameters
        ----------
        data : pd.DataFrame
            Input data Frame.
        primers_dict : dict[str, dict[str, str]] | None, optional
            Gene:Sequence dictionary to do pairwise alignment with.

        Returns
        -------
        pd.DataFrame
            Output data frame with c_call adjusted.
        """
        dat = data.copy()
        if primers_dict is None:
            if org == "human":
                primer_dict = {
                    "IGHG": {
                        "IGHG1": "GCCTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCACCCTCCTCCAAGAGCACCTCTGGGGGCACAGCGGCCCTGGGC",
                        "IGHG2": "GCCTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCCGAGAGCACAGCGGCCCTGGGC",
                        "IGHG3": "GCTTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCTGGGGGCACAGCGGCCCTGGGC",
                        "IGHG4": "GCTTCCACCAAGGGCCCATCCGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCCGAGAGCACAGCCGCCCTGGGC",
                    },
                    "IGHA": {
                        "IGHA1": "GCATCCCCGACCAGCCCCAAGGTCTTCCCGCTGAGCCTCTGCAGCACCCAGCCAGATGGGAACGTGGTCATCGCCTGC",
                        "IGHA2": "GCATCCCCGACCAGCCCCAAGGTCTTCCCGCTGAGCCTCGACAGCACCCCCCAAGATGGGAACGTGGTCGTCGCATGC",
                    },
                    "IGLC7": {
                        "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTG"
                        "TGTCTCATAA",
                        "IGLC7": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGT"
                        "GTGTCTCGTAA",
                    },
                    "IGLC3": {
                        "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTG"
                        "TGTCTCATAA",
                        "IGLC3": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGT"
                        "GTGTCTCATAA",
                    },
                    "IGLC6": {
                        "IGLC": "TCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCA",
                        "IGLC6": "TCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGCCTGA",
                    },
                }
            else:
                primer_dict = {
                    "IGHG2": {
                        "IGHG2A": "GCCAAAACAACAGCCCCATCGGTCTATCCACTGGCCCCTGTGTGTGGAGATACAACTGGC",
                        "IGHG2B": "GCCAAAACAACACCCCCATCAGTCTATCCACTGGCCCCTGGGTGTGGAGATACAACTGGT",
                        "IGHG2C": "GCCAAAACAACAGCCCCATCGGTCTATCCACTGGCCCCTGTGTGTGGAGGTACAACTGGC",
                    }
                }
        else:
            primer_dict = primers_dict

        for i in dat.index:
            if pd.notnull(dat.loc[i, "c_call"]):
                for k, genes in primer_dict.items():
                    if k in dat.loc[i, "c_call"]:
                        if "IGHG4A" not in dat.loc[i, "c_call"]:
                            gene_correction(dat, i, genes)
        return dat

    filePath = check_filepath(
        fasta, filename_prefix=filename_prefix, ends_with=".fasta"
    )
    if filePath is None:
        raise FileNotFoundError(
            "Path to fasta file is unknown. Please specify path to "
            + "fasta file or folder containing fasta file."
        )

    blast_out = run_blastn(
        fasta=filePath,
        database=blastdb,
        org=org,
        loci="ig",
        call="c",
        max_hsps=1,
        evalue=evalue,
        outfmt=(
            "6 qseqid sseqid pident length mismatch gapopen "
            + "qstart qend sstart send evalue bitscore qseq sseq"
        ),
        dust="no",
        additional_args=additional_args,
    )
    blast_out.drop_duplicates(subset="sequence_id", keep="first", inplace=True)

    _10xfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_annotations.csv",
    )
    _airrfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_igblast.tsv",
        sub_dir="tmp",
    )
    _processedfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_igblast_db-pass_genotyped.tsv",
        sub_dir="tmp",
    )
    if _processedfile is None:
        _processedfile = check_filepath(
            fasta,
            filename_prefix=filename_prefix,
            ends_with="_igblast_db-pass.tsv",
            sub_dir="tmp",
        )
        out_ex = "_igblast_db-pass.tsv"
    else:
        out_ex = "_igblast_db-pass_genotyped.tsv"
    dat = load_data(_processedfile)
    if _10xfile is not None:
        dat_10x = read_10x_vdj(_10xfile)
        res_10x = pd.DataFrame(dat_10x._data["c_call"].replace("", "None"))
    else:  # pragma: no cover
        res_10x = pd.DataFrame(dat["c_call"])
        res_10x["c_call"] = "None"
    for col in [
        "c_call",
        "c_sequence_alignment",
        "c_germline_alignment",
        "c_sequence_start",
        "c_sequence_end",
        "c_score",
        "c_identity",
    ]:
        dat[col] = pd.Series(blast_out[col])
    res_blast = pd.DataFrame(dat["c_call"].replace("", "None"))
    res_blast = res_blast.fillna(value="None")
    res_10x_sum = pd.DataFrame(
        res_10x["c_call"].value_counts(normalize=True) * 100
    )
    res_10x_sum["group"] = "10X"
    res_10x_sum.columns = ["counts", "group"]
    res_10x_sum.index = res_10x_sum.index.set_names(["c_call"])
    res_10x_sum.reset_index(drop=False, inplace=True)
    res_blast_sum = pd.DataFrame(
        res_blast["c_call"].value_counts(normalize=True) * 100
    )
    res_blast_sum["group"] = "blast"
    res_blast_sum.columns = ["counts", "group"]
    res_blast_sum.index = res_blast_sum.index.set_names(["c_call"])
    res_blast_sum.reset_index(drop=False, inplace=True)
    if (
        correct_c_call
    ):  # TODO: figure out if i need to set up a None correction?
        dat = _correct_c_call(dat, primers_dict=correction_dict, org=org)
        res_corrected = pd.DataFrame(dat["c_call"].replace("", "None"))
        res_corrected = res_corrected.fillna(value="None")
        res_corrected_sum = pd.DataFrame(
            res_corrected["c_call"].value_counts(normalize=True) * 100
        )
        res_corrected_sum["group"] = "corrected"
        res_corrected_sum.columns = ["counts", "group"]
        res_corrected_sum.index = res_corrected_sum.index.set_names(["c_call"])
        res_corrected_sum.reset_index(drop=False, inplace=True)
        res = pd.concat([res_10x_sum, res_blast_sum, res_corrected_sum])
    else:  # pragma: no cover
        res = pd.concat([res_10x_sum, res_blast_sum])
    res = res.reset_index(drop=True)
    res["c_call"] = res["c_call"].fillna(value="None")
    res["c_call"] = [re.sub("[*][0-9][0-9]", "", c) for c in res["c_call"]]
    res["c_call"] = res["c_call"].astype("category")
    res["c_call"] = res["c_call"].cat.reorder_categories(
        sorted(list(set(res["c_call"])), reverse=True)
    )
    dat["c_call_10x"] = pd.Series(res_10x["c_call"])
    # some minor adjustment to the final output table
    airr_output = load_data(_airrfile)
    cols_to_merge = [
        "junction_aa_length",
        "fwr1_aa",
        "fwr2_aa",
        "fwr3_aa",
        "fwr4_aa",
        "cdr1_aa",
        "cdr2_aa",
        "cdr3_aa",
        "sequence_alignment_aa",
        "v_sequence_alignment_aa",
        "d_sequence_alignment_aa",
        "j_sequence_alignment_aa",
    ]
    for x in cols_to_merge:
        dat[x] = pd.Series(airr_output[x])

    # remove allellic calls
    dat["c_call"] = dat["c_call"].fillna(value="")
    dat["c_call"] = [re.sub("[*][0-9][0-9]", "", c) for c in dat["c_call"]]

    write_airr(dat, _processedfile)
    if plot:
        options.figure_size = figsize
        if correct_c_call:
            p = (
                ggplot(res, aes(x="c_call", y="counts", fill="group"))
                + coord_flip()
                + theme_classic()
                + xlab("c_call")
                + ylab("% c calls")
                + geom_col(stat="identity", position="dodge")
                + scale_fill_manual(values=("#79706e", "#86bcb6", "#F28e2b"))
                + theme(legend_title=element_blank())
            )
        else:
            p = (
                ggplot(res, aes(x="c_call", y="counts", fill="group"))
                + coord_flip()
                + theme_classic()
                + xlab("c_call")
                + ylab("% c calls")
                + geom_col(stat="identity", position="dodge")
                + scale_fill_manual(values=("#79706e", "#86bcb6"))
                + theme(legend_title=element_blank())
            )
        if save_plot:
            _file3 = filePath.parent / "assign_isotype.pdf"
            save_as_pdf_pages([p], filename=_file3, verbose=False)
        if show_plot:  # pragma: no cover
            p.show()
    # move and rename
    move_to_tmp(fasta, filename_prefix)
    make_all(fasta, filename_prefix, loci="ig")
    rename_dandelion(fasta, filename_prefix, ends_with=out_ex, sub_dir="tmp")
    update_j_multimap(fasta, filename_prefix)


def assign_isotypes(
    fastas: list[Path | str] | Path | str,
    org: Literal["human", "mouse"] = "human",
    evalue: float = 1e4,
    correct_c_call: bool = True,
    correction_dict: dict[str, dict[str, str]] | None = None,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: tuple[float, float] = (4, 4),
    blastdb: Path | str | None = None,
    filename_prefix: list[str] | str | None = None,
    additional_args: list[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fastas : list[str]
        list of paths to fasta files.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    correct_c_call : bool, optional
        whether or not to adjust the c_calls after blast based on provided primers specified in `primer_dict` option.
    correction_dict : dict[str, dict[str, str]] | None, optional
        a nested dictionary contain isotype/c_genes as keys and primer sequences as records to use for correcting
        annotated c_calls. Defaults to a curated dictionary for human sequences if left as none.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plots.
    show_plot : bool, optional
        whether or not to show plots.
    figsize : tuple[float, float], optional
        size of figure.
    blastdb : Path | str | None, optional
        path to blast database. Defaults to `$BLASTDB` environmental variable.
    filename_prefix : list[str] | str | None, optional
        list of prefixes of file names preceding '_contig'. `None` defaults to 'all'.
    additional_args : list[str], optional
        additional arguments to pass to `blastn`.
    """
    fastas, filename_prefix = check_data(fastas, filename_prefix)

    logg.info("Assign isotypes \n")

    for i in range(0, len(fastas)):
        assign_isotype(
            fastas[i],
            org=org,
            evalue=evalue,
            correct_c_call=correct_c_call,
            correction_dict=correction_dict,
            plot=plot,
            save_plot=save_plot,
            show_plot=show_plot,
            figsize=figsize,
            blastdb=blastdb,
            filename_prefix=filename_prefix[i],
            additional_args=additional_args,
        )


def reannotate_genes(
    data: list[Path | str] | Path | str,
    igblast_db: str | None = None,
    germline: str | None = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    extended: bool = True,
    filename_prefix: list[str] | str | None = None,
    flavour: Literal["strict", "original"] = "strict",
    min_j_match: int = 7,
    min_d_match: int = 9,
    v_evalue: float = 1e-4,
    d_evalue: float = 1e-3,
    j_evalue: float = 1e-4,
    reassign_dj: bool = True,
    overwrite: bool = True,
    dust: str | None = "no",
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    additional_args: dict[str, list[str]] = {
        "assigngenes": [],
        "makedb": [],
        "igblastn": [],
        "blastn_j": [],
        "blastn_d": [],
    },
):
    """
    Reannotate cellranger fasta files with igblastn and parses to airr format.

    Parameters
    ----------
    data : list[str]
        list of fasta file locations, or folder name containing fasta files.
        if provided as a single string, it will first be converted to a list;
        this allows for the function to be run on single/multiple samples.
    igblast_db : str | None, optional
        path to igblast database folder. Defaults to `IGDATA` environmental
        variable.
    germline : str | None, optional
        path to germline database folder. Defaults to `GERMLINE` environmental
        variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    loci : Literal["ig", "tr"], optional
        mode for igblastn. 'ig' for BCRs, 'tr' for TCRs.
    extended : bool, optional
        whether or not to transfer additional 10X annotations to output file.
    filename_prefix : list[str] | str | None, optional
        list of prefixes of file names preceding '_contig'. `None` defaults
        to 'all'.
    flavour : Literal["strict", "original"], optional
        Either 'strict' or 'original'. Determines how igblastn should
        be run. Running in 'strict' flavour will add the additional the
        evalue and min_d_match options to the run.
    min_j_match : int, optional
        Minimum D gene nucleotide matches. This controls the threshold for
        D gene detection. You can set the minimal number of required
        consecutive nucleotide matches between the query sequence and the D
        genes based on your own criteria. Note that the matches do not include
        overlapping matches at V-D or D-J junctions.
    min_d_match : int, optional
        Minimum D gene nucleotide matches. This controls the threshold for
        D gene detection. You can set the minimal number of required
        consecutive nucleotide matches between the query sequence and the D
        genes based on your own criteria. Note that the matches do not include
        overlapping matches at V-D or D-J junctions.
    v_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for v gene.
    d_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for d gene.
    j_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for j gene.
    reassign_dj : bool, optional
        whether or not to perform a targetted blastn reassignment for D and J genes.
    overwrite : bool, optional
        whether or not to overwrite the assignment if flavour = 'strict'.
    dust : str | None, optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    db : Literal["imgt", "ogrdb"], optional
        database to use for igblastn. Defaults to 'imgt'.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args : dict[str, list[str]], optional
        additional arguments to pass to `AssignGenes.py`, `MakeDb.py`, `igblastn` and `blastn`.
        This accepts a dictionary with keys as the name of the sub-function (`assigngenes`, `makedb`,
        `igblastn`, `blastn_j` and `blastn_d`) and the records as lists of arguments to pass to the
        relevant scripts/tools.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    data, filename_prefix = check_data(data, filename_prefix)
    filePath = None
    for i in tqdm(
        range(0, len(data)),
        desc="Assigning genes ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        filePath = check_filepath(
            data[i], filename_prefix=filename_prefix[i], ends_with=".fasta"
        )
        if filePath is None:
            if filename_prefix[i] is not None:
                raise FileNotFoundError(
                    "Path to fasta file with filename prefix `{}_contig` is unknown. ".format(
                        filename_prefix[i]
                    )
                    + "Please specify path to fasta file or folder containing fasta file."
                )
            else:
                raise FileNotFoundError(
                    "Path to fasta file is unknown. "
                    + "Please specify path to fasta file or folder containing fasta file."
                )

        logg.info(f"Processing {str(filePath)} \n")
        if flavour == "original":
            assigngenes_igblast(
                filePath,
                igblast_db=igblast_db,
                org=org,
                loci=loci,
                additional_args=additional_args["assigngenes"],
            )
        elif flavour == "strict":
            run_igblastn(
                filePath,
                igblast_db=igblast_db,
                org=org,
                loci=loci,
                evalue=v_evalue,
                min_d_match=min_d_match,
                db=db,
                strain=strain,
                additional_args=additional_args["igblastn"],
            )
        db = "imgt" if flavour == "original" else db
        makedb_igblast(
            filePath,
            org=org,
            germline=germline,
            extended=extended,
            db=db,
            additional_args=additional_args["makedb"],
            loci=loci,
        )
        # block this for now, until I figure out if it's
        # worth it
        if flavour == "strict":
            if reassign_dj:
                assign_DJ(
                    fasta=filePath,
                    org=org,
                    loci=loci,
                    call="j",
                    database=igblast_db,
                    evalue=j_evalue,
                    filename_prefix=filename_prefix,
                    dust=dust,
                    word_size=min_j_match,
                    overwrite=overwrite,
                    db=db,
                    strain=strain,
                    additional_args=additional_args["blastn_j"],
                )
                assign_DJ(
                    fasta=filePath,
                    org=org,
                    loci=loci,
                    call="d",
                    database=igblast_db,
                    evalue=d_evalue,
                    filename_prefix=filename_prefix,
                    dust=dust,
                    word_size=min_d_match,
                    overwrite=overwrite,
                    db=db,
                    strain=strain,
                    additional_args=additional_args["blastn_d"],
                )
                ensure_columns_transferred(
                    fasta=filePath,
                    filename_prefix=filename_prefix,
                )

    if loci == "tr":
        change_file_location(data, filename_prefix)
        if flavour == "strict":
            mask_dj(data, filename_prefix, d_evalue, j_evalue)
        move_to_tmp(data, filename_prefix)
        make_all(data, filename_prefix, loci=loci)
        rename_dandelion(
            data, filename_prefix, ends_with="_igblast_db-pass.tsv"
        )
        update_j_multimap(data, filename_prefix)


def return_pass_fail_filepaths(
    fasta: Path | str,
    filename_prefix: str | None = None,
) -> tuple[Path, Path, Path]:
    """Return necessary file paths for internal use only.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. `None` defaults to 'all'.

    Returns
    -------
    tuple[Path, Path, Path]
        file paths for downstream functions.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    file_path = check_filepath(
        fasta, filename_prefix=filename_prefix, ends_with=".fasta"
    )
    if file_path is None:
        raise FileNotFoundError(
            "Path to fasta file is unknown. Please specify "
            + "path to fasta file or folder containing fasta file."
        )
    # read the original object
    pass_path = (
        file_path.parent / "tmp" / (file_path.stem + "_igblast_db-pass.tsv")
    )
    fail_path = (
        file_path.parent / "tmp" / (file_path.stem + "_igblast_db-fail.tsv")
    )
    return file_path, pass_path, fail_path


def ensure_columns_transferred(
    fasta: str,
    filename_prefix: str | None = None,
):
    """Ensure the additional columns are successfully populated.

    Parameters
    ----------
    fasta : str
        path to fasta file.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. `None` defaults to 'all'.
    """
    filePath, passfile, failfile = return_pass_fail_filepaths(
        fasta, filename_prefix=filename_prefix
    )
    addcols = [
        "_support_igblastn",
        "_score_igblastn",
        "_call_igblastn",
        "_call_blastn",
        "_identity_blastn",
        "_alignment_length_blastn",
        "_number_of_mismatches_blastn",
        "_number_of_gap_openings_blastn",
        "_sequence_start_blastn",
        "_sequence_end_blastn",
        "_germline_start_blastn",
        "_germline_end_blastn",
        "_support_blastn",
        "_score_blastn",
        "_sequence_alignment_blastn",
        "_germline_alignment_blastn",
        "_source",
    ]
    if passfile.is_file():
        db_pass = load_data(passfile)
    else:
        db_pass = None
    if failfile.is_file():
        db_fail = load_data(failfile)
    else:
        db_fail = None
    # load the 10x file
    _10xfile = check_filepath(
        fasta.parent / (fasta.stem + "_annotations.csv"),
        filename_prefix=filename_prefix,
        ends_with="_annotations.csv",
    )
    if _10xfile is not None:
        dat_10x = read_10x_vdj(_10xfile)
    else:
        dat_10x = None
    if db_pass is not None:
        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_pass:
                    db_pass[add_col] = ""
        if dat_10x is not None:
            for col in ["consensus_count", "umi_count"]:
                if all_missing(db_pass[col]):
                    db_pass[col] = pd.Series(dat_10x._data[col])
        db_pass = sanitize_data(db_pass)
        db_pass.to_csv(passfile, sep="\t", index=False)
    if db_fail is not None:
        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_fail:
                    db_fail[add_col] = ""
        if dat_10x is not None:
            for col in ["consensus_count", "umi_count"]:
                if all_missing(db_fail[col]):
                    db_fail[col] = pd.Series(dat_10x._data[col])
        db_fail = sanitize_data(db_fail)
        db_fail.to_csv(failfile, sep="\t", index=False)


def reassign_alleles(
    data: list[str],
    combined_folder: str,
    v_germline: str | None = None,
    germline: str | None = None,
    org: Literal["human", "mouse"] = "human",
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    novel: bool = True,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: tuple[float, float] = (4, 3),
    sample_id_dictionary: dict[str, str] | None = None,
    filename_prefix: list[str] | str | None = None,
    additional_args: dict[str, list[str]] = {
        "tigger": [],
        "creategermlines": [],
    },
):
    """
    Correct allele calls based on a personalized genotype using tigger.

    It uses a subject-specific genotype to correct correct preliminary allele
    assignments of a set of sequences derived from a single subject.

    Parameters
    ----------
    data : list[str]
        list of data folders containing the .tsv files. if provided as a single
        string, it will first be converted to a list; this allows for the
        function to be run on single/multiple samples.
    combined_folder : str
        name of folder for concatenated data file and genotyped files.
    v_germline : str | None, optional
        path to heavy chain v germline fasta. Defaults to IGHV fasta in
        `$GERMLINE` environmental variable.
    germline : str | None, optional
        path to germline database folder. `None` defaults to `GERMLINE` environmental
        variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    db : Literal["imgt", "ogrdb"], optional
        database to use for germline sequences.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    novel : bool, optional
        whether or not to run novel allele discovery during tigger-genotyping.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plot.
    show_plot : bool, optional
        whether or not to show plot.
    figsize : tuple[float, float], optional
        size of figure.
    sample_id_dictionary : dict[str, str] | None, optional
        dictionary for creating a sample_id column in the concatenated file.
    filename_prefix : list[str] | str | None, optional
        list of prefixes of file names preceding '_contig'. `None` defaults to
        'all'.
    additional_args : dict[str, list[str]], optional
        additional arguments to pass to `tigger-genotype.R` and `CreateGermlines.py`.
        This accepts a dictionary with keys as the name of the sub-function (`tigger` or `creategermlines`)
        and the records as lists of arguments to pass to the relevant scripts/tools.

    Raises
    ------
    FileNotFoundError
        if reannotated file is not found.
    """
    fileformat = "blast"
    data, filename_prefix = check_data(data, filename_prefix)

    informat_dict = {
        "changeo": "_igblast_db-pass.tsv",
        "blast": "_igblast_db-pass.tsv",
        "airr": "_igblast_gap.tsv",
    }
    germpass_dict = {
        "changeo": "_igblast_db-pass_germ-pass.tsv",
        "blast": "_igblast_db-pass_germ-pass.tsv",
        "airr": "_igblast_gap_germ-pass.tsv",
    }
    fileformat_dict = {
        "changeo": "_igblast_db-pass_genotyped.tsv",
        "blast": "_igblast_db-pass_genotyped.tsv",
        "airr": "_igblast_gap_genotyped.tsv",
    }
    fileformat_passed_dict = {
        "changeo": "_igblast_db-pass_genotyped_germ-pass.tsv",
        "blast": "_igblast_db-pass_genotyped_germ-pass.tsv",
        "airr": "_igblast_gap_genotyped_germ-pass.tsv",
    }
    inferred_fileformat_dict = {
        "changeo": "_igblast_db-pass_inferredGenotype.txt",
        "blast": "_igblast_db-pass_inferredGenotype.txt",
        "airr": "_igblast_gap_inferredGenotype.txt",
    }
    germline_dict = {
        "changeo": "_igblast_db-pass_genotype.fasta",
        "blast": "_igblast_db-pass_genotype.fasta",
        "airr": "_igblast_gap_genotype.fasta",
    }
    fform_dict = {"blast": "airr", "airr": "airr", "changeo": "changeo"}

    filepathlist_heavy = []
    filepathlist_light = []
    filePath = None
    sampleNames_dict = {}
    filePath_dict = {}
    for i in tqdm(
        range(0, len(data)),
        desc="Processing data file(s) ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with=informat_dict[fileformat],
            sub_dir="tmp",
        )
        if filePath is None:
            raise FileNotFoundError(
                "Path to .tsv file for {} is unknown. ".format(data[i])
                + "Please specify path to reannotated .tsv file or folder "
                + "containing reannotated .tsv file."
            )

        filePath_heavy = filePath.parent / (
            filePath.stem + "_heavy_parse-select.tsv"
        )
        filePath_light = filePath.parent / (
            filePath.stem + "_light_parse-select.tsv"
        )

        if sample_id_dictionary is not None:
            sampleNames_dict[filePath] = sample_id_dictionary[data[i]]
        else:
            sampleNames_dict[filePath] = str(data[i])

        filePath_dict[str(data[i])] = filePath

        # splitting up to heavy chain and light chain files
        parsedb_heavy(filePath)
        parsedb_light(filePath)

        # add to counter
        filepathlist_heavy.append(filePath_heavy)
        filepathlist_light.append(filePath_light)

    # make output directory
    out_dir = Path(str(combined_folder))
    out_dir.mkdir(parents=True, exist_ok=True)
    # concatenate
    if len(filepathlist_heavy) > 1:
        logg.info("Concatenating objects")
        try:
            cmd1 = " ".join(
                [
                    'awk "FNR==1 && NR!=1 { while (/^sequence_id/) getline; } 1 {print}"'
                ]
                + [f for f in filepathlist_heavy]
                + [">"]
                + [
                    str(
                        out_dir
                        / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                    )
                ]
            )
            cmd2 = " ".join(
                [
                    'awk "FNR==1 && NR!=1 { while (/^sequence_id/) getline; } 1 {print}"'
                ]
                + [f for f in filepathlist_light]
                + [">"]
                + [
                    str(
                        out_dir
                        / (out_dir.stem + "_light" + informat_dict[fileformat])
                    )
                ]
            )
            os.system(cmd1)
            os.system(cmd2)
        except:  # pragma: no cover
            fh = open(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
                "w",
            )
            fh.close()
            with open(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
                "a",
            ) as out_file:
                for filenum, filename in enumerate(filepathlist_heavy):
                    with open(filename) as in_file:
                        for line_num, line in enumerate(in_file):
                            if (line_num == 0) and (filenum > 0):
                                continue
                            out_file.write(line)
            fh = open(
                out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
                "w",
            )
            fh.close()
            with open(
                out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
                "a",
            ) as out_file:
                for filenum, filename in enumerate(filepathlist_light):
                    with open(filename) as in_file:
                        skip_next_line = False
                        for line_num, line in enumerate(in_file):
                            if (line_num == 0) and (filenum > 0):
                                continue
                            out_file.write(line)
    else:
        shutil.copyfile(
            Path(filepathlist_heavy[0]),
            out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
        )
        shutil.copyfile(
            Path(filepathlist_light[0]),
            out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
        )

    novel_dict = {True: "YES", False: "NO"}
    logg.info(
        "      Do not worry about ERROR appearing below. There is a check in place to ensure that the script continues to run."
    )
    if novel:
        try:
            logg.info(
                "      Running tigger-genotype with novel allele discovery."
            )
            tigger_genotype(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                ),
                v_germline=v_germline,
                org=org,
                fileformat=fform_dict[fileformat],
                novel_=novel_dict[novel],
                db=db,
                strain=strain,
                additional_args=additional_args["tigger"],
            )
            creategermlines(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + fileformat_dict[fileformat])
                ),
                germline=germline,
                org=org,
                genotyped_fasta=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                ),
                mode="heavy",
                db=db,
                strain=strain,
                additional_args=["--vf", "v_call_genotyped"]
                + additional_args["creategermlines"],
            )
            _ = load_data(
                out_dir
                / (out_dir.stem + "_heavy" + fileformat_passed_dict[fileformat])
            )
        except:
            try:
                logg.info("      Novel allele discovery execution halted.")
                logg.info(
                    "      Attempting to run tigger-genotype without novel allele discovery."
                )
                tigger_genotype(
                    airr_file=str(
                        out_dir
                        / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                    ),
                    v_germline=v_germline,
                    org=org,
                    fileformat=fform_dict[fileformat],
                    novel_=novel_dict[False],
                    db=db,
                    strain=strain,
                    additional_args=additional_args["tigger"],
                )
                creategermlines(
                    airr_file=str(
                        out_dir
                        / (
                            out_dir.stem
                            + "_heavy"
                            + fileformat_dict[fileformat]
                        )
                    ),
                    germline=germline,
                    org=org,
                    genotyped_fasta=str(
                        out_dir
                        / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                    ),
                    mode="heavy",
                    db=db,
                    strain=strain,
                    additional_args=["--vf", "v_call_genotyped"]
                    + additional_args["creategermlines"],
                )
                _ = load_data(
                    out_dir
                    / (
                        out_dir.stem
                        + "_heavy"
                        + fileformat_passed_dict[fileformat]
                    )
                )
            except:
                logg.info(
                    "     Insufficient contigs for running tigger-genotype. Defaulting to original heavy chain v_calls."
                )
                tigger_failed = ""
    else:
        try:
            logg.info(
                "      Running tigger-genotype without novel allele discovery."
            )
            tigger_genotype(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                ),
                v_germline=v_germline,
                org=org,
                fileformat=fform_dict[fileformat],
                novel_=novel_dict[False],
                db=db,
                strain=strain,
                additional_args=additional_args["tigger"],
            )
            creategermlines(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + fileformat_dict[fileformat])
                ),
                germline=germline,
                org=org,
                genotyped_fasta=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                ),
                mode="heavy",
                db=db,
                strain=strain,
                additional_args=["--vf", "v_call_genotyped"]
                + additional_args["creategermlines"],
            )
            _ = load_data(
                str(
                    out_dir
                    / (
                        out_dir.stem
                        + "_heavy"
                        + fileformat_passed_dict[fileformat]
                    )
                )
            )
        except:
            logg.info(
                "      Insufficient contigs for running tigger-genotype. Defaulting to original heavy chain v_calls."
            )
            tigger_failed = ""

    if "tigger_failed" in locals():
        creategermlines(
            airr_file=str(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat])
            ),
            germline=germline,
            org=org,
            genotyped_fasta=None,
            mode="heavy",
            db=db,
            strain=strain,
            additional_args=["--vf", "v_call"]
            + additional_args["creategermlines"],
        )
    creategermlines(
        airr_file=str(
            out_dir / (out_dir.stem + "_light" + informat_dict[fileformat])
        ),
        germline=germline,
        org=org,
        genotyped_fasta=None,
        mode="light",
        db=db,
        strain=strain,
        additional_args=["--vf", "v_call"] + additional_args["creategermlines"],
    )
    if "tigger_failed" in locals():
        try:
            heavy = load_data(
                out_dir / (out_dir.stem + "_heavy" + germpass_dict[fileformat])
            )
        except ValueError:
            # print error message and return
            warnings.warn(
                "Processing has failed for {}. Please check the error message for what went wrong.".format(
                    {
                        str(
                            out_dir
                            / (
                                out_dir.stem
                                + "_heavy"
                                + germpass_dict[fileformat]
                            )
                        )
                    }
                )
            )
            return
        logg.info(
            "      For convenience, entries for heavy chain in `v_call` are copied to `v_call_genotyped`."
        )
        heavy["v_call_genotyped"] = heavy["v_call"]
    else:
        heavy = load_data(
            out_dir
            / (out_dir.stem + "_heavy" + fileformat_passed_dict[fileformat])
        )

    logg.info(
        "      For convenience, entries for light chain `v_call` are copied to `v_call_genotyped`."
    )
    light = load_data(
        out_dir / (out_dir.stem + "_light" + germpass_dict[fileformat])
    )
    light["v_call_genotyped"] = light["v_call"]

    sampledict = {}
    heavy["sample_id"], light["sample_id"] = None, None
    for file in sampleNames_dict.keys():
        dat_f = load_data(file)
        dat_f["sample_id"] = sampleNames_dict[file]
        heavy.update(dat_f[["sample_id"]])
        light.update(dat_f[["sample_id"]])

    dat_ = pd.concat([heavy, light])
    if "cell_id" in dat_.columns:
        dat_.sort_values(by="cell_id", inplace=True)
    else:
        dat_.sort_values(by="sequence_id", inplace=True)

    if plot:
        if "tigger_failed" not in locals():
            logg.info("Returning summary plot")
            inferred_genotype = out_dir / (
                out_dir.stem + "_heavy" + inferred_fileformat_dict[fileformat]
            )
            inf_geno = pd.read_csv(inferred_genotype, sep="\t", dtype="object")
            s2 = set(inf_geno["gene"])
            results = []
            try:
                for samp in list(set(heavy["sample_id"])):
                    res_x = heavy[(heavy["sample_id"] == samp)]
                    V_ = [
                        re.sub("[*][0-9][0-9]", "", v) for v in res_x["v_call"]
                    ]
                    V_g = [
                        re.sub("[*][0-9][0-9]", "", v)
                        for v in res_x["v_call_genotyped"]
                    ]
                    s1 = set(
                        list(
                            ",".join(
                                [",".join(list(set(v.split(",")))) for v in V_]
                            ).split(",")
                        )
                    )
                    setdiff = s1 - s2
                    ambiguous = (
                        ["," in i for i in V_].count(True) / len(V_) * 100,
                        ["," in i for i in V_g].count(True) / len(V_g) * 100,
                    )
                    not_in_genotype = (
                        [i in setdiff for i in V_].count(True) / len(V_) * 100,
                        [i in setdiff for i in V_g].count(True)
                        / len(V_g)
                        * 100,
                    )

                    # FIXED: Create DataFrame correctly
                    # Each tuple represents (before, after) for a metric
                    stats = pd.DataFrame(
                        {
                            "before": [ambiguous[0], not_in_genotype[0]],
                            "after": [ambiguous[1], not_in_genotype[1]],
                        },
                        index=["ambiguous", "not_in_genotype"],
                    )

                    stats.index.set_names(["vgroup"], inplace=True)
                    stats.reset_index(drop=False, inplace=True)
                    stats["sample_id"] = samp
                    results.append(stats)
                results = pd.concat(results)
                ambiguous_table = results[results["vgroup"] == "ambiguous"]
                not_in_genotype_table = results[
                    results["vgroup"] == "not_in_genotype"
                ]
                ambiguous_table.reset_index(inplace=True, drop=True)
                not_in_genotype_table.reset_index(inplace=True, drop=True)
                # melting the dataframe
                ambiguous_table_before = ambiguous_table.drop("after", axis=1)
                ambiguous_table_before.rename(
                    columns={"before": "var"}, inplace=True
                )
                ambiguous_table_before["var_group"] = "before"
                ambiguous_table_after = ambiguous_table.drop("before", axis=1)
                ambiguous_table_after.rename(
                    columns={"after": "var"}, inplace=True
                )
                ambiguous_table_after["var_group"] = "after"
                ambiguous_table = pd.concat(
                    [ambiguous_table_before, ambiguous_table_after]
                )
                not_in_genotype_table_before = not_in_genotype_table.drop(
                    "after", axis=1
                )
                not_in_genotype_table_before.rename(
                    columns={"before": "var"}, inplace=True
                )
                not_in_genotype_table_before["var_group"] = "before"
                not_in_genotype_table_after = not_in_genotype_table.drop(
                    "before", axis=1
                )
                not_in_genotype_table_after.rename(
                    columns={"after": "var"}, inplace=True
                )
                not_in_genotype_table_after["var_group"] = "after"
                not_in_genotype_table = pd.concat(
                    [not_in_genotype_table_before, not_in_genotype_table_after]
                )
                ambiguous_table["var_group"] = ambiguous_table[
                    "var_group"
                ].astype("category")
                not_in_genotype_table["var_group"] = not_in_genotype_table[
                    "var_group"
                ].astype("category")
                ambiguous_table["var_group"] = ambiguous_table[
                    "var_group"
                ].cat.reorder_categories(["before", "after"])
                not_in_genotype_table["var_group"] = not_in_genotype_table[
                    "var_group"
                ].cat.reorder_categories(["before", "after"])

                options.figure_size = figsize
                final_table = pd.concat(
                    [ambiguous_table, not_in_genotype_table]
                )
                p = (
                    ggplot(
                        final_table,
                        aes(x="sample_id", y="var", fill="var_group"),
                    )
                    + coord_flip()
                    + theme_classic()
                    + xlab("sample_id")
                    + ylab("% allele calls")
                    + ggtitle("Genotype reassignment with TIgGER")
                    + geom_bar(stat="identity")
                    + facet_grid("~" + "vgroup", scales="free_y")
                    + scale_fill_manual(values=("#86bcb6", "#F28e2b"))
                    + theme(legend_title=element_blank())
                )
                if save_plot:
                    savefile = str(
                        out_dir / (out_dir.stem + "_reassign_alleles.pdf")
                    )
                    save_as_pdf_pages([p], filename=savefile, verbose=False)
                if show_plot:
                    p.show()
            except:
                logg.info("Error in plotting encountered. Skipping.")
                pass
        else:
            pass
    sleep(0.5)
    # if split_write_out:
    if "tigger_failed" in locals():
        logg.info(
            "Although tigger-genotype was not run successfully, file will still be saved with `_genotyped.tsv`"
            "extension for convenience."
        )
    for s in tqdm(
        data,
        desc="Writing out to individual folders ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        if sample_id_dictionary is not None:
            out_file = dat_[dat_["sample_id"] == sample_id_dictionary[s]]
        else:
            out_file = dat_[dat_["sample_id"] == s]
        outfilepath = filePath_dict[s]
        write_airr(
            out_file, outfilepath.parent / (outfilepath.stem + "_genotyped.tsv")
        )


def create_germlines(
    vdj_data: Dandelion | pd.DataFrame | str,
    # vdj_data: DandelionPolars | pd.DataFrame | str,
    germline: str | None = None,
    org: Literal["human", "mouse"] = "human",
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    genotyped_fasta: str | None = None,
    additional_args: list[str] = [],
    save: str | None = None,
) -> Dandelion:
    # ) -> DandelionPolars:
    """
    Run CreateGermlines.py to reconstruct the germline V(D)J sequence.

    Parameters
    ----------
    vdj_data : Dandelion | pd.DataFrame | str
        Dandelion object, pandas DataFrame in changeo/airr format, or file path to changeo/airr
        file after clones have been determined.
    germline : str | None, optional
        path to germline database folder. `None` defaults to  environmental variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    genotyped_fasta : str | None, optional
        location to corrected v genotyped fasta file.
    additional_args : list[str], optional
        additional arguments to pass to `CreateGermlines.py.`
    save : str | None, optional
        if provided, saves to specified file path.

    Returns
    -------
    Dandelion
        Dandelion object with `.germlines` slot populated.
    """
    start = logg.info("Reconstructing germline sequences")
    if not isinstance(vdj_data, Dandelion):
        # if not isinstance(vdj_data, DandelionPolars):
        tmpfile = (
            Path(vdj_data)
            if os.path.isfile(vdj_data)
            else Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        )
        if isinstance(vdj_data, pd.DataFrame):
            # _write_airr(data=vdj_data.germline, save=tmpfile)
            write_airr(data=vdj_data.germline, save=tmpfile)
        creategermlines(
            airr_file=tmpfile,
            germline=germline,
            org=org,
            genotyped_fasta=genotyped_fasta,
            db=db,
            strain=strain,
            additional_args=additional_args,
        )
    else:
        tmppath = Path(tempfile.TemporaryDirectory().name)
        tmppath.mkdir(parents=True, exist_ok=True)
        tmpfile = tmppath / "tmp.tsv"
        vdj_data.write_airr(filename=tmpfile)
        if len(vdj_data.germline) > 0:
            tmpgmlfile = tmppath / "germ.fasta"
            write_fasta(fasta_dict=vdj_data.germline, out_fasta=tmpgmlfile)
            creategermlines(
                airr_file=tmpfile,
                germline=tmpgmlfile,
                org=org,
                db=db,
                strain=strain,
                additional_args=additional_args,
            )
        else:
            creategermlines(
                airr_file=tmpfile,
                germline=germline,
                org=org,
                genotyped_fasta=genotyped_fasta,
                db=db,
                strain=strain,
                additional_args=additional_args,
            )
    # return as Dandelion object
    germpass_outfile = tmpfile.parent / (tmpfile.stem + "_germ-pass.tsv")
    if isinstance(vdj_data, Dandelion):
        # if isinstance(vdj_data, DandelionPolars):
        vdj_data.__init__(
            data=germpass_outfile,
            metadata=vdj_data._metadata,
            germline=vdj_data.germline,
            layout=vdj_data.layout,
            graph=vdj_data.graph,
            verbose=False,
        )
        out_vdj = vdj_data.copy()
    else:
        out_vdj = Dandelion(germpass_outfile, verbose=False)
        # out_vdj = DandelionPolars(germpass_outfile, verbose=False)
        out_vdj.store_germline_reference(
            corrected=genotyped_fasta, germline=germline, org=org
        )
    if save is not None:
        shutil.move(germpass_outfile, save)
    logg.info(
        " finished",
        time=start,
        deep=("Returning Dandelion object: \n"),
    )
    return out_vdj


def run_igblastn(
    fasta: Path | str,
    igblast_db: Path | str | None = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    evalue: float = 1e-4,
    min_d_match: int = 9,
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    additional_args: list[str] = [],
):
    """
    Reannotate with IgBLASTn.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file for reannotation.
    igblast_db : Path | str | None, optional
        path to igblast database.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    loci : Literal["ig", "tr"], optional
        `ig` or `tr` mode for running igblastn.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    min_d_match : int, optional
        minimum D nucleotide match.
    db : Literal["imgt", "ogrdb"], optional
        database to use for germline sequences.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args: list[str], optional
        additional arguments to pass to `igblastn`.
    """
    env, igdb, fasta = set_igblast_env(igblast_db=igblast_db, input_file=fasta)
    outfolder = fasta.parent / "tmp"
    outfolder.mkdir(parents=True, exist_ok=True)
    informat_dict = {"blast": "_igblast.fmt7", "airr": "_igblast.tsv"}
    if db == "ogrdb":
        _strain = "_" + strain if strain is not None else ""
        aux = "_gl_ogrdb.aux"
    else:
        _strain = ""
        aux = "_gl.aux"
    loci_type = {"ig": "Ig", "tr": "TCR"}
    outformat = {"blast": "7 std qseq sseq btop", "airr": "19"}

    dbpath = igdb / "database"
    db_org_loci = db + "_" + org + _strain + "_" + loci + "_"
    vpath = dbpath / (db_org_loci + "v")
    if strain in NO_DS:
        dpath = dbpath / (db + "_" + org + "_" + loci + "_" + "d")
    else:
        dpath = dbpath / (db_org_loci + "d")
    jpath = dbpath / (db_org_loci + "j")
    cpath = dbpath / ("imgt_" + org + "_" + loci + "_" + "c")  # only imgt
    auxpath = igdb / "optional_file" / (org + aux)
    for fileformat in ["blast", "airr"]:
        outfile = str(fasta.stem + informat_dict[fileformat])
        if loci == "tr":
            cmd = [
                "igblastn",
                "-germline_db_V",
                str(vpath),
                "-germline_db_D",
                str(dpath),
                "-germline_db_J",
                str(jpath),
                "-auxiliary_data",
                str(auxpath),
                "-domain_system",
                "imgt",
                "-ig_seqtype",
                loci_type[loci],
                "-organism",
                org,
                "-outfmt",
                outformat[fileformat],
                "-query",
                str(fasta),
                "-out",
                str(outfolder / outfile),
                "-evalue",
                str(evalue),
                "-min_D_match",
                str(min_d_match),
                "-D_penalty",
                str(-4),
                "-c_region_db",
                str(cpath),
            ]
        else:
            cmd = [
                "igblastn",
                "-germline_db_V",
                str(vpath),
                "-germline_db_D",
                str(dpath),
                "-germline_db_J",
                str(jpath),
                "-auxiliary_data",
                str(auxpath),
                "-domain_system",
                "imgt",
                "-ig_seqtype",
                loci_type[loci],
                "-organism",
                org,
                "-outfmt",
                outformat[fileformat],
                "-query",
                str(fasta),
                "-out",
                str(outfolder / outfile),
                "-evalue",
                str(evalue),
                "-min_D_match",
                str(min_d_match),
                "-c_region_db",
                str(cpath),
            ]
        cmd += additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def assign_DJ(
    fasta: Path | str,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "tr",
    call: Literal["d", "j"] = "j",
    database: str | None = None,
    evalue: float = 1e-4,
    max_hsps: int = 10,
    dust: str | None = None,
    word_size: int | None = None,
    outfmt: str = (
        "6 qseqid sseqid pident length mismatch gapopen "
        + "qstart qend sstart send evalue bitscore qseq sseq"
    ),
    filename_prefix: str | None = None,
    overwrite: bool = False,
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    additional_args: list[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    loci : Literal["ig", "tr"], optional
        locus. 'ig' or 'tr',
    call : Literal["d", "j"], optional
        Either 'd' of 'j' gene.
    database : str | None, optional
        path to database.
        Defaults to `IGDATA` environmental variable if v/d/j_call.
        Defaults to `BLASTDB` environmental variable if c_call.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    max_hsps : int, optional
        Maximum number of HSPs (alignments) to keep for any single query-subject pair.
        The HSPs shown will be the best as judged by expect value. This number should
        be an integer that is one or greater. Setting it to one will show only the best
        HSP for every query-subject pair. Only affects the output file in the tmp folder.
    dust : str | None, optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    word_size : int | None, optional
        Word size for wordfinder algorithm (length of best perfect match).
        Must be >=4. `None` defaults to 4.
    outfmt : str, optional
        specification of output format for blast.
    filename_prefix : str | None, optional
        prefix of file name preceding '_contig'. `None` defaults to 'all'.
    overwrite : bool, optional
        whether or not to overwrite the assignments.
    db : Literal["imgt", "ogrdb"], optional
        database to use for germline sequences.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args: list[str], optional
        additional arguments to pass to `blastn`.
    """
    # main function from here
    file_path, passfile, failfile = return_pass_fail_filepaths(
        fasta, filename_prefix=filename_prefix
    )

    # run blast
    blast_out = run_blastn(
        fasta=file_path,
        database=database,
        org=org,
        loci=loci,
        call=call,
        max_hsps=max_hsps,
        evalue=evalue,
        outfmt=outfmt,
        dust=dust,
        word_size=word_size,
        db=db,
        strain=strain,
        additional_args=additional_args,
    )

    transfer_assignment(
        passfile=passfile,
        failfile=failfile,
        blast_result=blast_out.drop_duplicates(
            subset="sequence_id", keep="first"
        ),
        call=call,
        overwrite=overwrite,
    )


def run_blastn(
    fasta: Path | str,
    database: str | None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    call: Literal["v", "d", "j", "c"] = "c",
    max_hsps: int = 10,
    evalue: float = 1e-4,
    outfmt: str = (
        "6 qseqid sseqid pident length mismatch gapopen "
        + "qstart qend sstart send evalue bitscore qseq sseq"
    ),
    dust: str | None = None,
    word_size: int | None = None,
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: (
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
        | None
    ) = None,
    additional_args: list[str] = [],
) -> pd.DataFrame:
    """
    Annotate contigs using blastn.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file.
    database : str | None
        path to database.
        Defaults to `IGDATA` environmental variable if v/d/j_call.
        Defaults to `BLASTDB` environmental variable if c_call.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    loci : Literal["ig", "tr"], optional
        locus. 'ig' or 'tr',
    call : Literal["v", "d", "j", "c"], optional
        Either 'v', 'd', 'j' or 'c' gene.
    max_hsps : int, optional
        Maximum number of HSPs (alignments) to keep for any single query-subject pair.
        The HSPs shown will be the best as judged by expect value. This number should
        be an integer that is one or greater. Setting it to one will show only the best
        HSP for every query-subject pair. Only affects the output file in the tmp folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    outfmt : str, optional
        blastn output format.
    dust : str | None, optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    word_size : int | None, optional
        Word size for wordfinder algorithm (length of best perfect match).
        Must be >=4. `None` defaults to 4.
    db : Literal["imgt", "ogrdb"], optional
        database to use for germline sequences.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args: list[str], optional
        additional arguments to pass to `blastn`.

    Returns
    -------
    pd.DataFrame
        reannotated information after blastn.
    """
    if call != "c":
        env, bdb, fasta = set_igblast_env(igblast_db=database, input_file=fasta)
        if db == "ogrdb":
            _strain = "_" + strain if strain is not None else ""
            bdb = (
                bdb
                / "database"
                / (db + "_" + org + _strain + "_" + loci + "_" + call)
            )
        else:
            bdb = bdb / "database" / (db + "_" + org + "_" + loci + "_" + call)
    else:
        env, bdb, fasta = set_blast_env(blast_db=database, input_file=fasta)
        bdb = bdb / org / (org + "_BCR_C.fasta") if database is None else bdb
        # else:
        #     if not bdb.stem.endswith("_" + loci + "_" + call):
        #         bdb = (
        #             bdb
        #             / "database"
        #             / (db + "_" + org + "_" + loci + "_" + call)
        #         )
    cmd = [
        "blastn",
        "-db",
        str(bdb),
        "-evalue",
        str(evalue),
        "-max_hsps",
        str(max_hsps),
        "-outfmt",
        outfmt,
        "-query",
        str(fasta),
    ]
    if dust is not None:
        cmd = cmd + ["-dust", str(dust)]
    if word_size is not None:
        cmd = cmd + ["-word_size", str(word_size)]
    cmd = cmd + additional_args
    blast_out = fasta.parent / "tmp" / (fasta.stem + "_" + call + "_blast.tsv")
    logg.info("Running command: %s\n" % (" ".join(cmd)))
    with open(blast_out, "w") as out:
        run(cmd, stdout=out, env=env)
    try:
        dat = pd.read_csv(blast_out, sep="\t", header=None)
        dat.columns = [
            "sequence_id",
            call + "_call",
            call + "_identity",
            call + "_alignment_length",
            call + "_number_of_mismatches",
            call + "_number_of_gap_openings",
            call + "_sequence_start",
            call + "_sequence_end",
            call + "_germline_start",
            call + "_germline_end",
            call + "_support",
            call + "_score",
            call + "_sequence_alignment",
            call + "_germline_alignment",
        ]
    except pd.errors.EmptyDataError:
        dat = pd.DataFrame(
            columns=[
                "sequence_id",
                call + "_call",
                call + "_identity",
                call + "_alignment_length",
                call + "_number_of_mismatches",
                call + "_number_of_gap_openings",
                call + "_sequence_start",
                call + "_sequence_end",
                call + "_germline_start",
                call + "_germline_end",
                call + "_support",
                call + "_score",
                call + "_sequence_alignment",
                call + "_germline_alignment",
            ]
        )
    write_blastn(data=dat, save=blast_out)
    dat = load_data(dat)
    return dat


def transfer_assignment(
    passfile: str,
    failfile: str,
    blast_result: pd.DataFrame,
    call: Literal["v", "d", "j", "c"] = "c",
    overwrite: bool = False,
):
    """Update gene calls with blastn results.

    Parameters
    ----------
    passfile : str
        path to db-pass.tsv file.
    failfile : str
        path to db-fail.tsv file.
    blast_result : pd.DataFrame
        path to blastn results file.
    call : Literal["v", "d", "j", "c"], optional
        which gene call.
    overwrite : bool, optional
        whether or not to overwrite.
    """
    if os.path.isfile(passfile):
        db_pass = load_data(passfile)
    else:
        db_pass = None
    if os.path.isfile(failfile):
        db_fail = load_data(failfile)
        # should be pretty safe to fill this in
        db_fail["vj_in_frame"] = db_fail["vj_in_frame"].fillna(value="F")
        db_fail["productive"] = db_fail["productive"].fillna(value="F")
        db_fail["c_call"] = db_fail["c_call"].fillna(value="")
        db_fail["v_call"] = db_fail["v_call"].fillna(value="")
        db_fail["d_call"] = db_fail["d_call"].fillna(value="")
        db_fail["j_call"] = db_fail["j_call"].fillna(value="")
        db_fail["locus"] = db_fail["locus"].fillna(value="")
        for i, row in db_fail.iterrows():
            if not present(row.locus):
                calls = list(
                    {
                        row.v_call[:3],
                        row.d_call[:3],
                        row.j_call[:3],
                        row.c_call[:3],
                    }
                )
                locus = "".join([c for c in calls if present(c)])
                if len(locus) == 3:
                    db_fail.at[i, "locus"] = locus
    else:
        db_fail = None
    if blast_result.shape[0] < 1:
        blast_result = None

    if blast_result is not None:
        if db_pass is not None:
            if call + "_support" in db_pass:
                db_pass_evalues = dict(db_pass[call + "_support"])
            if call + "_score" in db_pass:
                db_pass_scores = dict(db_pass[call + "_score"])
            db_pass[call + "_call"] = db_pass[call + "_call"].fillna(value="")
            db_pass_call = dict(db_pass[call + "_call"])
            if call + "_support" in db_pass:
                db_pass[call + "_support_igblastn"] = pd.Series(db_pass_evalues)
            if call + "_score" in db_pass:
                db_pass[call + "_score_igblastn"] = pd.Series(db_pass_scores)
            db_pass[call + "_call_igblastn"] = pd.Series(db_pass_call)
            db_pass[call + "_call_igblastn"] = db_pass[
                call + "_call_igblastn"
            ].fillna(value="")
            for col in blast_result:
                if col not in ["sequence_id", "cell_id"]:
                    db_pass[col + "_blastn"] = pd.Series(blast_result[col])
                    if col in [
                        call + "_call",
                        call + "_sequence_alignment",
                        call + "_germline_alignment",
                    ]:
                        db_pass[col + "_blastn"] = db_pass[
                            col + "_blastn"
                        ].fillna(value="")
            db_pass[call + "_source"] = ""
            if overwrite:
                for i in db_pass["sequence_id"]:
                    vend = db_pass.loc[i, "v_sequence_end"]
                    if not present(vend):
                        vend_ = 0
                    else:
                        vend_ = vend
                    jstart = db_pass.loc[i, "j_sequence_start"]
                    if not present(jstart):
                        jstart_ = 1000
                    else:
                        jstart_ = jstart
                    callstart = db_pass.loc[i, call + "_sequence_start_blastn"]
                    callend = db_pass.loc[i, call + "_sequence_end_blastn"]
                    if (callstart >= vend_) and (callend <= jstart_):
                        if call + "_support_igblastn" in db_pass:
                            eval1 = db_pass.loc[i, call + "_support_igblastn"]
                        else:
                            eval1 = 1
                        eval2 = db_pass.loc[i, call + "_support_blastn"]
                        if (
                            db_pass.loc[i, call + "_call_igblastn"]
                            != db_pass.loc[i, call + "_call_blastn"]
                        ):
                            if call + "_call_10x" in db_pass:
                                if (
                                    re.sub(
                                        "[*][0-9][0-9]",
                                        "",
                                        db_pass.loc[i, call + "_call_blastn"],
                                    )
                                    != db_pass.loc[i, call + "_call_10x"]
                                ):
                                    if present(eval1):
                                        if eval1 > eval2:
                                            db_pass.at[i, call + "_call"] = (
                                                db_pass.at[
                                                    i, call + "_call_blastn"
                                                ]
                                            )
                                            db_pass.at[
                                                i, call + "_sequence_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_end"
                                            ] = db_pass.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_end"
                                            ] = db_pass.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_pass.at[i, call + "_source"] = (
                                                "blastn"
                                            )
                                    else:
                                        if present(eval2):
                                            db_pass.at[i, call + "_call"] = (
                                                db_pass.at[
                                                    i, call + "_call_blastn"
                                                ]
                                            )
                                            db_pass.at[
                                                i, call + "_sequence_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_end"
                                            ] = db_pass.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_end"
                                            ] = db_pass.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_pass.at[i, call + "_source"] = (
                                                "blastn"
                                            )
                                else:
                                    db_pass.at[i, call + "_source"] = "10x"
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    if present(db_pass.loc[i, "junction_10x"]):
                                        if present(db_pass.loc[i, "junction"]):
                                            if (
                                                db_pass.loc[i, "junction"]
                                                != db_pass.loc[
                                                    i, "junction_10x"
                                                ]
                                            ):
                                                db_pass.at[i, "junction"] = (
                                                    db_pass.at[
                                                        i, "junction_10x"
                                                    ]
                                                )
                                                db_pass.at[i, "junction_aa"] = (
                                                    db_pass.at[
                                                        i, "junction_10x_aa"
                                                    ]
                                                )
                        else:
                            if present(eval1):
                                if eval1 > eval2:
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_pass.at[i, call + "_sequence_start"] = (
                                        db_pass.at[
                                            i, call + "_sequence_start_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_sequence_end"] = (
                                        db_pass.at[
                                            i, call + "_sequence_end_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_germline_start"] = (
                                        db_pass.at[
                                            i, call + "_germline_start_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_germline_end"] = (
                                        db_pass.at[
                                            i, call + "_germline_end_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_source"] = "blastn"
                            else:
                                if present(eval2):
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_pass.at[i, call + "_sequence_start"] = (
                                        db_pass.at[
                                            i, call + "_sequence_start_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_sequence_end"] = (
                                        db_pass.at[
                                            i, call + "_sequence_end_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_germline_start"] = (
                                        db_pass.at[
                                            i, call + "_germline_start_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_germline_end"] = (
                                        db_pass.at[
                                            i, call + "_germline_end_blastn"
                                        ]
                                    )
                                    db_pass.at[i, call + "_source"] = "blastn"

                vend = db_pass["v_sequence_end"]
                dstart = db_pass["d_sequence_start"]
                dend = db_pass["d_sequence_end"]
                jstart = db_pass["j_sequence_start"]

                np1 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (
                            (d - v) - 1
                            if pd.notnull(v) and pd.notnull(d)
                            else np.nan
                        )
                        for v, d in zip(vend, dstart)
                    ]
                ]
                np2 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (
                            (j - d) - 1
                            if pd.notnull(j) and pd.notnull(d)
                            else np.nan
                        )
                        for d, j in zip(dend, jstart)
                    ]
                ]

                db_pass["np1_length"] = np1
                db_pass["np2_length"] = np2

                for i in db_pass["sequence_id"]:
                    if not present(db_pass.loc[i, "np1_length"]):
                        vend = db_pass.loc[i, "v_sequence_end"]
                        if present(vend):
                            jstart = db_pass.loc[i, "j_sequence_start"]
                            if present(jstart):
                                np1l = (jstart - vend) - 1
                                if np1l >= 0:
                                    db_pass.loc[i, "np1_length"] = np1l
            # fill in blanks
            db_pass = sanitize_data(db_pass)
            db_pass.to_csv(passfile, sep="\t", index=False)

        if db_fail is not None:
            if call + "_support" in db_fail:
                db_fail_evalues = dict(db_fail[call + "_support"])
            if call + "_score" in db_fail:
                db_fail_scores = dict(db_fail[call + "_score"])
            db_fail[call + "_call"] = db_fail[call + "_call"].fillna(value="")
            db_fail_call = dict(db_fail[call + "_call"])
            if call + "_support" in db_fail:
                db_fail[call + "_support_igblastn"] = pd.Series(db_fail_evalues)
            if call + "_score" in db_fail:
                db_fail[call + "_score_igblastn"] = pd.Series(db_fail_scores)
            db_fail[call + "_call_igblastn"] = pd.Series(db_fail_call)
            db_fail[call + "_call_igblastn"] = db_fail[
                call + "_call_igblastn"
            ].fillna(value="")
            for col in blast_result:
                if col not in ["sequence_id", "cell_id"]:
                    db_fail[col + "_blastn"] = pd.Series(blast_result[col])
                    if col in [
                        call + "_call",
                        call + "_sequence_alignment",
                        call + "_germline_alignment",
                    ]:
                        db_fail[col + "_blastn"] = db_fail[
                            col + "_blastn"
                        ].fillna(value="")
            db_fail[call + "_source"] = ""
            if overwrite:
                for i in db_fail["sequence_id"]:
                    vend = db_fail.loc[i, "v_sequence_end"]
                    if not present(vend):
                        vend_ = 0
                    else:
                        vend_ = vend
                    jstart = db_fail.loc[i, "j_sequence_start"]
                    if not present(jstart):
                        jstart_ = 1000
                    else:
                        jstart_ = jstart
                    callstart = db_fail.loc[i, call + "_sequence_start_blastn"]
                    callend = db_fail.loc[i, call + "_sequence_end_blastn"]
                    if (callstart >= vend_) and (callend <= jstart_):
                        if call + "_support_igblastn" in db_fail:
                            eval1 = db_fail.loc[i, call + "_support_igblastn"]
                        else:
                            eval1 = 1
                        eval2 = db_fail.loc[i, call + "_support_blastn"]
                        if (
                            db_fail.loc[i, call + "_call_igblastn"]
                            != db_fail.loc[i, call + "_call_blastn"]
                        ):
                            if call + "_call_10x" in db_fail:
                                if (
                                    re.sub(
                                        "[*][0-9][0-9]",
                                        "",
                                        db_fail.loc[i, call + "_call_blastn"],
                                    )
                                    != db_fail.loc[i, call + "_call_10x"]
                                ):
                                    if present(eval1):
                                        if eval1 > eval2:
                                            db_fail.at[i, call + "_call"] = (
                                                db_fail.at[
                                                    i, call + "_call_blastn"
                                                ]
                                            )
                                            db_fail.at[
                                                i, call + "_sequence_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_end"
                                            ] = db_fail.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_end"
                                            ] = db_fail.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_fail.at[i, call + "_source"] = (
                                                "blastn"
                                            )
                                    else:
                                        if present(eval2):
                                            db_fail.at[i, call + "_call"] = (
                                                db_fail.at[
                                                    i, call + "_call_blastn"
                                                ]
                                            )
                                            db_fail.at[
                                                i, call + "_sequence_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_end"
                                            ] = db_fail.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_end"
                                            ] = db_fail.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_fail.at[i, call + "_source"] = (
                                                "blastn"
                                            )
                                else:
                                    db_fail.at[i, call + "_source"] = "10x"
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    if present(db_fail.loc[i, "junction_10x"]):
                                        if present(db_fail.loc[i, "junction"]):
                                            if (
                                                db_fail.loc[i, "junction"]
                                                != db_fail.loc[
                                                    i, "junction_10x"
                                                ]
                                            ):
                                                db_fail.at[i, "junction"] = (
                                                    db_fail.at[
                                                        i, "junction_10x"
                                                    ]
                                                )
                                                db_fail.at[i, "junction_aa"] = (
                                                    db_fail.at[
                                                        i, "junction_10x_aa"
                                                    ]
                                                )
                        else:
                            if present(eval1):
                                if eval1 > eval2:
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_fail.at[i, call + "_sequence_start"] = (
                                        db_fail.at[
                                            i, call + "_sequence_start_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_sequence_end"] = (
                                        db_fail.at[
                                            i, call + "_sequence_end_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_germline_start"] = (
                                        db_fail.at[
                                            i, call + "_germline_start_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_germline_end"] = (
                                        db_fail.at[
                                            i, call + "_germline_end_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_source"] = "blastn"
                            else:
                                if present(eval2):
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_fail.at[i, call + "_sequence_start"] = (
                                        db_fail.at[
                                            i, call + "_sequence_start_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_sequence_end"] = (
                                        db_fail.at[
                                            i, call + "_sequence_end_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_germline_start"] = (
                                        db_fail.at[
                                            i, call + "_germline_start_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_germline_end"] = (
                                        db_fail.at[
                                            i, call + "_germline_end_blastn"
                                        ]
                                    )
                                    db_fail.at[i, call + "_source"] = "blastn"

                vend = db_fail["v_sequence_end"]
                dstart = db_fail["d_sequence_start"]
                dend = db_fail["d_sequence_end"]
                jstart = db_fail["j_sequence_start"]

                np1 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (
                            (d - v) - 1
                            if pd.notnull(v) and pd.notnull(d)
                            else np.nan
                        )
                        for v, d in zip(vend, dstart)
                    ]
                ]
                np2 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (
                            (j - d) - 1
                            if pd.notnull(j) and pd.notnull(d)
                            else np.nan
                        )
                        for d, j in zip(dend, jstart)
                    ]
                ]
                db_fail["np1_length"] = np1
                db_fail["np2_length"] = np2

                # rescue the d blanks
                for i in db_fail["sequence_id"]:
                    if not present(db_fail.loc[i, "np1_length"]):
                        vend = db_fail.loc[i, "v_sequence_end"]
                        if present(vend):
                            jstart = db_fail.loc[i, "j_sequence_start"]
                            if present(jstart):
                                np1l = (jstart - vend) - 1
                                if np1l >= 0:
                                    db_fail.loc[i, "np1_length"] = np1l

            # fill in blanks
            db_fail = sanitize_data(db_fail)
            db_fail.to_csv(failfile, sep="\t", index=False)


def check_contigs(
    data: Dandelion | pd.DataFrame | str,
    # data: DandelionPolars | pd.DataFrame | str,
    adata: AnnData | None = None,
    productive_only: bool = True,
    library_type: Literal["ig", "tr-ab", "tr-gd"] | None = None,
    umi_foldchange_cutoff: int = 2,
    consensus_foldchange_cutoff: int = 5,
    ntop_vdj: int = 1,
    ntop_vj: int = 2,
    allow_exceptions: bool = True,
    filter_missing: bool = True,
    filter_extra: bool = True,
    filter_ambiguous: bool = False,
    save: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> tuple[Dandelion, AnnData]:
    # ) -> tuple[DandelionPolars, AnnData]:
    """
    Check contigs for whether they can be considered as ambiguous or not.

    Returns an `ambiguous` column with boolean T/F in the data. If the `sequence_alignment` is an exact match between
    contigs, the contigs will be merged into the one with the highest umi count, summing the umi/duplicate count. After
    this check, if there are still multiple contigs, cells with multiple contigs checked for whether there is a clear
    dominance in terms of UMI count resulting in two scenarios: 1) if true, all other contigs will be flagged as
    ambiguous; 2) if false, all contigs will be flagged as ambiguous. This is repeated for each cell, for their
    productive and non-productive VDJ and VJ contigs separately. Dominance is assessed by whether or not the umi counts
    demonstrate a > umi_foldchange_cutoff. There are some exceptions: 1) IgM and IgD are allowed to co-exist in the same
    B cell if no other isotypes are detected; 2) TRD and TRB contigs are allowed in the same cell because rearrangement
    of TRB and TRD loci happens at the same time during development and TRD variable region genes exhibits allelic
    inclusion. Thus this can potentially result in some situations where T cells expressing productive TRA-TRB chains
    can also express productive TRD chains. This can be toggled by `allow_exceptions` argument.

    Default behvaiour is to only consider productive contigs and remove all non-productive before checking, toggled by
    `productive_only` argument.

    If library_type is provided, it will remove all contigs that do not belong to the related loci. The rationale is
    that the choice of the library type should mean that the primers used would most likely amplify those related
    sequences and if there's any unexpected loci, they likely represent artifacts and shouldn't be analysed.

    If an `adata` object is provided, contigs with no corresponding cell barcode in the AnnData object is filtered in
    the output if filter_missing is True.

    Parameters
    ----------
    data : Dandelion | pd.DataFrame | str
        V(D)J AIRR data to check. Can be Dandelion, pandas DataFrame and file path to AIRR `.tsv` file.
    adata : AnnData | None, optional
        AnnData object to filter. If not provided, it will assume to keep all cells in the airr table and just return a
        Dandelion object.
    productive_only : bool, optional
        whether or not to retain only productive contigs.
    library_type : Literal["ig", "tr-ab", "tr-gd"] | None, optional
        if specified, it will first filter based on the expected type of contigs:
            `ig`:
                IGH, IGK, IGL
            `tr-ab`:
                TRA, TRB
            `tr-gd`:
                TRG, TRD
    umi_foldchange_cutoff : int, optional
        related to minimum fold change of UMI count, required to rescue contigs/barcode otherwise they will be marked as extra/ambiguous.
    consensus_foldchange_cutoff : int, optional
        related to minimum fold change of consensus count, required to rescue contigs/barcode otherwise they will be marked as extra/ambiguous.
    ntop_vdj : int, optional
        number of top VDJ contigs to consider for dominance check.
    ntop_vj : int, optional
        number of top VJ contigs to consider for dominance check.
    allow_exceptions : bool, optional
        whether or not to allow exceptions for certain loci.
    filter_missing : bool, optional
        cells in V(D)J data not found in AnnData object will removed from the dandelion object.
    filter_extra : bool, optional
        whether or not to remove contigs that are marked as extra.
    filter_ambiguous : bool, optional
        whether or not to remove contigs that are marked as ambiguous. This step is only for cleaning up the `.data` slot as the ambiguous contigs are not passed to the `.metadata` slot anyway.
    save : str | None, optional
        Only used if a pandas data frame or dandelion object is provided. Specifying will save the formatted vdj table
        with a `_checked.tsv` suffix extension.
    verbose : bool, optional
        whether to print progress when marking contigs.
    **kwargs
        additional kwargs passed to `dandelion.utilities._core.Dandelion`.

    Returns
    -------
    tuple[Dandelion, AnnData]
        checked dandelion V(D)J object and AnnData object.

    Raises
    ------
    IndexError
        if no contigs passed filtering.
    ValueError
        if save file name is not suitable.
    """
    start = logg.info("Filtering contigs")
    if isinstance(data, Dandelion):
        dat_ = load_data(data._data)
    # if isinstance(data, DandelionPolars):
    # dat_ = load_polars(data._data, as_pandas=True)
    else:
        dat_ = load_data(data)
        # dat_ = load_polars(data, as_pandas=True)
    # ensure that "unknown" are switched to blanks
    dat_.replace("unknown", "", inplace=True)

    if library_type is not None:
        acceptable = lib_type(library_type)
    else:
        if isinstance(data, Dandelion):
            # if isinstance(data, DandelionPolars):
            if data.library_type is not None:
                acceptable = lib_type(data.library_type)
            else:
                acceptable = None
        else:
            acceptable = None

    if productive_only:
        dat = dat_[dat_["productive"].isin(TRUES)].copy()
    else:
        dat = dat_.copy()

    if acceptable is not None:
        dat = dat[dat.locus.isin(acceptable)].copy()

    barcode = list(set(dat.cell_id))

    if adata is not None:
        adata_provided = True
        adata_ = adata.copy()
        contig_check = pd.DataFrame(index=adata_.obs_names)
        bc_ = {}
        for b in barcode:
            bc_.update({b: "True"})
        contig_check["has_contig"] = pd.Series(bc_)
        contig_check["has_contig"] = contig_check["has_contig"].replace(
            np.nan, "No_contig"
        )
        adata_.obs["has_contig"] = pd.Series(contig_check["has_contig"])
    else:
        adata_provided = False
        obs = pd.DataFrame(index=barcode)
        adata_ = ad.AnnData(obs=obs)
        adata_.obs["has_contig"] = "True"
    contig_status = MarkAmbiguousContigs(
        dat,
        umi_foldchange_cutoff,
        consensus_foldchange_cutoff,
        ntop_vdj,
        ntop_vj,
        allow_exceptions,
        verbose,
    )

    ambigous = contig_status.ambiguous_contigs.copy()
    extra = contig_status.extra_contigs.copy()
    umi_adjustment = contig_status.umi_adjustment.copy()
    if len(umi_adjustment) > 0:
        for k, v in umi_adjustment.items():
            if k in dat.index:
                dat.at[k, "umi_count"] = v

    ambi = {c: "F" for c in dat_.sequence_id}
    ambiguous_ = {x: "T" for x in ambigous}
    ambi.update(ambiguous_)
    dat["ambiguous"] = pd.Series(ambi)

    extr = {c: "F" for c in dat_.sequence_id}
    extra_ = {x: "T" for x in extra}
    extr.update(extra_)
    dat["extra"] = pd.Series(extr)

    if filter_missing:
        dat = dat[dat["cell_id"].isin(adata_.obs_names)].copy()

    if dat.shape[0] == 0:
        raise IndexError(
            "No contigs passed filtering. Are you sure that the cell barcodes are matching?"
        )
    if os.path.isfile(str(data)):
        data_path = Path(data)
        write_airr(
            dat, data_path.parent / "{}_checked.tsv".format(data_path.stem)
        )
    else:
        if save is not None:
            if save.endswith(".tsv"):
                # _write_airr(dat, str(save))
                write_airr(dat, save)
            else:
                raise ValueError(
                    "{} not suitable. Please provide a file name that ends with .tsv".format(
                        str(save)
                    )
                )

    if productive_only:
        for i, row in dat.iterrows():
            if i in dat_.index:
                dat_.at[i, "umi_count"] = row["umi_count"]
        for column in ["ambiguous", "extra"]:
            dat_[column] = dat[column]
            dat_[column] = dat_[column].fillna("T")
        dat = dat_.copy()

    if filter_extra:
        dat = dat[dat["extra"] == "F"].copy()
    if filter_ambiguous:
        dat = dat[dat["ambiguous"] == "F"].copy()

    logg.info("Initializing Dandelion object")
    out_dat = Dandelion(data=dat, verbose=False, **kwargs)
    # out_dat = DandelionPolars(data=dat, verbose=False, **kwargs)
    if isinstance(data, Dandelion):
        # if isinstance(data, DandelionPolars):
        out_dat.germline = data.germline
    if adata_provided:
        transfer(adata_, out_dat)
        return (out_dat, adata_)
    else:
        return out_dat


class MarkAmbiguousContigs:
    """
    `MarkAmbiguousContigs` class object.


    Attributes
    ----------
    Cell : dandelion.utilities._utilities.Tree
        nested dictionary of cells.
    ambiguous_contigs : list[str]
        list of `sequence_id`s that are ambiguous.
    extra_contigs : list[str]
        list of `sequence_id`s that are extra.
    umi_adjustment : dict[str, int]
        dictionary of `sequence_id`s with adjusted umi value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        umi_foldchange_cutoff: int | float,
        consensus_foldchange_cutoff: int | float,
        ntop_vdj: int,
        ntop_vj: int,
        allow_exceptions: bool,
        verbose: bool,
    ):
        """Init method for MarkAmbiguousContigs.

        Parameters
        ----------
        data : pd.DataFrame
            AIRR data frame in Dandelion.data.
        umi_foldchange_cutoff : int | float
            fold-change cut off for decision for umi count.
        consensus_foldchange_cutoff : int | float
            fold-change cut off for decision for consensus count.
        ntop_vdj : int
            number of top VDJ contigs to consider for dominance check.
        ntop_vj : int
            number of top VJ contigs to consider for dominance check.
        allow_exceptions : bool
            whether or not to allow exceptions.
        verbose : bool
            whether or not to print progress.
        """
        self.Cell = Tree()
        self.ambiguous_contigs = []
        self.extra_contigs = []
        self.umi_adjustment = {}
        if "v_call_genotyped" in data.columns:
            v_dict = dict(zip(data["sequence_id"], data["v_call_genotyped"]))
        else:
            v_dict = dict(zip(data["sequence_id"], data["v_call"]))
        d_dict = dict(zip(data["sequence_id"], data["d_call"]))
        j_dict = dict(zip(data["sequence_id"], data["j_call"]))
        c_dict = dict(zip(data["sequence_id"], data["c_call"]))
        for contig, row in tqdm(
            data.iterrows(),
            desc="Preparing data",
        ):
            cell = row["cell_id"]
            if row["locus"] in HEAVYLONG:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VDJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VDJ"]["NP"][contig].update(row)
            elif row["locus"] in LIGHTSHORT:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VJ"]["NP"][contig].update(row)

        for cell in tqdm(
            self.Cell,
            desc="Scanning for poor quality/ambiguous contigs",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        ):
            if len(self.Cell[cell]["VDJ"]["P"]) > 0:
                # VDJ productive
                data1 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["P"][x]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                )
                vdj_p = list(data1["sequence_id"])
                vdj_ccall_p = list(data1["c_call"])
                vdj_locus_p = list(data1["locus"])
                if len(vdj_p) > 1:
                    if "sequence_alignment" in data1:
                        (
                            data1,
                            vdj_p,
                            _,
                            vdj_ccall_p,
                            umi_adjust_vdj,
                            ambi_cont_vdj,
                        ) = check_update_same_seq(data1)
                        if len(umi_adjust_vdj) > 0:
                            self.umi_adjustment.update(umi_adjust_vdj)
                        if len(ambi_cont_vdj) > 0:
                            for avdj in ambi_cont_vdj:
                                self.ambiguous_contigs.append(avdj)
                    if len(vdj_p) > 1:
                        if allow_exceptions:
                            if "IGHD" in vdj_ccall_p:
                                if all(
                                    x in ["IGHM", "IGHD"] for x in vdj_ccall_p
                                ):
                                    if len(list(set(vdj_ccall_p))) == 2:
                                        # the v and j should be the same in order to keep as exception
                                        vdj_vcall_p = (
                                            list(data1["v_call_genotyped"])
                                            if "v_call_genotyped"
                                            in data1.columns
                                            else list(data1["v_call"])
                                        )
                                        vdj_jcall_p = list(data1["j_call"])
                                        if (
                                            len(set(vdj_vcall_p))
                                            == 1 & len(set(vdj_jcall_p))
                                            == 1
                                        ):
                                            vdj_ccall_p_igm_count = dict(
                                                data1[
                                                    data1["c_call"] == "IGHM"
                                                ]["umi_count"].astype(int)
                                            )
                                            vdj_ccall_c_igm_count = dict(
                                                data1[
                                                    data1["c_call"] == "IGHM"
                                                ]["consensus_count"].astype(int)
                                            )
                                            vdj_ccall_p_igd_count = dict(
                                                data1[
                                                    data1["c_call"] == "IGHD"
                                                ]["umi_count"].astype(int)
                                            )
                                            vdj_ccall_c_igd_count = dict(
                                                data1[
                                                    data1["c_call"] == "IGHD"
                                                ]["consensus_count"].astype(int)
                                            )
                                        else:
                                            (
                                                vdj_ccall_p_igm_count,
                                                vdj_ccall_p_igd_count,
                                                vdj_ccall_c_igm_count,
                                                vdj_ccall_c_igd_count,
                                            ) = ({}, {}, {}, {})
                                            vdj_ccall_p_count = dict(
                                                data1["umi_count"].astype(int)
                                            )
                                            vdj_ccall_c_count = dict(
                                                data1["consensus_count"].astype(
                                                    int
                                                )
                                            )
                                            if len(vdj_ccall_p_count) > 1:
                                                (
                                                    vdj_p,
                                                    extra_vdj,
                                                    ambiguous_vdj,
                                                ) = check_productive_chain(
                                                    umi_counts=vdj_ccall_p_count,
                                                    consensus_counts=vdj_ccall_c_count,
                                                    umi_foldchange_cutoff=umi_foldchange_cutoff,
                                                    consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                                    ntop=ntop_vdj,
                                                )
                                            else:
                                                (
                                                    vdj_p,
                                                    extra_vdj,
                                                    ambiguous_vdj,
                                                ) = (
                                                    [],
                                                    [],
                                                    [],
                                                )
                                    else:
                                        (
                                            vdj_ccall_p_igm_count,
                                            vdj_ccall_p_igd_count,
                                            vdj_ccall_c_igm_count,
                                            vdj_ccall_c_igd_count,
                                        ) = ({}, {}, {}, {})

                                    if len(vdj_ccall_p_igm_count) > 1:
                                        (
                                            keep_igm,
                                            extra_igm,
                                            ambiguous_igm,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_ccall_p_igm_count,
                                            consensus_counts=vdj_ccall_c_igm_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        keep_igm, extra_igm, ambiguous_igm = (
                                            [],
                                            [],
                                            [],
                                        )

                                    if len(vdj_ccall_p_igd_count) > 1:
                                        (
                                            keep_igd,
                                            extra_igd,
                                            ambiguous_igd,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_ccall_p_igd_count,
                                            consensus_counts=vdj_ccall_c_igd_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        keep_igd, extra_igd, ambiguous_igd = (
                                            [],
                                            [],
                                            [],
                                        )
                                    if "vdj_p" not in locals():
                                        vdj_p = keep_igm + keep_igd
                                    if "extra_vdj" not in locals():
                                        extra_vdj = extra_igm + extra_igd
                                    if "ambiguous_vdj" not in locals():
                                        ambiguous_vdj = (
                                            ambiguous_igm + ambiguous_igd
                                        )
                                else:
                                    vdj_ccall_p_count = dict(
                                        data1["umi_count"].astype(int)
                                    )
                                    vdj_ccall_c_count = dict(
                                        data1["consensus_count"].astype(int)
                                    )
                                    if len(vdj_ccall_p_count) > 1:
                                        (
                                            vdj_p,
                                            extra_vdj,
                                            ambiguous_vdj,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_ccall_p_count,
                                            consensus_counts=vdj_ccall_c_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        vdj_p, extra_vdj, ambiguous_vdj = (
                                            [],
                                            [],
                                            [],
                                        )
                            elif all(x in ["TRB", "TRD"] for x in vdj_locus_p):
                                if len(list(set(vdj_locus_p))) == 2:
                                    vdj_locus_p_trb_count = dict(
                                        data1[data1["locus"] == "TRB"][
                                            "umi_count"
                                        ].astype(int)
                                    )
                                    vdj_locus_p_trd_count = dict(
                                        data1[data1["locus"] == "TRD"][
                                            "umi_count"
                                        ].astype(int)
                                    )
                                    vdj_locus_c_trb_count = dict(
                                        data1[data1["locus"] == "TRB"][
                                            "consensus_count"
                                        ].astype(int)
                                    )
                                    vdj_locus_c_trd_count = dict(
                                        data1[data1["locus"] == "TRD"][
                                            "consensus_count"
                                        ].astype(int)
                                    )
                                    if len(vdj_locus_p_trb_count) > 1:
                                        (
                                            keep_trb,
                                            extra_trb,
                                            ambiguous_trb,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_locus_p_trb_count,
                                            consensus_counts=vdj_locus_c_trb_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        keep_trb, extra_trb, ambiguous_trb = (
                                            [],
                                            [],
                                            [],
                                        )

                                    if len(vdj_locus_p_trd_count) > 1:
                                        (
                                            keep_trd,
                                            extra_trd,
                                            ambiguous_trd,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_locus_p_trd_count,
                                            consensus_counts=vdj_locus_c_trd_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        keep_trd, extra_trd, ambiguous_trd = (
                                            [],
                                            [],
                                            [],
                                        )

                                    vdj_p = keep_trb + keep_trd
                                    extra_vdj = extra_trb + extra_trd
                                    ambiguous_vdj = (
                                        ambiguous_trb + ambiguous_trd
                                    )
                                else:
                                    vdj_ccall_p_count = dict(
                                        data1["umi_count"].astype(int)
                                    )
                                    vdj_ccall_c_count = dict(
                                        data1["consensus_count"].astype(int)
                                    )
                                    if len(vdj_ccall_p_count) > 1:
                                        (
                                            vdj_p,
                                            extra_vdj,
                                            ambiguous_vdj,
                                        ) = check_productive_chain(
                                            umi_counts=vdj_ccall_p_count,
                                            consensus_counts=vdj_ccall_c_count,
                                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                            ntop=ntop_vdj,
                                        )
                                    else:
                                        vdj_p, extra_vdj, ambiguous_vdj = (
                                            [],
                                            [],
                                            [],
                                        )
                            else:
                                vdj_ccall_p_count = dict(
                                    data1["umi_count"].astype(int)
                                )
                                vdj_ccall_c_count = dict(
                                    data1["consensus_count"].astype(int)
                                )
                                if len(vdj_ccall_p_count) > 1:
                                    (
                                        vdj_p,
                                        extra_vdj,
                                        ambiguous_vdj,
                                    ) = check_productive_chain(
                                        umi_counts=vdj_ccall_p_count,
                                        consensus_counts=vdj_ccall_c_count,
                                        umi_foldchange_cutoff=umi_foldchange_cutoff,
                                        consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                        ntop=ntop_vdj,
                                    )
                        else:
                            vdj_ccall_p_count = dict(
                                data1["umi_count"].astype(int)
                            )
                            vdj_ccall_c_count = dict(
                                data1["consensus_count"].astype(int)
                            )
                            if len(vdj_ccall_p_count) > 1:
                                (
                                    vdj_p,
                                    extra_vdj,
                                    ambiguous_vdj,
                                ) = check_productive_chain(
                                    umi_counts=vdj_ccall_p_count,
                                    consensus_counts=vdj_ccall_c_count,
                                    umi_foldchange_cutoff=umi_foldchange_cutoff,
                                    consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                    ntop=ntop_vdj,
                                )
                if "ambiguous_vdj" not in locals():
                    ambiguous_vdj = []

                if len(ambiguous_vdj) > 0:
                    for a in ambiguous_vdj:
                        self.ambiguous_contigs.append(a)
            else:
                data1 = None
                vdj_p = []
            # VDJ non-productive
            if len(self.Cell[cell]["VDJ"]["NP"]) > 0:
                data2 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["NP"][x]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                )
                vdj_np = list(data2["sequence_id"])
                if len(vdj_np) > 1:
                    if "sequence_alignment" in data2:
                        (
                            data2,
                            vdj_np,
                            _,
                            _,
                            umi_adjust_vdjnp,
                            ambi_cont_vdjnp,
                        ) = check_update_same_seq(data2)
                        if len(umi_adjust_vdjnp) > 0:
                            self.umi_adjustment.update(umi_adjust_vdjnp)
                        if len(ambi_cont_vdjnp) > 0:
                            for avdj in ambi_cont_vdjnp:
                                self.ambiguous_contigs.append(avdj)
                    if len(vdj_np) > 1:
                        vdj_ccall_np_count = dict(
                            data2["umi_count"].astype(int)
                        )
                        vdj_ccall_c_count = dict(
                            data2["consensus_count"].astype(int)
                        )
                        if len(vdj_ccall_np_count) > 1:
                            vdj_np, extra_vdjnp, ambiguous_vdjnp = (
                                check_productive_chain(
                                    umi_counts=vdj_ccall_np_count,
                                    consensus_counts=vdj_ccall_c_count,
                                    umi_foldchange_cutoff=umi_foldchange_cutoff,
                                    consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                    ntop=ntop_vdj,
                                )
                            )
                            if len(ambiguous_vdjnp) > 0:
                                for a in ambiguous_vdjnp:
                                    self.ambiguous_contigs.append(a)
            else:
                data2 = None
                vdj_np = []
            # VJ productive
            if len(self.Cell[cell]["VJ"]["P"]) > 0:
                data3 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["P"][x]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                )
                vj_p = list(data3["sequence_id"])
                if len(vj_p) > 1:
                    if "sequence_alignment" in data3:
                        (
                            data3,
                            vj_p,
                            _,
                            _,
                            umi_adjust_vj,
                            ambi_cont_vj,
                        ) = check_update_same_seq(data3)
                        if len(umi_adjust_vj) > 0:
                            self.umi_adjustment.update(umi_adjust_vj)
                        if len(ambi_cont_vj) > 0:
                            for avj in ambi_cont_vj:
                                self.ambiguous_contigs.append(avj)
                    if len(vj_p) > 1:
                        vj_ccall_p_count = dict(data3["umi_count"].astype(int))
                        vj_ccall_c_count = dict(
                            data3["consensus_count"].astype(int)
                        )
                        vj_p, extra_vj, ambiguous_vj = check_productive_chain(
                            umi_counts=vj_ccall_p_count,
                            consensus_counts=vj_ccall_c_count,
                            umi_foldchange_cutoff=umi_foldchange_cutoff,
                            consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                            ntop=ntop_vj,  # maximum keep 2 as default?
                        )
                if "ambiguous_vj" not in locals():
                    ambiguous_vj = []
                if len(ambiguous_vj) > 0:
                    for a in ambiguous_vj:
                        self.ambiguous_contigs.append(a)
            else:
                data3 = None
                vj_p = []
            # VJ non-productive
            if len(self.Cell[cell]["VJ"]["NP"]) > 0:
                data4 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["NP"][x]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                )
                vj_np = list(data4["sequence_id"])
                if len(vj_np) > 1:
                    if "sequence_alignment" in data4:
                        (
                            data4,
                            vj_np,
                            _,
                            _,
                            umi_adjust_vjnp,
                            ambi_cont_vjnp,
                        ) = check_update_same_seq(data4)
                        if len(umi_adjust_vjnp) > 0:
                            self.umi_adjustment.update(umi_adjust_vjnp)
                        if len(ambi_cont_vjnp) > 0:
                            for avj in ambi_cont_vjnp:
                                self.ambiguous_contigs.append(avj)
                    if len(vj_np) > 1:
                        vj_ccall_np_count = dict(data4["umi_count"].astype(int))
                        vj_ccall_c_count = dict(
                            data4["consensus_count"].astype(int)
                        )
                        if len(vj_ccall_np_count) > 1:
                            vj_np, extra_vjnp, ambiguous_vjnp = (
                                check_productive_chain(
                                    umi_counts=vj_ccall_np_count,
                                    consensus_counts=vj_ccall_c_count,
                                    umi_foldchange_cutoff=umi_foldchange_cutoff,
                                    consensus_foldchange_cutoff=consensus_foldchange_cutoff,
                                    ntop=ntop_vj,
                                )
                            )
                            if len(ambiguous_vjnp) > 0:
                                for a in ambiguous_vjnp:
                                    self.ambiguous_contigs.append(a)
            else:
                data4 = None
                vj_np = []
            extra_vdj_ = mark_ntop_contigs(
                productive_data=data1,
                nonproductive_data=data2,
                productive_contig=vdj_p,
                nonproductive_contig=vdj_np,
                ntop=ntop_vdj,
            )
            for ex in extra_vdj_:
                self.extra_contigs.append(ex)
            extra_vj_ = mark_ntop_contigs(
                productive_data=data3,
                nonproductive_data=data4,
                productive_contig=vj_p,
                nonproductive_contig=vj_np,
                ntop=ntop_vj,
            )
            for ex in extra_vj_:
                self.extra_contigs.append(ex)

            if "vdj_p" not in locals():
                vdj_p = []
            if "vj_p" not in locals():
                vj_p = []
            if "vdj_np" not in locals():
                vdj_np = []
            if "vj_np" not in locals():
                vj_np = []
            if "extra_vdj" not in locals():
                extra_vdj = []
            if "extra_vj" not in locals():
                extra_vj = []
            if "extra_vdjnp" not in locals():
                extra_vdjnp = []
            if "extra_vjnp" not in locals():
                extra_vjnp = []

            # check here for bad combinations
            # marking poor bcr quality, defined as those with conflicting assignment of
            # locus and V(D)J v-, d-, j- and c- calls, and also those that are missing
            # j calls.
            if len(vdj_p) > 0:
                for vdjx in vdj_p:
                    v = v_dict[vdjx]
                    d = d_dict[vdjx]
                    j = j_dict[vdjx]
                    c = c_dict[vdjx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(vdjx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(vdjx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(vdjx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(vdjx)
                    else:
                        self.ambiguous_contigs.append(vdjx)

            if len(vdj_np) > 0:
                for vdjx in vdj_np:
                    v = v_dict[vdjx]
                    d = d_dict[vdjx]
                    j = j_dict[vdjx]
                    c = c_dict[vdjx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(vdjx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(vdjx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(vdjx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(vdjx)
                    else:
                        self.ambiguous_contigs.append(vdjx)
            if len(vj_p) > 0:
                for vjx in vj_p:
                    v = v_dict[vjx]
                    j = j_dict[vjx]
                    c = c_dict[vjx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(vjx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(vjx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(vjx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(vjx)
                    else:
                        self.ambiguous_contigs.append(vjx)

            if len(vj_np) > 0:
                for vjx in vj_np:
                    v = v_dict[vjx]
                    j = j_dict[vjx]
                    c = c_dict[vjx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(vjx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(vjx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(vjx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(vjx)
                    else:
                        self.ambiguous_contigs.append(vjx)

            if len(extra_vdj) > 0:
                for evdj in extra_vdj:
                    v = v_dict[evdj]
                    d = d_dict[evdj]
                    j = j_dict[evdj]
                    c = c_dict[evdj]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(evdj)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(evdj)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(evdj)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(evdj)
                    else:
                        self.ambiguous_contigs.append(evdj)
                    self.extra_contigs.append(evdj)

            if len(extra_vdjnp) > 0:
                for evdj in extra_vdjnp:
                    v = v_dict[evdj]
                    d = d_dict[evdj]
                    j = j_dict[evdj]
                    c = c_dict[evdj]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(evdj)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(evdj)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(evdj)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(evdj)
                    else:
                        self.ambiguous_contigs.append(evdj)
                    self.extra_contigs.append(evdj)

            if len(extra_vj) > 0:
                for evj in extra_vj:
                    v = v_dict[evj]
                    j = j_dict[evj]
                    c = c_dict[evj]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(evj)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(evj)
                    self.extra_contigs.append(evj)

            if len(extra_vjnp) > 0:
                for evj in extra_vjnp:
                    v = v_dict[evj]
                    j = j_dict[evj]
                    c = c_dict[evj]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(evj)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(evj)
                    self.extra_contigs.append(evj)


def check_productive_chain(
    umi_counts: dict[str, int],
    consensus_counts: dict[str, int],
    umi_foldchange_cutoff: int | float,
    consensus_foldchange_cutoff: int | float,
    ntop: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Function to keep top two productive vj chains because of allelic inclusions.

    Parameters
    ----------
    umi_counts : dict[str, int]
        dictionary of contigs with umi count.
    consensus_counts : dict[str, int]
        dictionary of contigs with consensus count.
    umi_foldchange_cutoff : int | float
        fold-change cut off for decision for umi count.
    consensus_foldchange_cutoff : int | float
        fold-change cut off for decision for consensus count.
    ntop : int
        number of top contigs to keep.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        lists of contigs to keep, are extra or are ambiguous.
    """
    keep_contigs, extra_contigs, ambiguous_contigs = [], [], []
    # Look at all the contigs, assessing if their umi counts are >= cutoff compared to the lowest umi count or 3, whichever is lower.
    # if none pass the cutoff, then mark all as ambiguous
    # Allow up to two to pass the cutoff, keep those two contigs, and the rest that pass the cut off in the extra contigs, and mark the rest as ambiguous
    # If there are ties in the umi counts, then look at the consensus counts to break the tie using the same logic as above, but with a cutoff of 5.
    min_umi_count, min_con_count = min(min(umi_counts.values()), 3), min(
        min(consensus_counts.values()), 5
    )
    # test the umi fold change
    umi_test = {
        k: (
            ((umi_counts[k] / min_umi_count) >= umi_foldchange_cutoff)
            & (umi_counts[k] >= 3)
        )
        for k in umi_counts
    }
    # test the consensus fold change
    con_test = {
        k: (
            (
                (consensus_counts[k] / min_con_count)
                >= consensus_foldchange_cutoff
            )
            & (consensus_counts[k] >= 5)
        )
        for k in consensus_counts
    }
    filtered_contigs = {k: v for k, v in umi_counts.items() if umi_test[k]}
    if any(umi_test.values()):
        # get the top n contigs
        if len(filtered_contigs) <= ntop:
            keep_contigs.extend(list(filtered_contigs.keys()))
            failed_contigs = {
                k: v for k, v in umi_counts.items() if not umi_test[k]
            }
            ambiguous_contigs.extend(list(failed_contigs.keys()))
        else:
            # check the consensus fold change
            if any(con_test.values()):
                keep_contigs, extra_contigs, ambiguous_contigs = (
                    keep_if_consensus_count_is_high(
                        filtered_contigs=filtered_contigs,
                        umi_counts=umi_counts,
                        consensus_counts=consensus_counts,
                        keep_contigs=keep_contigs,
                        extra_contigs=extra_contigs,
                        ambiguous_contigs=ambiguous_contigs,
                        consensus_test=con_test,
                        ntop=ntop,
                    )
                )
            else:
                # not enough counts to confidently classify
                ambiguous_contigs.extend(list(umi_counts.keys()))
    else:
        if any(con_test.values()):
            keep_contigs, extra_contigs, ambiguous_contigs = (
                keep_if_consensus_count_is_high(
                    filtered_contigs=filtered_contigs,
                    umi_counts=umi_counts,
                    consensus_counts=consensus_counts,
                    keep_contigs=keep_contigs,
                    extra_contigs=extra_contigs,
                    ambiguous_contigs=ambiguous_contigs,
                    consensus_test=con_test,
                    ntop=ntop,
                )
            )
        else:
            # Not enough counts to confidently classify
            ambiguous_contigs.extend(umi_counts.keys())

    return keep_contigs, extra_contigs, ambiguous_contigs


def keep_if_consensus_count_is_high(
    filtered_contigs: dict[str, int],
    umi_counts: dict[str, int],
    consensus_counts: dict[str, int],
    keep_contigs: list[str],
    extra_contigs: list[str],
    ambiguous_contigs: list[str],
    consensus_test: dict[str, bool],
    ntop: int,
) -> tuple[list[str], list[str], list[str]]:
    """Accessories function to keep the contigs based on consensus counts."""
    # sort the umi counts and use the consensus counts to break the tie
    sorted_contigs = dict(
        sorted(
            umi_counts.items(),
            key=lambda k: (k[1], consensus_counts[k[0]]),
            reverse=True,
        )
    )
    # get the top 2 contigs
    _keep = list(sorted_contigs.keys())[:ntop]
    _keep_filtered = [k for k in _keep if consensus_test[k]]
    # get the extras
    _extra = [k for k in filtered_contigs if k not in _keep_filtered]
    # get the ambiguous
    _ambi = list({k for k in umi_counts if k not in _keep_filtered + _extra})
    # extend the lists
    keep_contigs.extend(_keep_filtered)
    # get the extra contigs
    extra_contigs.extend(_extra)
    # get the ambiguous contigs
    ambiguous_contigs.extend(_ambi)
    return keep_contigs, extra_contigs, ambiguous_contigs


def check_update_same_seq(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame, list[str], list[int], list[str], dict[str, int], list[str]
]:
    """Check if sequences are the same.

    Parameters
    ----------
    data : pd.DataFrame
        AIRR data frame in Dandelion.data.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[int], list[str], dict[str, int], list[str]]
        updated  AIRR data frame, lists of contigs to keep, their umi counts, their c_calls,
        adjusted umi counts, and list of ambiguous contigs.
    """
    keep_id = []
    keep_ccall = []
    umi_adjust = {}
    ambi_cont = []
    sequencecol = (
        "sequence_alignment" if "sequence_alignment" in data else "sequence"
    )
    if sequencecol in data:
        seq_ = list(data[sequencecol])
        seq_2 = [s for s in seq_ if present(s)]
        if len(set(seq_2)) < len(seq_2):
            _seq = {
                k: r for k, r in dict(data[sequencecol]).items() if present(r)
            }
            _count = {
                k: r
                for k, r in dict(data.umi_count.astype(int)).items()
                if k in _seq
            }
            rep_seq = [
                seq
                for seq in set(_seq.values())
                if countOf(_seq.values(), seq) > 1
            ]
            keep_seqs = [
                seq
                for seq in set(_seq.values())
                if countOf(_seq.values(), seq) == 1
            ]
            keep_seqs_ids = [i for i, seq in _seq.items() if seq in keep_seqs]
            if len(rep_seq) > 0:
                for rep in rep_seq:
                    dup_keys = [k for k, v in _seq.items() if v == rep]
                    keep_index_vj = dup_keys[0]
                    keep_index_count = int(_count[keep_index_vj])
                    sum_rep_counts = np.sum([_count[k] for k in dup_keys[1:]])
                    umi_adjust.update(
                        {
                            keep_index_vj: int(
                                np.sum(
                                    [
                                        sum_rep_counts,
                                        keep_index_count,
                                    ],
                                )
                            )
                        }
                    )
                    for dk in dup_keys[1:]:
                        ambi_cont.append(dk)
                    keep_seqs_ids.append(keep_index_vj)
                    if keep_index_vj in data.index:
                        data.at[keep_index_vj, "umi_count"] = keep_index_count
                # refresh
                empty_seqs_ids = [
                    k
                    for k, r in dict(data[sequencecol]).items()
                    if not present(r)
                ]
                if len(empty_seqs_ids) > 0:
                    keep_seqs_ids = keep_seqs_ids + empty_seqs_ids
                data = data.loc[keep_seqs_ids]
        keep_id = list(data.sequence_id)
        keep_umi = [int(x) for x in pd.to_numeric(data.umi_count)]
        keep_ccall = list(data.c_call)
    else:
        keep_id = list(data.sequence_id)
        keep_umi = [int(x) for x in pd.to_numeric(data.umi_count)]
        keep_ccall = list(data.c_call)

    return (data, keep_id, keep_umi, keep_ccall, umi_adjust, ambi_cont)


def choose_segments(
    starts: pd.Series, ends: pd.Series, scores: pd.Series
) -> list[str]:
    """Choose left most segment

    Parameters
    ----------
    starts : pd.Series
        nucleotide start positions.
    ends : pd.Series
        nucleotide end positions.
    scores : pd.Series
        alignment scores.

    Returns
    -------
    list[str]
        list of chosen segments.
    """
    starts = np.array(starts)
    ends = np.array(ends)
    scores = np.array(scores)
    ind = np.arange(len(starts))
    chosen = []
    while len(ind) > 0:
        best = np.argmax(scores)
        chosen.append(ind[best])
        overlap = (starts <= ends[best]) & (ends >= starts[best])
        ind = ind[~overlap]
        starts = starts[~overlap]
        ends = ends[~overlap]
        scores = scores[~overlap]
    return chosen


def multimapper(filename: str) -> pd.DataFrame:
    """Select the left more segment as the final call

    Parameters
    ----------
    filename : str
        path to multimapper file.

    Returns
    -------
    pd.DataFrame
        Mapped multimapper data frame.
    """
    df = pd.read_csv(filename, delimiter="\t")
    df_new = df.loc[
        df["j_support"] < 1e-3, :
    ]  # maybe not needing to filter if j_support has already been filtered

    tmp = pd.DataFrame(
        index=list(set(df_new["sequence_id"])),
        columns=[
            "multimappers",
            "multiplicity",
            "sequence_start_multimappers",
            "sequence_end_multimappers",
            "support_multimappers",
        ],
    )

    # Define a function to apply to each group
    def process_group(group: pd.DataFrame) -> pd.Series:
        """
        Create a dictionary for the multimappers results.

        Parameters
        ----------
        group : pd.DataFrame
            input dataframe for a given sequence_id.

        Returns
        -------
        pd.Series
            A pandas series with the multimappers results.
        """
        starts = group["j_sequence_start"]
        ends = group["j_sequence_end"]
        scores = -group["j_support"]
        chosen_ind = choose_segments(starts, ends, scores)
        group = group.iloc[chosen_ind, :]
        group = group.sort_values(by=["j_sequence_start"], ascending=True)

        return pd.Series(
            {
                "multimappers": json.dumps(list(group["j_call"])),
                "multiplicity": group.shape[0],
                "sequence_start_multimappers": json.dumps(
                    list(group["j_sequence_start"])
                ),
                "sequence_end_multimappers": json.dumps(
                    list(group["j_sequence_end"])
                ),
                "support_multimappers": json.dumps(list(group["j_support"])),
            }
        )

    # Group by "sequence_id" and apply the processing function, then reset the index
    mapped = df_new.groupby("sequence_id").apply(process_group).reset_index()
    # Set the index explicitly
    mapped.set_index("sequence_id", drop=True, inplace=True)
    # change multiplicity to integer
    mapped["multiplicity"] = mapped["multiplicity"].astype("Int64")
    mapped = mapped.reindex(tmp.index)

    return mapped


def update_j_multimap(data: list[str], filename_prefix: list[str]):
    """Update j multimapper call.

    Parameters
    ----------
    data : list[str]
        input folders.
    filename_prefix : list[str]
        prefixes to append to front of files.
    """
    if not isinstance(data, list):
        data = [data]
    if not isinstance(filename_prefix, list):
        filename_prefix = [filename_prefix]
    for i in range(0, len(data)):
        filePath0 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_j_blast.tsv",
            sub_dir="tmp",
        )
        filePath1 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass.tsv",
            sub_dir="tmp",
        )
        filePath1g = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass_genotyped.tsv",
            sub_dir="tmp",
        )
        filePath2 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-all.tsv",
            sub_dir="tmp",
        )
        filePath3 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-fail.tsv",
            sub_dir="tmp",
        )
        filePath4 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_dandelion.tsv",
        )

        jmm_transfer_cols = [
            "multimappers",
            "multiplicity",
            "sequence_start_multimappers",
            "sequence_end_multimappers",
            "support_multimappers",
        ]
        check_multimapper(filePath0, filePath2)
        if filePath0 is not None:
            jmulti = multimapper(filePath0)
            if filePath1 is not None:
                dbpass = load_data(filePath1)
                for col in jmm_transfer_cols:
                    update_j_col_df(dbpass, jmulti, col)
                # _write_airr(dbpass, filePath1)
                write_airr(dbpass, filePath1)
            if filePath1g is not None:
                dbpassg = load_data(filePath1g)
                for col in jmm_transfer_cols:
                    update_j_col_df(dbpassg, jmulti, col)
                # _write_airr(dbpassg, filePath1g)
                write_airr(dbpassg, filePath1g)
            if filePath2 is not None:
                dbfail = load_data(filePath2)
                for col in jmm_transfer_cols:
                    update_j_col_df(dbfail, jmulti, col)
                for i in dbfail.index:
                    if not present(dbfail.loc[i, "v_call"]):
                        jmmappers = safe_json_load(
                            dbfail.at[i, "j_call_multimappers"]
                        )
                        jmmappersstart = safe_json_load(
                            dbfail.at[i, "j_call_sequence_start_multimappers"]
                        )
                        jmmappersend = safe_json_load(
                            dbfail.at[i, "j_call_sequence_end_multimappers"]
                        )
                        jmmapperssupport = safe_json_load(
                            dbfail.at[i, "j_call_support_multimappers"]
                        )
                        if len(jmmappers) > 1:
                            dbfail.at[i, "j_call"] = jmmappers[0]
                            dbfail.at[i, "j_sequence_start"] = float(
                                jmmappersstart[0]
                            )
                            dbfail.at[i, "j_sequence_end"] = float(
                                jmmappersend[0]
                            )
                            dbfail.at[i, "j_support"] = float(
                                jmmapperssupport[0]
                            )
                # _write_airr(dbfail, filePath2)
                write_airr(dbfail, filePath2)
            if filePath3 is not None:
                dball = load_data(filePath3)
                for col in jmm_transfer_cols:
                    update_j_col_df(dball, jmulti, col)
                for i in dball.index:
                    if not present(dball.loc[i, "v_call"]):
                        jmmappers = safe_json_load(
                            dball.at[i, "j_call_multimappers"]
                        )
                        jmmappersstart = safe_json_load(
                            dball.at[i, "j_call_sequence_start_multimappers"]
                        )
                        jmmappersend = safe_json_load(
                            dball.at[i, "j_call_sequence_end_multimappers"]
                        )
                        jmmapperssupport = safe_json_load(
                            dball.at[i, "j_call_support_multimappers"]
                        )
                        if len(jmmappers) > 1:
                            dball.at[i, "j_call"] = jmmappers[0]
                            dball.at[i, "j_sequence_start"] = float(
                                jmmappersstart[0]
                            )
                            dball.at[i, "j_sequence_end"] = float(
                                jmmappersend[0]
                            )
                            dball.at[i, "j_support"] = float(
                                jmmapperssupport[0]
                            )
                # _write_airr(dball, filePath3)
                write_airr(dball, filePath3)
            if filePath4 is not None:
                dandy = load_data(filePath4)
                for col in jmm_transfer_cols:
                    update_j_col_df(dandy, jmulti, col)
                # _write_airr(dandy, filePath4)
                write_airr(dandy, filePath4)


def check_multimapper(
    filename1: str,
    filename2: str,
) -> pd.DataFrame:
    """Select the left more segment as the final call
    Parameters
    ----------
    filename1 : str
        path to multimapper file.
    filename2 : str
        path to reference file containing all information.
    Returns
    -------
    pd.DataFrame
        Mapped multimapper data frame.
    """
    if filename1 is not None:
        if filename2 is not None:
            df = pd.read_csv(filename1, sep="\t")
            df_new = df[
                df["j_support"] < 1e-3
            ]  # maybe not needing to filter if j_support has already been filtered
            df_ref = load_data(filename2)
            mapped = list(set(df_new["sequence_id"]))
            keep = []
            for j in mapped:
                tmp = df_new[df_new["sequence_id"] == j][
                    [
                        "j_sequence_start",
                        "j_sequence_end",
                        "j_support",
                        "j_call",
                    ]
                ]
                if j in df_ref.index:
                    vend = df_ref.loc[j, "v_sequence_end"]
                    vend_ = 0 if not present(vend) else vend
                    for i in tmp.index:
                        callstart = tmp.loc[i, "j_sequence_start"]
                        if callstart >= vend_:
                            keep.append(i)
            keepdf = df_new.loc[keep]
            keepdf.to_csv(filename1, sep="\t", index=False)


def update_j_col_df(airrdata: pd.DataFrame, jmulti: pd.DataFrame, col: str):
    """
    Update the j_call column in the dataframe with the values from the jmulti dataframe without triggering future warning.

    Parameters
    ----------
    airrdata : pd.DataFrame
        The airr dataframe to update.
    jmulti : pd.DataFrame
        The jmulti dataframe to update from.
    col : str
        The column to update.
    """
    airrdata["j_call_" + col] = ""
    df = pd.DataFrame(index=airrdata.index)
    df[col] = ""
    df.update(jmulti[[col]])
    df["j_call_" + col] = df[col]
    df = df[["j_call_" + col]]
    airrdata.update(df[["j_call_" + col]])
    if col == "multiplicity":
        airrdata["j_call_" + col] = airrdata["j_call_" + col].replace("", 0)
        airrdata["j_call_" + col] = sanitize_column(
            airrdata["j_call_" + col], "integer"
        )


def mark_ntop_contigs(
    productive_data: pd.DataFrame | None,
    nonproductive_data: pd.DataFrame | None,
    productive_contig: list[str],
    nonproductive_contig: list[str],
    ntop: int,
) -> list[str]:
    """
    Function to mark the not top contigs based on umi count and productive status.

    Parameters
    ----------
    productive_data : pd.DataFrame
        dataframe containing productive contigs.
    nonproductive_data : pd.DataFrame
        dataframe containing nonproductive contigs.
    productive_contig : list[str]
        filtered productive contigs.
    nonproductive_contig : list[str]
        filtered nonproductive contigs.
    ntop : int
        number of top contigs to keep.

    Returns
    -------
    list[str]
        list of additional contigs that are not in the top ntop contigs to mark as extra.
    """
    additional_extras = []
    if productive_data is not None:
        if nonproductive_data is not None:
            data_concat = pd.concat([productive_data, nonproductive_data])
        else:
            data_concat = productive_data
    else:
        if nonproductive_data is not None:
            data_concat = nonproductive_data
        else:
            return additional_extras

    # sort by productive and then by umi count
    data_concat = data_concat.sort_values(
        ["productive", "umi_count"], ascending=[False, False]
    )
    # keep only the top ntop contigs
    data_concat = data_concat.head(ntop)
    # any contigs not found in data_concat are extra
    if len(productive_contig) > 0:
        for contig in productive_contig:
            if contig not in data_concat["sequence_id"]:
                additional_extras.append(contig)
    if len(nonproductive_contig) > 0:
        for contig in nonproductive_contig:
            if contig not in data_concat["sequence_id"]:
                additional_extras.append(contig)
    return additional_extras


def mask_dj(
    data: list[Path | str],
    filename_prefix: str,
    d_evalue_threshold: float,
    j_evalue_threshold: float,
) -> None:
    """Mask d/j assignment."""
    for i in range(0, len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass.tsv",
        )
        if filePath is not None:
            dat = load_data(filePath)
            if "d_support_blastn" in dat:
                dat["d_call"] = [
                    "" if s > d_evalue_threshold else c
                    for c, s in zip(dat["d_call"], dat["d_support_blastn"])
                ]
            if "j_support_blastn" in dat:
                dat["j_call"] = [
                    "" if s > j_evalue_threshold else c
                    for c, s in zip(dat["j_call"], dat["j_support_blastn"])
                ]
            # _write_airr(dat, filePath)
            write_airr(dat, filePath)


def change_file_location(
    data: list[Path | str],
    filename_prefix: list[str] | str | None = None,
) -> None:
    """
    Move file from tmp folder to dandelion folder.

    Only used for TCR data.

    Parameters
    ----------
    data : list[Path | str]
        list of data folders containing the .tsv files. if provided as a single string, it will first be converted to a
        list; this allows for the function to be run on single/multiple samples.
    filename_prefix : list[str] | str | None, optional
        list of prefixes of file names preceding '_contig'. None defaults to 'all'.
    """
    fileformat = "blast"
    data, filename_prefix = check_data(data, filename_prefix)

    informat_dict = {
        "changeo": "_igblast_db-pass.tsv",
        "blast": "_igblast_db-pass.tsv",
        "airr": "_igblast_gap.tsv",
    }
    filePath = None

    for i in range(0, len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with=informat_dict[fileformat],
            sub_dir="tmp",
        )
        if filePath is not None:
            tmp = load_data(filePath)
            tmp = check_travdv(tmp)
            _airrfile = str(filePath).replace("_db-pass.tsv", ".tsv")
            airr_output = load_data(_airrfile)
            cols_to_merge = [
                "junction_aa_length",
                "fwr1_aa",
                "fwr2_aa",
                "fwr3_aa",
                "fwr4_aa",
                "cdr1_aa",
                "cdr2_aa",
                "cdr3_aa",
                "sequence_alignment_aa",
                "v_sequence_alignment_aa",
                "d_sequence_alignment_aa",
                "j_sequence_alignment_aa",
            ]
            for x in cols_to_merge:
                tmp[x] = pd.Series(airr_output[x])
            # _write_airr(tmp, filePath)
            write_airr(tmp, filePath)
            fp = Path(filePath)
            shutil.copyfile(fp, fp.parent.parent / fp.name)


def move_to_tmp(
    data: list[Path | str], filename_prefix: list[str] | str | None = None
) -> None:
    """Move file to tmp."""
    data, filename_prefix = check_data(data, filename_prefix)

    for i in range(0, len(data)):
        filePath1 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_annotations.csv",
        )
        filePath2 = check_filepath(
            data[i], filename_prefix=filename_prefix[i], ends_with=".fasta"
        )
        for fp in [filePath1, filePath2]:
            fp = Path(fp)
            shutil.move(fp, fp.parent / "tmp" / fp.name)


def make_all(
    data: list[Path | str],
    filename_prefix: list[str] | str | None = None,
    loci: Literal["ig", "tr"] = "tr",
) -> None:
    """Construct db-all tsv file."""
    data, filename_prefix = check_data(data, filename_prefix)

    for i in range(0, len(data)):
        if loci == "tr":
            filePath1 = check_filepath(
                data[i],
                filename_prefix=filename_prefix[i],
                ends_with="_igblast_db-pass.tsv",
                sub_dir="tmp",
            )
        else:
            filePath1 = check_filepath(
                data[i],
                filename_prefix=filename_prefix[i],
                ends_with="_igblast_db-pass_genotyped.tsv",
                sub_dir="tmp",
            )
            if filePath1 is None:
                filePath1 = check_filepath(
                    data[i],
                    filename_prefix=filename_prefix[i],
                    ends_with="_igblast_db-pass.tsv",
                    sub_dir="tmp",
                )
                out_ex = "db-pass.tsv"
            else:
                out_ex = "db-pass_genotyped.tsv"
        filePath2 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-fail.tsv",
            sub_dir="tmp",
        )
        if filePath1 is not None:
            df1 = pd.read_csv(filePath1, sep="\t")
            df1 = check_complete(df1)
            # _write_airr(df1, filePath1)
            write_airr(df1, filePath1)
            if filePath2 is not None:
                df2 = pd.read_csv(filePath2, sep="\t")
                df2 = check_complete(df2)
                df = pd.concat([df1, df2])
                if loci == "tr":
                    # _write_airr(
                    write_airr(
                        df,
                        filePath1.parent
                        / (
                            filePath1.name.rsplit("db-pass.tsv")[0]
                            + "db-all.tsv"
                        ),
                    )
                else:
                    # _write_airr(
                    write_airr(
                        df,
                        filePath1.parent
                        / (filePath1.name.rsplit(out_ex)[0] + "db-all.tsv"),
                    )
                # _write_airr(df2, filePath2)
                write_airr(df2, filePath2)
            else:
                if loci == "tr":
                    # _write_airr(
                    write_airr(
                        df1,
                        filePath1.parent
                        / (
                            filePath1.name.rsplit("db-pass.tsv")[0]
                            + "db-all.tsv"
                        ),
                    )
                else:
                    # _write_airr(
                    write_airr(
                        df1,
                        filePath1.parent
                        / (filePath1.name.rsplit(out_ex)[0] + "db-all.tsv"),
                    )


def rename_dandelion(
    data: list[Path | str],
    filename_prefix: list[str] | str | None = None,
    ends_with: str = "_igblast_db-pass_genotyped.tsv",
    sub_dir: str | None = None,
) -> None:
    """Rename final dandlion file."""
    data, filename_prefix = check_data(data, filename_prefix)

    for i in range(0, len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with=ends_with,
            sub_dir=sub_dir,
        )  # must be whatever's after contig
        if sub_dir is None:
            fp = filePath.parent / filePath.name.rsplit(ends_with)[0]
        else:
            fp = filePath.parent.parent / filePath.name.rsplit(ends_with)[0]
        shutil.move(filePath, Path(str(fp) + "_dandelion.tsv"))


def check_complete(df: pd.DataFrame) -> pd.DataFrame:
    """check if contig contains cdr3.

    Parameters
    ----------
    df : pd.DataFrame
        airr data frame.

    Returns
    -------
    pd.DataFrame
        completed airr data frame
    """
    if "complete_vdj" not in df:
        df["complete_vdj"] = ""
    for i in df.index:
        junc = df.loc[i, "junction"]
        if not present(junc):
            df.at[i, "productive"] = "F"
            df.at[i, "complete_vdj"] = "F"
    return df


def safe_json_load(s: list | str | None) -> list:
    """Safely load json arrays."""
    if not s:  # empty string or None
        return []  # fallback empty list
    return json.loads(s)


def assert_dj_numeric_sequence_cols(
    data: pd.DataFrame,
) -> None:
    """Assert that specified columns in the DataFrame are numeric.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to check.

    Raises
    ------
    ValueError
        If any of the specified columns contain non-numeric values.
    """
    for call in ["d", "j"]:
        for col in [
            call + "_sequence_start",
            call + "_sequence_end",
            call + "_support",
            call + "_sequence_start_blastn",
            call + "_sequence_end_blastn",
            call + "_support_igblastn",
        ]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
    return data
