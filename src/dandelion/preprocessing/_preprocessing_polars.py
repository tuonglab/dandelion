import os
import json
import shutil
import tempfile
import warnings

import numpy as np
import polars as pl

from Bio import Align
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
    write_fasta,
)

from dandelion.utilities._polars import (
    DandelionPolars,
    load_polars,
    read_10x_vdj_polars,
    _check_travdv_polars,
    _sanitize_data_polars,
    _write_airr,
    all_missing_polars,
)
from dandelion.utilities._io import (
    fasta_iterator,
)
from dandelion.utilities._utilities import (
    TRUES,
    NO_DS,
    DEFAULT_PREFIX,
    check_filepath,
    present,
    set_igblast_env,
    set_blast_env,
    check_data,
)


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
    data = pl.read_csv(anno).with_columns(pl.col("*").cast(pl.String))
    if prefix is not None:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(prefix)
                        + separator
                        + str(c).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(c).split("_contig")[1],
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(prefix)
                        + separator
                        + str(b).split("-")[0]
                        + separator
                        + str(suffix),
                        return_dtype=pl.String,
                    ),
                )
            else:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(prefix)
                        + separator
                        + str(c).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(c).split("_contig")[1],
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(prefix)
                        + separator
                        + str(b)
                        + separator
                        + str(suffix),
                        return_dtype=pl.String,
                    ),
                )
        else:
            if remove_trailing_hyphen_number:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(prefix)
                        + separator
                        + str(c).split("_contig")[0].split("-")[0]
                        + "_contig"
                        + str(c).split("_contig")[1],
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(prefix)
                        + separator
                        + str(b).split("-")[0],
                        return_dtype=pl.String,
                    ),
                )
            else:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(prefix) + separator + str(c),
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(prefix) + separator + str(b),
                        return_dtype=pl.String,
                    ),
                )
    else:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(c).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(c).split("_contig")[1],
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(b).split("-")[0]
                        + separator
                        + str(suffix),
                        return_dtype=pl.String,
                    ),
                )
            else:
                data = data.with_columns(
                    pl.col("contig_id").map_elements(
                        lambda c: str(c).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(c).split("_contig")[1],
                        return_dtype=pl.String,
                    ),
                    pl.col("barcode").map_elements(
                        lambda b: str(b) + separator + str(suffix),
                        return_dtype=pl.String,
                    ),
                )
        else:
            data = data.with_columns(
                pl.col("contig_id").cast(pl.String),
                pl.col("barcode").cast(pl.String),
            )
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
        hiconf_contigs = data.filter(pl.col("high_confidence").is_in(TRUES))[
            "contig_id"
        ].to_list()
        seqs = {hiconf: seqs[hiconf] for hiconf in hiconf_contigs}
        data = data.filter(pl.col("contig_id").is_in(hiconf_contigs))
    write_fasta(fasta_dict=seqs, out_fasta=out_fasta)
    data.write_csv(out_anno, quote_style="never")


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

    def _correct_c_call(
        data: pl.DataFrame,
        org: Literal["human", "mouse"] = "human",
        primers_dict: dict[str, dict[str, str]] | None = None,
    ) -> pl.DataFrame:
        """Vectorized pairwise alignment for c genes using Polars."""

        # Build default primer dictionary if none provided
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
                        "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCATAA",
                        "IGLC7": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCGTAA",
                    },
                    "IGLC3": {
                        "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCATAA",
                        "IGLC3": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCATAA",
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

        # Helper function to compute alignment and return best match(es)
        def compute_best_match(
            c_call: str | None, c_seq: str | None
        ) -> str | None:
            if c_call is None or c_seq is None or "IGHG4A" in c_call:
                return c_call

            seq = c_seq.replace("-", "")
            if not seq:
                return c_call

            # Find matching primer group
            for key, genes in primer_dict.items():
                if key in c_call:
                    scores = {
                        gene: aligner.align(seq, primer_seq).score
                        for gene, primer_seq in genes.items()
                    }
                    max_score = max(scores.values())
                    return ",".join(
                        gene
                        for gene, score in scores.items()
                        if score == max_score
                    )

            return c_call

        # Apply vectorized map_elements (more efficient than row iteration)
        result = data.with_columns(
            pl.struct(["c_call", "c_sequence_alignment"])
            .map_elements(
                lambda row: compute_best_match(
                    row["c_call"], row["c_sequence_alignment"]
                ),
                return_dtype=pl.String,
            )
            .alias("c_call")
        )

        return result

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
    # Convert to DataFrame if LazyFrame and deduplicate
    if isinstance(blast_out, pl.LazyFrame):
        blast_out = blast_out.collect()
    blast_out = blast_out.unique(subset=["sequence_id"], keep="first")

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
    dat = load_polars(_processedfile)
    if isinstance(dat, pl.LazyFrame):
        dat = dat.collect()
    if _10xfile is not None:
        dat_10x = read_10x_vdj_polars(_10xfile)
        # Extract c_call and sequence_id from 10x data
        # dat_10x._data is now a Polars LazyFrame
        dat_10x_data = (
            dat_10x._data.collect()
            if isinstance(dat_10x._data, pl.LazyFrame)
            else dat_10x._data
        )
        c_call_10x_df = dat_10x_data.select(
            [
                "sequence_id",
                pl.col("c_call")
                .fill_null("None")
                .replace("", "None")
                .alias("c_call"),
            ]
        )
        # Join with dat using sequence_id as the key
        res_10x = dat.select("sequence_id").join(
            c_call_10x_df, on="sequence_id", how="left"
        )
    else:  # pragma: no cover
        # Create res_10x from dat if no 10x file
        res_10x = dat.select(["c_call", "sequence_id"]).with_columns(
            pl.col("c_call").fill_null("None")
        )
    # Replace columns from blast_out in dat (mimicking pandas behavior: dat[col] = blast_out[col])
    # First, identify which columns to replace
    blast_cols = [
        "c_call",
        "c_sequence_alignment",
        "c_germline_alignment",
        "c_sequence_start",
        "c_sequence_end",
        "c_score",
        "c_identity",
    ]
    cols_to_replace = [
        col for col in blast_cols if col in blast_out.collect_schema()
    ]

    # Drop existing columns from dat to avoid conflicts
    dat = dat.drop(
        [col for col in cols_to_replace if col in dat.collect_schema()]
    )

    # Join blast_out columns into dat
    if cols_to_replace:
        dat = dat.join(
            blast_out.select(["sequence_id"] + cols_to_replace),
            on="sequence_id",
            how="left",
        )

    # Create res_blast from dat's c_call, replace empty with "None"
    res_blast = dat.select("c_call").with_columns(
        pl.col("c_call").fill_null("None").replace("", "None").alias("c_call")
    )

    # Aggregate counts for res_10x_sum, res_blast_sum, etc using polars
    res_10x_sum = (
        res_10x.group_by("c_call")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("counts")
        )
        .drop("count")
        .with_columns(pl.lit("10X").alias("group"))
    )

    res_blast_sum = (
        res_blast.group_by("c_call")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("counts")
        )
        .drop("count")
        .with_columns(pl.lit("blast").alias("group"))
    )

    if (
        correct_c_call
    ):  # TODO: figure out if i need to set up a None correction?
        dat = _correct_c_call(dat, primers_dict=correction_dict, org=org)
        res_corrected = dat.select("c_call").with_columns(
            pl.col("c_call")
            .fill_null("None")
            .replace("", "None")
            .alias("c_call")
        )
        res_corrected_sum = (
            res_corrected.group_by("c_call")
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / pl.col("count").sum() * 100).alias("counts")
            )
            .drop("count")
            .with_columns(pl.lit("corrected").alias("group"))
        )
        res = pl.concat([res_10x_sum, res_blast_sum, res_corrected_sum])
    else:  # pragma: no cover
        res = pl.concat([res_10x_sum, res_blast_sum])

    # Clean up res dataframe
    res = res.with_columns(
        pl.col("c_call")
        .fill_null("None")
        .str.replace_all(r"[*][0-9][0-9]", "")
        .alias("c_call")
    )
    # Add c_call_10x to dat (direct column assignment like pandas)
    # Drop existing c_call_10x column if it exists, then join
    if "c_call_10x" in dat.collect_schema():
        dat = dat.drop("c_call_10x")
    dat = dat.join(
        res_10x.select(["sequence_id", "c_call"]).rename(
            {"c_call": "c_call_10x"}
        ),
        on="sequence_id",
        how="left",
    )

    # Load airr_output and merge columns
    airr_output = load_polars(_airrfile)
    if isinstance(airr_output, pl.LazyFrame):
        airr_output = airr_output.collect()

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
        if x in airr_output.collect_schema():
            dat = dat.join(
                airr_output.select(["sequence_id", x]),
                on="sequence_id",
                how="left",
            )

    # Clean up c_call - remove allelic calls
    dat = dat.with_columns(
        pl.col("c_call")
        .fill_null("")
        .str.replace_all(r"[*][0-9][0-9]", "")
        .alias("c_call")
    )

    _write_airr(dat, _processedfile)
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
        db_pass = load_polars(passfile)
    else:
        db_pass = None
    if failfile.is_file():
        db_fail = load_polars(failfile)
    else:
        db_fail = None
    # load the 10x file
    _10xfile = check_filepath(
        fasta.parent / (fasta.stem + "_annotations.csv"),
        filename_prefix=filename_prefix,
        ends_with="_annotations.csv",
    )
    if _10xfile is not None:
        dat_10x = read_10x_vdj_polars(_10xfile)
    else:
        dat_10x = None
    if db_pass is not None:
        # Ensure it's eager for column access
        if isinstance(db_pass, pl.LazyFrame):
            db_pass = db_pass.collect()

        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_pass.collect_schema():
                    db_pass = db_pass.with_columns(pl.lit("").alias(add_col))
        if dat_10x is not None:
            dat_10x_data = (
                dat_10x._data.collect()
                if isinstance(dat_10x._data, pl.LazyFrame)
                else dat_10x._data
            )
            for col in ["consensus_count", "umi_count"]:
                if all_missing_polars(db_pass[col]):
                    db_pass = (
                        db_pass.join(
                            dat_10x_data.select(["sequence_id", col]),
                            on="sequence_id",
                            how="left",
                            suffix="_10x",
                        )
                        .with_columns(
                            pl.coalesce(
                                [pl.col(col), pl.col(col + "_10x")]
                            ).alias(col)
                        )
                        .drop(col + "_10x")
                    )
        db_pass = _sanitize_data_polars(db_pass)
        if isinstance(db_pass, pl.LazyFrame):
            db_pass = db_pass.collect()
        db_pass.write_csv(passfile, separator="\t", quote_style="never")
    if db_fail is not None:
        if isinstance(db_fail, pl.LazyFrame):
            db_fail = db_fail.collect()
        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_fail.collect_schema():
                    db_fail = db_fail.with_columns(pl.lit("").alias(add_col))
        if dat_10x is not None:
            dat_10x_data = (
                dat_10x._data.collect()
                if isinstance(dat_10x._data, pl.LazyFrame)
                else dat_10x._data
            )
            for col in ["consensus_count", "umi_count"]:
                if all_missing_polars(db_fail[col]):
                    db_fail = (
                        db_fail.join(
                            dat_10x_data.select(["sequence_id", col]),
                            on="sequence_id",
                            how="left",
                            suffix="_10x",
                        )
                        .with_columns(
                            pl.coalesce(
                                [pl.col(col), pl.col(col + "_10x")]
                            ).alias(col)
                        )
                        .drop(col + "_10x")
                    )
        db_fail = _sanitize_data_polars(db_fail)
        if isinstance(db_fail, pl.LazyFrame):
            db_fail = db_fail.collect()
        db_fail.write_csv(failfile, separator="\t", quote_style="never")


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
            _ = load_polars(
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
                _ = load_polars(
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
            _ = load_polars(
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
            heavy = load_polars(
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
        # Convert LazyFrame to DataFrame if needed
        if isinstance(heavy, pl.LazyFrame):
            heavy = heavy.collect()
        heavy = heavy.with_columns(pl.col("v_call").alias("v_call_genotyped"))
    else:
        heavy = load_polars(
            out_dir
            / (out_dir.stem + "_heavy" + fileformat_passed_dict[fileformat])
        )
        if isinstance(heavy, pl.LazyFrame):
            heavy = heavy.collect()

    logg.info(
        "      For convenience, entries for light chain `v_call` are copied to `v_call_genotyped`."
    )
    light = load_polars(
        out_dir / (out_dir.stem + "_light" + germpass_dict[fileformat])
    )
    if isinstance(light, pl.LazyFrame):
        light = light.collect()
    light = light.with_columns(pl.col("v_call").alias("v_call_genotyped"))

    # Initialize sample_id column to None
    heavy = heavy.with_columns(pl.lit(None).alias("sample_id"))
    light = light.with_columns(pl.lit(None).alias("sample_id"))

    for file, sample_id in sampleNames_dict.items():
        dat_f = load_polars(file)
        if isinstance(dat_f, pl.LazyFrame):
            dat_f = dat_f.collect()
        # Add sample_id to dat_f if not present
        if "sample_id" not in dat_f.collect_schema():
            dat_f = dat_f.with_columns(pl.lit(sample_id).alias("sample_id"))
        # Update sample_id in heavy and light based on sequence_id match
        sample_map = dat_f.select(["sequence_id", "sample_id"])
        heavy = (
            heavy.join(
                sample_map,
                on="sequence_id",
                how="left",
                suffix="_new",
            )
            .with_columns(
                pl.coalesce(
                    [pl.col("sample_id_new"), pl.col("sample_id")]
                ).alias("sample_id")
            )
            .drop("sample_id_new")
        )
        light = (
            light.join(
                sample_map,
                on="sequence_id",
                how="left",
                suffix="_new",
            )
            .with_columns(
                pl.coalesce(
                    [pl.col("sample_id_new"), pl.col("sample_id")]
                ).alias("sample_id")
            )
            .drop("sample_id_new")
        )

    # Concatenate heavy and light with aligned schemas to handle differences
    # Ensure eager DataFrames
    if isinstance(heavy, pl.LazyFrame):
        heavy = heavy.collect()
    if isinstance(light, pl.LazyFrame):
        light = light.collect()

    # Align columns: add missing columns to each with nulls and enforce same order
    heavy_cols = set(heavy.collect_schema().names())
    light_cols = set(light.collect_schema().names())
    missing_in_heavy = sorted(list(light_cols - heavy_cols))
    missing_in_light = sorted(list(heavy_cols - light_cols))

    if missing_in_heavy:
        heavy = heavy.with_columns(
            [pl.lit(None).alias(c) for c in missing_in_heavy]
        )
    if missing_in_light:
        light = light.with_columns(
            [pl.lit(None).alias(c) for c in missing_in_light]
        )

    # Enforce identical column order across both frames
    all_cols = sorted(list(heavy_cols | light_cols))
    heavy = heavy.select(all_cols)
    light = light.select(all_cols)

    dat_ = pl.concat([heavy, light], how="vertical_relaxed")

    # Sort using polars
    if "cell_id" in dat_.collect_schema():
        dat_ = dat_.sort("cell_id")
    else:
        dat_ = dat_.sort("sequence_id")

    if plot:
        if "tigger_failed" not in locals():
            logg.info("Returning summary plot")
            # options.figure_size = figsize
            inferred_genotype = out_dir / (
                out_dir.stem + "_heavy" + inferred_fileformat_dict[fileformat]
            )
            # Read inferred genotype table; avoid deprecated/invalid dtypes argument
            inf_geno = pl.read_csv(inferred_genotype, separator="\t")
            # Match original pandas semantics: use TIgGER gene names as-is
            s2 = set(inf_geno["gene"])
            if isinstance(heavy, pl.LazyFrame):
                heavy = heavy.collect()
            heavy = heavy.with_columns(
                [
                    pl.col("v_call")
                    .str.replace(r"\*[0-9]{2}", "")
                    .alias("v_call_clean"),
                    pl.col("v_call_genotyped")
                    .str.replace(r"\*[0-9]{2}", "")
                    .alias("v_call_genotyped_clean"),
                ]
            )
            # Pure Polars computation mirroring pandas semantics exactly
            # 1) Ambiguity: percent of rows with comma-separated calls
            amb_tbl = heavy.group_by("sample_id").agg(
                [
                    pl.col("v_call_clean")
                    .str.contains(",")
                    .mean()
                    .mul(100)
                    .alias("ambiguous_before"),
                    pl.col("v_call_genotyped_clean")
                    .str.contains(",")
                    .mean()
                    .mul(100)
                    .alias("ambiguous_after"),
                    pl.len().alias("total_rows"),
                ]
            )

            # 2) Build per-sample observed allele set from original calls (split/explode, dedup)
            observed_alleles = (
                heavy.select(
                    [
                        pl.col("sample_id"),
                        pl.col("v_call_clean").str.split(",").alias("alleles"),
                    ]
                )
                .explode("alleles")
                .with_columns(pl.col("alleles").alias("allele"))
                .select(["sample_id", "allele"])
                .unique()
            )
            # 3) setdiff across sample: alleles observed but not in inferred genotype
            setdiff_df = observed_alleles.filter(
                ~pl.col("allele").is_in(list(s2))
            )

            # 4) Rows with single allele before and after
            rows_before = heavy.select(
                [
                    pl.col("sample_id"),
                    pl.col("v_call_clean").str.contains(",").alias("has_comma"),
                    pl.col("v_call_clean").alias("allele"),
                ]
            )
            rows_after = heavy.select(
                [
                    pl.col("sample_id"),
                    pl.col("v_call_genotyped_clean")
                    .str.contains(",")
                    .alias("has_comma"),
                    pl.col("v_call_genotyped_clean").alias("allele"),
                ]
            )

            # 5) Identify single-allele rows whose allele is not in genotype (per sample)
            not_in_before_rows = rows_before.filter(~pl.col("has_comma")).join(
                setdiff_df, on=["sample_id", "allele"], how="semi"
            )
            not_in_after_rows = rows_after.filter(~pl.col("has_comma")).join(
                setdiff_df, on=["sample_id", "allele"], how="semi"
            )

            not_in_before_counts = (
                not_in_before_rows.group_by("sample_id")
                .len()
                .rename({"len": "not_in_before"})
            )
            not_in_after_counts = (
                not_in_after_rows.group_by("sample_id")
                .len()
                .rename({"len": "not_in_after"})
            )

            # 6) Combine to percentages using total row counts
            stats = (
                amb_tbl.join(not_in_before_counts, on="sample_id", how="left")
                .join(not_in_after_counts, on="sample_id", how="left")
                .with_columns(
                    [
                        (
                            pl.col("not_in_before").fill_null(0)
                            / pl.col("total_rows")
                            * 100
                        ).alias("not_in_genotype_before"),
                        (
                            pl.col("not_in_after").fill_null(0)
                            / pl.col("total_rows")
                            * 100
                        ).alias("not_in_genotype_after"),
                    ]
                )
                .select(
                    [
                        "sample_id",
                        "ambiguous_before",
                        "ambiguous_after",
                        "not_in_genotype_before",
                        "not_in_genotype_after",
                    ]
                )
            )
            final_table = (
                stats.unpivot(
                    index="sample_id", variable_name="metric", value_name="var"
                )
                .with_columns(
                    [
                        pl.when(pl.col("metric").str.ends_with("_before"))
                        .then(pl.lit("before"))
                        .otherwise(pl.lit("after"))
                        .alias("var_group"),
                        pl.col("metric")
                        .str.replace("_before", "")
                        .str.replace("_after", "")
                        .alias("vgroup"),
                    ]
                )
                .select("sample_id", "vgroup", "var_group", "var")
                .with_columns(pl.col("var_group").cast(pl.Categorical))
            )
            options.figure_size = figsize
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
            sample_id_val = sample_id_dictionary[s]
        else:
            sample_id_val = s

        # Filter using polars
        out_file = dat_.filter(pl.col("sample_id") == sample_id_val)

        outfilepath = filePath_dict[s]
        _write_airr(
            out_file, outfilepath.parent / (outfilepath.stem + "_genotyped.tsv")
        )


def create_germlines(
    vdj_data: DandelionPolars | pl.DataFrame | pl.LazyFrame | str,
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
) -> DandelionPolars:
    """
    Run CreateGermlines.py to reconstruct the germline V(D)J sequence.

    Parameters
    ----------
    vdj_data : DandelionPolars | pl.DataFrame | pl.LazyFrame | str
        DandelionPolars object, polars DataFrame/LazyFrame in changeo/airr format, or file path to changeo/airr
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
    DandelionPolars
        DandelionPolars object with `.germlines` slot populated.
    """
    start = logg.info("Reconstructing germline sequences")
    if not isinstance(vdj_data, DandelionPolars):
        tmpfile = (
            Path(vdj_data)
            if os.path.isfile(vdj_data)
            else Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        )
        if isinstance(vdj_data, (pl.DataFrame, pl.LazyFrame)):
            _write_airr(data=vdj_data, save=tmpfile)
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
        vdj_data._write_airr(filename=tmpfile)
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
    # return as DandelionPolars object
    germpass_outfile = tmpfile.parent / (tmpfile.stem + "_germ-pass.tsv")
    if isinstance(vdj_data, DandelionPolars):
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
        out_vdj = DandelionPolars(germpass_outfile, verbose=False)
        out_vdj.store_germline_reference(
            corrected=genotyped_fasta, germline=germline, org=org
        )
    if save is not None:
        shutil.move(germpass_outfile, save)
    logg.info(
        " finished",
        time=start,
        deep=("Returning DandelionPolars object: \n"),
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
        blast_result=blast_out.unique(subset=["sequence_id"], keep="first"),
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
) -> pl.LazyFrame:
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
    pl.LazyFrame
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
        dat = pl.read_csv(
            blast_out,
            separator="\t",
            has_header=False,
            new_columns=[
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
            ],
        ).lazy()
    except Exception:
        dat = pl.DataFrame(
            {
                "sequence_id": [],
                call + "_call": [],
                call + "_identity": [],
                call + "_alignment_length": [],
                call + "_number_of_mismatches": [],
                call + "_number_of_gap_openings": [],
                call + "_sequence_start": [],
                call + "_sequence_end": [],
                call + "_germline_start": [],
                call + "_germline_end": [],
                call + "_support": [],
                call + "_score": [],
                call + "_sequence_alignment": [],
                call + "_germline_alignment": [],
            }
        ).lazy()
    # Write directly without AIRR validation (these are just raw BLAST results)
    dat_collected = dat.collect()
    if len(dat_collected) > 0:
        dat_collected.write_csv(blast_out, separator="\t", quote_style="never")
    return dat


def transfer_assignment(
    passfile: str,
    failfile: str,
    blast_result: pl.LazyFrame,
    call: Literal["v", "d", "j", "c"] = "c",
    overwrite: bool = False,
):
    """Update gene calls with blastn results using pure polars operations.

    Parameters
    ----------
    passfile : str
        path to db-pass.tsv file.
    failfile : str
        path to db-fail.tsv file.
    blast_result : pl.LazyFrame
        polars LazyFrame with blastn results.
    call : Literal["v", "d", "j", "c"], optional
        which gene call.
    overwrite : bool, optional
        whether or not to overwrite.
    """
    if os.path.isfile(passfile):
        db_pass = load_polars(passfile)
    else:
        db_pass = None
    if os.path.isfile(failfile):
        db_fail = load_polars(failfile)
        if isinstance(db_fail, pl.LazyFrame):
            db_fail = db_fail.collect()
        # Fill in missing values
        db_fail = db_fail.with_columns(
            pl.col("vj_in_frame").fill_null("F"),
            pl.col("productive").fill_null("F"),
            pl.col("c_call").fill_null(""),
            pl.col("v_call").fill_null(""),
            pl.col("d_call").fill_null(""),
            pl.col("j_call").fill_null(""),
            pl.col("locus").fill_null(""),
        )

        # Fill in locus from v/d/j/c calls where missing
        def infer_locus(row):
            if not present(row["locus"]):
                calls = {
                    (
                        row["v_call"][:3]
                        if present(row["v_call"]) and len(row["v_call"]) >= 3
                        else None
                    ),
                    (
                        row["d_call"][:3]
                        if present(row["d_call"]) and len(row["d_call"]) >= 3
                        else None
                    ),
                    (
                        row["j_call"][:3]
                        if present(row["j_call"]) and len(row["j_call"]) >= 3
                        else None
                    ),
                    (
                        row["c_call"][:3]
                        if present(row["c_call"]) and len(row["c_call"]) >= 3
                        else None
                    ),
                }
                calls.discard(None)
                locus = "".join([c for c in calls if present(c)])
                return locus if len(locus) == 3 else row["locus"]
            return row["locus"]

        db_fail = db_fail.with_columns(
            pl.struct(["locus", "v_call", "d_call", "j_call", "c_call"])
            .map_elements(infer_locus, return_dtype=pl.String)
            .alias("locus")
        )
    else:
        db_fail = None

    # Collect LazyFrame
    if isinstance(blast_result, pl.LazyFrame):
        blast_result = blast_result.collect()
    if blast_result.shape[0] < 1:
        blast_result = None

    # Collect DataFrames if needed
    if db_pass is not None and isinstance(db_pass, pl.LazyFrame):
        db_pass = db_pass.collect()

    if blast_result is not None:
        # Pre-rename blast_result columns to add _blastn suffix (except sequence_id)
        # Polars 0.20+: use a simple mapping dict with DataFrame.rename
        rename_map = {
            col: f"{col}_blastn"
            for col in blast_result.collect_schema()
            if col != "sequence_id"
        }
        blast_result_renamed = blast_result.rename(rename_map)

        if db_pass is not None:
            # Preserve row order by adding an index before join
            db_pass = db_pass.with_row_index("_row_order")

            # Join blast results with db_pass on sequence_id (no suffix needed since we pre-renamed)
            db_pass = (
                db_pass.join(
                    blast_result_renamed,
                    left_on="sequence_id",
                    right_on="sequence_id",
                    how="left",
                )
                .sort("_row_order")
                .drop("_row_order")
            )

            # Store original values and create igblastn columns
            if (
                call + "_support" in db_pass.collect_schema()
                and call + "_support_blastn" not in db_pass.collect_schema()
            ):
                db_pass = db_pass.with_columns(
                    pl.col(call + "_support").alias(call + "_support_igblastn")
                )
            if (
                call + "_score" in db_pass.collect_schema()
                and call + "_score_blastn" not in db_pass.collect_schema()
            ):
                db_pass = db_pass.with_columns(
                    pl.col(call + "_score").alias(call + "_score_igblastn")
                )

            # Create igblastn call column from original call
            db_pass = db_pass.with_columns(
                pl.col(call + "_call")
                .fill_null("")
                .alias(call + "_call_igblastn")
            )

            # Fill null values in blast columns
            for col in [
                call + "_call_blastn",
                call + "_sequence_alignment_blastn",
                call + "_germline_alignment_blastn",
            ]:
                if col in db_pass.collect_schema():
                    db_pass = db_pass.with_columns(pl.col(col).fill_null(""))

            # Initialize _source column
            source_col_name = call + "_source"
            db_pass = db_pass.with_columns(pl.lit("").alias(source_col_name))

            if overwrite:
                # Apply complex assignment logic using when/then chains
                # This updates v/d/j positions if blast results are more reliable
                db_pass = _apply_blast_overrides_polars(
                    db_pass, blast_result_renamed, call
                )

                # Recalculate np1 and np2 lengths based on potentially updated v/d/j positions from blast overrides
                # This mirrors the pandas logic: convert negative values to empty strings, then apply fallback
                db_pass = db_pass.with_columns(
                    pl.when(
                        (pl.col("v_sequence_end").is_not_null())
                        & (pl.col("d_sequence_start").is_not_null())
                    )
                    .then(
                        pl.when(
                            (
                                (
                                    pl.col("d_sequence_start")
                                    - pl.col("v_sequence_end")
                                )
                                - 1
                            )
                            >= 0
                        )
                        .then(
                            (
                                (
                                    pl.col("d_sequence_start")
                                    - pl.col("v_sequence_end")
                                )
                                - 1
                            )
                            .cast(pl.Int64)
                            .cast(pl.String)
                        )
                        .otherwise(pl.lit(""))
                    )
                    .otherwise(pl.lit(None))
                    .alias("np1_length")
                )

                # Recalculate np2_length similarly
                db_pass = db_pass.with_columns(
                    pl.when(
                        (pl.col("d_sequence_end").is_not_null())
                        & (pl.col("j_sequence_start").is_not_null())
                    )
                    .then(
                        pl.when(
                            (
                                (
                                    pl.col("j_sequence_start")
                                    - pl.col("d_sequence_end")
                                )
                                - 1
                            )
                            >= 0
                        )
                        .then(
                            (
                                (
                                    pl.col("j_sequence_start")
                                    - pl.col("d_sequence_end")
                                )
                                - 1
                            )
                            .cast(pl.Int64)
                            .cast(pl.String)
                        )
                        .otherwise(pl.lit(""))
                    )
                    .otherwise(pl.lit(None))
                    .alias("np2_length")
                )

                # Fallback: if np1_length is empty (no D region found) and v/j bounds exist, compute using j-v distance
                db_pass = db_pass.with_columns(
                    pl.when(
                        (pl.col("np1_length") == "")
                        | pl.col("np1_length").is_null()
                    )
                    .then(
                        pl.when(
                            (pl.col("v_sequence_end").is_not_null())
                            & (pl.col("j_sequence_start").is_not_null())
                        )
                        .then(
                            pl.when(
                                (
                                    (
                                        pl.col("j_sequence_start")
                                        - pl.col("v_sequence_end")
                                    )
                                    - 1
                                )
                                >= 0
                            )
                            .then(
                                (
                                    (
                                        pl.col("j_sequence_start")
                                        - pl.col("v_sequence_end")
                                    )
                                    - 1
                                )
                                .cast(pl.Int64)
                                .cast(pl.String)
                            )
                            .otherwise(pl.col("np1_length"))
                        )
                        .otherwise(pl.col("np1_length"))
                    )
                    .otherwise(pl.col("np1_length"))
                    .alias("np1_length")
                )

            # Sanitize and write
            db_pass = _sanitize_data_polars(db_pass)
            _write_airr(db_pass, passfile)

        if db_fail is not None:
            # Preserve row order by adding an index before join
            db_fail = db_fail.with_row_index("_row_order")

            # Same join and processing for db_fail using pre-renamed blast_result
            db_fail = (
                db_fail.join(
                    blast_result_renamed,
                    left_on="sequence_id",
                    right_on="sequence_id",
                    how="left",
                )
                .sort("_row_order")
                .drop("_row_order")
            )

            # Store original values and create igblastn columns
            if (
                call + "_support" in db_fail.collect_schema()
                and call + "_support_blastn" not in db_fail.collect_schema()
            ):
                db_fail = db_fail.with_columns(
                    pl.col(call + "_support").alias(call + "_support_igblastn")
                )
            if (
                call + "_score" in db_fail.collect_schema()
                and call + "_score_blastn" not in db_fail.collect_schema()
            ):
                db_fail = db_fail.with_columns(
                    pl.col(call + "_score").alias(call + "_score_igblastn")
                )

            db_fail = db_fail.with_columns(
                pl.col(call + "_call")
                .fill_null("")
                .alias(call + "_call_igblastn")
            )

            # Fill null values
            for col in [
                call + "_call_blastn",
                call + "_sequence_alignment_blastn",
                call + "_germline_alignment_blastn",
            ]:
                if col in db_fail.collect_schema():
                    db_fail = db_fail.with_columns(pl.col(col).fill_null(""))
            # Initialize _source column
            source_col_name = call + "_source"
            db_fail = db_fail.with_columns(pl.lit("").alias(source_col_name))

            if overwrite:
                db_fail = _apply_blast_overrides_polars(
                    db_fail, blast_result_renamed, call
                )

                # Recalculate np1 and np2 lengths after blast overrides
                # First ensure columns are numeric
                numeric_cols = [
                    "v_sequence_end",
                    "d_sequence_start",
                    "d_sequence_end",
                    "j_sequence_start",
                ]
                for col in numeric_cols:
                    if col in db_fail.collect_schema():
                        db_fail = db_fail.with_columns(
                            pl.col(col).cast(pl.Int64, strict=False)
                        )

                db_fail = db_fail.with_columns(
                    pl.when(
                        (pl.col("v_sequence_end").is_not_null())
                        & (pl.col("d_sequence_start").is_not_null())
                    )
                    .then(
                        pl.when(
                            (
                                (
                                    pl.col("d_sequence_start")
                                    - pl.col("v_sequence_end")
                                )
                                - 1
                            )
                            >= 0
                        )
                        .then(
                            (
                                (
                                    pl.col("d_sequence_start")
                                    - pl.col("v_sequence_end")
                                )
                                - 1
                            )
                            .cast(pl.Int64)
                            .cast(pl.String)
                        )
                        .otherwise(pl.lit(""))
                    )
                    .otherwise(pl.lit(None))
                    .alias("np1_length")
                )

                # Fallback: if np1_length is empty/null and v/j bounds exist, compute using j-v distance
                db_fail = db_fail.with_columns(
                    pl.when(
                        (pl.col("np1_length") == "")
                        | pl.col("np1_length").is_null()
                    )
                    .then(
                        pl.when(
                            pl.col("v_sequence_end").is_not_null()
                            & pl.col("j_sequence_start").is_not_null()
                        )
                        .then(
                            pl.when(
                                (
                                    (
                                        pl.col("j_sequence_start")
                                        - pl.col("v_sequence_end")
                                    )
                                    - 1
                                )
                                >= 0
                            )
                            .then(
                                (
                                    (
                                        pl.col("j_sequence_start")
                                        - pl.col("v_sequence_end")
                                    )
                                    - 1
                                )
                                .cast(pl.Int64)
                                .cast(pl.String)
                            )
                            .otherwise(pl.col("np1_length"))
                        )
                        .otherwise(pl.col("np1_length"))
                    )
                    .otherwise(pl.col("np1_length"))
                    .alias("np1_length")
                )

                # Recalculate np2_length similarly
                db_fail = db_fail.with_columns(
                    pl.when(
                        (pl.col("d_sequence_end").is_not_null())
                        & (pl.col("j_sequence_start").is_not_null())
                    )
                    .then(
                        pl.when(
                            (
                                (
                                    pl.col("j_sequence_start")
                                    - pl.col("d_sequence_end")
                                )
                                - 1
                            )
                            >= 0
                        )
                        .then(
                            (
                                (
                                    pl.col("j_sequence_start")
                                    - pl.col("d_sequence_end")
                                )
                                - 1
                            )
                            .cast(pl.Int64)
                            .cast(pl.String)
                        )
                        .otherwise(pl.lit(""))
                    )
                    .otherwise(pl.lit(None))
                    .alias("np2_length")
                )

            # Sanitize and write
            db_fail = _sanitize_data_polars(db_fail)
            _write_airr(db_fail, failfile)


def _apply_blast_overrides_polars(
    db: pl.DataFrame, blast_result: pl.DataFrame, call: str
) -> pl.DataFrame:
    """Apply blast overrides to calls using polars when/then logic.

    Parameters
    ----------
    db : pl.DataFrame
        Database frame with blast columns joined.
    blast_result : pl.DataFrame
        Blast results.
    call : str
        Which call ('v', 'd', 'j', or 'c').

    Returns
    -------
    pl.DataFrame
        DataFrame with overridden calls where appropriate.
    """
    # Ensure igblastn columns exist
    if (call + "_support_igblastn") not in db.collect_schema():
        if (call + "_support") in db.collect_schema():
            db = db.with_columns(
                pl.col(call + "_support")
                .cast(pl.Float64, strict=False)
                .alias(call + "_support_igblastn")
            )
        else:
            db = db.with_columns(pl.lit(None).alias(call + "_support_igblastn"))
    else:
        # Ensure it's numeric
        db = db.with_columns(
            pl.col(call + "_support_igblastn").cast(pl.Float64, strict=False)
        )

    if (call + "_score_igblastn") not in db.collect_schema():
        if (call + "_score") in db.collect_schema():
            db = db.with_columns(
                pl.col(call + "_score")
                .cast(pl.Float64, strict=False)
                .alias(call + "_score_igblastn")
            )
        else:
            db = db.with_columns(pl.lit(None).alias(call + "_score_igblastn"))
    else:
        # Ensure it's numeric
        db = db.with_columns(
            pl.col(call + "_score_igblastn").cast(pl.Float64, strict=False)
        )

    # Ensure blastn support and score columns are numeric
    if (call + "_support_blastn") in db.collect_schema():
        db = db.with_columns(
            pl.col(call + "_support_blastn").cast(pl.Float64, strict=False)
        )
    if (call + "_score_blastn") in db.collect_schema():
        db = db.with_columns(
            pl.col(call + "_score_blastn").cast(pl.Float64, strict=False)
        )

    # Prepare conditions for override logic
    vend = pl.col("v_sequence_end").fill_null(0)
    jstart = pl.col("j_sequence_start").fill_null(1000)
    callstart = pl.col(call + "_sequence_start_blastn")
    callend = pl.col(call + "_sequence_end_blastn")

    # Check if blast hit is in valid region
    in_valid_region = (callstart >= vend) & (callend <= jstart)

    # Check if calls differ
    calls_differ = pl.col(call + "_call_igblastn") != pl.col(
        call + "_call_blastn"
    )

    # Support values
    eval1 = pl.col(call + "_support_igblastn").fill_null(1)
    eval2 = pl.col(call + "_support_blastn")

    # Check if the call_10x column exists and build comparison with stripped allele
    has_call_10x = (call + "_call_10x") in db.collect_schema()
    if has_call_10x:
        blastn_stripped = (
            pl.col(call + "_call_blastn")
            .fill_null("")
            .str.replace_all(r"\*[0-9][0-9]", "")
        )
        matches_10x = (
            (blastn_stripped == pl.col(call + "_call_10x"))
            .fill_null(False)
            .cast(pl.Boolean)
        )
    else:
        matches_10x = pl.lit(False)
    has_call_10x_expr = pl.lit(True) if has_call_10x else pl.lit(False)

    # Evalue presence and comparison conditions
    eval1_present = pl.col(call + "_support_igblastn").is_not_null()
    eval2_present = pl.col(call + "_support_blastn").is_not_null()
    eval1 = pl.col(call + "_support_igblastn")
    eval2 = pl.col(call + "_support_blastn")

    # Only apply override if igblastn call exists OR if eval comparison justifies it
    # Note: when eval1 not present but eval2 is, still apply override (matches pandas logic)
    override_eval = (eval1_present & (eval1 > eval2)) | (
        ~eval1_present & eval2_present
    )

    # Branch conditions
    # When override_eval is True (either eval1 > eval2 OR eval2 present without eval1),
    # apply the override regardless of whether igblastn call exists
    differ_override = (
        in_valid_region
        & calls_differ
        & (
            (has_call_10x_expr & (~matches_10x) & override_eval)
            | ((~has_call_10x_expr) & override_eval)
        )
    )
    same_override = in_valid_region & (~calls_differ) & override_eval
    tenx_match = (
        in_valid_region & calls_differ & (has_call_10x_expr & matches_10x)
    )

    # Apply updates conditionally: calls and coordinates
    db = db.with_columns(
        pl.when(differ_override | same_override)
        .then(pl.col(call + "_call_blastn"))
        .when(tenx_match)
        .then(pl.col(call + "_call_blastn"))
        .otherwise(pl.col(call + "_call"))
        .alias(call + "_call"),
        pl.when(differ_override | same_override)
        .then(pl.col(call + "_sequence_start_blastn"))
        .otherwise(pl.col(call + "_sequence_start"))
        .alias(call + "_sequence_start"),
        pl.when(differ_override | same_override)
        .then(pl.col(call + "_sequence_end_blastn"))
        .otherwise(pl.col(call + "_sequence_end"))
        .alias(call + "_sequence_end"),
        pl.when(differ_override | same_override)
        .then(pl.col(call + "_germline_start_blastn"))
        .otherwise(pl.col(call + "_germline_start"))
        .alias(call + "_germline_start"),
        pl.when(differ_override | same_override)
        .then(pl.col(call + "_germline_end_blastn"))
        .otherwise(pl.col(call + "_germline_end"))
        .alias(call + "_germline_end"),
        pl.when(differ_override | same_override)
        .then(pl.lit("blastn"))
        .when(tenx_match)
        .then(pl.lit("10x"))
        .otherwise(pl.col(call + "_source"))
        .alias(call + "_source"),
    )

    # If 10x match branch, update junction and junction_aa when present and different
    if ("junction_10x" in db.collect_schema()) and (
        "junction" in db.collect_schema()
    ):
        db = db.with_columns(
            pl.when(
                tenx_match
                & pl.col("junction_10x").is_not_null()
                & pl.col("junction").is_not_null()
                & (pl.col("junction") != pl.col("junction_10x"))
            )
            .then(pl.col("junction_10x"))
            .otherwise(pl.col("junction"))
            .alias("junction")
        )
    if ("junction_10x_aa" in db.collect_schema()) and (
        "junction_aa" in db.collect_schema()
    ):
        db = db.with_columns(
            pl.when(
                tenx_match
                & pl.col("junction_10x_aa").is_not_null()
                & pl.col("junction_aa").is_not_null()
                & (pl.col("junction_aa") != pl.col("junction_10x_aa"))
            )
            .then(pl.col("junction_10x_aa"))
            .otherwise(pl.col("junction_aa"))
            .alias("junction_aa")
        )

    return db


def choose_segments(
    starts: pl.Series, ends: pl.Series, scores: pl.Series
) -> list[int]:
    """Choose left most segment using greedy algorithm.

    Parameters
    ----------
    starts : pl.Series
        nucleotide start positions.
    ends : pl.Series
        nucleotide end positions.
    scores : pl.Series
        alignment scores.

    Returns
    -------
    list[int]
        list of chosen segment indices.
    """
    starts = starts.to_numpy()
    ends = ends.to_numpy()
    scores = scores.to_numpy()
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


def multimapper(filename: str) -> pl.DataFrame:
    """Select the left most segment as the final call.

    Parameters
    ----------
    filename : str
        path to multimapper file.

    Returns
    -------
    pl.DataFrame
        Mapped multimapper data frame.
    """
    # Read with Polars
    df = pl.read_csv(filename, separator="\t")

    # Filter rows where j_support < 1e-3
    df_filtered = df.filter(pl.col("j_support") < 1e-3)

    # Get all unique sequence_ids for reindexing later
    all_sequence_ids = df_filtered["sequence_id"].unique().sort()

    # Group and process
    mapped = (
        df_filtered.group_by("sequence_id")
        .agg(
            [
                # Polars automatically creates lists in aggregation
                pl.col("j_sequence_start"),
                pl.col("j_sequence_end"),
                pl.col("j_support"),
                pl.col("j_call"),
            ]
        )
        .with_columns(
            # Now map_elements works on each row (group) where values are lists
            pl.struct(
                [
                    "j_sequence_start",
                    "j_sequence_end",
                    "j_support",
                    "j_call",
                ]
            )
            .map_elements(
                lambda group_struct: process_group_vectorized(group_struct),
                return_dtype=pl.Struct(
                    {
                        "multimappers": pl.String,
                        "multiplicity": pl.Int64,
                        "sequence_start_multimappers": pl.String,
                        "sequence_end_multimappers": pl.String,
                        "support_multimappers": pl.String,
                    }
                ),
            )
            .alias("result")
        )
        .unnest("result")
        .select(
            [
                "sequence_id",
                "multimappers",
                "multiplicity",
                "sequence_start_multimappers",
                "sequence_end_multimappers",
                "support_multimappers",
            ]
        )
    )

    # Reindex to include all sequence_ids (with nulls for missing)
    result = pl.DataFrame({"sequence_id": all_sequence_ids}).join(
        mapped, on="sequence_id", how="left"
    )

    return result


def process_group_vectorized(group_data: dict) -> dict:
    """Process a single group to select non-overlapping segments.

    Parameters
    ----------
    group_data : dict
        Dictionary containing the struct fields (as lists) for a group.

    Returns
    -------
    dict
        Dictionary with aggregated results.
    """
    # Extract lists from struct
    # In Polars, when map_elements is called on struct of lists, each field is a list
    starts_list = group_data["j_sequence_start"]
    ends_list = group_data["j_sequence_end"]
    supports_list = group_data["j_support"]
    calls_list = group_data["j_call"]

    # Convert lists to Polars Series
    starts = pl.Series(starts_list)
    ends = pl.Series(ends_list)
    supports = pl.Series(supports_list)
    calls = pl.Series(calls_list)

    # Use negative scores for maximization (greedy algorithm selects highest score)
    scores = -supports

    # Get chosen indices
    chosen_ind = choose_segments(starts, ends, scores)

    # Select and sort the chosen segments
    selected = pl.DataFrame(
        {
            "j_call": calls[chosen_ind],
            "j_sequence_start": starts[chosen_ind],
            "j_sequence_end": ends[chosen_ind],
            "j_support": supports[chosen_ind],
        }
    ).sort("j_sequence_start")

    return {
        "multimappers": json.dumps(selected["j_call"].to_list()),
        "multiplicity": len(selected),
        "sequence_start_multimappers": json.dumps(
            selected["j_sequence_start"].to_list()
        ),
        "sequence_end_multimappers": json.dumps(
            selected["j_sequence_end"].to_list()
        ),
        "support_multimappers": json.dumps(selected["j_support"].to_list()),
    }


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

    for i in range(len(data)):
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

            # Process db-pass file
            if filePath1 is not None:
                dbpass = load_polars(filePath1)
                if isinstance(dbpass, pl.LazyFrame):
                    dbpass = dbpass.collect()
                dbpass = update_j_cols_polars(dbpass, jmulti, jmm_transfer_cols)
                _write_airr(dbpass, filePath1)

            # Process db-pass genotyped file
            if filePath1g is not None:
                dbpassg = load_polars(filePath1g)
                if isinstance(dbpassg, pl.LazyFrame):
                    dbpassg = dbpassg.collect()
                dbpassg = update_j_cols_polars(
                    dbpassg, jmulti, jmm_transfer_cols
                )
                _write_airr(dbpassg, filePath1g)

            # Process db-all file (with additional logic for missing v_call)
            if filePath2 is not None:
                dbfail = load_polars(filePath2)
                if isinstance(dbfail, pl.LazyFrame):
                    dbfail = dbfail.collect()
                dbfail = update_j_cols_polars(dbfail, jmulti, jmm_transfer_cols)
                dbfail = update_missing_vcall_polars(dbfail)
                _write_airr(dbfail, filePath2)

            # Process db-fail file
            if filePath3 is not None:
                dball = load_polars(filePath3)
                if isinstance(dball, pl.LazyFrame):
                    dball = dball.collect()
                dball = update_j_cols_polars(dball, jmulti, jmm_transfer_cols)
                dball = update_missing_vcall_polars(dball)
                _write_airr(dball, filePath3)

            # Process dandelion file
            if filePath4 is not None:
                dandy = load_polars(filePath4)
                if isinstance(dandy, pl.LazyFrame):
                    dandy = dandy.collect()
                dandy = update_j_cols_polars(dandy, jmulti, jmm_transfer_cols)
                _write_airr(dandy, filePath4)


def update_j_cols_polars(
    airrdata: pl.DataFrame, jmulti: pl.DataFrame, cols: list[str]
) -> pl.DataFrame:
    """Update j_call columns using vectorized Polars operations.

    Parameters
    ----------
    airrdata : pl.DataFrame
        The airr dataframe to update.
    jmulti : pl.DataFrame
        The jmulti dataframe to update from.
    cols : list[str]
        The columns to update.

    Returns
    -------
    pl.DataFrame
        Updated dataframe.
    """
    # Prepare jmulti with renamed columns
    jmulti_renamed = jmulti.select(
        [
            pl.col("sequence_id"),
            *[pl.col(col).alias(f"j_call_{col}") for col in cols],
        ]
    )

    # Left join to preserve all rows in airrdata
    result = airrdata.join(jmulti_renamed, on="sequence_id", how="left")

    # Fill missing values and handle multiplicity
    for col in cols:
        new_col = f"j_call_{col}"
        if new_col not in airrdata.collect_schema():
            # Column didn't exist, keep the joined values with null fill
            if col == "multiplicity":
                result = result.with_columns(
                    pl.col(new_col).fill_null(0).cast(pl.Int64)
                )
            else:
                result = result.with_columns(pl.col(new_col).fill_null(""))
        else:
            # Column existed, update only non-null values from jmulti
            if col == "multiplicity":
                result = result.with_columns(
                    pl.when(pl.col(f"{new_col}_right").is_not_null())
                    .then(pl.col(f"{new_col}_right"))
                    .otherwise(pl.col(new_col))
                    .fill_null(0)
                    .cast(pl.Int64)
                    .alias(new_col)
                ).drop(f"{new_col}_right")
            else:
                result = result.with_columns(
                    pl.when(pl.col(f"{new_col}_right").is_not_null())
                    .then(pl.col(f"{new_col}_right"))
                    .otherwise(pl.col(new_col))
                    .fill_null("")
                    .alias(new_col)
                ).drop(f"{new_col}_right")

    return result


def update_missing_vcall_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Update j_call fields for rows with missing v_call using vectorized operations.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to update.

    Returns
    -------
    pl.DataFrame
        Updated dataframe.
    """
    # Check if v_call is missing (null or empty string)
    has_missing_vcall = pl.col("v_call").is_null() | (
        pl.col("v_call").cast(pl.String) == ""
    )

    # Only try to extract if we have rows with missing v_call and multiplicity > 1
    condition = has_missing_vcall & (pl.col("j_call_multiplicity") > 1)

    # Check if any condition is true to avoid unnecessary processing
    if df.filter(condition).height == 0:
        # No rows match the condition, return df unchanged
        return df

    # Safe JSON extraction helper
    def safe_extract_first(col_name: str, dtype: pl.DataType) -> pl.Expr:
        """Safely extract first element from JSON array."""
        return (
            pl.when(pl.col(col_name).is_null() | (pl.col(col_name) == ""))
            .then(None)
            .otherwise(
                pl.col(col_name)
                .str.json_decode(dtype=pl.List(dtype))
                .list.first()
            )
        )

    # Update j_call fields for rows where v_call is missing and multiplicity > 1
    result = df.with_columns(
        [
            pl.when(condition)
            .then(safe_extract_first("j_call_multimappers", pl.String))
            .otherwise(pl.col("j_call"))
            .alias("j_call"),
            pl.when(condition)
            .then(
                safe_extract_first(
                    "j_call_sequence_start_multimappers", pl.Float64
                ).cast(pl.Float64)
            )
            .otherwise(pl.col("j_sequence_start"))
            .alias("j_sequence_start"),
            pl.when(condition)
            .then(
                safe_extract_first(
                    "j_call_sequence_end_multimappers", pl.Float64
                ).cast(pl.Float64)
            )
            .otherwise(pl.col("j_sequence_end"))
            .alias("j_sequence_end"),
            pl.when(condition)
            .then(
                safe_extract_first(
                    "j_call_support_multimappers", pl.Float64
                ).cast(pl.Float64)
            )
            .otherwise(pl.col("j_support"))
            .alias("j_support"),
        ]
    )

    return result


def check_multimapper(filename1: str, filename2: str) -> None:
    """Filter multimapper file based on reference file using vectorized operations.

    Parameters
    ----------
    filename1 : str
        Path to multimapper file.
    filename2 : str
        Path to reference file containing all information.
    """
    if filename1 is None or filename2 is None:
        return

    # Read multimapper data
    df = pl.read_csv(filename1, separator="\t")
    df_filtered = df.filter(pl.col("j_support") < 1e-3)

    # Read reference data
    df_ref = load_polars(filename2)
    if isinstance(df_ref, pl.LazyFrame):
        df_ref = df_ref.collect()

    # Join to get v_sequence_end for each sequence_id
    df_with_vend = df_filtered.join(
        df_ref.select(["sequence_id", "v_sequence_end"]),
        on="sequence_id",
        how="left",
    )

    # Filter: keep rows where j_sequence_start >= v_sequence_end (or v_sequence_end is null/0)
    keep_df = df_with_vend.filter(
        pl.col("j_sequence_start") >= pl.col("v_sequence_end").fill_null(0)
    ).drop("v_sequence_end")

    # Write filtered data back
    keep_df.write_csv(filename1, separator="\t")


def mask_dj(
    data: list[Path | str],
    filename_prefix: list[str],
    d_evalue_threshold: float,
    j_evalue_threshold: float,
) -> None:
    """Mask d/j assignment using vectorized operations.

    Parameters
    ----------
    data : list[Path | str]
        Input folders.
    filename_prefix : list[str]
        Prefixes to append to front of files.
    d_evalue_threshold : float
        Threshold for d_support_blastn.
    j_evalue_threshold : float
        Threshold for j_support_blastn.
    """
    for i in range(len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass.tsv",
        )

        if filePath is not None:
            dat = load_polars(filePath)
            if isinstance(dat, pl.LazyFrame):
                dat = dat.collect()

            # Mask d_call based on d_support_blastn threshold
            if "d_support_blastn" in dat.collect_schema():
                dat = dat.with_columns(
                    pl.when(pl.col("d_support_blastn") > d_evalue_threshold)
                    .then(pl.lit(""))
                    .otherwise(pl.col("d_call"))
                    .alias("d_call")
                )

            # Mask j_call based on j_support_blastn threshold
            if "j_support_blastn" in dat.collect_schema():
                dat = dat.with_columns(
                    pl.when(pl.col("j_support_blastn") > j_evalue_threshold)
                    .then(pl.lit(""))
                    .otherwise(pl.col("j_call"))
                    .alias("j_call")
                )

            _write_airr(dat, filePath)


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

    for i in range(len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with=informat_dict[fileformat],
            sub_dir="tmp",
        )

        if filePath is not None:
            tmp = load_polars(filePath)
            if isinstance(tmp, pl.LazyFrame):
                tmp = tmp.collect()

            tmp = _check_travdv_polars(tmp, lazy=False)

            _airrfile = str(filePath).replace("_db-pass.tsv", ".tsv")
            airr_output = load_polars(_airrfile)
            if isinstance(airr_output, pl.LazyFrame):
                airr_output = airr_output.collect()

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

            # Merge columns from airr_output into tmp using vectorized operations
            # First, ensure both DataFrames have the same number of rows
            if len(tmp) == len(airr_output):
                # Select only the columns that exist in airr_output
                existing_cols = [
                    col
                    for col in cols_to_merge
                    if col in airr_output.collect_schema()
                ]

                # Drop existing columns from tmp to avoid conflicts
                tmp = tmp.drop(
                    [
                        col
                        for col in existing_cols
                        if col in tmp.collect_schema()
                    ]
                )

                # Add columns from airr_output
                for col in existing_cols:
                    tmp = tmp.with_columns(airr_output[col].alias(col))

            _write_airr(tmp, filePath)

            fp = Path(filePath)
            shutil.copyfile(fp, fp.parent.parent / fp.name)


def make_all(
    data: list[Path | str],
    filename_prefix: list[str] | str | None = None,
    loci: Literal["ig", "tr"] = "tr",
) -> None:
    """Construct db-all tsv file using vectorized Polars operations."""
    data, filename_prefix = check_data(data, filename_prefix)

    for i in range(len(data)):
        if loci == "tr":
            filePath1 = check_filepath(
                data[i],
                filename_prefix=filename_prefix[i],
                ends_with="_igblast_db-pass.tsv",
                sub_dir="tmp",
            )
            out_ex = "db-pass.tsv"
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
            # Load and process df1
            df1 = pl.read_csv(filePath1, separator="\t")
            df1 = check_complete_polars(df1)
            _write_airr(df1, filePath1)

            # Construct output path
            output_path = filePath1.parent / (
                filePath1.name.rsplit(out_ex)[0] + "db-all.tsv"
            )

            if filePath2 is not None:
                # Load and process df2
                df2 = pl.read_csv(filePath2, separator="\t")
                df2 = check_complete_polars(df2)

                # Align schemas before concatenating
                df1_aligned, df2_aligned = align_schemas(df1, df2)

                # Concatenate with aligned schemas
                df = pl.concat([df1_aligned, df2_aligned], how="diagonal")

                _write_airr(df, output_path)
                _write_airr(df2, filePath2)
            else:
                # Just write df1 as db-all
                _write_airr(df1, output_path)


def align_schemas(
    df1: pl.DataFrame, df2: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align schemas between two DataFrames by casting to compatible types.

    Parameters
    ----------
    df1 : pl.DataFrame
        First DataFrame.
    df2 : pl.DataFrame
        Second DataFrame.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Both DataFrames with aligned schemas.
    """
    # Get common columns
    common_cols = set(df1.collect_schema().names()) & set(
        df2.collect_schema().names()
    )

    # Cast common columns to compatible types
    for col in common_cols:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype

        if dtype1 != dtype2:
            # If one is String and other is numeric, cast both to String
            if (
                dtype1 == pl.String
                and dtype2 in [pl.Int64, pl.Float64, pl.Int32]
            ) or (
                dtype2 == pl.String
                and dtype1 in [pl.Int64, pl.Float64, pl.Int32]
            ):
                df1 = df1.with_columns(pl.col(col).cast(pl.String))
                df2 = df2.with_columns(pl.col(col).cast(pl.String))

            # If both are numeric but different types, cast to Float64
            elif dtype1 in [pl.Int64, pl.Float64, pl.Int32] and dtype2 in [
                pl.Int64,
                pl.Float64,
                pl.Int32,
            ]:
                df1 = df1.with_columns(pl.col(col).cast(pl.Float64))
                df2 = df2.with_columns(pl.col(col).cast(pl.Float64))

    return df1, df2


def check_complete_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Check if contig contains cdr3 using vectorized operations.

    Parameters
    ----------
    df : pl.DataFrame
        airr data frame.

    Returns
    -------
    pl.DataFrame
        completed airr data frame
    """
    # Ensure complete_vdj column exists
    if "complete_vdj" not in df.collect_schema():
        df = df.with_columns(pl.lit("").alias("complete_vdj"))

    # Check if junction is missing (null or empty)
    junction_missing = pl.col("junction").is_null() | (
        pl.col("junction").cast(pl.String) == ""
    )

    # Update productive and complete_vdj for rows with missing junction
    result = df.with_columns(
        [
            pl.when(junction_missing)
            .then(pl.lit("F"))
            .otherwise(pl.col("productive"))
            .alias("productive"),
            pl.when(junction_missing)
            .then(pl.lit("F"))
            .otherwise(pl.col("complete_vdj"))
            .alias("complete_vdj"),
        ]
    )

    return result


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


def safe_json_load(s: list | str | None) -> list:
    """Safely load json arrays."""
    if not s:  # empty string or None
        return []  # fallback empty list
    return json.loads(s)
