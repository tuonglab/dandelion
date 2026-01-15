#!/usr/bin/env python
import os
import sys
import shutil
import tempfile

import polars as pl

from changeo.Gene import getGene
from pathlib import Path
from scanpy import logging as logg
from subprocess import run
from typing import Literal

from dandelion.utilities._polars import (
    DandelionPolars,
    load_polars,
    _write_airr,
)
from dandelion.utilities._core import write_fasta
from dandelion.utilities._utilities import (
    set_germline_env,
    set_igblast_env,
    NO_DS,
)


def assigngenes_igblast(
    fasta: Path | str,
    igblast_db: Path | str | None = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
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
    additional_args : list[str], optional
        Additional arguments to pass to `AssignGenes.py`.
    """
    env, igdb, fasta = set_igblast_env(igblast_db=igblast_db, input_file=fasta)
    outfolder = fasta.parent / "tmp"
    outfolder.mkdir(parents=True, exist_ok=True)

    informat_dict = {"blast": "_igblast.fmt7", "airr": "_igblast.tsv"}

    for fileformat in ["blast", "airr"]:
        outfile = fasta.stem + informat_dict[fileformat]
        cmd = [
            "AssignGenes.py",
            "igblast",
            "-s",
            str(fasta),
            "-b",
            str(igdb),
            "--organism",
            org,
            "--loci",
            loci,
            "--format",
            fileformat,
            "-o",
            str(outfolder / outfile),
        ]
        cmd += additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def makedb_igblast(
    fasta: Path | str,
    igblast_output: Path | str | None = None,
    germline: str | None = None,
    org: Literal["human", "mouse"] = "human",
    db: Literal["imgt", "ogrdb"] = "imgt",
    extended: bool = True,
    additional_args: list[str] = [],
    loci: Literal["ig", "tr"] = "ig",
):
    """
    Parse IgBLAST output to AIRR format.

    Parameters
    ----------
    fasta : Path | str
        path to fasta file used for reannotation.
    igblast_output : Path | str | None, optional
        path to igblast output file.
    germline : str | None, optional
        path to germline database.
    org : Literal["human", "mouse"], optional
        organism of germline sequences.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database for running igblastn.
    extended : bool, optional
        whether or not to parse extended 10x annotations.
    additional_args: list[str], optional
        Additional arguments to pass to `MakeDb.py`.
    """
    env, gml, fasta = set_germline_env(
        germline=germline,
        org=org,
        input_file=fasta,
        db=db,
    )
    if igblast_output is None:
        indir = fasta.parent / "tmp"
        infile = fasta.stem + "_igblast.fmt7"
        igbo = indir / infile
    else:
        igbo = Path(igblast_output)

    cellranger_annotation = fasta.parent / (fasta.stem + "_annotations.csv")

    if (org == "mouse") and (loci.lower() == "tr"):
        cmd = [
            "MakeDb_gentle.py",
            "igblast",
            "-i",
            str(igbo),
            "-s",
            str(fasta),
            "-r",
            str(gml),
            "--10x",
            str(cellranger_annotation),
        ]
    else:
        cmd = [
            "MakeDb.py",
            "igblast",
            "-i",
            str(igbo),
            "-s",
            str(fasta),
            "-r",
            str(gml),
            "--10x",
            str(cellranger_annotation),
        ]
    if extended:
        cmd = cmd + ["--extended"]
    for add_cmd in [[], ["--failed"]]:
        cmd = cmd + add_cmd + additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def parsedb_heavy(airr_file: Path | str):
    """
    Parse AIRR tsv file (heavy chain contigs only).

    Parameters
    ----------
    airr_file : Path | str
        path to AIRR tsv file.
    """
    outname = Path(airr_file).stem + "_heavy"
    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        str(airr_file),
        "-f",
        "locus",
        "-u",
        "IGH",
        "--logic",
        "all",
        "--regex",
        "--outname",
        outname,
    ]

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)  # logs are printed to terminal


def parsedb_light(airr_file: Path | str):
    """
    Parse AIRR tsv file (light chain contigs only).

    Parameters
    ----------
    airr_file : Path | str
        path to AIRR tsv file.
    """
    outname = Path(airr_file).stem + "_light"
    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        str(airr_file),
        "-f",
        "locus",
        "-u",
        "IG[LK]",
        "--logic",
        "all",
        "--regex",
        "--outname",
        outname,
    ]

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)  # logs are printed to terminal


def creategermlines(
    airr_file: Path | str,
    germline: list[str] | None = None,
    org: Literal["human", "mouse"] = "human",
    genotyped_fasta: str | None = None,
    mode: Literal["heavy", "light"] | None = None,
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
    Wrapper for CreateGermlines.py for reconstructing germline sequences.

    Parameters
    ----------
    airr_file : Path | str
        path to AIRR tsv file.
    germline : list[str] | None, optional
        location to germline fasta files as a list.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    genotyped_fasta : str | None, optional
        location to V genotyped fasta file.
    mode : Literal["heavy", "light"] | None, optional
        whether to run on heavy or light mode. If left as None, heavy and
        light will be run together.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args : list[str], optional
        Additional arguments to pass to `CreateGermlines.py`.
    """
    env, gml, airr_file = set_germline_env(
        germline=germline,
        org=org,
        input_file=airr_file,
        db=db,
    )
    _strain = "_" + strain if strain is not None else ""
    if germline is None:
        if mode == "heavy":
            if genotyped_fasta is None:
                gml_ref = [
                    str(gml / ("imgt_" + org + _strain + "_IGHV.fasta")),
                ]
            else:
                gml_ref = [
                    str(genotyped_fasta),
                ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGHJ.fasta")),
            ]
            _strainD = "" if strain not in NO_DS else _strain
            gml_ref += [
                str(gml / ("imgt_" + org + _strainD + "_IGHD.fasta")),
            ]
        elif mode == "light":
            gml_ref = [
                str(gml / ("imgt_" + org + _strain + "_IGKV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGKJ.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLJ.fasta")),
            ]
        elif mode is None:
            if genotyped_fasta is None:
                gml_ref = [
                    str(gml / ("imgt_" + org + _strain + "_IGHV.fasta")),
                ]
            else:
                gml_ref = [
                    str(genotyped_fasta),
                ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGHJ.fasta")),
            ]
            _strainD = "" if strain not in NO_DS else _strain
            gml_ref += [
                str(gml / ("imgt_" + org + _strainD + "_IGHD.fasta")),
            ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGKV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGKJ.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLJ.fasta")),
            ]
    else:
        if not isinstance(germline, list):
            germline = [str(germline)]
        gml_ref = germline
    cmd = [
        "CreateGermlines.py",
        "-d",
        str(airr_file),
        "-r",
    ]
    cmd = cmd + gml_ref + additional_args

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal


def define_clones(
    vdj: DandelionPolars | pl.DataFrame | pl.LazyFrame | str,
    dist: float,
    action: Literal["first", "set"] = "set",
    model: Literal[
        "ham",
        "aa",
        "hh_s1f",
        "hh_s5f",
        "mk_rs1nf",
        "mk_rs5nf",
        "hs1f_compat",
        "m1n_compat",
    ] = "ham",
    norm: Literal["len", "mut", "none"] = "len",
    doublets: Literal["drop", "count"] = "drop",
    fileformat: Literal["changeo", "airr"] = "airr",
    n_cpus: int | None = None,
    outFilePrefix: str | None = None,
    key_added: str | None = None,
    out_dir: Path | str | None = None,
    additional_args: list[str] = [],
) -> DandelionPolars:
    """
    Find clones using changeo's `DefineClones.py <https://changeo.readthedocs.io/en/stable/tools/DefineClones.html>`__.

    Only callable for BCR data at the moment.

    Parameters
    ----------
    vdj : DandelionPolars | pl.DataFrame | pl.LazyFrame | str
        DandelionPolars object, Polars DataFrame/LazyFrame in changeo/airr format, or file path to changeo/airr file after
        clones have been determined.
    dist : float
        The distance threshold for clonal grouping.
    action : Literal["first", "set"], optional
        Specifies how to handle multiple V(D)J assignments for initial grouping. Default is 'set'.
        The "first" action will use only the first gene listed. The "set" action will use all gene assignments and
        construct a larger gene grouping composed of any sequences sharing an assignment or linked to another sequence
        by a common assignment (similar to single-linkage).
    model : Literal["ham", "aa", "hh_s1f", "hh_s5f", "mk_rs1nf", "mk_rs5nf", "hs1f_compat", "m1n_compat", ], optional
        Specifies which substitution model to use for calculating distance between sequences. Default is 'ham'.
        The "ham" model is nucleotide Hamming distance and "aa" is amino acid Hamming distance. The "hh_s1f" and
        "hh_s5f" models are human specific single nucleotide and 5-mer content models, respectively, from Yaari et al,
        2013. The "mk_rs1nf" and "mk_rs5nf" models are mouse specific single nucleotide and 5-mer content models,
        respectively, from Cui et al, 2016. The "m1n_compat" and "hs1f_compat" models are deprecated models provided
        backwards compatibility with the "m1n" and "hs1f" models in Change-O v0.3.3 and SHazaM v0.1.4. Both 5-mer
        models should be considered experimental.
    norm : Literal["len", "mut", "none"], optional
        Specifies how to normalize distances. Default is 'len'. 'none' (do not normalize), 'len' (normalize by length),
        or 'mut' (normalize by number of mutations between sequences).
    doublets : Literal["drop", "count"], optional
        Option to control behaviour when dealing with heavy chain 'doublets'. Default is 'drop'. 'drop' will filter out
        the doublets while 'count' will retain only the highest umi count contig.
    fileformat : Literal["changeo", "airr"], optional
        Format of V(D)J file/objects. Default is 'airr'. Also accepts 'changeo'.
    n_cpus : int | None, optional
        Number of cpus for parallelization. Default is 1, no parallelization.
    outFilePrefix : str | None, optional
        If specified, the out file name will have this prefix. `None` defaults to 'dandelion_define_clones'
    key_added : str | None, optional
        Column name to add for define_clones.
    out_dir : Path | str | None, optional
        If specified, the files will be written to this directory.
    additional_args : list[str], optional
        Additional arguments to pass to `DefineClones.py`.

    Returns
    -------
    DandelionPolars
        DandelionPolars object with clone_id annotated in `.data` slot and `.metadata` initialized.
    """
    start = logg.info("Finding clones")
    if n_cpus is None:
        nproc = 1
    else:
        nproc = n_cpus

    clone_key = key_added if key_added is not None else "clone_id"

    # Load data as Polars
    if isinstance(vdj, DandelionPolars):
        dat_ = load_polars(vdj._data)
    else:
        dat_ = load_polars(vdj)

    # Ensure we have eager DataFrame for operations
    if isinstance(dat_, pl.LazyFrame):
        dat_ = dat_.collect()

    # Filter ambiguous sequences
    if "ambiguous" in dat_.columns:
        dat = dat_.filter(pl.col("ambiguous") == "F")
    else:
        dat = dat_

    # Split into heavy and light chains
    dat_h = dat.filter(pl.col("locus") == "IGH")
    dat_l = dat.filter(pl.col("locus").is_in(["IGK", "IGL"]))

    # Setup output directories
    if os.path.isfile(str(vdj)):
        vdj_path = Path(vdj)
        tmpFolder = vdj_path.parent / "tmp"
        outFolder = vdj_path.parent
    elif out_dir is not None:
        vdj_path = Path(out_dir)
        tmpFolder = vdj_path / "tmp"
        outFolder = vdj_path
    else:
        outFolder = Path(tempfile.TemporaryDirectory().name)
        tmpFolder = outFolder / "tmp"

    for folder in [outFolder, tmpFolder]:
        folder.mkdir(parents=True, exist_ok=True)

    # Setup file paths
    if "vdj_path" in locals():
        h_file1 = tmpFolder / (vdj_path.stem + "_heavy-clone.tsv")
        h_file2 = outFolder / (vdj_path.stem + "_heavy-clone.tsv")
        l_file = tmpFolder / (vdj_path.stem + "_light.tsv")
        outfile = outFolder / (vdj_path.stem + "_clone.tsv")
    else:
        out_FilePrefix = (
            "dandelion_define_clones"
            if outFilePrefix is None
            else outFilePrefix
        )
        h_file1 = tmpFolder / (out_FilePrefix + "_heavy-clone.tsv")
        h_file2 = outFolder / (out_FilePrefix + "_heavy-clone.tsv")
        l_file = tmpFolder / (out_FilePrefix + "_light.tsv")
        outfile = outFolder / (out_FilePrefix + "_clone.tsv")

    # Write files
    _write_airr(dat_h, h_file1)
    _write_airr(dat_l, l_file)

    # Determine v_call field
    v_field = (
        "v_call_genotyped" if "v_call_genotyped" in dat.columns else "v_call"
    )

    # Build DefineClones.py command
    cmd = [
        "DefineClones.py",
        "-d",
        str(h_file1),
        "-o",
        str(h_file2),
        "--act",
        action,
        "--model",
        model,
        "--norm",
        norm,
        "--dist",
        str(dist),
        "--nproc",
        str(nproc),
        "--vf",
        v_field,
    ]
    cmd = cmd + additional_args

    def clusterLinkage(cell_series, group_series):
        """
        Return a dictionary of {cell_id : cluster_id}.

        that identifies clusters of cells by analyzing their shared
        features (group_series) using single linkage.

        Arguments:
        cell_series (iter): iter of cell ids.
        group_series (iter): iter of group ids.

        Returns:
        dict:  dictionary of {cell_id : cluster_id}.

        """
        # assign initial clusters
        # initial_dict = {cluster1: [cell1], cluster2: [cell1]}
        initial_dict = {}
        for cell, group in zip(cell_series, group_series):
            try:
                initial_dict[group].append(cell)
            except KeyError:
                initial_dict[group] = [cell]

        # naive single linkage clustering (ON^2 best case, ON^3 worst case) ...ie for cells with multiple light chains
        # cluster_dict = {cluster1: [cell1, cell2]}, 2 cells belong in same group if they share 1 light chain
        while True:
            cluster_dict = {}
            for i, group in enumerate(initial_dict.keys()):
                cluster_dict[i] = initial_dict[group]
                for cluster in cluster_dict:
                    # if initial_dict[group] and cluster_dict[cluster] share common cells, add initial_dict[group] to
                    # cluster
                    if cluster != i and any(
                        cell in initial_dict[group]
                        for cell in cluster_dict[cluster]
                    ):
                        cluster_dict[cluster] = (
                            cluster_dict[cluster] + initial_dict[group]
                        )
                        del cluster_dict[i]
                        break
            # break if clusters stop changing, otherwise restart
            if len(cluster_dict.keys()) == len(initial_dict.keys()):
                break
            else:
                initial_dict = cluster_dict.copy()

        # invert cluster_dict for return
        assign_dict = {
            cell: k for k, v in cluster_dict.items() for cell in set(v)
        }

        return assign_dict

    def _lightCluster(heavy_file, light_file, out_file, doublets, fileformat):
        """
        Split heavy chain clones based on light chains using Polars.

        Arguments:
        heavy_file (str): heavy chain input file.
        light_file (str): light chain input file.
        out_file (str): heavy chain output file.
        doublets (str): method for handling multiple heavy chains per cell. one of 'drop' or 'count'.
        fileformat (str): file format. one of 'changeo' or 'airr'.
        """
        # Set column names
        if fileformat == "changeo":
            cell_id = "cell_id"
            clone_id = "clone_id"
            v_call = "v_call"
            j_call = "j_call"
            junction_length = "junction_length"
            umi_count = "umicount"
        elif fileformat == "airr":
            cell_id = "cell_id"
            clone_id = "clone_id"
            v_call = "v_call"
            j_call = "j_call"
            junction_length = "junction_length"
            umi_count = "umi_count"
        else:
            sys.exit("Invalid format %s" % fileformat)

        # Read in heavy and light DataFrames using Polars
        heavy_df = pl.read_csv(
            heavy_file,
            separator="\t",
            null_values=["", "None", "NA"],
            infer_schema_length=10000,
        )
        light_df = pl.read_csv(
            light_file,
            separator="\t",
            null_values=["", "None", "NA"],
            infer_schema_length=10000,
        )

        # Column checking
        expected_heavy_columns = [
            cell_id,
            clone_id,
            v_call,
            j_call,
            junction_length,
            umi_count,
        ]
        if not set(expected_heavy_columns).issubset(heavy_df.columns):
            raise ValueError(
                "Missing one or more columns in heavy chain file: "
                + ", ".join(expected_heavy_columns)
            )
        expected_light_columns = [
            cell_id,
            v_call,
            j_call,
            junction_length,
            umi_count,
        ]
        if not set(expected_light_columns).issubset(light_df.columns):
            raise ValueError(
                "Missing one or more columns in light chain file: "
                + ", ".join(expected_light_columns)
            )

        # Fix types for junction_length
        try:
            heavy_df = heavy_df.with_columns(
                pl.col(junction_length).cast(pl.Int64)
            )
            light_df = light_df.with_columns(
                pl.col(junction_length).cast(pl.Int64)
            )
        except Exception:
            # Already nullable integer type
            pass

        # Filter multiple heavy chains
        if doublets == "drop":
            # Keep only cells with exactly one heavy chain
            cell_counts = heavy_df.group_by(cell_id).len()
            single_cells = cell_counts.filter(pl.col("len") == 1)[cell_id]
            heavy_df = heavy_df.filter(pl.col(cell_id).is_in(single_cells))
            if len(heavy_df) == 0:
                raise ValueError(
                    "Empty heavy chain data, after doublets drop. Are you combining experiments "
                    "in a single file? If so, split your data into multiple files."
                )
        elif doublets == "count":
            # Keep only the highest UMI count contig per cell
            heavy_df = heavy_df.with_columns(pl.col(umi_count).cast(pl.Int64))
            heavy_df = (
                heavy_df.sort(umi_count, descending=True)
                .group_by(cell_id, maintain_order=True)
                .first()
            )

        # Transfer clone IDs from heavy chain df to light chain df
        # Ensure clone_id is string type for consistent handling
        heavy_df = heavy_df.with_columns(pl.col(clone_id).cast(pl.String))
        clone_dict = dict(
            zip(
                heavy_df[cell_id].to_list(),
                heavy_df[clone_id].to_list(),
            )
        )
        light_df = light_df.filter(
            pl.col(cell_id).is_in(list(clone_dict.keys()))
        )
        light_df = light_df.with_columns(
            pl.col(cell_id)
            .map_elements(lambda x: clone_dict.get(x), return_dtype=pl.String)
            .alias(clone_id)
        )

        # Generate a "cluster_dict" of CELL:CLONE dictionary from light df
        # Build grouping key from v_call, j_call, junction_length, and clone_id
        light_grouping = light_df.with_columns(
            (
                pl.col(v_call).map_elements(
                    lambda x: getGene(x) if x else "", return_dtype=pl.String
                )
                + pl.lit(",")
                + pl.col(j_call).map_elements(
                    lambda x: getGene(x) if x else "", return_dtype=pl.String
                )
                + pl.lit(",")
                + pl.col(junction_length).cast(pl.String)
                + pl.lit(",")
                + pl.col(clone_id)
            ).alias("grouping_key")
        )

        cluster_dict = clusterLinkage(
            light_grouping[cell_id].to_list(),
            light_grouping["grouping_key"].to_list(),
        )

        # Add assignments to heavy_df
        heavy_df = heavy_df.filter(
            pl.col(cell_id).is_in(list(cluster_dict.keys()))
        )
        heavy_df = heavy_df.with_columns(
            (
                pl.col(clone_id)
                + pl.lit("_")
                + pl.col(cell_id).map_elements(
                    lambda x: str(cluster_dict.get(x, "")),
                    return_dtype=pl.String,
                )
            ).alias(clone_id)
        )

        # Write heavy chains
        _write_airr(heavy_df, out_file)
        return (heavy_df, light_df)

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)

    h_df, l_df = _lightCluster(
        h_file2, l_file, outfile, doublets=doublets, fileformat=fileformat
    )

    h_df = load_polars(h_df)
    if isinstance(h_df, pl.LazyFrame):
        h_df = h_df.collect()

    # Create a dictionary for cell_id : clone_id from h_df
    linked_clones = dict(
        zip(h_df["cell_id"].to_list(), h_df["clone_id"].to_list())
    )

    # Create a clone_reference
    clone_ref = list(set(h_df["clone_id"].to_list()))
    clone_ref = [
        c.split("_")[1] if c and str(c) != "nan" else c for c in clone_ref
    ]

    l_df = load_polars(l_df)
    if isinstance(l_df, pl.LazyFrame):
        l_df = l_df.collect()

    # Update light chain clone_ids based on linkage
    def update_light_clone(cell_id, clone_id):
        if clone_id in clone_ref:
            return linked_clones.get(cell_id, cell_id + "_notlinked")
        else:
            return cell_id + "_notlinked"

    l_df = l_df.with_columns(
        pl.struct(["cell_id", "clone_id"])
        .map_elements(
            lambda row: update_light_clone(row["cell_id"], row["clone_id"]),
            return_dtype=pl.String,
        )
        .alias("clone_id")
    )

    # Concatenate heavy and light chains
    cloned_ = pl.concat([h_df, l_df], how="diagonal_relaxed")

    # Transfer the new clone_id to the original data
    clone_mapping = dict(
        zip(cloned_["sequence_id"].to_list(), cloned_["clone_id"].to_list())
    )

    dat_ = dat_.with_columns(
        pl.col("sequence_id")
        .map_elements(
            lambda x: clone_mapping.get(x, ""),
            return_dtype=pl.String,
        )
        .alias(str(clone_key))
    )

    # Ensure clone_key has empty strings instead of nulls
    dat_ = dat_.with_columns(pl.col(str(clone_key)).fill_null(""))

    if isinstance(vdj, DandelionPolars):
        vdj._data = dat_
        vdj.update_metadata(clone_key=str(clone_key))
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated DandelionPolars object: \n"
                "   'data', contig-indexed AIRR table\n"
                "   'metadata', cell-indexed observations table\n"
            ),
        )
        return vdj
    else:
        out = DandelionPolars(
            data=dat_,
            verbose=False,
        )
        out.update_metadata(clone_key=str(clone_key))
        logg.info(
            " finished",
            time=start,
            deep=(
                "Returning DandelionPolars object: \n"
                "   'data', contig-indexed AIRR table\n"
                "   'metadata', cell-indexed observations table\n"
            ),
        )
        return out


def create_germlines(
    vdj: DandelionPolars | pl.DataFrame | pl.LazyFrame | str,
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
    vdj : DandelionPolars | pl.DataFrame | pl.LazyFrame | str
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
    if not isinstance(vdj, DandelionPolars):
        tmpfile = (
            Path(vdj)
            if os.path.isfile(vdj)
            else Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        )
        if isinstance(vdj, (pl.DataFrame, pl.LazyFrame)):
            _write_airr(data=vdj, save=tmpfile)
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
        vdj.write_airr(filename=tmpfile)
        if len(vdj.germline) > 0:
            tmpgmlfile = tmppath / "germ.fasta"
            write_fasta(fasta_dict=vdj.germline, out_fasta=tmpgmlfile)
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
    if isinstance(vdj, DandelionPolars):
        vdj.__init__(
            data=germpass_outfile,
            metadata=vdj._metadata,
            germline=vdj.germline,
            layout=vdj.layout,
            graph=vdj.graph,
            verbose=False,
        )
        out_vdj = vdj.copy()
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
