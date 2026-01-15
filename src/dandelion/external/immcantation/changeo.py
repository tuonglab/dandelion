import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from scanpy import logging as logg
from subprocess import run
from typing import Literal

from changeo.Gene import getGene
from dandelion.utilities._core import Dandelion, load_data, write_fasta
from dandelion.utilities._io import write_airr
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
    vdj: Dandelion | pd.DataFrame | str,
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
    outFilePrefix: int | None = None,
    key_added: int | None = None,
    out_dir: Path | str | None = None,
    additional_args: list[str] = [],
) -> Dandelion:
    """
    Find clones using changeo's `DefineClones.py <https://changeo.readthedocs.io/en/stable/tools/DefineClones.html>`__.

    Only callable for BCR data at the moment.

    Parameters
    ----------
    vdj : Dandelion | pd.DataFrame | str
        Dandelion object, pandas DataFrame in changeo/airr format, or file path to changeo/airr file after
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
    outFilePrefix : int | None, optional
        If specified, the out file name will have this prefix. `None` defaults to 'dandelion_define_clones'
    key_added : int | None, optional
        Column name to add for define_clones.
    out_dir : Path | str | None, optional
        If specified, the files will be written to this directory.
    additional_args : list[str], optional
        Additional arguments to pass to `DefineClones.py`.

    Returns
    -------
    Dandelion
        Dandelion object with clone_id annotated in `.data` slot and `.metadata` initialized.
    """
    start = logg.info("Finding clones")
    if n_cpus is None:
        nproc = 1
    else:
        nproc = n_cpus

    clone_key = key_added if key_added is not None else "clone_id"

    if isinstance(vdj, Dandelion):
        dat_ = load_data(vdj._data)
    else:
        dat_ = load_data(vdj)
    if "ambiguous" in dat_:
        dat = dat_[dat_["ambiguous"] == "F"].copy()
    else:
        dat = dat_.copy()
    dat_h = dat[dat["locus"] == "IGH"]
    dat_l = dat[dat["locus"].isin(["IGK", "IGL"])]

    if os.path.isfile(str(vdj)):
        vdj_path = Path(vdj)
        tmpFolder = vdj_path.parent / "tmp"
        outFolder = vdj_path.parent
    elif out_dir is not None:
        vdj_path = Path(out_dir)
        tmpFolder = vdj_path / "tmp"
        outFolder = vdj_path
    else:
        import tempfile

        outFolder = Path(tempfile.TemporaryDirectory().name)
        tmpFolder = outFolder / "tmp"

    for _ in [outFolder, tmpFolder]:
        _.mkdir(parents=True, exist_ok=True)

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
    write_airr(dat_h, h_file1)
    write_airr(dat_l, l_file)
    v_field = (
        "v_call_genotyped" if "v_call_genotyped" in dat.columns else "v_call"
    )

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

    # TODO: might need to remove this function to drop requirement to maintain this as a dependency internally
    def _lightCluster(heavy_file, light_file, out_file, doublets, fileformat):
        """
        Split heavy chain clones based on light chains.

        Arguments:
        heavy_file (str): heavy chain input file.
        light_file (str): light chain input file.
        out_file (str): heavy chain output file.
        doublets (str): method for handling multiple heavy chains per cell. one of 'drop' or 'count'.
        format (str): file format. one of 'changeo' or 'airr'.
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

        # read in heavy and light DFs
        heavy_df = pd.read_csv(
            heavy_file, dtype="object", na_values=["", "None", "NA"], sep="\t"
        )
        light_df = pd.read_csv(
            light_file, dtype="object", na_values=["", "None", "NA"], sep="\t"
        )

        # column checking
        expected_heavy_columns = [
            cell_id,
            clone_id,
            v_call,
            j_call,
            junction_length,
            umi_count,
        ]
        if set(expected_heavy_columns).issubset(heavy_df.columns) is False:
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
        if set(expected_light_columns).issubset(light_df.columns) is False:
            raise ValueError(
                "Missing one or more columns in light chain file: "
                + ", ".join(expected_light_columns)
            )

        # Fix types
        try:
            heavy_df[junction_length] = heavy_df[junction_length].astype("int")
            light_df[junction_length] = light_df[junction_length].astype("int")
        except:
            heavy_df[junction_length] = heavy_df[junction_length].replace(
                np.nan, pd.NA
            )
            light_df[junction_length] = light_df[junction_length].replace(
                np.nan, pd.NA
            )
            heavy_df[junction_length] = heavy_df[junction_length].astype(
                "Int64"
            )
            light_df[junction_length] = light_df[junction_length].astype(
                "Int64"
            )

        # filter multiple heavy chains
        if doublets == "drop":
            heavy_df = heavy_df.drop_duplicates(cell_id, keep=False)
            if heavy_df.empty is True:
                raise ValueError(
                    "Empty heavy chain data, after doublets drop. Are you combining experiments "
                    "in a single file? If so, split your data into multiple files."
                )
        elif doublets == "count":
            heavy_df[umi_count] = heavy_df[umi_count].astype("int")
            heavy_df = heavy_df.groupby(cell_id, sort=False).apply(
                lambda x: x.nlargest(1, umi_count)
            )

        # transfer clone IDs from heavy chain df to light chain df
        clone_dict = {
            v[cell_id]: v[clone_id]
            for k, v in heavy_df[[clone_id, cell_id]].T.to_dict().items()
        }
        light_df = light_df.loc[
            light_df[cell_id].apply(lambda x: x in clone_dict.keys()),
        ]
        light_df[clone_id] = light_df.apply(
            lambda row: clone_dict[row[cell_id]], axis=1
        )

        # generate a "cluster_dict" of CELL:CLONE dictionary from light df  (TODO: use receptor object V/J gene names)
        cluster_dict = clusterLinkage(
            light_df[cell_id],
            light_df.apply(
                lambda row: getGene(row[v_call])
                + ","
                + getGene(row[j_call])
                + ","
                + str(row[junction_length])
                + ","
                + row[clone_id],
                axis=1,
            ),
        )

        # add assignments to heavy_df
        heavy_df = heavy_df.loc[
            heavy_df[cell_id].apply(lambda x: x in cluster_dict.keys()), :
        ]
        heavy_df[clone_id] = (
            heavy_df[clone_id]
            + "_"
            + heavy_df.apply(
                lambda row: str(cluster_dict[row[cell_id]]), axis=1
            )
        )

        # write heavy chains
        write_airr(heavy_df, out_file)
        return (heavy_df, light_df)

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)

    h_df, l_df = _lightCluster(
        h_file2, l_file, outfile, doublets=doublets, fileformat=fileformat
    )

    h_df = load_data(h_df)
    # create a dictionary for cell_id : clone_id from h_df
    linked_clones = dict(zip(h_df["cell_id"], h_df["clone_id"]))

    # create a clone_reference
    clone_ref = list(set(h_df["clone_id"]))
    clone_ref = [c.split("_")[1] if c is not np.nan else c for c in clone_ref]
    l_df = load_data(l_df)

    for x in l_df.index:
        if l_df.loc[x, "clone_id"] in clone_ref:
            l_df.at[x, "clone_id"] = linked_clones[l_df.loc[x, "cell_id"]]
        else:
            try:
                l_df.at[x, "clone_id"] = l_df.loc[x, "cell_id"] + "_notlinked"
            except:
                pass

    cloned_ = pd.concat([h_df, l_df])
    # transfer the new clone_id to the heavy + light file
    dat_[str(clone_key)] = pd.Series(cloned_["clone_id"])
    dat_[str(clone_key)] = dat_[str(clone_key)].fillna("")
    if isinstance(vdj, Dandelion):
        vdj._data[str(clone_key)] = dat_[str(clone_key)]
        vdj.update_metadata(clone_key=str(clone_key))
    else:
        out = Dandelion(
            data=dat_,
            clone_key=clone_key,
            verbose=False,
        )
        return out
    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig-indexed AIRR table\n"
            "   'metadata', cell-indexed observations table\n"
        ),
    )


def create_germlines(
    vdj: Dandelion | pd.DataFrame | str,
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
    """
    Run CreateGermlines.py to reconstruct the germline V(D)J sequence.

    Parameters
    ----------
    vdj : Dandelion | pd.DataFrame | str
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
    if not isinstance(vdj, Dandelion):
        tmpfile = (
            Path(vdj)
            if os.path.isfile(vdj)
            else Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        )
        if isinstance(vdj, pd.DataFrame):
            write_airr(data=vdj.germline, save=tmpfile)
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
    # return as Dandelion object
    germpass_outfile = tmpfile.parent / (tmpfile.stem + "_germ-pass.tsv")
    if isinstance(vdj, Dandelion):
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
        out_vdj = Dandelion(germpass_outfile, verbose=False)
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
