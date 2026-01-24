from pathlib import Path
from importlib.resources import files
from scanpy import logging as logg
from subprocess import run
from typing import Literal

from dandelion.utilities._utilities import set_germline_env


def tigger_genotype(
    airr_file: Path | str,
    v_germline: Path | str | None = None,
    outdir: Path | str | None = None,
    org: Literal["human", "mouse"] = "human",
    fileformat: Literal["airr", "changeo"] = "airr",
    novel_: Literal["YES", "NO"] = "YES",
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
    Reassign alleles with TIgGER in R.

    Parameters
    ----------
    airr_file : Path | str
        path to AIRR tsv file.
    v_germline : Path | str | None, optional
        fasta file containing IMGT-gapped V segment reference germlines.
    outdir : Path | str | None, optional
        output directory. Will be created if it does not exist.
        Defaults to the current working directory.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    fileformat : Literal["airr", "changeo"], optional
        format for running tigger. Default is 'airr'. Also accepts 'changeo'.
    novel_ : Literal["YES", "NO"], optional
        whether or not to run novel allele discovery.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database.
    strain : Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"] | None, optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args : list[str], optional
        Additional arguments to pass to `tigger-genotype.R`.
    """
    env, gml, airr_file = set_germline_env(
        germline=v_germline,
        org=org,
        input_file=airr_file,
        db=db,
    )
    _strain = "_" + strain if strain is not None else ""
    if v_germline is None:
        v_gml = gml / (db + "_" + org + _strain + "_IGHV.fasta")
    else:
        v_gml = Path(v_germline)
    if outdir is not None:
        out_dir = Path(outdir)
    else:
        out_dir = airr_file.parent
    r_script = files("dandelion").joinpath(
        "external/immcantation/tigger-genotype.R"
    )

    cmd = [
        "Rscript",
        # "tigger-genotype.R",
        str(r_script),
        "-d",
        str(airr_file),
        "-r",
        str(v_gml),
        "-n",
        airr_file.stem,
        "-N",
        novel_,
        "-o",
        str(out_dir),
        "-f",
        fileformat,
    ]
    cmd = cmd + additional_args

    print("      Reassigning alleles")
    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal
