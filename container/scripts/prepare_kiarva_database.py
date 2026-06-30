import argparse
import logging
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from utils import fasta_iterator, write_fasta

# ---------------------------------------------------------------------------
# KIARVA BCR germline database
# https://kiarva.scilifelab.se/api/fasta
#
# Human BCR V D J genes only.
# Headers are already in igblast-ready format (e.g. TRAV1-1*01) —
# no edit_imgt_file.pl processing required.
# ---------------------------------------------------------------------------

KIARVA_BASE_URL = "https://kiarva.scilifelab.se/api/fasta"
KIARVA_HEADERS = {
    "x-api-key": "kiarvafrontend",
}

# Maps (locus, segment) -> filename on the KIARVA server.
# Key is the 3-letter locus prefix (lowercase) and single-char gene type.
ORI_KIARVA_FILES = {
    ("igh", "v"): "IGHV",
    ("igh", "d"): "IGHD",
    ("igh", "j"): "IGHJ",
}

KIARVA_FILES_IMGT = {
    ("igh", "v"): "kiarva_imgt_human_IGHV.fasta",
    ("igh", "d"): "kiarva_imgt_human_IGHD.fasta",
    ("igh", "j"): "kiarva_imgt_human_IGHJ.fasta",
    ("igk", "v"): "kiarva_imgt_human_IGKV.fasta",
    ("igk", "j"): "kiarva_imgt_human_IGKJ.fasta",
    ("igl", "v"): "kiarva_imgt_human_IGLV.fasta",
    ("igl", "j"): "kiarva_imgt_human_IGLJ.fasta",
}

KIARVA_FILES_OGRDB = {
    ("igh", "v"): "kiarva_ogrdb_human_IGHV.fasta",
    ("igh", "d"): "kiarva_ogrdb_human_IGHD.fasta",
    ("igh", "j"): "kiarva_ogrdb_human_IGHJ.fasta",
    ("igk", "v"): "kiarva_ogrdb_human_IGKV.fasta",
    ("igk", "j"): "kiarva_ogrdb_human_IGKJ.fasta",
    ("igl", "v"): "kiarva_ogrdb_human_IGLV.fasta",
    ("igl", "j"): "kiarva_ogrdb_human_IGLJ.fasta",
}

# Light chain segments that need to be borrowed from another germline
# database, since KIARVA does not provide them.
LIGHT_CHAIN_SEGMENTS = ["IGKV", "IGKJ", "IGLV", "IGLJ"]


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download KIARVA human BCR germline sequences and build "
            "igblastn databases."
        )
    )
    parser.add_argument(
        "--outdir",
        default="./database",
        help="Output directory for downloaded files. Defaults to ./database.",
    )
    parser.add_argument(
        "--makeblastdb_bin",
        default=None,
        help="Path to makeblastdb binary. Defaults to None (resolved from PATH).",
    )
    return parser.parse_args()


def download_kiarva_fasta(locus: str, segment: str) -> str | None:
    family = ORI_KIARVA_FILES[(locus, segment)]

    url = f"{KIARVA_BASE_URL}/genomic?file_name={family}"

    request = Request(url, headers=KIARVA_HEADERS)

    try:
        with urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")

    except URLError as e:
        logging.error(f"Failed downloading {family}: {e}")
        return None


def download_and_write_germlines(
    germline_dir: Path, imgt_ref_dir: Path
) -> bool:
    """
    Download all KIARVA BCR fasta files and write them to *germline_dir*.

    Skips files that already exist and are non-empty.  Returns ``True`` if
    every file was obtained successfully, ``False`` if any download failed.

    Parameters
    ----------
    germline_dir : Path
        Directory to write raw per-segment fasta files into.
    imgt_ref_dir : Path
        Directory containing IMGT reference sequences for gapping.

    Returns
    -------
    bool
        ``True`` if all downloads succeeded.
    """
    germline_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for (locus, segment), filename in ORI_KIARVA_FILES.items():
        dest = germline_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            logging.info(f"Skipping {filename} — already exists.")
            continue

        logging.info(f"Downloading {filename} ...")
        content = download_kiarva_fasta(locus, segment)

        if content is None:
            logging.warning(f"Download failed for {filename}.")
            all_ok = False
            continue

        # Strip empty lines
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            logging.warning(f"Downloaded content for {filename} is empty.")
            all_ok = False
            continue

        dest.write_text("\n".join(lines) + "\n")
        # create gapped sequences
        # make sure imgt database was actually downloaded - just check the file exists and is not empty
        imgt_reference = Path(imgt_ref_dir) / (
            ("imgt_human_" + dest.stem) + ".fasta"
        )
        assert imgt_reference.exists() and imgt_reference.stat().st_size > 0, (
            f"IMGT reference fasta {imgt_reference} not found or empty, which is required for gapping.\n"
            "Please run the prepare_imgt_database.py script first to download the IMGT reference sequences."
        )
        for lc_db in ["imgt", "ogrdb"]:
            gap_sequence(
                in_fasta=dest,
                out_fasta=dest.parent
                / f"kiarva_{lc_db}_human_{filename}.fasta",
                reference_fasta=imgt_reference,
            )
        dest.unlink()  # remove the original un-gapped fasta
        logging.info(
            f"Saved {filename} ({len([l for l in lines if l.startswith('>')])} sequences)."
        )

    return all_ok


def copy_light_chain_germlines(
    germline_dir: Path,
    light_chain_dir: Path,
    light_chain_db: str,
) -> bool:
    """
    Borrow IGK/IGL V and J germline references from another database.

    KIARVA only provides heavy chain (IGH) sequences, so light chain V and J
    references (already IMGT-gapped) are copied across from *light_chain_dir*
    and renamed to follow the ``kiarva_human_*`` naming convention expected
    by downstream tools (e.g. ``tigger-genotype.R``, ``CreateGermlines.py``).

    Parameters
    ----------
    germline_dir : Path
        Directory to write the renamed light chain fasta files into
        (``germlines/kiarva/human/vdj``).
    light_chain_dir : Path
        Directory containing the source database's germline references
        (e.g. ``germlines/imgt/human/vdj`` or ``germlines/ogrdb/human/vdj``).
    light_chain_db : str
        Name of the source database (``imgt`` or ``ogrdb``), used to look up
        the source filenames.

    Returns
    -------
    bool
        ``True`` if every light chain segment was copied successfully.
    """
    germline_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for segment in LIGHT_CHAIN_SEGMENTS:
        src = light_chain_dir / f"{light_chain_db}_human_{segment}.fasta"
        dest = germline_dir / f"kiarva_{light_chain_db}_human_{segment}.fasta"

        if not src.exists() or src.stat().st_size == 0:
            logging.warning(
                f"Light chain reference {src} not found or empty. "
                f"Please run prepare_{light_chain_db}_database.py first, "
                "or choose a different --light_chain_db."
            )
            all_ok = False
            continue

        shutil.copyfile(src, dest)
        print("Writing", dest)
        logging.info(f"Copied light chain reference {src.name} -> {dest.name}")
        # also fix the header names to ensure that it's clean
        seqs = {}
        fh = open(dest)
        for header, sequence in fasta_iterator(fh):
            parts = header.split("|")
            if len(parts) > 3:
                # Raw IMGT format: filter pseudo-genes
                if parts[3] != "P":
                    seqs[parts[1].rstrip()] = sequence.upper()
            else:
                # Already-processed format (GitHub backup): keep as-is
                seqs[header.rstrip()] = sequence.upper()
        fh.close()
        write_fasta(seqs, dest, overwrite=True)

    return all_ok


def deduplicate_fasta(path: Path) -> None:
    """
    Remove duplicate FASTA records in-place.

    Rules
    -----
    * If a header is repeated with an identical sequence, keep only the first.
    * If a header is repeated with a different sequence, rename subsequent
      records by appending '_1', '_2', ... to the header.

    Parameters
    ----------
    path : Path
        FASTA file to clean.
    """
    records = []

    # header -> first sequence seen
    first_sequence = {}

    # header -> number of renamed variants assigned
    variant_count = {}

    with open(path) as fh:
        for header, sequence in fasta_iterator(fh):

            sequence = sequence.strip()

            if header not in first_sequence:
                first_sequence[header] = sequence
                variant_count[header] = 0
                records.append((header, sequence))
                continue

            # duplicate with identical sequence -> ignore
            if first_sequence[header] == sequence:
                continue

            # duplicate with different sequence -> rename
            variant_count[header] += 1
            new_header = f"{header}_{variant_count[header]}"

            logging.warning(
                f"Duplicate header '{header}' has a different sequence. "
                f"Renaming to '{new_header}'."
            )

            records.append((new_header, sequence))

    with open(path, "w") as fh:
        for header, sequence in records:
            fh.write(f">{header}\n{sequence}\n")


def build_igblast_fastas(germline_dir: Path, igblast_fasta_dir: Path) -> None:
    """
    Merge per-segment fasta files into the combined igblast input fastas.

    For each (locus, segment) pair we produce a file named::

        kiarva_{light_chain_db}_human_{locus}_{segment}.fasta

    e.g. ``kiarva_human_tr_v.fasta``, which concatenates TRAV + TRBV + TRDV
    + TRGV.  igblastn's ``-germline_db_V/D/J`` arguments accept a single
    file, so same-segment files across loci are merged together.  Sequences
    containing IMGT gaps (``"."``) are stripped before writing.

    Parameters
    ----------
    germline_dir : Path
        Directory containing the per-segment fasta files written by
        :func:`download_and_write_germlines`.
    igblast_fasta_dir : Path
        Output directory for the merged igblast fasta files.
    """
    igblast_fasta_dir.mkdir(parents=True, exist_ok=True)

    # Collect sequences by segment type across all loci
    by_segment: dict[str, dict[str, str]] = {"v": {}, "d": {}, "j": {}}

    for k_files, lc_db in zip(
        [KIARVA_FILES_IMGT, KIARVA_FILES_OGRDB], ["imgt", "ogrdb"]
    ):
        for (locus, segment), filename in k_files.items():
            src = germline_dir / filename
            if not src.exists() or src.stat().st_size == 0:
                logging.warning(f"Skipping missing or empty file: {src.name}")
                continue

            with open(src) as fh:
                for header, sequence in fasta_iterator(fh):
                    # Strip IMGT alignment gaps if present
                    clean_seq = sequence.replace(".", "").upper().rstrip()
                    if header not in by_segment[segment]:
                        by_segment[segment][header] = clean_seq

        for segment, seqs in by_segment.items():
            out_path = (
                igblast_fasta_dir / f"kiarva_{lc_db}_human_ig_{segment}.fasta"
            )
            if not seqs:
                # Write an empty placeholder so makeblastdb doesn't fail
                logging.warning(
                    f"No sequences collected for segment '{segment}' — "
                    f"writing empty file: {out_path.name}"
                )
                out_path.write_text("")
                continue

            write_fasta(seqs, out_path)
            logging.info(f"Wrote {len(seqs)} sequences to {out_path.name}")


def build_igblast_aux(igblast_fasta_dir: Path, optional_file_dir: Path) -> None:
    """
    Generate the igblastn auxiliary file for the KIARVA J genes.

    Runs ``annotate_j`` on the combined IG J fasta to produce
    ``human_gl_kiarva_<lc_db>.aux`` in *optional_file_dir*.

    Parameters
    ----------
    igblast_fasta_dir : Path
        Directory containing ``kiarva_<lc_db>_human_ig_j.fasta``.
    optional_file_dir : Path
        Directory where the ``.aux`` file will be written.
    """
    optional_file_dir.mkdir(parents=True, exist_ok=True)
    j_fastas = [
        igblast_fasta_dir / "kiarva_imgt_human_ig_j.fasta",
        igblast_fasta_dir / "kiarva_ogrdb_human_ig_j.fasta",
    ]
    aux_outs = [
        optional_file_dir / "human_gl_kiarva_imgt.aux",
        optional_file_dir / "human_gl_kiarva_ogrdb.aux",
    ]

    for j_fasta, aux_out in zip(j_fastas, aux_outs):
        logging.info(f"Generating auxiliary file: {aux_out.name}")
        if not j_fasta.exists() or j_fasta.stat().st_size == 0:
            logging.warning(
                f"Skipping missing or empty J fasta file: {j_fasta.name}"
            )
            continue
        cmd = ["annotate_j", str(j_fasta), str(aux_out)]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            logging.warning(
                f"annotate_j returned non-zero exit code: "
                f"{res.stderr.decode('utf-8')}"
            )
    else:
        logging.info(res.stdout.decode("utf-8"))


def gap_sequence(
    in_fasta: Path, out_fasta: Path, reference_fasta: Path
) -> None:
    """
    Add IMGT alignment gaps to sequences in a fasta file.

    Parameters
    ----------
    in_fasta : Path
        Input fasta file with unaligned sequences.
    out_fasta : Path
        Output fasta file with IMGT-aligned sequences.
    """
    if not in_fasta.exists() or in_fasta.stat().st_size == 0:
        logging.warning(
            f"Input fasta not found or empty ({in_fasta}); skipping."
        )
        return

    logging.info(f"Adding IMGT gaps to sequences in {in_fasta.name}")
    cmd = ["gap_sequences", str(in_fasta), str(reference_fasta), str(out_fasta)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        logging.warning(
            f"gap_sequences returned non-zero exit code: "
            f"{res.stderr.decode('utf-8')}"
        )
    else:
        logging.info(res.stdout.decode("utf-8"))


def build_blast_databases(
    igblast_fasta_dir: Path,
    igblastdb_dir: Path,
    makeblastdb: Path,
) -> None:
    """
    Run ``makeblastdb`` on every KIARVA fasta to produce igblastn databases.

    Parameters
    ----------
    igblast_fasta_dir : Path
        Directory containing the merged per-segment fasta files.
    igblastdb_dir : Path
        Output directory for the blast database files.
    makeblastdb : Path
        Path to the ``makeblastdb`` binary.
    """
    igblastdb_dir.mkdir(parents=True, exist_ok=True)

    for fasta_file in sorted(igblast_fasta_dir.iterdir()):
        if not fasta_file.stem.startswith("kiarva"):
            continue
        if fasta_file.stat().st_size == 0:
            logging.warning(f"Skipping empty fasta file: {fasta_file.name}")
            continue
        deduplicate_fasta(fasta_file)
        db_out = igblastdb_dir / fasta_file.stem
        logging.info(f"Building blast database: {db_out.name}")
        cmd = [
            str(makeblastdb),
            "-parse_seqids",
            "-dbtype",
            "nucl",
            "-input_type",
            "fasta",
            "-in",
            str(fasta_file),
            "-out",
            str(db_out),
        ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(res.stdout.decode("utf-8"))
        if res.returncode != 0:
            logging.warning(
                f"makeblastdb failed for {fasta_file.name}: "
                f"{res.stderr.decode('utf-8')}"
            )


def main():
    """Main function."""
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    makeblastdb = (
        Path(sys.executable).parent / "makeblastdb"
        if args.makeblastdb_bin is None
        else Path(args.makeblastdb_bin)
    )

    # Directory layout mirrors the ogrdb prepare script:
    #   germlines/kiarva/human/          <- raw per-segment fastas
    #   igblast/fasta/                   <- merged fastas for makeblastdb
    #   igblast/database/                <- blast databases
    #   igblast/optional_file/           <- aux files
    germline_dir = out_dir / "germlines" / "kiarva" / "human" / "vdj"
    imgt_ref_dir = out_dir / "germlines" / "imgt" / "human" / "vdj"
    # ogrdb_ref_dir = out_dir / "germlines" / "ogrdb" / "human" / "vdj"
    igblast_fasta_dir = out_dir / "igblast" / "fasta"
    igblastdb_dir = out_dir / "igblast" / "database"
    optional_file_dir = out_dir / "igblast" / "optional_file"

    # Logging
    log_file = out_dir / "kiarva_database.log"
    log_file.write_text("")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    start_time = datetime.now()
    logging.info(f"Source:        {KIARVA_BASE_URL}")
    logging.info(f"Out directory: {out_dir.absolute()}")
    logging.info(f"Download date: {start_time.strftime('%Y-%m-%d')}")
    logging.info("Species: human (BCR VDJ only)")

    # 1. Download raw fastas
    logging.info("--- Downloading KIARVA BCR germline sequences ---")
    ok = download_and_write_germlines(germline_dir, imgt_ref_dir)
    if not ok:
        logging.warning(
            "One or more downloads failed. The resulting databases may be "
            "incomplete. Re-run to retry failed files."
        )
    # 1b. Borrow light chain (IGK/IGL) V and J germlines, since KIARVA
    # does not provide them.
    for light_chain_db in ["imgt", "ogrdb"]:
        light_chain_dir = (
            out_dir / "germlines" / light_chain_db / "human" / "vdj"
        )
        copy_light_chain_germlines(
            germline_dir, light_chain_dir, light_chain_db
        )

    # 2. Merge into per-segment igblast fastas
    logging.info("--- Building merged igblast fasta files ---")
    build_igblast_fastas(germline_dir, igblast_fasta_dir)

    # 3. Generate auxiliary J annotation file
    logging.info("--- Generating igblastn auxiliary file ---")
    build_igblast_aux(igblast_fasta_dir, optional_file_dir)

    # 4. Build blast databases
    logging.info("--- Building blast databases ---")
    build_blast_databases(igblast_fasta_dir, igblastdb_dir, makeblastdb)

    end_time = datetime.now()
    logging.info(f"Download finished: {end_time}")
    logging.info(f"Total execution time: {end_time - start_time}")


if __name__ == "__main__":
    if not shutil.which("annotate_j"):
        print("Please install receptor-utils with `pip install receptor-utils`")

    main()
