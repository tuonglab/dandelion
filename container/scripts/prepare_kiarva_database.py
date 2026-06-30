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

KIARVA_FILES = {
    ("igh", "v"): "kiarva_human_IGHV.fasta",
    ("igh", "d"): "kiarva_human_IGHD.fasta",
    ("igh", "j"): "kiarva_human_IGHJ.fasta",
}


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
        gap_sequence(
            in_fasta=dest,
            out_fasta=dest.parent / f"kiarva_human_{filename}.fasta",
            reference_fasta=imgt_reference,
        )
        dest.unlink()  # remove the original un-gapped fasta
        logging.info(
            f"Saved {filename} ({len([l for l in lines if l.startswith('>')])} sequences)."
        )

    return all_ok


def build_igblast_fastas(germline_dir: Path, igblast_fasta_dir: Path) -> None:
    """
    Merge per-segment fasta files into the combined igblast input fastas.

    For each (locus, segment) pair we produce a file named::

        kiarva_human_{locus}_{segment}.fasta

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

    for (locus, segment), filename in KIARVA_FILES.items():
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
        out_path = igblast_fasta_dir / f"kiarva_human_ig_{segment}.fasta"
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
    ``human_gl_kiarva.aux`` in *optional_file_dir*.

    Parameters
    ----------
    igblast_fasta_dir : Path
        Directory containing ``kiarva_human_ig_j.fasta``.
    optional_file_dir : Path
        Directory where the ``.aux`` file will be written.
    """
    optional_file_dir.mkdir(parents=True, exist_ok=True)
    j_fasta = igblast_fasta_dir / "kiarva_human_ig_j.fasta"
    aux_out = optional_file_dir / "human_gl_kiarva.aux"

    if not j_fasta.exists() or j_fasta.stat().st_size == 0:
        logging.warning(
            f"J fasta not found or empty ({j_fasta}); skipping aux file generation."
        )
        return

    logging.info(f"Generating auxiliary file: {aux_out.name}")
    cmd = ["annotate_j", str(j_fasta), str(aux_out)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
