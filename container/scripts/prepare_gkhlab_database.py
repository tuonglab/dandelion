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
# GKHlab TCR germline database
# Karlsson Hedestam lab, Karolinska Institutet
# https://gkhlab.gitlab.io/tcr/sequences/
#
# Human TCR only (TRA, TRB, TRD, TRG).
# Headers are already in igblast-ready format (e.g. TRAV1-1*01) —
# no edit_imgt_file.pl processing required.
# ---------------------------------------------------------------------------

GKHLAB_BASE_URL = "https://gkhlab.gitlab.io/tcr/sequences"

# Maps (locus, segment) -> filename on the GKHlab server.
# Key is the 3-letter locus prefix (lowercase) and single-char gene type.
GKHLAB_FILES: dict[tuple[str, str], str] = {
    ("tra", "v"): "TRAV.fasta",
    ("tra", "j"): "TRAJ.fasta",
    ("trb", "v"): "TRBV.fasta",
    ("trb", "d"): "TRBD.fasta",
    ("trb", "j"): "TRBJ.fasta",
    ("trd", "v"): "TRDV.fasta",
    ("trd", "d"): "TRDD.fasta",
    ("trd", "j"): "TRDJ.fasta",
    ("trg", "v"): "TRGV.fasta",
    ("trg", "j"): "TRGJ.fasta",
}


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download GKHlab human TCR germline sequences and build "
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


def download_gkhlab_fasta(locus: str, segment: str) -> str | None:
    """
    Download a single GKHlab TCR fasta file.

    Parameters
    ----------
    locus : str
        Lowercase three-letter locus code, e.g. ``"tra"``.
    segment : str
        Single-character gene segment, e.g. ``"v"``.

    Returns
    -------
    str | None
        Raw fasta content as a string, or ``None`` on failure.
    """
    filename = GKHLAB_FILES[(locus, segment)]
    url = f"{GKHLAB_BASE_URL}/{filename}"
    request = Request(url, headers={"Accept": "text/plain"})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read().decode("utf-8")
        return data
    except URLError as e:
        logging.error(f"Failed to download {url}: {e.reason}")
        return None


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
                logging.info(
                    f"Removing duplicate record '{header}' from {path.name}"
                )
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


def download_and_write_germlines(germline_dir: Path) -> bool:
    """
    Download all GKHlab TCR fasta files and write them to *germline_dir*.

    Skips files that already exist and are non-empty.  Returns ``True`` if
    every file was obtained successfully, ``False`` if any download failed.

    Parameters
    ----------
    germline_dir : Path
        Directory to write raw per-segment fasta files into.

    Returns
    -------
    bool
        ``True`` if all downloads succeeded.
    """
    germline_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for (locus, segment), filename in GKHLAB_FILES.items():
        dest = germline_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            logging.info(f"Skipping {filename} — already exists.")
            continue

        logging.info(f"Downloading {filename} ...")
        content = download_gkhlab_fasta(locus, segment)

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
        deduplicate_fasta(dest)
        logging.info(
            f"Saved {filename} ({len([l for l in lines if l.startswith('>')])} sequences)."
        )

    return all_ok


def build_igblast_fastas(germline_dir: Path, igblast_fasta_dir: Path) -> None:
    """
    Merge per-segment fasta files into the combined igblast input fastas.

    For each (locus, segment) pair we produce a file named::

        gkhlab_human_{locus}_{segment}.fasta

    e.g. ``gkhlab_human_tr_v.fasta``, which concatenates TRAV + TRBV + TRDV
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

    for (locus, segment), filename in GKHLAB_FILES.items():
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
        out_path = igblast_fasta_dir / f"gkhlab_human_tr_{segment}.fasta"
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
    Generate the igblastn auxiliary file for the GKHlab J genes.

    Runs ``annotate_j`` on the combined TR J fasta to produce
    ``human_gl_gkhlab.aux`` in *optional_file_dir*.

    Parameters
    ----------
    igblast_fasta_dir : Path
        Directory containing ``gkhlab_human_tr_j.fasta``.
    optional_file_dir : Path
        Directory where the ``.aux`` file will be written.
    """
    optional_file_dir.mkdir(parents=True, exist_ok=True)
    j_fasta = igblast_fasta_dir / "gkhlab_human_tr_j.fasta"
    aux_out = optional_file_dir / "human_gl_gkhlab.aux"

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


def build_blast_databases(
    igblast_fasta_dir: Path,
    igblastdb_dir: Path,
    makeblastdb: Path,
) -> None:
    """
    Run ``makeblastdb`` on every GKHlab fasta to produce igblastn databases.

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
        if not fasta_file.stem.startswith("gkhlab"):
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
    #   germlines/gkhlab/human/          <- raw per-segment fastas
    #   igblast/fasta/                   <- merged fastas for makeblastdb
    #   igblast/database/                <- blast databases
    #   igblast/optional_file/           <- aux files
    germline_dir = out_dir / "germlines" / "gkhlab" / "human"
    igblast_fasta_dir = out_dir / "igblast" / "fasta"
    igblastdb_dir = out_dir / "igblast" / "database"
    optional_file_dir = out_dir / "igblast" / "optional_file"

    # Logging
    log_file = out_dir / "gkhlab_database.log"
    log_file.write_text("")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    start_time = datetime.now()
    logging.info(f"Source:        {GKHLAB_BASE_URL}")
    logging.info(f"Out directory: {out_dir.absolute()}")
    logging.info(f"Download date: {start_time.strftime('%Y-%m-%d')}")
    logging.info("Species: human (TCR VDJ only)")

    # 1. Download raw fastas
    logging.info("--- Downloading GKHlab TCR germline sequences ---")
    ok = download_and_write_germlines(germline_dir)
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
