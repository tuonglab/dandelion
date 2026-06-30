import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from utils import Tree, fasta_iterator, write_fasta

MERGE_STRAINS = ["BALB_c", "BALB_c_ByJ", "C57BL_6", "C57BL_6J"]
MERGE_STRAINS_DICT = {
    "BALB_c": "balbc",
    "BALB_c_ByJ": "balbc",
    "C57BL_6": "c57bl6",
    "C57BL_6J": "c57bl6",
}
OGRDB_IGHC_URL = (
    "https://ogrdb.airr-community.org/download_germline_set/"
    "Homo%20sapiens/IGHC/published/ungapped_ex"
)


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./database",
        help="Output directory for downloaded files. Defaults to current directory.",
    )
    parser.add_argument(
        "--makeblastdb_bin",
        default=None,
        help="Path to makeblastdb. Defaults to None.",
    )
    args = parser.parse_args()
    return args


def query_ogrdb_set_info(species: str):
    """
    Query the ogrdb API for germline set info.

    Parameters
    ----------
    species : str
        Name of species.

    """
    species_dict = {"human": "homo%20sapiens", "mouse": "mus%20musculus"}
    url = f"https://ogrdb.airr-community.org/api/germline/sets/{species_dict[species]}"
    headers = {"accept": "application/json"}
    # Create a request object with the URL and headers
    request = Request(url, headers=headers)
    try:
        # Perform the GET request
        with urlopen(request, timeout=60) as response:
            # Read and decode the response data
            data = response.read().decode("utf-8")
            # Parse the JSON data
            json_data = json.loads(data)
            return json_data
    except URLError as e:
        print(f"Error: {e.reason}")
        return None


def download_ogrdb_set_fasta(set_id: str):
    """
    Download the fasta file for a given set id.

    Parameters
    ----------
    set_id : str
        OGRDB germline set id.
    """
    url = f"https://ogrdb.airr-community.org/api/germline/set/{set_id}/published/gapped"
    headers = {"accept": "application/json"}
    # Create a request object with the URL and headers
    request = Request(url, headers=headers)
    try:
        # Perform the GET request
        with urlopen(request, timeout=60) as response:
            filename = response.getheader("Content-disposition").split("=")[1]
            # Read and decode the response data
            data = response.read().decode("utf-8")
            # Parse the JSON data
            return data, filename
    except URLError as e:
        print(f"Error: {e.reason}")
        return None


def return_ogrdb_info(species: str) -> tuple[list[str], dict[str, str]]:
    """
    Return the info required from ogrdb set.

    Parameters
    ----------
    species : str
        Species name.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        List of ogrdb germline set ids and the species subgroup.
    """
    query_sets = query_ogrdb_set_info(species)
    set_ids, set_subgroup = [], {}
    for s in query_sets:
        if s["germline_set_id"] not in set_ids:
            set_ids.append(s["germline_set_id"])
        set_subgroup.update({s["germline_set_id"]: s["species_subgroup"]})
    return set_ids, set_subgroup


def copy_ogrdb_aux_to_igblast(
    igblast_out: str | Path,
    out_dir: str | Path,
):
    """
    Copy files in optional_file to where igblast expects them.

    Parameters
    ----------
    igblast_out : str | Path
        Location of downloaded fasta files for igblast.
    out_dir : str | Path
        Location of new database folder.
    """
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for org in ["human", "mouse"]:
        cmd = [
            "annotate_j",
            str(Path(igblast_out) / f"ogrdb_{org}_ig_j.fasta"),
            str(OUT_DIR / f"{org}_gl_ogrdb.aux"),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        logging.info(res.stdout.decode("utf-8"))


def download_germline_and_process(
    species: str,
    file_path: str | Path,
):
    """
    Download sequence from imgt and write to fasta file.

    Parameters
    ----------
    species : str
        Species name.
    file_path : str | Path
        Path to write fasta file to.
    """
    if os.path.exists(file_path / "vdj"):
        logging.info(f"Skipping download of files as it already exists.")
        return
    set_ids, _ = return_ogrdb_info(species)
    for set_id in set_ids:
        content, file_name = download_ogrdb_set_fasta(set_id)
        new_file_name = set_id + ".fasta"
        # Stop if the file already exists
        if os.path.exists(file_path / set_id):
            logging.info(
                f"Skipping download of {file_name} as it already exists."
            )
            return
        # Check if the downloaded content is empty
        if content.rstrip() == "":
            logging.warning(
                f"Downloaded content for {new_file_name} is empty. Skipping processing."
            )
            fh = open(new_file_name, "w")
            fh.close()
            return
        # Remove empty lines
        content_lines = [line for line in content.splitlines() if line.strip()]

        content = "\n".join(content_lines)
        logging.info(f"Downloading {file_name}.")
        with open(file_path / new_file_name, "w") as output_file:
            output_file.write(content)


def process_ogrdb_fasta(species: str, file_path: str | Path):
    """
    Process the downloaded fasta files so that it's in the correct format for igblast.

    Parameters
    ----------
    species : str
        Species name.
    file_path : str | Path
        Path to write fasta file to.
    """
    new_file_path = file_path / "vdj"
    if os.path.exists(new_file_path):
        logging.info(f"Skipping download of files as it already exists.")
        return
    # generate the split v/d/j files
    # find the "all" file first
    if species == "mouse":
        all_v_seqs, all_j_seqs = {}, {}
        for file in sorted(file_path.iterdir()):
            if file.is_file():
                _, subgroups = return_ogrdb_info(species)
                if file.stat().st_size != 0:
                    set_id = file.stem
                    strain = re.sub("\\/| ", "_", subgroups[set_id])
                    strain = "all" if strain == "" else strain
                    if strain == "all":
                        fh = open(file)
                        for header, sequence in fasta_iterator(fh):
                            locus, gene = header[:3], header[3]
                            if gene == "V":
                                if header not in all_v_seqs:
                                    all_v_seqs[header] = sequence
                            if gene == "J":
                                if header not in all_j_seqs:
                                    all_j_seqs[header] = sequence
                        fh.close()
    new_file_path.mkdir(parents=True, exist_ok=True)
    for file in sorted(file_path.iterdir()):
        if file.is_file():
            v_seqs, d_seqs, j_seqs = {}, {}, {}
            if file.stat().st_size != 0:
                fh = open(file)
                for header, sequence in fasta_iterator(fh):
                    locus, gene = header[:3], header[3]
                    if gene == "V":
                        if header not in v_seqs:
                            v_seqs[header] = sequence
                    if gene == "D":
                        if header not in d_seqs:
                            d_seqs[header] = sequence
                    if gene == "J":
                        if header not in j_seqs:
                            j_seqs[header] = sequence
                fh.close()
                # make a grouped one for mice.
                if len(v_seqs) > 0:
                    write_fasta(
                        v_seqs,
                        new_file_path / f"ogrdb_{species}_{locus}V.fasta",
                    )
                if len(d_seqs) > 0:
                    write_fasta(
                        d_seqs,
                        new_file_path / f"ogrdb_{species}_{locus}D.fasta",
                    )
                if len(j_seqs) > 0:
                    write_fasta(
                        j_seqs,
                        new_file_path / f"ogrdb_{species}_{locus}J.fasta",
                    )
            if species == "human":
                file.unlink()
    if species == "mouse":
        _, subgroups = return_ogrdb_info(species)
        for file in sorted(file_path.iterdir()):
            if file.is_file():
                set_id = file.stem
                strain = re.sub("\\/| ", "_", subgroups[set_id])
                strain = "all" if strain == "" else strain
                if strain != "all":
                    v_seqs, d_seqs, j_seqs = {}, {}, {}
                    if file.stat().st_size != 0:
                        fh = open(file)
                        for header, sequence in fasta_iterator(fh):
                            locus, gene = header[:3], header[3]
                            if gene == "V":
                                if header not in v_seqs:
                                    v_seqs[header] = sequence
                            if gene == "D":
                                if header not in d_seqs:
                                    d_seqs[header] = sequence
                            if gene == "J":
                                if header not in j_seqs:
                                    j_seqs[header] = sequence
                        v_seqs.update(all_v_seqs)
                        j_seqs.update(all_j_seqs)
                        if len(v_seqs) > 0:
                            write_fasta(
                                v_seqs,
                                new_file_path
                                / f"ogrdb_{species}_{strain}_{locus}V.fasta",
                            )
                            if strain in MERGE_STRAINS:
                                _strain = MERGE_STRAINS_DICT[strain]
                                write_fasta(
                                    v_seqs,
                                    new_file_path
                                    / f"ogrdb_{species}_{_strain}_{locus}V.fasta",
                                )
                        else:
                            fh1 = open(
                                new_file_path
                                / f"ogrdb_{species}_{strain}_{locus}V.fasta",
                                "w",
                            )
                            fh1.close()
                        if len(d_seqs) > 0:
                            write_fasta(
                                d_seqs,
                                new_file_path
                                / f"ogrdb_{species}_{strain}_{locus}D.fasta",
                            )
                            if strain in MERGE_STRAINS:
                                _strain = MERGE_STRAINS_DICT[strain]
                                write_fasta(
                                    d_seqs,
                                    new_file_path
                                    / f"ogrdb_{species}_{_strain}_{locus}D.fasta",
                                )
                        else:
                            if locus == "IGH":
                                fh1 = open(
                                    new_file_path
                                    / f"ogrdb_{species}_{strain}_{locus}D.fasta",
                                    "w",
                                )
                                fh1.close()
                        if len(j_seqs) > 0:
                            write_fasta(
                                j_seqs,
                                new_file_path
                                / f"ogrdb_{species}_{strain}_{locus}J.fasta",
                            )
                            if strain in MERGE_STRAINS:
                                _strain = MERGE_STRAINS_DICT[strain]
                                write_fasta(
                                    j_seqs,
                                    new_file_path
                                    / f"ogrdb_{species}_{_strain}_{locus}J.fasta",
                                )
                        else:
                            fh1 = open(
                                new_file_path
                                / f"ogrdb_{species}_{strain}_{locus}J.fasta",
                                "w",
                            )
                            fh1.close()
                        fh.close()
                file.unlink()


def download_ogrdb_ighc(out_dir: str | Path) -> Path | None:
    """
    Download the human IGHC germline set from OGRDB (set 90) and build a
    blastn database from it.

    Parameters
    ----------
    out_dir : str | Path
        Root database output directory (same as --outdir).

    Returns
    -------
    Path | None
        Path to the downloaded fasta, or None on failure.
    """
    fasta_name = "Homo_sapiens_IGHC_rev_1_ungapped_ex.fasta"
    germline_dir = Path(out_dir) / "germlines" / "ogrdb" / "human"
    germline_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = germline_dir / fasta_name

    if fasta_path.exists() and fasta_path.stat().st_size > 0:
        logging.info(f"Skipping IGHC download – {fasta_path} already exists.")
        return fasta_path

    logging.info(f"Downloading human OGRDB IGHC set from {OGRDB_IGHC_URL}")
    request = Request(OGRDB_IGHC_URL, headers={"accept": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read().decode("utf-8")
    except URLError as e:
        logging.warning(f"Failed to download OGRDB IGHC: {e.reason}")
        return None

    content_lines = [line for line in data.splitlines() if line.strip()]
    if not content_lines:
        logging.warning("Downloaded OGRDB IGHC content is empty – skipping.")
        return None

    fasta_path.write_text("\n".join(content_lines) + "\n")
    logging.info(f"Saved IGHC fasta to {fasta_path}")
    return fasta_path


def build_ogrdb_ighc_blastdb(
    fasta_path: Path,
    out_dir: str | Path,
    makeblastdb: str | Path,
) -> None:
    """
    Run makeblastdb on the OGRDB IGHC fasta and place the database under
    <out_dir>/blast/database/ as ``ogrdb_human_ig_c``.

    Parameters
    ----------
    fasta_path : Path
        Path to the downloaded IGHC fasta.
    out_dir : str | Path
        Root database output directory.
    makeblastdb : str | Path
        Path to the makeblastdb binary.
    """
    igblastdb_out = Path(out_dir) / "igblast" / "database"
    igblastdb_out.mkdir(parents=True, exist_ok=True)
    db_name = igblastdb_out / "ogrdb_human_ig_c"

    logging.info(f"Building OGRDB IGHC blast database at {db_name}")
    cmd = [
        str(makeblastdb),
        "-parse_seqids",
        "-dbtype",
        "nucl",
        "-input_type",
        "fasta",
        "-in",
        str(fasta_path),
        "-out",
        str(db_name),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(res.stdout.decode("utf-8"))
    if res.returncode != 0:
        logging.warning(
            f"makeblastdb for IGHC returned non-zero exit code: "
            f"{res.stderr.decode('utf-8')}"
        )


def main():
    """Main function."""
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.makeblastdb_bin is None:
        makeblastdb = Path(sys.executable).parent / "makeblastdb"
    else:
        makeblastdb = Path(args.makeblastdb_bin)
    # germline folder
    germline_out = out_dir / "germlines" / "ogrdb"
    # igblast folder
    igblast_out = out_dir / "igblast" / "fasta"
    igblastdb_out = out_dir / "igblast" / "database"
    # Set up logging
    log_file = out_dir / "ogrdb_database.log"
    fh = open(log_file, "w")
    fh.close()
    source = "https://ogrdb.airr-community.org"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Log the start time
    start_time = datetime.now()
    logging.info(f"Source:  {source}")
    logging.info(f"Out directory:  {Path(args.outdir).absolute()}")
    logging.info(f"Download date: {start_time.strftime('%Y-%m-%d')}")
    logging.info(f"Species:")

    # For each species
    for species in [
        "human",
        "mouse",
    ]:
        logging.info(f"    - {species}")
        file_path = germline_out / species
        file_path.mkdir(parents=True, exist_ok=True)
        download_germline_and_process(
            species,
            file_path,
        )
        process_ogrdb_fasta(species, file_path)
        logging.info(f"Converting to igblast database for {species}")
        igblast_out.mkdir(parents=True, exist_ok=True)
        folder = "vdj"
        file_tree, out_filename_tree = Tree(), Tree()
        file_path = germline_out / species / folder
        for file in sorted(file_path.iterdir()):
            file_code = file.stem.rsplit("_", 1)[1].lower()
            chain, segment = file_code[:2], file_code[3]
            out_filename = (
                igblast_out / f"ogrdb_{species}_{chain}_{segment}.fasta"
            )
            file_tree[chain + segment][file].value = 1
            out_filename_tree[chain + segment][out_filename].value = 1
            if species == "mouse":
                if file.stem.count("_") != 2:
                    strain = file.stem.rsplit("_", 1)[0].split("_", 2)[2]
                    out_filename = (
                        igblast_out
                        / f"ogrdb_{species}_{strain}_{chain}_{segment}.fasta"
                    )
                    file_tree[chain + segment + strain][file].value = 1
                    out_filename_tree[chain + segment + strain][
                        out_filename
                    ].value = 1
        for chain_segment in file_tree:
            in_files = list(file_tree[chain_segment])
            out_file = list(out_filename_tree[chain_segment])[0]
            fh = open(out_file, "w")
            fh.close()
            seqs = {}
            for file in in_files:
                if file.stat().st_size != 0:
                    fh = open(file)
                    for header, sequence in fasta_iterator(fh):
                        if header not in seqs:
                            seqs[header] = (
                                sequence.replace(".", "").upper().rstrip()
                            )
                    fh.close()
            write_fasta(seqs, out_file)
        if species == "human":
            # Download and build blast DB for OGRDB human IGHC (no mouse equivalent)
            logging.info("Downloading OGRDB human IGHC germline set")
            ighc_fasta = download_ogrdb_ighc(out_dir)
            if ighc_fasta is not None:
                build_ogrdb_ighc_blastdb(ighc_fasta, out_dir, makeblastdb)
            else:
                logging.warning(
                    "OGRDB IGHC download failed – ogrdb_human_ig_c blast database "
                    "will not be available. Constant gene annotation with db='ogrdb' "
                    "will fall back to IMGT."
                )

    logging.info("Preparing auxiliary files for igblast")
    copy_ogrdb_aux_to_igblast(
        igblast_out,
        out_dir / "igblast" / "optional_file",
    )
    # convert to igblast database
    igblastdb_out.mkdir(parents=True, exist_ok=True)
    for fastafile in [
        f for f in sorted(igblast_out.iterdir()) if f.stem.startswith("ogrdb")
    ]:
        cmd = [
            str(makeblastdb),
            "-parse_seqids",
            "-dbtype",
            "nucl",
            "-input_type",
            "fasta",
            "-in",
            str(fastafile),
            "-out",
            str(igblastdb_out / fastafile.stem),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        logging.info(res.stdout.decode("utf-8"))

    # Log the end time
    end_time = datetime.now()
    logging.info(f"Download finished: {end_time}")
    logging.info(f"Total execution time: {end_time - start_time}")


if __name__ == "__main__":
    if not shutil.which("annotate_j"):
        print("Please install receptor-utils with `pip install receptor-utils`")

    main()
