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
IMGT_HUMAN_LIGHT_CHAIN_CONSTANT_FILES = {
    "IGKC": "imgt_human_IGKC.fasta",
    "IGLC": "imgt_human_IGLC.fasta",
}
IMGT_IG_LIGHT_CHAIN_CONSTANT_SEGMENTS = ["IGKC", "IGLC"]
IMGT_TR_CONSTANT_SEGMENTS = ["TRAC", "TRBC", "TRDC", "TRGC"]


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
        logging.info("Skipping download of files as it already exists.")
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
        logging.info("Skipping download of files as it already exists.")
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
    germline_dir = Path(out_dir) / "germlines" / "ogrdb" / "human" / "constant"
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


def _read_constant_fasta(path: Path) -> dict[str, str]:
    """Read a constant fasta and normalize headers for igblast databases."""
    seqs = {}
    if not path.exists() or path.stat().st_size == 0:
        return seqs
    fh = open(path)
    for header, sequence in fasta_iterator(fh):
        parts = header.split("|")
        if len(parts) > 3:
            # Raw IMGT format: filter pseudo-genes and keep the gene name field.
            if parts[3] != "P":
                seqs[parts[1].rstrip()] = sequence.replace(".", "").upper()
        else:
            seqs[header.rstrip()] = sequence.replace(".", "").upper()
    fh.close()
    return seqs


def build_ogrdb_human_ig_constant_fasta(
    fasta_path: Path,
    out_dir: str | Path,
) -> Path | None:
    """
    Create a unified OGRDB human IG constant fasta for igblast.

    The output contains OGRDB IGHC (heavy chain) plus IGKC/IGLC from IMGT,
    written to ``<out_dir>/igblast/fasta/ogrdb_human_ig_c.fasta``.

    Parameters
    ----------
    fasta_path : Path
        Path to the downloaded IGHC fasta.
    out_dir : str | Path
        Root database output directory.

    Returns
    -------
    Path | None
        Path to the merged constant fasta, or None if no sequences are found.
    """
    igblast_out = Path(out_dir) / "igblast" / "fasta"
    igblast_out.mkdir(parents=True, exist_ok=True)
    out_fasta = igblast_out / "ogrdb_human_ig_c.fasta"

    seqs = _read_constant_fasta(fasta_path)
    imgt_constant_dir = (
        Path(out_dir) / "germlines" / "imgt" / "human" / "constant"
    )
    for chain, filename in IMGT_HUMAN_LIGHT_CHAIN_CONSTANT_FILES.items():
        imgt_file = imgt_constant_dir / filename
        light_chain_seqs = _read_constant_fasta(imgt_file)
        if len(light_chain_seqs) == 0:
            logging.warning(
                f"IMGT {chain} constant file missing or empty: {imgt_file}. "
                "Unified OGRDB constant fasta may be incomplete."
            )
            continue
        for header, sequence in light_chain_seqs.items():
            if header not in seqs:
                seqs[header] = sequence

    if len(seqs) == 0:
        logging.warning("No constant sequences found for ogrdb_human_ig_c.")
        return None

    write_fasta(seqs, out_fasta, overwrite=True)
    logging.info(
        f"Wrote unified constant fasta with {len(seqs)} sequences to {out_fasta}"
    )
    return out_fasta


def build_ogrdb_constant_fastas(
    out_dir: str | Path,
    human_ighc_fasta: Path | None = None,
) -> None:
    """
    Build unified OGRDB constant fasta files for human/mouse and IG/TR.

    Outputs are written to ``<out_dir>/igblast/fasta``:
    * ``ogrdb_human_ig_c.fasta``
    * ``ogrdb_mouse_ig_c.fasta``
    * ``ogrdb_human_tr_c.fasta``
    * ``ogrdb_mouse_tr_c.fasta``

    Notes
    -----
    OGRDB only provides human IGHC constants. Missing constants are borrowed
    from IMGT so all expected ``ogrdb_*_c`` databases can be built.
    """
    out_dir = Path(out_dir)
    igblast_out = out_dir / "igblast" / "fasta"
    igblast_out.mkdir(parents=True, exist_ok=True)

    for species in ["human", "mouse"]:
        imgt_constant_dir = (
            out_dir / "germlines" / "imgt" / species / "constant"
        )

        # Build IG constants.
        ig_seqs: dict[str, str] = {}
        if species == "human" and human_ighc_fasta is not None:
            ig_seqs.update(_read_constant_fasta(human_ighc_fasta))
        else:
            ig_seqs.update(
                _read_constant_fasta(
                    imgt_constant_dir / f"imgt_{species}_IGHC.fasta"
                )
            )
        for segment in IMGT_IG_LIGHT_CHAIN_CONSTANT_SEGMENTS:
            ig_seqs.update(
                _read_constant_fasta(
                    imgt_constant_dir / f"imgt_{species}_{segment}.fasta"
                )
            )

        ig_out = igblast_out / f"ogrdb_{species}_ig_c.fasta"
        if len(ig_seqs) > 0:
            write_fasta(ig_seqs, ig_out, overwrite=True)
            logging.info(
                f"Wrote unified constant fasta with {len(ig_seqs)} sequences to {ig_out}"
            )

            # For mouse, also mirror constants to every available strain-specific
            # OGRDB IG FASTA prefix so makeblastdb produces matching *_ig_c DBs.
            if species == "mouse":
                strain_labels = set()
                for fasta_file in igblast_out.iterdir():
                    if (
                        not fasta_file.is_file()
                        or fasta_file.suffix != ".fasta"
                    ):
                        continue
                    match = re.match(
                        r"^ogrdb_mouse_(.+)_ig_[vdj]$", fasta_file.stem
                    )
                    if match is not None:
                        strain_labels.add(match.group(1))

                for strain in sorted(strain_labels):
                    strain_out = (
                        igblast_out / f"ogrdb_mouse_{strain}_ig_c.fasta"
                    )
                    write_fasta(ig_seqs, strain_out, overwrite=True)
                if len(strain_labels) > 0:
                    logging.info(
                        "Wrote strain-specific mouse IG constant fastas for "
                        + f"{len(strain_labels)} strains."
                    )
        else:
            logging.warning(
                f"No IG constant sequences found for {species}; skipping {ig_out.name}."
            )

        # Build TR constants from IMGT.
        tr_seqs: dict[str, str] = {}
        for segment in IMGT_TR_CONSTANT_SEGMENTS:
            tr_seqs.update(
                _read_constant_fasta(
                    imgt_constant_dir / f"imgt_{species}_{segment}.fasta"
                )
            )

        tr_out = igblast_out / f"ogrdb_{species}_tr_c.fasta"
        if len(tr_seqs) > 0:
            write_fasta(tr_seqs, tr_out, overwrite=True)
            logging.info(
                f"Wrote unified constant fasta with {len(tr_seqs)} sequences to {tr_out}"
            )
        else:
            logging.warning(
                f"No TR constant sequences found for {species}; skipping {tr_out.name}."
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
    logging.info("Species:")

    # For each species
    human_ighc_fasta = None
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
            # Download OGRDB IGHC and merge with IMGT IGKC/IGLC constants.
            logging.info("Preparing unified OGRDB human IG constant fasta")
            ighc_fasta = download_ogrdb_ighc(out_dir)
            if ighc_fasta is not None:
                human_ighc_fasta = ighc_fasta
                build_ogrdb_human_ig_constant_fasta(ighc_fasta, out_dir)
            else:
                logging.warning(
                    "OGRDB IGHC download failed – unified ogrdb_human_ig_c fasta "
                    "cannot be created. Constant gene annotation with db='ogrdb' "
                    "will fall back to IMGT."
                )

    logging.info("Preparing unified OGRDB constant-region fasta files")
    build_ogrdb_constant_fastas(out_dir, human_ighc_fasta=human_ighc_fasta)

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
