import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from utils import Tree, fasta_iterator, write_fasta

GITHUB_BACKUP_BASE = "https://raw.githubusercontent.com/tuonglab/dandelion/master/container/database"


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./database",
        help="Output directory for downloaded files. Defaults to current directory.",
    )
    parser.add_argument(
        "--igblast_dir",
        default=None,
        help="Igblast directory if not downloaded through conda/mamba. Defaults to None.",
    )
    parser.add_argument(
        "--makeblastdb_bin",
        default=None,
        help="Path to makeblastdb. Defaults to None.",
    )
    args = parser.parse_args()
    return args


def download_from_github_backup(fallback_url: str, dest_path: str) -> bool:
    """
    Download a file from the GitHub backup database.

    Parameters
    ----------
    fallback_url : str
        Raw GitHub URL to download from.
    dest_path : str
        Local path to write the downloaded file to.

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    try:
        logging.warning(
            f"IMGT unavailable — trying GitHub backup: {fallback_url}"
        )
        with urlopen(fallback_url, timeout=60) as response:
            content = response.read().decode("utf-8")
        with open(dest_path, "w") as fh:
            fh.write(content)
        logging.info(f"Downloaded from GitHub backup: {dest_path}")
        return True
    except Exception as e:
        logging.error(f"GitHub backup also failed for {dest_path}: {str(e)}")
        return False


def copy_db_from_igblast(
    out_dir: str | Path, igblast_loc: str | Path | None = None
):
    """
    Copy internal data from igblast's folder to the new database folder.

    Parameters
    ----------
    out_dir : str | Path
        Location of new database folder.
    igblast_loc : str | Path | None, optional
        Location of igblast database folder, by default None.
    """
    if igblast_loc is None:
        lib_path = (
            Path(sys.executable).parent.parent / "share" / "igblast"
        )  # if installed with conda/mamba
    else:
        lib_path = Path(igblast_loc)
    for folder in ["optional_file", "internal_data"]:
        shutil.copytree(
            lib_path / folder, Path(out_dir) / folder, dirs_exist_ok=True
        )


def download_germline_and_process(
    species: str,
    query: str,
    chain: str,
    file_path: str | Path,
    source: str,
    query_type: str,
    add_prefix: str,
    add_suffix: str,
    url_suffix: str,
    fallback_url: str | None = None,
):
    """
    Download sequence from imgt and write to fasta file.

    Parameters
    ----------
    species : str
        Species name.
    query : str
        Query species name for url.
    chain : str
        Chain name.
    file_path : str | Path
        Path to write fasta file to.
    source : str
        Source url.
    query_type : str
        Query type to specify in url.
    add_prefix : str
        Prefix to add in file name, after `imgt_`.
    add_suffix : str
        Suffix to add in file name, before `.fasta`.
    url_suffix : str
        Suffix to add in url.
    fallback_url : str | None, optional
        GitHub raw URL to fall back to if the IMGT download fails.
    """
    imgt_out_dict = {
        "human": ["Homo sapiens", "Homo_sapiens"],
        "mouse": ["Mus musculus", "Mus_musculus"],
        # "rat": ["Rattus norvegicus", "Rattus_norvegicus"], # seems like there's an issue retrieving this from IMGT/GENE-DB 18/07/2024
        "rabbit": ["Oryctolagus cuniculus", "Oryctolagus_cuniculus"],
        # "rhesus_monkey": ["Macaca mulatta", "Macaca_mulatta"], # seems like there's an issue retrieving this from IMGT/GENE-DB 18/07/2024
    }
    url = f"{source}/GENElect?query={query_type}+{chain}&species={query}{url_suffix}"
    file_name = (
        f"{str(file_path)}/imgt_{add_prefix}{species}_{chain}{add_suffix}.fasta"
    )

    # Stop if the file already exists
    if os.path.exists(file_name):
        logging.info(f"Skipping download of {file_name} as it already exists.")
        return

    try:
        # else download the file
        with urlopen(url, timeout=60) as response:
            content = (
                response.read()
                .decode("utf-8")
                .split("<pre>")[2]
                .split("</pre>")[0]
            )

        # Check if the downloaded content is empty
        if content.rstrip() == "":
            logging.warning(
                f"Downloaded content for {url} is empty. Skipping processing."
            )
            fh = open(file_name, "w")
            fh.close()
            return

        # Remove empty lines
        content_lines = [line for line in content.splitlines() if line.strip()]

        # Substitute patterns in lines starting with '>'
        for i, line in enumerate(content_lines):
            if line.startswith(">"):
                # Example substitution: Homo sapiens to Homo_sapiens
                line = re.sub(
                    imgt_out_dict[species][0], imgt_out_dict[species][1], line
                )
                content_lines[i] = line

        content = "\n".join(content_lines)

        with open(file_name, "w") as output_file:
            output_file.write(content)
        logging.info(f"Downloaded from IMGT: {file_name}")

    except Exception as e:
        logging.error(
            f"Failed to download {species} {chain} from IMGT: {str(e)}"
        )
        if fallback_url is not None:
            download_from_github_backup(fallback_url, file_name)


def download_bcr_constant_and_process(
    species: str,
    query: str,
    file_path: str | Path,
    source: str,
    fallback_url: str | None = None,
):
    """
    Download BCR CH1/constant region sequences from imgt.

    Parameters
    ----------
    species : str
        Species name.
    query : str
        Query species name for url.
    file_path : str | Path
        Path to write fasta file to.
    source : str
        Source url.
    fallback_url : str | None, optional
        GitHub raw URL to fall back to if the IMGT download fails.
    """
    urls = [
        f"{source}/GENElect?query=8.1+IGHC&species={query}&IMGTlabel=CH1",
        f"{source}/GENElect?query=8.1+IGKC&species={query}&IMGTlabel=C-REGION",
        f"{source}/GENElect?query=8.1+IGLC&species={query}&IMGTlabel=C-REGION",
    ]
    file_name = Path(f"{str(file_path)}/{species}_BCR_C.fasta")
    file_path.mkdir(parents=True, exist_ok=True)
    # Stop if the file already exists
    if os.path.exists(file_name):
        logging.info(
            f"Skipping download of {file_name.stem} as it already exists."
        )
        return
    else:
        fh = open(file_name, "w")
        fh.close()

    try:
        contents = ""
        newline = ""
        for url in urls:
            with urlopen(url, timeout=60) as response:
                content = (
                    response.read()
                    .decode("utf-8")
                    .split("<pre>")[2]
                    .split("</pre>")[0]
                )
            # Remove empty lines
            content_lines = [
                line for line in content.splitlines() if line.strip()
            ]
            contents += newline + "\n".join(content_lines)
            newline = "\n"
        with open(file_name, "a") as output_file:
            output_file.write(contents)
        logging.info(f"Downloaded from IMGT: {file_name}")
    except Exception as e:
        logging.error(
            f"Failed to download BCR constant sequences for {species} from IMGT: {str(e)}"
        )
        if fallback_url is not None:
            if not download_from_github_backup(fallback_url, str(file_name)):
                return
        else:
            return

    seqs = {}
    if file_name.stat().st_size != 0:
        fh = open(file_name)
        for header, sequence in fasta_iterator(fh):
            parts = header.split("|")
            if len(parts) > 3:
                # Raw IMGT format: filter pseudo-genes and short sequences
                if parts[3] != "P" and len(sequence) >= 150:
                    seqs[parts[1].rstrip()] = sequence.upper()
            else:
                # Already-processed format (GitHub backup): keep as-is
                seqs[header.rstrip()] = sequence.upper()
        fh.close()
        write_fasta(seqs, file_name, overwrite=True)


def main():
    """Main function."""
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.makeblastdb_bin is None:
        makeblastdb = Path(sys.executable).parent / "makeblastdb"
    else:
        makeblastdb = Path(args.makeblastdb_bin)
    species_dict = {
        "human": "Homo+sapiens",
        "mouse": "Mus",
        # "rat": "Rattus+norvegicus", # seems like there's an issue retrieving this from IMGT/GENE-DB 18/07/2024
        "rabbit": "Oryctolagus+cuniculus",
        # "rhesus_monkey": "Macaca+mulatta", # seems like there's an issue retrieving this from IMGT/GENE-DB 18/07/2024
    }
    igblast_out_dict = {
        "vdj": "",
        "vdj_aa": "aa_",
        "constant": "",
    }
    # germline folder
    germline_out = out_dir / "germlines" / "imgt"
    # igblast folder
    igblast_out = out_dir / "igblast" / "fasta"
    igblastdb_out = out_dir / "igblast" / "database"
    # blast folder
    blast_out = out_dir / "blast"
    # Set up logging
    log_file = out_dir / "imgt_database.log"
    fh = open(log_file, "w")
    fh.close()
    source = "https://www.imgt.org/genedb"
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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for species, query in species_dict.items():
            logging.info(f"    - {species}")
            for folder in [
                "vdj",
                "vdj_aa",
                # "leader_vexon",
                # "leader",
                "constant",
            ]:
                futures = []
                logging.info(f"        - {folder}")
                file_path = germline_out / species / folder
                file_path.mkdir(parents=True, exist_ok=True)
                if folder == "vdj":
                    query_type, add_prefix, add_suffix, url_suffix = (
                        "7.1",
                        "",
                        "",
                        "",
                    )
                    for chain in [
                        "IGHV",
                        "IGHD",
                        "IGHJ",
                        "IGKV",
                        "IGKJ",
                        "IGLV",
                        "IGLJ",
                        "TRAV",
                        "TRAJ",
                        "TRBV",
                        "TRBD",
                        "TRBJ",
                        "TRDV",
                        "TRDD",
                        "TRDJ",
                        "TRGV",
                        "TRGJ",
                    ]:
                        fallback_url = f"{GITHUB_BACKUP_BASE}/germlines/imgt/{species}/{folder}/imgt_{add_prefix}{species}_{chain}{add_suffix}.fasta"
                        futures.append(
                            executor.submit(
                                download_germline_and_process,
                                species,
                                query,
                                chain,
                                file_path,
                                source,
                                query_type,
                                add_prefix,
                                add_suffix,
                                url_suffix,
                                fallback_url,
                            )
                        )
                elif folder == "vdj_aa":
                    query_type, add_prefix, add_suffix, url_suffix = (
                        "7.3",
                        "aa_",
                        "",
                        "",
                    )
                    for chain in [
                        "IGHV",
                        "IGKV",
                        "IGLV",
                        "TRAV",
                        "TRBV",
                        "TRDV",
                        "TRGV",
                    ]:
                        fallback_url = f"{GITHUB_BACKUP_BASE}/germlines/imgt/{species}/{folder}/imgt_{add_prefix}{species}_{chain}{add_suffix}.fasta"
                        futures.append(
                            executor.submit(
                                download_germline_and_process,
                                species,
                                query,
                                chain,
                                file_path,
                                source,
                                query_type,
                                add_prefix,
                                add_suffix,
                                url_suffix,
                                fallback_url,
                            )
                        )
                # elif folder == "leader_vexon":
                #     query_type, add_prefix, add_suffix, url_suffix = (
                #         "8.1",
                #         "lv_",
                #         "",
                #         "&IMGTlabel=L-PART1+V-EXON",
                #     )
                #     for chain in [
                #         "IGHV",
                #         "IGKV",
                #         "IGLV",
                #         "TRAV",
                #         "TRBV",
                #         "TRDV",
                #         "TRGV",
                #     ]:
                #         futures.append(
                #             executor.submit(
                #                 download_germline_and_process,
                #                 species,
                #                 query,
                #                 chain,
                #                 file_path,
                #                 source,
                #                 query_type,
                #                 add_prefix,
                #                 add_suffix,
                #                 url_suffix,
                #             )
                #         )
                # elif folder == "leader":
                #     query_type, add_prefix, add_suffix, url_suffix = (
                #         "8.1",
                #         "",
                #         "L",
                #         "&IMGTlabel=L-PART1+L-PART2",
                #     )
                #     for chain in [
                #         "IGHV",
                #         "IGKV",
                #         "IGLV",
                #         "TRAV",
                #         "TRBV",
                #         "TRDV",
                #         "TRGV",
                #     ]:
                #         futures.append(
                #             executor.submit(
                #                 download_germline_and_process,
                #                 species,
                #                 query,
                #                 chain,
                #                 file_path,
                #                 source,
                #                 query_type,
                #                 add_prefix,
                #                 add_suffix,
                #                 url_suffix,
                #             )
                #         )
                elif folder == "constant":
                    query_type, add_prefix, add_suffix, url_suffix = (
                        "7.5",
                        "",
                        "",
                        "",
                    )
                    fallback_url = f"{GITHUB_BACKUP_BASE}/germlines/imgt/{species}/{folder}/imgt_{add_prefix}{species}_IGHC{add_suffix}.fasta"
                    futures.append(
                        executor.submit(
                            download_germline_and_process,
                            species,
                            query,
                            "IGHC",
                            file_path,
                            source,
                            "14.1",
                            add_prefix,
                            add_suffix,
                            url_suffix,
                            fallback_url,
                        )
                    )
                    for chain in [
                        "IGKC",
                        "IGLC",
                        "TRAC",
                        "TRBC",
                        "TRDC",
                        "TRGC",
                    ]:
                        fallback_url = f"{GITHUB_BACKUP_BASE}/germlines/imgt/{species}/{folder}/imgt_{add_prefix}{species}_{chain}{add_suffix}.fasta"
                        futures.append(
                            executor.submit(
                                download_germline_and_process,
                                species,
                                query,
                                chain,
                                file_path,
                                source,
                                query_type,
                                add_prefix,
                                add_suffix,
                                url_suffix,
                                fallback_url,
                            )
                        )
                # Wait for all futures to complete
                concurrent.futures.wait(futures)
        for species, query in species_dict.items():
            logging.info(f"Converting to igblast database for {species}")
            igblast_out.mkdir(parents=True, exist_ok=True)
            for folder in [
                "vdj",
                "vdj_aa",
                "constant",
            ]:
                file_tree, out_filename_tree = Tree(), Tree()
                file_path = germline_out / species / folder
                for file in sorted(file_path.iterdir()):
                    file_code = file.stem.rsplit("_", 1)[1].lower()
                    chain, segment = file_code[:2], file_code[3]
                    out_filename = (
                        igblast_out
                        / f"imgt_{igblast_out_dict[folder]}{species}_{chain}_{segment}.fasta"
                    )
                    file_tree[chain + segment][file].value = 1
                    out_filename_tree[chain + segment][out_filename].value = 1
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
                                new_header = header.split("|")[1].strip()
                                if new_header not in seqs:
                                    seqs[new_header] = (
                                        sequence.replace(".", "")
                                        .upper()
                                        .rstrip()
                                    )
                            fh.close()
                    write_fasta(seqs, out_file)
        # convert to igblast database
        igblastdb_out.mkdir(parents=True, exist_ok=True)
        for fastafile in [
            f
            for f in sorted(igblast_out.iterdir())
            if f.stem.startswith("imgt")
        ]:
            cmd = [
                str(makeblastdb),
                "-parse_seqids",
                "-dbtype",
                "prot" if re.search("aa", fastafile.stem) else "nucl",
                "-input_type",
                "fasta",
                "-in",
                str(fastafile),
                "-out",
                str(igblastdb_out / fastafile.stem),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE)
            logging.info(res.stdout.decode("utf-8"))
    # copying igblast internal data to igblast folder
    copy_db_from_igblast(
        out_dir=out_dir / "igblast", igblast_loc=args.igblast_dir
    )
    # download for blast
    for species, query in species_dict.items():
        logging.info(
            f"Downloading IMGT BCR constant sequences for blast database for {species}"
        )
        fallback_url = (
            f"{GITHUB_BACKUP_BASE}/blast/{species}/{species}_BCR_C.fasta"
        )
        download_bcr_constant_and_process(
            species, query, blast_out / species, source, fallback_url
        )
    for species, query in species_dict.items():
        logging.info(f"Converting to blast database for {species}")
        fastafile = blast_out / species / f"{species}_BCR_C.fasta"
        cmd = [
            str(makeblastdb),
            "-parse_seqids",
            "-dbtype",
            "nucl",
            "-input_type",
            "fasta",
            "-in",
            str(fastafile),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        logging.info(res.stdout.decode("utf-8"))

    # Log the end time
    end_time = datetime.now()
    logging.info(f"Download finished: {end_time}")
    logging.info(f"Total execution time: {end_time - start_time}")


if __name__ == "__main__":
    main()
