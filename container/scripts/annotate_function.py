import argparse
import pandas as pd

from pathlib import Path
from utils import fasta_iterator


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        help="Path to input FASTA file.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    fasta_path = args.fasta
    if not fasta_path:
        raise ValueError(
            "Please provide a path to the input FASTA file using --fasta."
        )

    fasta_file = Path(fasta_path)
    if not fasta_file.exists():
        raise FileNotFoundError(
            f"The specified FASTA file does not exist: {fasta_file}"
        )

    fh = open(fasta_file)
    rows = []

    for header, sequence in fasta_iterator(fh):
        parts = header.split("|")

        gene = parts[1].strip()
        functionality = parts[3].strip()

        rows.append({"gene": gene, "functionality": functionality})
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(fasta_file.with_suffix(".csv"), index=False)
