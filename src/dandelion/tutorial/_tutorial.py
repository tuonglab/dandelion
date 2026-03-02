import subprocess

from pathlib import Path
from urllib.request import Request, urlopen


def _download_file(url: str, dest: Path | str, chunk_size: int = 8192):
    """Download a file using urllib.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dest : Path | str
        Destination file path to write the downloaded content.
    chunk_size : int, optional
        Number of bytes to read per chunk. Defaults to 8192.
    """
    req = Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; Python urllib)"}
    )
    with urlopen(req) as response, open(dest, "wb") as out_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def setup_dandelion_tutorial_bcr(path: Path | str | None = None) -> None:
    """Download example BCR datasets for Dandelion tutorial.

    Downloads 10x Genomics PBMC BCR datasets into a local directory for use
    in the dandelion BCR preprocessing tutorial.

    Parameters
    ----------
    path : Path | str | None, optional
        Root directory to download datasets into.
        Defaults to ``./dandelion_tutorial``.
    """
    base = Path("./dandelion_tutorial") if path is None else Path(path)
    base.mkdir(parents=True, exist_ok=True)

    datasets = {
        "vdj_v1_hs_pbmc3_b": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_v1_hs_pbmc3/vdj_v1_hs_pbmc3_filtered_feature_bc_matrix.h5",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_v1_hs_pbmc3/vdj_v1_hs_pbmc3_b_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_v1_hs_pbmc3/vdj_v1_hs_pbmc3_b_filtered_contig.fasta",
        },
        "vdj_nextgem_hs_pbmc3_b": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_nextgem_hs_pbmc3/vdj_nextgem_hs_pbmc3_filtered_feature_bc_matrix.h5",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_nextgem_hs_pbmc3/vdj_nextgem_hs_pbmc3_b_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_nextgem_hs_pbmc3/vdj_nextgem_hs_pbmc3_b_filtered_contig.fasta",
        },
        "sc5p_v2_hs_PBMC_10k_b": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_b_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_b_filtered_contig.fasta",
            "airr_rearrangement.tsv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_b_airr_rearrangement.tsv",
        },
        "sc5p_v2_hs_PBMC_1k_b": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_1k/sc5p_v2_hs_PBMC_1k_filtered_feature_bc_matrix.h5",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_1k/sc5p_v2_hs_PBMC_1k_b_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_1k/sc5p_v2_hs_PBMC_1k_b_filtered_contig.fasta",
        },
    }

    for dirname, files in datasets.items():
        dirpath = base / dirname
        dirpath.mkdir(parents=True, exist_ok=True)

        for filename, url in files.items():
            outfile = dirpath / filename
            if outfile.exists():
                continue
            print(f"Downloading {filename} → {outfile}")
            _download_file(url, outfile)


def setup_dandelion_tutorial_tcr(path: Path | str | None = None) -> None:
    """Download example TCR datasets for Dandelion tutorial.

    Downloads 10x Genomics PBMC and melanoma TCR datasets into a local
    directory for use in the dandelion TCR preprocessing tutorial.

    Parameters
    ----------
    path : Path | str | None, optional
        Root directory to download datasets into.
        Defaults to ``./dandelion_tutorial``.
    """
    base = Path("./dandelion_tutorial") if path is None else Path(path)
    base.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sc5p_v2_hs_PBMC_10k_t": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5",
            "airr_rearrangement.tsv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_t_airr_rearrangement.tsv",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_t_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_t_filtered_contig.fasta",
        },
        "sc5p_v1p1_hs_melanoma_10k_t": {
            "filtered_feature_bc_matrix.h5": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_filtered_feature_bc_matrix.h5",
            "airr_rearrangement.tsv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_t_airr_rearrangement.tsv",
            "filtered_contig_annotations.csv": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_t_filtered_contig_annotations.csv",
            "filtered_contig.fasta": "https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v1p1_hs_melanoma_10k/sc5p_v1p1_hs_melanoma_10k_t_filtered_contig.fasta",
        },
    }

    for dirname, files in datasets.items():
        dirpath = base / dirname
        dirpath.mkdir(parents=True, exist_ok=True)

        for filename, url in files.items():
            outfile = dirpath / filename
            if outfile.exists():
                continue
            print(f"Downloading {filename} → {outfile}")
            _download_file(url, outfile)


def setup_dandelion_tutorial_trajectory(path: Path | str | None = None) -> None:
    """Download example datasets for Dandelion V(D)J trajectory tutorial.

    Downloads panfetal B-cell trajectory GEX and VDJ data from Google Drive
    using ``gdown``.

    Parameters
    ----------
    path : Path | str | None, optional
        Root directory to download datasets into.
        Defaults to ``./dandelion_tutorial``.

    Raises
    ------
    ImportError
        If ``gdown`` is not installed.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download the trajectory tutorial data. Please install it via `pip install gdown`."
        )
    base = Path("./dandelion_tutorial") if path is None else Path(path)
    base.mkdir(parents=True, exist_ok=True)

    gex_id = "1-LbAinwhAhJW3Y60wpO9GWJJcaMa_liy"
    vdj_id = "1lyScJWdGopW2nLoIhZmfUGVSWLWI_qWg"
    datasets = {
        "panfetal_trajectory": {
            "demo-pseudobulk.h5ad": f"https://drive.google.com/uc?id={gex_id}",
            "demo-vdj-traj.tsv.gz": f"https://drive.google.com/uc?id={vdj_id}",
        }
    }
    for dirname, files in datasets.items():
        dirpath = base / dirname
        dirpath.mkdir(parents=True, exist_ok=True)
        for filename, url in files.items():
            outfile = dirpath / filename
            if outfile.exists():
                continue
            print(f"Downloading {filename} → {outfile}")
            gdown.download(url, str(outfile), quiet=False)


def setup_dandelion_tutorial_parse(path: Path | str | None = None) -> None:
    """Download the extremely large dataset from Parse Biosciences for Dandelion tutorial.

    Downloads the Parse Biosciences 1M human BCR dataset (cell metadata CSV
    and AIRR rearrangement TSV) into a local directory.

    Parameters
    ----------
    path : Path | str | None, optional
        Root directory to download datasets into.
        Defaults to ``./dandelion_tutorial``.
    """
    base = Path("./dandelion_tutorial") if path is None else Path(path)
    base.mkdir(parents=True, exist_ok=True)
    datasets = {
        "human-bcr-1m": {
            "cell_metadata.csv": "https://cdn.parsebiosciences.com/bcr/human-bcr-1m/cell_metadata.csv",
            "bcr_annotation_airr.tsv": "https://cdn.parsebiosciences.com/bcr/human-bcr-1m/bcr_annotation_airr.tsv",
        }
    }

    for dirname, files in datasets.items():
        dirpath = base / dirname
        dirpath.mkdir(parents=True, exist_ok=True)
        for filename, url in files.items():
            outfile = dirpath / filename
            if outfile.exists():
                continue
            print(f"Downloading {filename} → {outfile}")
            _download_file(url, outfile)


def setup_colab_singularity() -> None:  # pragma: no cover
    """Install and configure Apptainer/Singularity in a Google Colab environment.

    Installs ``apptainer-suid`` from the official PPA, wraps it with
    ``unshare -r`` so that it operates correctly inside Colab's unprivileged
    container, and registers the Sylabs remote. Safe to re-run; existing
    files are backed up rather than overwritten.

    Raises
    ------
    subprocess.CalledProcessError
        If any step of the installation script exits with a non-zero status.
    """

    bash_script = r"""
set -e

echo "Installing Apptainer..."

sudo apt update
sudo apt install -y software-properties-common

if ! grep -q apptainer /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt update
fi

sudo apt install -y apptainer-suid

echo "Configuring fakeroot..."
sudo apptainer config fakeroot --add root || true

echo "Creating singularity wrapper..."

sudo bash -c 'cat <<EOF > /usr/bin/singularity_test
unshare -r apptainer "$@"
EOF'

sudo chmod +x /usr/bin/singularity_test

if [ -f /usr/bin/singularity ]; then
    sudo mv /usr/bin/singularity /usr/bin/singularity_backup || true
fi

sudo mv /usr/bin/singularity_test /usr/bin/singularity

echo "Adding Sylabs remote..."
singularity remote add --no-login SylabsCloud cloud.sylabs.io || true

echo "Done."
"""

    subprocess.run(["bash", "-c", bash_script], check=True)

    print("\n✅ Singularity / Apptainer ready!")
