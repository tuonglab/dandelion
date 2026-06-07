#!/opt/conda/envs/sc-dandelion-container/bin/python
import os
import shutil
import pandas as pd
import dandelion as ddl
import numpy as np
import anndata as ad

from unittest.mock import patch
from subprocess import run


def test_callscript():
    """Test script to run preprocessing."""
    p = run(
        ["python", "/share/dandelion_preprocess.py", "-h"],
        capture_output=True,
        encoding="utf8",
    )
    assert p.returncode == 0
    assert p.stdout != ""


def test_container():
    """Test script to run container."""
    os.system(
        "cd /tests; python /share/dandelion_preprocess.py --meta test.csv --file_prefix filtered;"
    )
    dat = pd.read_csv(
        "/tests/sample_test_10x/dandelion/filtered_contig_dandelion.tsv",
        sep="\t",
    )
    assert not dat["c_call"].empty
    assert not dat["v_call_genotyped"].empty
    assert not dat["mu_count"].empty
    assert not dat["mu_freq"].empty
    vdj = None
    try:
        vdj = ddl.Dandelion(dat)
    except Exception:
        pass
    assert vdj is not None


def test_container_skip_tigger():
    """Test script to run container but skip tigger."""
    os.system(
        "cd /tests; python /share/dandelion_preprocess.py --meta test.csv --file_prefix filtered --skip_tigger;"
    )
    dat = pd.read_csv(
        "/tests/sample_test_10x/dandelion/filtered_contig_dandelion.tsv",
        sep="\t",
    )
    assert not dat["c_call"].empty
    assert not dat["mu_count"].empty
    assert not dat["mu_freq"].empty
    vdj = None
    try:
        vdj = ddl.Dandelion(dat)
    except Exception:
        pass
    assert vdj is not None


def _append_id_tag(cell_id: str, tag: str) -> str:
    """Append tag while preserving trailing 10x '-N' suffix when present."""
    if "-" in cell_id:
        base, suffix = cell_id.rsplit("-", 1)
        if suffix.isdigit():
            return f"{base}{tag}-{suffix}"
    return f"{cell_id}{tag}"


def _rewrite_duplicated_sample_ids(sample_dir: str, file_prefix: str, tag: str):
    """Rewrite barcodes/contig IDs in duplicated 10x files with a deterministic tag."""
    ann_path = os.path.join(sample_dir, f"{file_prefix}_contig_annotations.csv")
    fa_path = os.path.join(sample_dir, f"{file_prefix}_contig.fasta")

    ann = pd.read_csv(ann_path)
    ann["barcode"] = ann["barcode"].astype(str)
    ann["contig_id"] = ann["contig_id"].astype(str)

    old_to_new = {}
    for old_contig in ann["contig_id"].unique():
        if "_contig_" in old_contig:
            old_bc, contig_suffix = old_contig.split("_contig_", 1)
            new_bc = _append_id_tag(old_bc, tag)
            old_to_new[old_contig] = f"{new_bc}_contig_{contig_suffix}"

    ann["barcode"] = ann["barcode"].map(lambda x: _append_id_tag(x, tag))
    ann["contig_id"] = ann["contig_id"].map(lambda x: old_to_new.get(x, x))
    ann.to_csv(ann_path, index=False)

    rewritten = []
    with open(fa_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                new_header = old_to_new.get(header, header)
                rewritten.append(f">{new_header}\n")
            else:
                rewritten.append(line)
    with open(fa_path, "w") as f:
        f.writelines(rewritten)


def test_container_demultiplex_h5ad_metadata_skip_tigger():
    """Simulate multiplexed preprocessing using per-cell .h5ad metadata."""
    tests_dir = "/tests"
    file_prefix = "filtered"

    src_sample = os.path.join(tests_dir, "sample_test_10x")
    dup_sample_name = "sample_test_10x_dup"
    dup_sample = os.path.join(tests_dir, dup_sample_name)
    h5ad_path = os.path.join(tests_dir, "demux_meta.h5ad")

    out_a = os.path.join(tests_dir, "mux_individual_a")
    out_b = os.path.join(tests_dir, "mux_individual_b")

    for path in [dup_sample, out_a, out_b]:
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists(h5ad_path):
        os.remove(h5ad_path)

    try:
        shutil.copytree(src_sample, dup_sample)
        _rewrite_duplicated_sample_ids(dup_sample, file_prefix, "_DUP")

        src_ann = pd.read_csv(
            os.path.join(src_sample, f"{file_prefix}_contig_annotations.csv")
        )
        dup_ann = pd.read_csv(
            os.path.join(dup_sample, f"{file_prefix}_contig_annotations.csv")
        )

        src_cells = sorted(src_ann["barcode"].astype(str).unique().tolist())
        dup_cells = sorted(dup_ann["barcode"].astype(str).unique().tolist())

        obs = pd.DataFrame(
            {
                "cell_id": src_cells + dup_cells,
                "individual": ["ind_a"] * len(src_cells)
                + ["ind_b"] * len(dup_cells),
                "source_sample": ["sample_test_10x"] * len(src_cells)
                + [dup_sample_name] * len(dup_cells),
                "output_sample": ["mux_individual_a"] * len(src_cells)
                + ["mux_individual_b"] * len(dup_cells),
            }
        )
        ad.AnnData(
            X=np.zeros((len(obs), 1), dtype=np.float32),
            obs=obs.set_index("cell_id", drop=False),
        ).write_h5ad(h5ad_path)

        os.system(
            "cd /tests; python /share/dandelion_preprocess.py "
            "--meta demux_meta.h5ad "
            "--meta_cell_id_col cell_id "
            "--meta_individual_col individual "
            "--meta_sample_col source_sample "
            "--meta_output_col output_sample "
            "--file_prefix filtered "
            "--skip_tigger;"
        )

        for out_folder in ["mux_individual_a", "mux_individual_b"]:
            out_tsv = os.path.join(
                tests_dir,
                out_folder,
                "dandelion",
                f"{file_prefix}_contig_dandelion.tsv",
            )
            dat = pd.read_csv(out_tsv, sep="\t")
            assert not dat["c_call"].empty
            assert not dat["mu_count"].empty
            assert not dat["mu_freq"].empty
            vdj = None
            try:
                vdj = ddl.Dandelion(dat)
            except Exception:
                pass
            assert vdj is not None
    finally:
        for path in [dup_sample, out_a, out_b]:
            if os.path.exists(path):
                shutil.rmtree(path)
        if os.path.exists(h5ad_path):
            os.remove(h5ad_path)


def test_container_demultiplex_csv_metadata_skip_tigger():
    """Simulate multiplexed preprocessing using per-cell CSV metadata."""
    tests_dir = "/tests"
    file_prefix = "filtered"

    src_sample = os.path.join(tests_dir, "sample_test_10x")
    dup_sample_name = "sample_test_10x_dup_csv"
    dup_sample = os.path.join(tests_dir, dup_sample_name)
    csv_path = os.path.join(tests_dir, "demux_meta.csv")

    out_a = os.path.join(tests_dir, "mux_csv_individual_a")
    out_b = os.path.join(tests_dir, "mux_csv_individual_b")

    for path in [dup_sample, out_a, out_b]:
        if os.path.exists(path):
            shutil.rmtree(path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    try:
        shutil.copytree(src_sample, dup_sample)
        _rewrite_duplicated_sample_ids(dup_sample, file_prefix, "_DUPCSV")

        src_ann = pd.read_csv(
            os.path.join(src_sample, f"{file_prefix}_contig_annotations.csv")
        )
        dup_ann = pd.read_csv(
            os.path.join(dup_sample, f"{file_prefix}_contig_annotations.csv")
        )

        src_cells = sorted(src_ann["barcode"].astype(str).unique().tolist())
        dup_cells = sorted(dup_ann["barcode"].astype(str).unique().tolist())

        demux_meta = pd.DataFrame(
            {
                "cell_id": src_cells + dup_cells,
                "individual": ["ind_a"] * len(src_cells)
                + ["ind_b"] * len(dup_cells),
                "source_sample": ["sample_test_10x"] * len(src_cells)
                + [dup_sample_name] * len(dup_cells),
                "output_sample": ["mux_csv_individual_a"] * len(src_cells)
                + ["mux_csv_individual_b"] * len(dup_cells),
            }
        )
        demux_meta.to_csv(csv_path, index=False)

        os.system(
            "cd /tests; python /share/dandelion_preprocess.py "
            "--meta demux_meta.csv "
            "--meta_cell_id_col cell_id "
            "--meta_individual_col individual "
            "--meta_sample_col source_sample "
            "--meta_output_col output_sample "
            "--file_prefix filtered "
            "--skip_tigger;"
        )

        for out_folder in ["mux_csv_individual_a", "mux_csv_individual_b"]:
            out_tsv = os.path.join(
                tests_dir,
                out_folder,
                "dandelion",
                f"{file_prefix}_contig_dandelion.tsv",
            )
            dat = pd.read_csv(out_tsv, sep="\t")
            assert not dat["c_call"].empty
            assert not dat["mu_count"].empty
            assert not dat["mu_freq"].empty
            vdj = None
            try:
                vdj = ddl.Dandelion(dat)
            except Exception:
                pass
            assert vdj is not None
    finally:
        for path in [dup_sample, out_a, out_b]:
            if os.path.exists(path):
                shutil.rmtree(path)
        if os.path.exists(csv_path):
            os.remove(csv_path)


@patch("matplotlib.pyplot.show")
def test_threshold(mock_show):
    """Test script to run container."""
    os.system(
        "cd /tests; python /share/changeo_clonotypes.py --h5ddl sample_test_10x/demo-vdj.h5ddl;"
    )
    dat = ddl.read_h5ddl("/tests/sample_test_10x/demo-vdj_changeo.h5ddl")
    assert dat._data.collect()["changeo_clone_id"].len() > 0
