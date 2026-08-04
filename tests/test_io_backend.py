#!/usr/bin/env python
"""
Backend-agnostic IO tests.

Imports from the top-level ``dandelion`` namespace so that the active backend
(controlled by the ``DANDELION_BACKEND`` env-var) determines which code path
is exercised.  CI runs this file once per backend to cover both
``dandelion.base`` and ``dandelion.polars``.
"""

import json
import os
import pytest
import polars as pl

import dandelion as ddl

from dandelion.utilities._utilities import write_fasta

# -- read functions -----------------------------------------------------------


@pytest.mark.usefixtures("create_testfolder", "airr_10x")
def test_write_airr(create_testfolder, airr_10x):
    out_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_10x.to_csv(out_file, sep="\t", index=False)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_readwrite_ddl(create_testfolder, airr_reannotated):
    vdj = ddl.Dandelion(airr_reannotated)
    out_file = create_testfolder / "test_backend.ddl"
    vdj.write(out_file)
    ddl.read(out_file)


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite10xairr(create_testfolder):
    airr_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_file2 = create_testfolder / "test_airr_rearrangements2.tsv"
    vdj = ddl.read_10x_airr(airr_file)
    vdj.write_airr(airr_file2)
    ddl.Dandelion(airr_file2)
    # os.remove(airr_file2)


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6")
def test_read10xvdj_json(create_testfolder, json_10x_cr6):
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    ddl.read_10x_vdj(json_file, filename_prefix="test_all")
    os.remove(json_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    os.remove(json_file)
    write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj(create_testfolder, annotation_10x, fasta_10x):
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    ddl.read_10x_vdj(annot_file)
    write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    ddl.read_10x_vdj(annot_file)
    os.remove(fasta_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6_folder(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    os.remove(json_file)
    write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj_folder(create_testfolder, annotation_10x, fasta_10x):
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_io_prefix_suffix_combinations(create_testfolder, annotation_10x):
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    airr_file = create_testfolder / "test_airr_rearrangements.tsv"
    annotation_10x.to_csv(annot_file, index=False)
    ddl.read_10x_vdj(annot_file, suffix="x")
    ddl.read_10x_vdj(annot_file, prefix="y")
    ddl.read_10x_vdj(
        annot_file,
        suffix="x",
        remove_trailing_hyphen_number=True,
        filename_prefix="filtered",
    )
    ddl.read_10x_vdj(
        annot_file,
        prefix="x",
        remove_trailing_hyphen_number=True,
        filename_prefix="filtered",
    )
    ddl.read_10x_airr(airr_file, suffix="x")
    ddl.read_10x_airr(airr_file, prefix="y")
    ddl.read_10x_airr(airr_file, suffix="x", remove_trailing_hyphen_number=True)
    ddl.read_10x_airr(airr_file, prefix="x", remove_trailing_hyphen_number=True)


@pytest.mark.usefixtures("airr_bd")
def test_read_bd(airr_bd):
    ddl.read_bd_airr(airr_bd)


@pytest.mark.usefixtures("airr_parse")
def test_read_parse(airr_parse):
    ddl.read_parse_airr(airr_parse)


@pytest.mark.usefixtures("airr_bd")
def test_read_standard(airr_bd):
    ddl.read_airr(airr_bd)


_10X_SUFFIXED_COLS = [
    "is_cell_10x",
    "high_confidence_10x",
    "sequence_length_10x",
    "raw_consensus_id_10x",
    "exact_subclonotype_id_10x",
]


def _data_columns(vdj) -> list:
    """Return column names regardless of backend (pandas/polars eager/lazy)."""
    if hasattr(vdj._data, "collect_schema"):
        return vdj._data.collect_schema().names()
    return list(vdj._data.columns)


def _metadata_columns(vdj) -> list:
    """Return metadata column names regardless of backend (pandas/polars eager/lazy)."""
    if hasattr(vdj._metadata, "collect_schema"):
        return vdj._metadata.collect_schema().names()
    return list(vdj._metadata.columns)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_read10xvdj_has_productive_metadata_columns(
    create_testfolder, annotation_10x
):
    """read_10x_vdj should initialize productive metadata columns in all backends."""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    cols = _metadata_columns(vdj)
    assert "productive_VDJ" in cols
    assert "productive_VJ" in cols


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_simplify_preserves_user_metadata_columns(
    create_testfolder, annotation_10x
):
    """simplify should not drop user-added metadata columns."""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)

    if hasattr(vdj._metadata, "collect_schema"):
        vdj._metadata = vdj._metadata.with_columns(
            pl.lit("KEEP_ME").alias("user_added_col")
        )
    else:
        vdj.metadata["user_added_col"] = "KEEP_ME"

    vdj.simplify()
    cols = _metadata_columns(vdj)
    assert "user_added_col" in cols


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_cr6")
def test_read_seekgene_vdj_csv(create_testfolder, annotation_10x_cr6):
    """read_seekgene_vdj must strip _10x-suffixed columns from CSV reads."""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    vdj = ddl.read_seekgene_vdj(annot_file, filename_prefix="test_filtered")
    cols = _data_columns(vdj)
    for col in _10X_SUFFIXED_COLS:
        assert col not in cols, f"unexpected _10x column: {col}"
    assert "is_cell" in cols
    assert "high_confidence" in cols


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6")
def test_read_seekgene_vdj_json(create_testfolder, json_10x_cr6):
    """read_seekgene_vdj must strip _10x-suffixed columns from JSON reads."""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_seekgene_vdj(json_file, filename_prefix="test_all")
    cols = _data_columns(vdj)
    for col in _10X_SUFFIXED_COLS:
        assert col not in cols, f"unexpected _10x column: {col}"
    os.remove(json_file)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated", "dummy_adata")
def test_readwrite_h5ddl(create_testfolder, airr_reannotated, dummy_adata):
    """Round-trip write_h5ddl / read_h5ddl via backend dispatcher."""
    vdj = ddl.Dandelion(airr_reannotated)
    vdj, _ = ddl.pp.check_contigs(vdj, dummy_adata)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, layout_method="mod_fr")

    out_file = create_testfolder / "test_h5ddl_backend.h5ddl"
    vdj.write_h5ddl(out_file)
    vdj2 = ddl.read_h5ddl(out_file)

    assert vdj2._data is not None
    assert vdj2._metadata is not None
    assert vdj2.graph is not None
    assert vdj2.layout is not None

    # gzip compression round-trip
    vdj.write_h5ddl(out_file, compression="gzip")
    vdj3 = ddl.read_h5ddl(out_file)
    assert vdj3._data is not None


# -- Dandelion constructor ----------------------------------------------------


@pytest.mark.usefixtures("airr_reannotated")
def test_dandelion_constructor(airr_reannotated):
    ddl.Dandelion(airr_reannotated)
