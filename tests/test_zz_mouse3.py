#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import pytest
from pathlib import Path
import sys


@pytest.fixture(scope="module", autouse=True)
def setup_testfolder_files(
    create_testfolder, fasta_10x_mouse, annotation_10x_mouse
):
    """Write test files once for all tests in this module."""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x_mouse, out_fasta=out_fasta)
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    return create_testfolder


@pytest.fixture(scope="module")
def formatted_testfolder(setup_testfolder_files, create_testfolder):
    """Format fastas once for tests that need formatted data."""
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    return create_testfolder


@pytest.mark.usefixtures("setup_testfolder_files", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test write fasta"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    # Verify file exists and has expected number of sequences
    assert out_fasta.exists()
    # Verify content by counting FASTA headers efficiently
    header_count = 0
    with open(out_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                header_count += 1
    assert header_count == len(fasta_10x_mouse)


@pytest.mark.usefixtures("setup_testfolder_files", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test write annot"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    # Verify file exists
    assert out_file.exists()
    # Verify content by reading back and comparing
    written_data = pd.read_csv(out_file)
    assert len(written_data) == len(annotation_10x_mouse)
    assert list(written_data.columns) == list(annotation_10x_mouse.columns)


@pytest.mark.usefixtures("formatted_testfolder")
def test_formatfasta(create_testfolder):
    """test format fasta"""
    # Already formatted by fixture, just verify
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("formatted_testfolder", "database_paths_mouse")
def test_reannotategenes_strict(create_testfolder, database_paths_mouse):
    """test reannotate"""
    # format_fastas already called by fixture
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        flavour="strict",
        reassign_dj=True,
        org="mouse",
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 6


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reassignalleles(create_testfolder, database_paths_mouse):
    """test reassign"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / "test_mouse",
        germline=database_paths_mouse["germline"],
        org="mouse",
        novel=True,
        plot=False,
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 9


@pytest.mark.usefixtures("database_paths_mouse")
def test_updateblastdb(database_paths_mouse):
    """test update blast"""
    ddl.utl.makeblastdb(database_paths_mouse["blastdb_fasta"])


@pytest.mark.usefixtures(
    "create_testfolder", "database_paths_mouse", "balbc_ighg_primers"
)
def test_assignsisotypes(
    create_testfolder, database_paths_mouse, balbc_ighg_primers
):
    """test assign isotype"""
    ddl.pp.assign_isotypes(
        create_testfolder,
        blastdb=database_paths_mouse["blastdb_fasta"],
        correction_dict=balbc_ighg_primers,
        plot=False,
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "dummy_adata_mouse"
)
def test_checkcontigs(create_testfolder, processed_files, dummy_adata_mouse):
    """test check contigs"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    dat = pd.read_csv(f, sep="\t")
    vdj, adata = ddl.pp.check_contigs(dat, dummy_adata_mouse)
    f1 = create_testfolder / "test.h5ddl"
    # f1 = create_testfolder / "test.zipddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_h5ddl(f1)
    # vdj.write_zipddl(f1)
    adata.write_h5ad(f2)


@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
@pytest.mark.usefixtures("create_testfolder")
def test_generate_network_sfdp(create_testfolder):
    """test generate network sfdp"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    with pytest.raises(ValueError):
        ddl.tl.generate_network(vdj, compute_layout=False)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, layout_method="sfdp")
    assert vdj.layout is not None
