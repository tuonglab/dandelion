#!/usr/bin/env python
import pytest
import dandelion as ddl


@pytest.fixture(scope="module", autouse=True)
def setup_formatted_fastas(create_testfolder, fasta_10x_mouse, annotation_10x_mouse):
    """Setup formatted fastas once for all tests in this module."""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x_mouse, out_fasta=out_fasta)
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    return create_testfolder


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test_write_fasta"""
    # Files already written by autouse fixture, just verify
    out_fasta = create_testfolder / "filtered_contig.fasta"
    assert out_fasta.exists()


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test_write_annotation"""
    # Files already written by autouse fixture, just verify
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    assert out_file.exists()


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test_formatfasta"""
    # Already formatted by autouse fixture, just verify
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("database_paths_mouse")
def test_updateblastdb(database_paths_mouse):
    """test update blast"""
    ddl.utl.makeblastdb(database_paths_mouse["blastdb_fasta"])


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_original(create_testfolder, database_paths_mouse):
    """test_reannotategenes_original"""
    # format_fastas already called by autouse fixture
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        flavour="original",
        org="mouse",
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 4


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_other(create_testfolder, database_paths_mouse):
    """test_reannotategenes_other"""
    # format_fastas already called by autouse fixture
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        extended=False,
        org="mouse",
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 6
