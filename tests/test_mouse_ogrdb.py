#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_testfolder_files(create_testfolder, fasta_10x_mouse, annotation_10x_mouse):
    """Write and format test files once for all tests in this module."""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x_mouse, out_fasta=out_fasta)
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    return create_testfolder


@pytest.mark.usefixtures("setup_testfolder_files", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test_write_fasta - verify files written by autouse fixture"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    assert out_fasta.exists()
    # Verify content by counting FASTA headers
    header_count = 0
    with open(out_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header_count += 1
    assert header_count == len(fasta_10x_mouse)


@pytest.mark.usefixtures("setup_testfolder_files", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test_write_annotation - verify files written by autouse fixture"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    assert out_file.exists()
    # Verify content
    written_data = pd.read_csv(out_file)
    assert len(written_data) == len(annotation_10x_mouse)
    assert list(written_data.columns) == list(annotation_10x_mouse.columns)


@pytest.mark.usefixtures("setup_testfolder_files")
def test_formatfasta(create_testfolder):
    """test_formatfasta - verify formatting done by autouse fixture"""
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.slow
@pytest.mark.parametrize("strain,check_igblast_file", [
    (None, False),
    ("balbc", False),
    ("BALB_c_ByJ", True),
])
@pytest.mark.usefixtures("setup_testfolder_files", "database_paths_mouse")
def test_reannotategenes_nod(
    create_testfolder,
    database_paths_mouse,
    strain,
    check_igblast_file,
):
    """test_reannotategenes - parametrized for different strains"""
    # different igblast versions/references may give different results and lead to regression.
    # disabling those checks would work for now.
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=strain,
        filename_prefix="filtered",
    )
    
    # Only check igblast file existence for BALB_c_ByJ strain
    if check_igblast_file:
        assert (
            create_testfolder / "dandelion" / "tmp" / "filtered_contig_igblast.fmt7"
        ).exists()


@pytest.mark.expensive
@pytest.mark.parametrize("strain", [None, "balbc", "BALB_c_ByJ"])
@pytest.mark.usefixtures("setup_testfolder_files", "database_paths_mouse")
def test_reassignalleles(
    create_testfolder,
    database_paths_mouse,
    strain,
):
    """test_reassignalleles - parametrized for different strains"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / "test_mouse",
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=strain,
        novel=True,
        plot=False,
        filename_prefix="filtered",
    )


@pytest.mark.usefixtures(
    "setup_testfolder_files", "database_paths_mouse", "balbc_ighg_primers"
)
def test_assignsisotypes(
    create_testfolder, database_paths_mouse, balbc_ighg_primers
):
    """test_assignsisotypes"""
    ddl.pp.assign_isotypes(
        create_testfolder,
        org="mouse",
        blastdb=database_paths_mouse["blastdb_fasta"],
        correction_dict=balbc_ighg_primers,
        plot=False,
        filename_prefix="filtered",
    )


@pytest.mark.parametrize("strain", [None, "balbc", "BALB_c_ByJ"])
@pytest.mark.usefixtures(
    "setup_testfolder_files", "processed_files", "database_paths_mouse"
)
def test_create_germlines(
    create_testfolder,
    processed_files,
    database_paths_mouse,
    strain,
):
    """test_create_germlines - parametrized for different strains"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    ddl.pp.create_germlines(
        f,
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=strain,
    )
    f2 = create_testfolder / "dandelion" / processed_files["germ-pass"]
    dat = pd.read_csv(f2, sep="\t")
    assert not dat["germline_alignment_d_mask"].empty
