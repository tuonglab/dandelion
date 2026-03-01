#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import pytest

from dandelion.utilities._utilities import write_fasta


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test_write_fasta"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    write_fasta(fasta_dict=fasta_10x_mouse, out_fasta=out_fasta)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test_write_annotation"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test_formatfasta"""
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_nod(
    create_testfolder,
    database_paths_mouse,
):
    """test_reannotategenes"""
    # different igblast versions/references may give different results and lead to regression.
    # disabling those checks would work for now.
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=None,
        filename_prefix="filtered",
    )


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reassignalleles(
    create_testfolder,
    database_paths_mouse,
):
    """test_reassignalleles"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / "test_mouse",
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=None,
        novel=True,
        plot=False,
        filename_prefix="filtered",
    )


@pytest.mark.usefixtures(
    "create_testfolder", "database_paths_mouse", "balbc_ighg_primers"
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


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths_mouse"
)
def test_create_germlines(
    create_testfolder,
    processed_files,
    database_paths_mouse,
):
    """test_create_germlines"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    ddl.pp.create_germlines(
        f,
        germline=database_paths_mouse["ogrdb"],
        org="mouse",
        db="ogrdb",
        strain=None,
    )
    f2 = create_testfolder / "dandelion" / processed_files["germ-pass"]
    dat = pd.read_csv(f2, sep="\t")
    assert not dat["germline_alignment_d_mask"].empty
