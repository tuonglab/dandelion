#!/usr/bin/env python
import pytest
import os
import pandas as pd
import dandelion as ddl

try:
    os.environ.pop("IGDATA")
    os.environ.pop("GERMLINE")
    os.environ.pop("BLASTDB")
except KeyError:
    pass

from dandelion.utilities._utilities import write_fasta, makeblastdb


@pytest.mark.usefixtures("create_testfolder", "fasta_10x")
def test_write_fasta(create_testfolder, fasta_10x):
    """test_write_fasta"""
    fastafilename = str(create_testfolder / "all_contig.fasta")
    write_fasta(fasta_dict=fasta_10x, out_fasta=fastafilename)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_write_annotation(
    create_testfolder,
    annotation_10x,
):
    """test_write_annotation"""
    annofilename = str(create_testfolder / "all_contig_annotations.csv")
    annotation_10x.to_csv(annofilename, index=False)


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test_formatfasta"""
    ddl.pp.format_fastas(create_testfolder, filename_prefix="all")


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_reannotategenes(create_testfolder, database_paths):
    """test_reannotategenes"""
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths["igblast_db"],
        germline=database_paths["germline"],
        filename_prefix="all",
    )


@pytest.mark.usefixtures("database_paths")
def test_updateblastdb(database_paths):
    """test_updateblastdb"""
    makeblastdb(database_paths["blastdb_fasta"])


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_assignsisotypes(create_testfolder, database_paths):
    """test_assignsisotypes"""
    ddl.pp.assign_isotypes(
        create_testfolder,
        blastdb=database_paths["blastdb_fasta"],
        filename_prefix="all",
        save_plot=True,
        show_plot=False,
    )


@pytest.mark.usefixtures("create_testfolder", "processed_files")
def test_checkccall(create_testfolder, processed_files):
    """test_checkccall"""
    f = create_testfolder / "dandelion" / processed_files["all"]
    dat = pd.read_csv(f, sep="\t")
    assert not dat["c_call"].empty
