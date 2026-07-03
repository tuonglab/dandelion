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

from dandelion.utilities._utilities import write_fasta


@pytest.mark.usefixtures("create_testfolder", "fasta_10x")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 1), pytest.param("all", 2)]
)
def test_write_fasta(create_testfolder, fasta_10x, filename, expected):
    """test_write_fasta"""
    out_fasta = create_testfolder / (filename + "_contig.fasta")
    write_fasta(fasta_dict=fasta_10x, out_fasta=out_fasta)
    assert len(list(create_testfolder.iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 3), pytest.param("all", 4)]
)
def test_write_annotation(
    create_testfolder, annotation_10x, filename, expected
):
    """test_write_annotation"""
    out_file = create_testfolder / (filename + "_contig_annotations.csv")
    annotation_10x.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "filename,expected",
    [
        pytest.param(None, 2),
        pytest.param("all", 2),
        pytest.param("filtered", 4),
    ],
)
def test_formatfasta(create_testfolder, filename, expected):
    """test_formatfasta"""
    ddl.pp.format_fastas(create_testfolder, filename_prefix=filename)
    assert len(list((create_testfolder / "dandelion").iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder", "database_paths")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 5), pytest.param("all", 10)]
)
def test_reannotategenes(create_testfolder, database_paths, filename, expected):
    """test_reannotategenes"""
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths["igblast_db"],
        germline=database_paths["germline"],
        filename_prefix=filename,
    )
    assert (
        len(list((create_testfolder / "dandelion" / "tmp").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder", "database_paths")
@pytest.mark.parametrize(
    "filename, expected", [pytest.param("filtered", 5), pytest.param("all", 4)]
)
def test_assignsisotypes(create_testfolder, database_paths, filename, expected):
    """test_assignsisotypes"""
    ddl.pp.assign_isotypes(
        create_testfolder,
        blastdb=database_paths["blastdb_fasta"],
        filename_prefix=filename,
        save_plot=True,
        show_plot=False,
    )
    assert len(list((create_testfolder / "dandelion").iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder", "processed_files", "dummy_adata")
@pytest.mark.parametrize(
    "filename",
    [
        "filtered",
        "all",
    ],
)
def test_filter_pseudo(
    create_testfolder, processed_files, dummy_adata, filename, database_paths
):
    """test_filter_pseudo"""
    f = create_testfolder / "dandelion" / processed_files[filename]
    dat = ddl.Dandelion(f)
    annotated = ddl.pp.annotate_functionality(
        dat, germline=database_paths["germline"]
    )
    if annotated is not None:
        dat = annotated
    vdj, adata = ddl.pp.check_contigs(
        dat,
        dummy_adata,
        filter_pseudo={"v": ["P", "ORF"], "d": ["P", "ORF"], "j": ["P", "ORF"]},
    )
    assert vdj.n_contigs == 8
    assert vdj.n_obs == 5
    assert adata.n_obs == 5
    vdj, adata = ddl.pp.check_contigs(
        dat,
        dummy_adata,
        filter_pseudo={"v": ["F"], "d": ["F"], "j": ["F"]},
    )
    assert vdj.n_contigs == 0
    assert vdj.n_obs == 0
    assert adata.n_obs == 5
