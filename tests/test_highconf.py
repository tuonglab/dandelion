#!/usr/bin/env python
import pytest
import dandelion as ddl

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
    ddl.pp.format_fastas(
        create_testfolder,
        filename_prefix=filename,
        high_confidence_filtering=True,
    )
    assert len(list((create_testfolder / "dandelion").iterdir())) == expected
