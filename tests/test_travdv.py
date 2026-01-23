#!/usr/bin/env python
import dandelion as ddl
import pandas as pd
import polars as pl
import pytest

from dandelion.utilities._utilities import write_fasta
from dandelion.base.core._core import Dandelion, check_travdv


@pytest.mark.usefixtures("create_testfolder")
def test_loadtravdv(airr_travdv):
    """test_loadtravdv"""
    temp = check_travdv(airr_travdv)
    assert temp.shape[0] == 6
    assert all([i == "TRA" for i in airr_travdv["locus"]])
    assert all([i == "TRD" for i in temp["locus"]])


@pytest.mark.usefixtures("airr_travdv")
def test_loadtravdv2(airr_travdv):
    """test_loadtravdv2"""
    vdj = Dandelion(airr_travdv)
    assert vdj._data.shape[0] == 6
    assert all([i == "TRD" for i in vdj._data["locus"]])


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_travdv")
def test_write_fasta_tr(create_testfolder, fasta_10x_travdv):
    """testwrite_fasta_tr"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    write_fasta(fasta_dict=fasta_10x_travdv, out_fasta=out_fasta)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_travdv")
def test_write_annotation_tr(create_testfolder, annotation_10x_travdv):
    """test write annot"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_travdv.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test format fasta"""
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_reannotategenes(create_testfolder, database_paths):
    """test reannotate"""
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths["igblast_db"],
        germline=database_paths["germline"],
        loci="tr",
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 9
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_loadtravdv_reannotated(create_testfolder):
    """test check tradv"""
    vdj = ddl.Dandelion(
        create_testfolder / "dandelion" / "filtered_contig_dandelion.tsv"
    )
    if isinstance(vdj._data, pd.DataFrame):
        assert len([i for i in vdj._data["locus"] if i == "TRD"]) == 0
    else:
        assert (
            vdj._data.filter(pl.col("locus") == "TRD")
            .select(pl.len())
            .collect()
            .item()
            == 0
        )


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_travdv")
def test_travdv_filter(create_testfolder, dummy_adata_travdv):
    """test check tradv filter"""
    vdj = ddl.Dandelion(
        create_testfolder / "dandelion" / "filtered_contig_dandelion.tsv"
    )
    if isinstance(vdj._data, pd.DataFrame):
        assert len([i for i in vdj._data["locus"] if i == "TRD"]) == 0
    else:
        assert (
            vdj._data.filter(pl.col("locus") == "TRD")
            .select(pl.len())
            .collect()
            .item()
            == 0
        )
    vdj2, adata = ddl.pp.check_contigs(vdj, dummy_adata_travdv)
    if isinstance(vdj2._data, pd.DataFrame):
        assert vdj2._data.shape[0] > 0
    else:
        assert vdj2._data.collect().height > 0
