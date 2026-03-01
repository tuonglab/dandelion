#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import pytest

from dandelion.utilities._utilities import write_fasta, makeblastdb


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
def test_reannotategenes(create_testfolder, database_paths_mouse):
    """test_reannotategenes"""
    # different igblast versions/references may give different results and lead to regression.
    # disabling those checks would work for now.
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        org="mouse",
        reassign_dj=False,
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 4


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reassignalleles(create_testfolder, database_paths_mouse):
    """test_reassignalleles"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / "test_mouse",
        germline=database_paths_mouse["germline"],
        org="mouse",
        novel=True,
        plot=False,
        filename_prefix="filtered",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 7


@pytest.mark.usefixtures("database_paths_mouse")
def test_updateblastdb(database_paths_mouse):
    """test_updateblastdb"""
    makeblastdb(database_paths_mouse["blastdb_fasta"])


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
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths_mouse"
)
def test_create_germlines(
    create_testfolder, processed_files, database_paths_mouse
):
    """test_create_germlines"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    ddl.pp.create_germlines(
        f, org="mouse", germline=database_paths_mouse["germline"]
    )
    f2 = create_testfolder / "dandelion" / processed_files["germ-pass"]
    dat = pd.read_csv(f2, sep="\t")
    assert not dat["germline_alignment_d_mask"].empty


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "dummy_adata_mouse"
)
def test_filtercontigs(create_testfolder, processed_files, dummy_adata_mouse):
    """test_filtercontigs"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    dat = pd.read_csv(f, sep="\t")
    vdj, adata = ddl.pp.check_contigs(dat, dummy_adata_mouse)
    f1 = create_testfolder / "test.ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_ddl(f1)
    adata.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_generate_network(create_testfolder):
    """test_generate_network"""
    f = create_testfolder / "test.ddl"
    vdj = ddl.read_ddl(f)
    with pytest.raises(ValueError):
        ddl.tl.generate_network(vdj, compute_layout=False)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, layout_method="mod_fr", n_cpus=2)
    # assert vdj.n_obs == 448
    assert vdj.layout is not None
    assert vdj.graph is not None
    # assert len(vdj.graph[1]) == 8
    ddl.tl.generate_network(vdj, compute_layout=False, min_size=3)
    # assert len(vdj.graph[1]) == 0


@pytest.mark.usefixtures("create_testfolder")
def test_generate_network_new(create_testfolder):
    """test_generate_network"""
    f = create_testfolder / "test.ddl"
    vdj = ddl.read_ddl(f)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, layout_method="mod_fr", expanded_only=True)
    # assert vdj.n_obs == 448
    assert vdj.layout is not None
    assert vdj.graph is not None
    # assert len(vdj.graph[1]) == 8
    ddl.tl.generate_network(vdj)
    assert vdj.layout is not None
    assert vdj.graph is not None
