#!/usr/bin/env python
"""
Backend-agnostic tools and core tests.

Imports from the top-level ``dandelion`` namespace so that the active backend
(controlled by the ``DANDELION_BACKEND`` env-var) determines which code path
is exercised.  CI runs this file once per backend to cover both
``dandelion.base`` and ``dandelion.polars``.
"""

import pytest
import scanpy as sc

import dandelion as ddl

# -- core / preprocessing ----------------------------------------------------


@pytest.mark.usefixtures(
    "create_testfolder", "airr_reannotated", "airr_reannotated2", "dummy_adata"
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """Create Dandelion objects via check_contigs and persist for later tests."""
    vdj, adata = ddl.pp.check_contigs(airr_reannotated, dummy_adata)
    vdj2 = ddl.pp.check_contigs(airr_reannotated2)
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj.write(f)
    vdj2.write(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj2 = ddl.read(f2)
    ddl.tl.find_clones(vdj)
    ddl.tl.find_clones(vdj2)
    vdj.write(f)
    vdj2.write(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.find_clones(vdj, key_added="test_clone")


@pytest.mark.usefixtures("create_testfolder")
def test_clone_size(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_size(vdj)
    ddl.tl.clone_size(vdj, max_size=3)


@pytest.mark.usefixtures("create_testfolder")
def test_generate_network(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj2 = ddl.read(f2)
    ddl.tl.generate_network(vdj2, layout_method="mod_fr")
    vdj2.write(f2)
    adata = ddl.tl.to_scirpy(vdj, to_mudata=False)
    ddl.tl.generate_network(vdj, adata=adata, layout_method="mod_fr")
    ddl.tl.transfer(adata, vdj)
    vdj.write(f)
    f3 = create_testfolder / "test_backend.h5ad"
    adata.write_h5ad(f3)


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    f = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj, adata = ddl.pp.check_contigs(vdj, dummy_adata2)
    ddl.tl.transfer(dummy_adata2, vdj)
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.transfer(dummy_adata2, vdj)


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.extract_edge_weights(vdj)
    ddl.tl.extract_edge_weights(vdj, expanded_only=True)


# -- diversity ----------------------------------------------------------------


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("method", ["chao1", "shannon", "gini"])
def test_diversity_anndata(create_testfolder, method):
    f = create_testfolder / "test_backend.h5ad"
    adata = sc.read_h5ad(f)
    ddl.tl.clone_diversity(adata, groupby="sample_id", method=method, n_boot=5)


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_ddl(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_diversity(vdj, groupby="sample_id", key="sequence", n_boot=5)


@pytest.mark.usefixtures("create_testfolder")
def test_clone_rarefaction(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_rarefaction(vdj, groupby="sample_id")


# -- scirpy / concat ----------------------------------------------------------


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_from_scirpy(create_testfolder, annotation_10x, fasta_10x):
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    adata = ddl.tl.to_scirpy(vdj)
    ddl.tl.from_scirpy(adata)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_concat(create_testfolder, annotation_10x):
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    ddl.tl.concat([vdj, vdj], prefixes=["x", "y"])


# -- find_clones options ------------------------------------------------------


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_from_dataframe(airr_reannotated):
    vdj = ddl.Dandelion(airr_reannotated)
    ddl.tl.find_clones(vdj)


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_by_alleles(airr_reannotated):
    vdj = ddl.Dandelion(airr_reannotated)
    ddl.tl.find_clones(vdj, by_alleles=True)


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_after_network(airr_reannotated):
    vdj = ddl.pp.check_contigs(airr_reannotated)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="junction_aa", layout_method="mod_fr")
    vdj2 = vdj.copy()
    ddl.tl.find_clones(vdj2)
    ddl.tl.find_clones(vdj2, key_added="cloned_idx")
