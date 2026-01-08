#!/usr/bin/env python
import os
import json
import pytest
import dandelion as ddl
import pandas as pd

from dandelion.tools._tools import to_ak, from_ak


@pytest.mark.usefixtures("create_testfolder", "airr_10x")
def test_write_airr(create_testfolder, airr_10x):
    """test_write_airr"""
    out_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_10x.to_csv(out_file, sep="\t", index=False)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder")
def test_loaddata(create_testfolder):
    """test_loaddata"""
    file1 = create_testfolder / "test_airr_rearrangements.tsv"
    file2 = create_testfolder / "test_airr_rearrangements2.tsv"
    dat = ddl.utl.load_data(file1)
    assert isinstance(dat, pd.DataFrame)
    with pytest.raises(ValueError):
        dat2 = ddl.utl.load_data(file2)
    with pytest.raises(ValueError):
        dat2 = ddl.utl.load_data("something.tsv")
    dat2 = pd.read_csv(file1, sep="\t")
    dat2.drop("sequence_id", inplace=True, axis=1)
    with pytest.raises(KeyError):
        dat2 = ddl.utl.load_data(dat2)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_write_annotated(create_testfolder, airr_reannotated):
    """test_write_annotated"""
    out_file = create_testfolder / "test_airr_reannotated.tsv"
    airr_reannotated.to_csv(out_file, sep="\t", index=False)
    assert not airr_reannotated.np1_length.empty
    assert not airr_reannotated.np2_length.empty
    assert not airr_reannotated.junction_length.empty


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite_h5ddl(create_testfolder):
    """test_readwrite_h5"""
    out_file1 = create_testfolder / "test_airr_reannotated.tsv"
    out_file2 = create_testfolder / "test_airr_reannotated.h5ddl"
    vdj = ddl.Dandelion(out_file1)
    assert not vdj._data.np1_length.empty
    assert not vdj._data.np2_length.empty
    assert not vdj._data.junction_length.empty
    vdj.write_h5ddl(out_file2)
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2._data.np1_length.empty
    assert not vdj2._data.np2_length.empty
    assert not vdj2._data.junction_length.empty
    vdj.write_h5ddl(out_file2)
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2._data.np1_length.empty
    assert not vdj2._data.np2_length.empty
    assert not vdj2._data.junction_length.empty
    vdj.write_h5ddl(out_file2)
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2._data.np1_length.empty
    assert not vdj2._data.np2_length.empty
    assert not vdj2._data.junction_length.empty
    with pytest.raises(ValueError):
        vdj.write_h5ddl(out_file2, compression="blosc")


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite_pkl(create_testfolder):
    """test_readwrite_pkl"""
    out_file1 = create_testfolder / "test_airr_reannotated.tsv"
    out_file2 = create_testfolder / "test_airr_reannotated.pkl"
    out_file3 = create_testfolder / "test_airr_reannotated.pkl.gz"
    out_file4 = create_testfolder / "test_airr_reannotated.pkl.pbz2"
    vdj = ddl.Dandelion(out_file1)
    assert not vdj._data.np1_length.empty
    assert not vdj._data.np2_length.empty
    assert not vdj._data.junction_length.empty
    vdj.write_pkl(out_file2)
    vdj3 = ddl.read_pkl(out_file2)
    assert not vdj3._data.np1_length.empty
    assert not vdj3._data.np2_length.empty
    assert not vdj3._data.junction_length.empty
    vdj.write_pkl(out_file3)
    vdj4 = ddl.read_pkl(out_file3)
    assert not vdj4._data.np1_length.empty
    assert not vdj4._data.np2_length.empty
    assert not vdj4._data.junction_length.empty
    vdj.write_pkl(out_file4)
    vdj5 = ddl.read_pkl(out_file4)
    assert not vdj5._data.np1_length.empty
    assert not vdj5._data.np2_length.empty
    assert not vdj5._data.junction_length.empty


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite10xairr(create_testfolder):
    """test_readwrite10xairr"""
    airr_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_file2 = create_testfolder / "test_airr_rearrangements2.tsv"
    vdj = ddl.read_10x_airr(airr_file)
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    vdj.write_airr(airr_file2)
    vdj2 = ddl.read_10x_airr(airr_file2)
    assert vdj2._data.shape[0] == 9
    assert vdj2._metadata.shape[0] == 5
    os.remove(airr_file2)


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6")
def test_read10xvdj_json(create_testfolder, json_10x_cr6):
    """test_read10xvdj_json"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(json_file, filename_prefix="test_all")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    os.remove(json_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    """test_read10xvdj_cr6"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    assert not vdj._data.sequence.empty
    os.remove(json_file)
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(annot_file, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    assert not vdj._data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj(create_testfolder, annotation_10x, fasta_10x):
    """test_read10xvdj"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    assert not vdj._data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6_folder(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    """test_read10xvdj_cr6_folder"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    assert not vdj._data.sequence.empty
    os.remove(json_file)
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    assert not vdj._data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj_folder(create_testfolder, annotation_10x, fasta_10x):
    """test_read10xvdj_folder"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    assert not vdj._data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_io_prefix_suffix_combinations(create_testfolder, annotation_10x):
    """test_io_prefix_suffix_combinations"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    airr_file = create_testfolder / "test_airr_rearrangements.tsv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file, suffix="x")
    vdj = ddl.read_10x_vdj(annot_file, prefix="y")
    vdj = ddl.read_10x_vdj(
        annot_file,
        suffix="x",
        remove_trailing_hyphen_number=True,
        filename_prefix="filtered",
    )
    vdj = ddl.read_10x_vdj(
        annot_file,
        prefix="x",
        remove_trailing_hyphen_number=True,
        filename_prefix="filtered",
    )
    vdj = ddl.read_10x_airr(airr_file, suffix="x")
    vdj = ddl.read_10x_airr(airr_file, prefix="y")
    vdj = ddl.read_10x_airr(
        airr_file, suffix="x", remove_trailing_hyphen_number=True
    )
    vdj = ddl.read_10x_airr(
        airr_file, prefix="x", remove_trailing_hyphen_number=True
    )
    vdj = ddl.read_10x_vdj(annot_file, filename_prefix="filtered")
    _ = ddl.tl.concat([vdj, vdj], prefixes=["x", "y"])
    with pytest.raises(ValueError):
        _ = ddl.tl.concat([vdj, vdj], suffixes=["x"])
    with pytest.raises(ValueError):
        _ = ddl.tl.concat([vdj, vdj], prefixes=["y"])
    with pytest.raises(ValueError):
        _ = ddl.tl.concat([vdj, vdj], suffixes=["x", "y"], prefixes=["y", "z"])
    # also test with the different metadata.
    vdj1 = vdj.copy()
    vdj1._metadata["new_col"] = "test"
    _ = ddl.tl.concat([vdj, vdj1])


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_scirpy(create_testfolder, annotation_10x, fasta_10x):
    """test_to_scirpy"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="filtered")
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    adata = ddl.tl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 5
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="filtered")
    assert vdj._data.shape[0] == 9
    assert vdj._metadata.shape[0] == 5
    assert not vdj._data.sequence.empty
    adata = ddl.tl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 5
    os.remove(fasta_file)
    vdjx = ddl.tl.from_scirpy(adata)
    assert vdjx._data.shape[0] == 9


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x_cr6", "json_10x_cr6"
)
def test_tofro_scirpy_cr6(create_testfolder, annotation_10x_cr6, json_10x_cr6):
    """test_tofro_scirpy_cr6"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_all_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    adata = ddl.tl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 10
    vdjx = ddl.tl.from_scirpy(adata)
    assert vdjx._data.shape[0] == 26


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "json_10x_cr6")
def test_tofro_scirpy_cr6_transfer(
    create_testfolder, annotation_10x_cr6, json_10x_cr6
):
    """test_tofro_scirpy_cr6_transfer"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_all_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj._data.shape[0] == 26
    assert vdj._metadata.shape[0] == 10
    adata = ddl.tl.to_scirpy(vdj, transfer=True)
    assert adata.obs.shape[0] == 10
    vdjx = ddl.tl.from_scirpy(adata)
    assert vdjx._data.shape[0] == 26


@pytest.mark.usefixtures("airr_generic")
def test_librarytype(airr_generic):
    """test library type"""
    tmp = ddl.Dandelion(airr_generic)
    assert tmp._data.shape[0] == 130
    assert tmp._metadata.shape[0] == 45

    tmp = ddl.Dandelion(airr_generic, library_type="ig")
    assert tmp._data.shape[0] == 68
    assert tmp._metadata.shape[0] == 25

    tmp = ddl.Dandelion(airr_generic, library_type="tr-ab")
    assert tmp._data.shape[0] == 37
    assert tmp._metadata.shape[0] == 20

    tmp = ddl.Dandelion(airr_generic, library_type="tr-gd")
    assert tmp._data.shape[0] == 25
    assert tmp._metadata.shape[0] == 15


@pytest.mark.usefixtures("create_testfolder")
def test_convert_obsm_airr_to_data(create_testfolder):
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    mdata = ddl.tl.to_scirpy(vdj)

    result = from_ak(mdata["airr"].obsm["airr"])

    assert result.shape == vdj._data.shape
    assert result.shape[0] == 26


def test_convert_data_to_obsm_airr(create_testfolder):
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    anndata = ddl.tl.to_scirpy(vdj, to_mudata=False)
    obsm_airr, obs = to_ak(vdj._data)
    assert len(anndata.obsm["airr"]) == len(obsm_airr)
    # assert anndata.obsm["airr"].type.show() == obsm_airr.type.show()
    assert anndata.obs.shape == obs.shape


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_scirpy_v2(create_testfolder, annotation_10x, fasta_10x):
    """test_to_scirpy"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 35
    assert vdj._metadata.shape[0] == 15
    adata = ddl.tl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 15
    ddl.utl._core.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    assert vdj._data.shape[0] == 35
    assert vdj._metadata.shape[0] == 15
    assert not vdj._data.sequence.empty
    adata = ddl.tl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 15
    mdata = ddl.tl.to_scirpy(vdj, to_mudata=True)
    assert mdata.mod["airr"].shape[0] == 15
    os.remove(fasta_file)
    vdjx = ddl.tl.from_scirpy(adata)
    assert vdjx._data.shape[0] == 35
    vdjx = ddl.tl.from_scirpy(mdata)
    assert vdjx._data.shape[0] == 35


@pytest.mark.usefixtures("airr_generic")
def test_locus_productive(airr_generic):
    """Just test if this works. don't care about the output."""
    tmp = ddl.Dandelion(airr_generic, report_status_productive=True)
    tmp = ddl.Dandelion(airr_generic, report_status_productive=False)


@pytest.mark.usefixtures("airr_generic")
def test_write_10x(airr_generic, create_testfolder):
    vdj = ddl.Dandelion(airr_generic)
    vdj.write_10x(folder=create_testfolder / "test_10x")


@pytest.mark.usefixtures("airr_bd")
def test_read_bd(airr_bd):
    vdj = ddl.read_bd_airr(airr_bd)
    vdj2 = ddl.pp.check_contigs(vdj)
    assert vdj2._metadata.shape[0] == 10


@pytest.mark.usefixtures("airr_parse")
def test_read_parse(airr_parse):
    vdj = ddl.read_parse_airr(airr_parse)
    vdj2 = ddl.pp.check_contigs(vdj)
    assert vdj2._metadata.shape[0] == 10


@pytest.mark.usefixtures("airr_bd")
def test_read_standard(airr_bd):
    vdj = ddl.read_airr(airr_bd)
    vdj2 = ddl.pp.check_contigs(vdj)
    assert vdj2._metadata.shape[0] == 10


@pytest.mark.skip(reason="can't install dependencies on github actions.")
@pytest.mark.usefixtures("create_testfolder")
def test_legacy_write(create_testfolder):
    """check i can read and write in legacy mode."""
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="junction")
    vdj.write_h5ddl(create_testfolder / "legacy.h5ddl", version=3)
    _ = ddl.read_h5ddl(create_testfolder / "legacy.h5ddl")
