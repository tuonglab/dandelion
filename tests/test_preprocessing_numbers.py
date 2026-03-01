#!/usr/bin/env python
import pytest
import sys
import os
import pandas as pd
import dandelion as ddl
from pathlib import Path

try:
    os.environ.pop("IGDATA")
    os.environ.pop("GERMLINE")
    os.environ.pop("BLASTDB")
except KeyError:
    pass

from dandelion.base.core import Dandelion
from dandelion.base.io import read_h5ddl
from dandelion.base.preprocessing import (
    assign_isotypes,
    check_contigs,
    create_germlines,
    format_fastas,
    quantify_mutations,
    reannotate_genes,
    reassign_alleles,
)
from dandelion.utilities._utilities import write_fasta, makeblastdb


@pytest.mark.usefixtures("create_testfolder_number", "fasta_10x")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 1), pytest.param("all", 2)]
)
def test_write_fasta(create_testfolder_number, fasta_10x, filename, expected):
    """test_write_fasta"""
    out_fasta = create_testfolder_number / (filename + "_contig.fasta")
    write_fasta(fasta_dict=fasta_10x, out_fasta=out_fasta)
    assert len(list(create_testfolder_number.iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder_number", "annotation_10x")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 3), pytest.param("all", 4)]
)
def test_write_annotation(
    create_testfolder_number, annotation_10x, filename, expected
):
    """test_write_annotation"""
    out_file = create_testfolder_number / (filename + "_contig_annotations.csv")
    annotation_10x.to_csv(out_file, index=False)
    assert len(list(create_testfolder_number.iterdir())) == expected


@pytest.mark.usefixtures("create_testfolder_number")
@pytest.mark.parametrize(
    "filename,expected",
    [
        pytest.param(None, 2),
        pytest.param("all", 2),
        pytest.param("filtered", 4),
    ],
)
def test_formatfasta(create_testfolder_number, filename, expected):
    """test_formatfasta"""
    format_fastas(create_testfolder_number, filename_prefix=filename)
    assert (
        len(list((create_testfolder_number / "dandelion").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
def test_reannotate_fails(create_testfolder_number, database_paths):
    """test_reannotate_fails"""
    with pytest.raises(KeyError):
        reannotate_genes(create_testfolder_number, filename_prefix="filtered")
    with pytest.raises(KeyError):
        reannotate_genes(
            create_testfolder_number,
            igblast_db=database_paths["igblast_db"],
            filename_prefix="filtered",
        )
    with pytest.raises(KeyError):
        reannotate_genes(
            create_testfolder_number,
            germline=database_paths["germline"],
            filename_prefix="filtered",
        )


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
@pytest.mark.parametrize(
    "filename,expected", [pytest.param("filtered", 5), pytest.param("all", 10)]
)
def test_reannotategenes(
    create_testfolder_number, database_paths, filename, expected
):
    """test_reannotategenes"""
    reannotate_genes(
        create_testfolder_number,
        igblast_db=database_paths["igblast_db"],
        germline=database_paths["germline"],
        filename_prefix=filename,
    )
    assert (
        len(list((create_testfolder_number / "dandelion" / "tmp").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
def test_reassign_alleles_fails(create_testfolder_number, database_paths):
    """test_reassign_alleles_fails"""
    with pytest.raises(TypeError):
        reassign_alleles(
            str(create_testfolder_number), filename_prefix="filtered"
        )
    with pytest.raises(KeyError):
        reassign_alleles(
            str(create_testfolder_number),
            combined_folder=create_testfolder_number / "reassigned_filtered",
            filename_prefix="filtered",
        )


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
@pytest.mark.parametrize(
    "filename,combine,expected",
    [
        pytest.param("filtered", "reassigned_filtered", 13),
        pytest.param("all", "reassigned_all", 16),
    ],
)
def test_reassignalleles(
    create_testfolder_number, database_paths, filename, combine, expected
):
    """test_reassignalleles"""
    reassign_alleles(
        str(create_testfolder_number),
        combined_folder=create_testfolder_number / combine,
        germline=database_paths["germline"],
        filename_prefix=filename,
        novel=True,
        save_plot=True,
        show_plot=False,
    )
    assert (
        len(list((create_testfolder_number / "dandelion" / "tmp").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
def test_reassign_alleles_combined_number(
    create_testfolder_number, database_paths
):
    """test_reassign_alleles_fails"""
    reassign_alleles(
        str(create_testfolder_number),
        combined_folder=create_testfolder_number / str(2),
        germline=database_paths["germline"],
        filename_prefix="all",
        novel=True,
        save_plot=True,
        show_plot=False,
    )
    assert (
        len(list((create_testfolder_number / "dandelion" / "tmp").iterdir()))
        == 16
    )


@pytest.mark.usefixtures("database_paths")
def test_updateblastdb(database_paths):
    """test_updateblastdb"""
    makeblastdb(database_paths["blastdb_fasta"])


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
@pytest.mark.parametrize(
    "filename, expected", [pytest.param("filtered", 5), pytest.param("all", 4)]
)
def test_assignsisotypes(
    create_testfolder_number, database_paths, filename, expected
):
    """test_assignsisotypes"""
    assign_isotypes(
        create_testfolder_number,
        blastdb=database_paths["blastdb_fasta"],
        filename_prefix=filename,
        save_plot=True,
        show_plot=False,
    )
    assert (
        len(list((create_testfolder_number / "dandelion").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
@pytest.mark.parametrize("filename", ["all", "filtered"])
def test_checkccall(create_testfolder_number, processed_files, filename):
    """test_checkccall"""
    f = create_testfolder_number / "dandelion" / processed_files[filename]
    dat = pd.read_csv(f, sep="\t")
    assert not dat["c_call"].empty


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
def test_create_germlines_fails(create_testfolder_number, processed_files):
    """test_create_germlines_fails"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    with pytest.raises(KeyError):
        create_germlines(f)


@pytest.mark.usefixtures(
    "create_testfolder_number", "processed_files", "database_paths"
)
def test_create_germlines(
    create_testfolder_number, processed_files, database_paths
):
    """test_create_germlines"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    create_germlines(f, germline=database_paths["germline"])
    f2 = create_testfolder_number / "dandelion" / processed_files["germ-pass"]
    dat = pd.read_csv(f2, sep="\t")
    assert not dat["germline_alignment_d_mask"].empty


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
def test_store_germline_references_fail2(
    create_testfolder_number, processed_files
):
    """test_store_germline_references_fail2"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder_number", "processed_files", "database_paths"
)
def test_store_germline_references(
    create_testfolder_number, processed_files, database_paths
):
    """test_store_germline_references"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
@pytest.mark.parametrize(
    "freq,colname,dtype",
    [
        pytest.param(True, "mu_freq", float),
        pytest.param(False, "mu_count", int),
    ],
)
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_quantify_mut(
    create_testfolder_number, processed_files, freq, colname, dtype
):
    """test_quantify_mut"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    try:
        quantify_mutations(f, frequency=freq)
    except:
        pytest.skip("R package 'shazam' not installed")
    dat = pd.read_csv(f, sep="\t")
    assert not dat[colname].empty
    assert dat[colname].dtype == dtype


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
@pytest.mark.parametrize(
    "freq,colname",
    [pytest.param(True, "mu_freq"), pytest.param(False, "mu_count")],
)
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_quantify_mut_2(
    create_testfolder_number, processed_files, freq, colname
):
    """test_quantify_mut_2"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    try:
        quantify_mutations(vdj, frequency=freq)
    except:
        pytest.skip("R package 'shazam' not installed")
    assert not vdj._data[colname].empty
    if colname == "mu_freq":
        assert vdj._data[colname].dtype == float
    else:
        assert vdj._data[colname].dtype == int


@pytest.mark.usefixtures(
    "create_testfolder_number", "processed_files", "dummy_adata"
)
@pytest.mark.parametrize(
    "filename,size",
    [
        pytest.param("filtered", 8),
        pytest.param("all", 8),
    ],
)
def test_checkcontigs(
    create_testfolder_number,
    processed_files,
    dummy_adata,
    filename,
    size,
):
    """test_checkcontigs"""
    f = create_testfolder_number / "dandelion" / processed_files[filename]
    dat = pd.read_csv(f, sep="\t")
    vdj, adata = check_contigs(dat, dummy_adata)
    assert dat.shape[0] == 9
    assert vdj._data.shape[0] == size
    assert vdj._metadata.shape[0] == 5
    assert adata.n_obs == 5


@pytest.mark.usefixtures("create_testfolder_number", "database_paths")
def test_assign_isotypes_fails(create_testfolder_number, database_paths):
    """test_assign_isotypes_fails"""
    with pytest.raises(FileNotFoundError):
        assign_isotypes(
            create_testfolder_number, filename_prefix="filtered", plot=False
        )
    format_fastas(create_testfolder_number, filename_prefix="filtered")
    with pytest.raises(KeyError):
        assign_isotypes(
            create_testfolder_number, filename_prefix="filtered", plot=False
        )


@pytest.mark.usefixtures("create_testfolder_number")
@pytest.mark.parametrize(
    "prefix,suffix,sep,remove",
    [
        pytest.param("test", None, None, True),
        pytest.param("test", None, None, False),
        pytest.param(None, "test", None, True),
        pytest.param(None, "test", None, False),
        pytest.param(None, None, "-", True),
        pytest.param(None, None, "-", False),
        pytest.param("test", "test", "-", True),
        pytest.param("test", "test", "-", False),
    ],
)
def test_formatfasta2(create_testfolder_number, prefix, suffix, sep, remove):
    """test_formatfasta2"""
    format_fastas(
        create_testfolder_number,
        filename_prefix="filtered",
        prefix=prefix,
        suffix=suffix,
        sep=sep,
        remove_trailing_hyphen_number=remove,
    )
    f = (
        create_testfolder_number
        / "dandelion"
        / "filtered_contig_annotations.csv"
    )
    df = pd.read_csv(f)
    contig = list(df["contig_id"])[0]
    if prefix is None:
        if remove:
            if suffix is not None:
                if sep is None:
                    assert contig.split("_contig")[0].endswith("_" + suffix)
                else:
                    assert contig.split("_contig")[0].endswith(sep + suffix)
            else:
                assert contig.split("_contig")[0].endswith("-1")
        else:
            if suffix is None:
                assert contig.split("_contig")[0].endswith("-1")
            else:
                if sep is None:
                    assert contig.split("_contig")[0].endswith("_" + suffix)
                else:
                    assert contig.split("_contig")[0].endswith(sep + suffix)
    else:
        if sep is None:
            assert contig.startswith(prefix + "_")
        else:
            assert contig.startswith(prefix + sep)


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
def test_store_germline_references_fail(
    create_testfolder_number, processed_files
):
    """test_store_germline_references_fail"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder_number", "processed_files", "database_paths", "fasta_10x"
)
def test_store_germline_references2(
    create_testfolder_number, processed_files, database_paths, fasta_10x
):
    """test_store_germline_references2"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0
    out_file = create_testfolder_number / "test_airr_reannotated.h5ddl"
    vdj.write_h5ddl(out_file)
    tmp = read_h5ddl(out_file)
    assert len(tmp.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"],
        corrected=create_testfolder_number / "filtered_contig.fasta",
    )
    assert len(vdj.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"], corrected=fasta_10x
    )
    assert len(vdj.germline) > 0
    with pytest.raises(TypeError):
        vdj.store_germline_reference(
            germline=database_paths["germline"], corrected=[]
        )


@pytest.mark.usefixtures("create_testfolder_number", "processed_files")
def test_store_germline_reference_fail(
    create_testfolder_number, processed_files
):
    """test_store_germline_reference_fail"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder_number", "processed_files", "database_paths", "fasta_10x"
)
def test_store_germline_reference2(
    create_testfolder_number, processed_files, database_paths, fasta_10x
):
    """test_store_germline_references2"""
    f = create_testfolder_number / "dandelion" / processed_files["filtered"]
    vdj = Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0
    out_file = create_testfolder_number / "test_airr_reannotated.h5ddl"
    vdj.write_h5ddl(out_file)
    tmp = read_h5ddl(out_file)
    assert len(tmp.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"],
        corrected=create_testfolder_number / "filtered_contig.fasta",
    )
    assert len(vdj.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"], corrected=fasta_10x
    )
    assert len(vdj.germline) > 0
    with pytest.raises(TypeError):
        vdj.store_germline_reference(
            germline=database_paths["germline"], corrected=[]
        )
