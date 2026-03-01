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
def test_reannotate_fails(create_testfolder, database_paths):
    """test_reannotate_fails"""
    with pytest.raises(KeyError):
        ddl.pp.reannotate_genes(create_testfolder, filename_prefix="filtered")
    with pytest.raises(KeyError):
        ddl.pp.reannotate_genes(
            create_testfolder,
            igblast_db=database_paths["igblast_db"],
            filename_prefix="filtered",
        )
    with pytest.raises(KeyError):
        ddl.pp.reannotate_genes(
            create_testfolder,
            germline=database_paths["germline"],
            filename_prefix="filtered",
        )


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
def test_reassign_alleles_fails(create_testfolder, database_paths):
    """test_reassign_alleles_fails"""
    with pytest.raises(TypeError):
        ddl.pp.reassign_alleles(
            str(create_testfolder), filename_prefix="filtered"
        )
    with pytest.raises(KeyError):
        ddl.pp.reassign_alleles(
            str(create_testfolder),
            combined_folder=create_testfolder / "reassigned_filtered",
            filename_prefix="filtered",
        )


@pytest.mark.usefixtures("create_testfolder", "database_paths")
@pytest.mark.parametrize(
    "filename,combine,expected",
    [
        pytest.param("filtered", "reassigned_filtered", 13),
        pytest.param("all", "reassigned_all", 16),
    ],
)
def test_reassignalleles(
    create_testfolder, database_paths, filename, combine, expected
):
    """test_reassignalleles"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / combine,
        germline=database_paths["germline"],
        filename_prefix=filename,
        novel=True,
        save_plot=True,
        show_plot=False,
    )
    assert (
        len(list((create_testfolder / "dandelion" / "tmp").iterdir()))
        == expected
    )


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_reassign_alleles_combined_number(create_testfolder, database_paths):
    """test_reassign_alleles_fails"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder=create_testfolder / str(2),
        germline=database_paths["germline"],
        filename_prefix="all",
        novel=True,
        save_plot=True,
        show_plot=False,
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 16


@pytest.mark.usefixtures("database_paths")
def test_updateblastdb(database_paths):
    """test_updateblastdb"""
    makeblastdb(database_paths["blastdb_fasta"])


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


@pytest.mark.usefixtures("create_testfolder", "processed_files")
@pytest.mark.parametrize("filename", ["all", "filtered"])
def test_checkccall(create_testfolder, processed_files, filename):
    """test_checkccall"""
    f = create_testfolder / "dandelion" / processed_files[filename]
    dat = pd.read_csv(f, sep="\t")
    assert not dat["c_call"].empty


@pytest.mark.usefixtures("create_testfolder", "processed_files")
def test_create_germlines_fails(create_testfolder, processed_files):
    """test_create_germlines_fails"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    with pytest.raises(KeyError):
        ddl.pp.create_germlines(f)


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths"
)
def test_create_germlines(create_testfolder, processed_files, database_paths):
    """test_create_germlines"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    ddl.pp.create_germlines(f, germline=database_paths["germline"])
    f2 = create_testfolder / "dandelion" / processed_files["germ-pass"]
    dat = pd.read_csv(f2, sep="\t")
    assert not dat["germline_alignment_d_mask"].empty


@pytest.mark.usefixtures("create_testfolder", "processed_files")
def test_store_germline_references_fail2(create_testfolder, processed_files):
    """test_store_germline_references_fail2"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths"
)
def test_store_germline_references(
    create_testfolder, processed_files, database_paths
):
    """test_store_germline_references"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0


@pytest.mark.usefixtures("create_testfolder", "processed_files")
@pytest.mark.parametrize(
    "freq,colname,dtype",
    [
        pytest.param(True, "mu_freq", float),
        pytest.param(False, "mu_count", int),
    ],
)
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_quantify_mut(create_testfolder, processed_files, freq, colname, dtype):
    """test_quantify_mut"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    try:
        ddl.pp.quantify_mutations(f, frequency=freq)
    except:
        pytest.skip("R package 'shazam' not installed")
    dat = pd.read_csv(f, sep="\t")
    assert not dat[colname].empty
    # assert dat[colname].dtype == dtype


@pytest.mark.usefixtures("create_testfolder", "processed_files")
@pytest.mark.parametrize(
    "freq,colname",
    [pytest.param(True, "mu_freq"), pytest.param(False, "mu_count")],
)
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_quantify_mut_2(create_testfolder, processed_files, freq, colname):
    """test_quantify_mut_2"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    try:
        ddl.pp.quantify_mutations(vdj, frequency=freq)
    except:
        pytest.skip("R package 'shazam' not installed")
    if isinstance(vdj._data, pd.DataFrame):
        assert not vdj._data[colname].empty
    else:
        assert not vdj._data[colname].is_null().all()


@pytest.mark.usefixtures("create_testfolder", "processed_files", "dummy_adata")
@pytest.mark.parametrize(
    "filename",
    [
        "filtered",
        "all",
    ],
)
def test_checkcontigs(
    create_testfolder, processed_files, dummy_adata, filename
):
    """test_checkcontigs"""
    f = create_testfolder / "dandelion" / processed_files[filename]
    dat = pd.read_csv(f, sep="\t")
    vdj, adata = ddl.pp.check_contigs(dat, dummy_adata)
    assert dat.shape[0] == 9
    assert vdj.n_contigs == 8
    assert vdj.n_obs == 5
    assert adata.n_obs == 5


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_assign_isotypes_fails(create_testfolder, database_paths):
    """test_assign_isotypes_fails"""
    with pytest.raises(FileNotFoundError):
        ddl.pp.assign_isotypes(
            create_testfolder, filename_prefix="filtered", plot=False
        )
    ddl.pp.format_fastas(create_testfolder, filename_prefix="filtered")
    with pytest.raises(KeyError):
        ddl.pp.assign_isotypes(
            create_testfolder, filename_prefix="filtered", plot=False
        )


@pytest.mark.usefixtures("create_testfolder")
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
def test_formatfasta2(create_testfolder, prefix, suffix, sep, remove):
    """test_formatfasta2"""
    ddl.pp.format_fastas(
        create_testfolder,
        filename_prefix="filtered",
        prefix=prefix,
        suffix=suffix,
        sep=sep,
        remove_trailing_hyphen_number=remove,
    )
    f = create_testfolder / "dandelion" / "filtered_contig_annotations.csv"
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


@pytest.mark.usefixtures("create_testfolder", "processed_files")
def test_store_germline_references_fail(create_testfolder, processed_files):
    """test_store_germline_references_fail"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths", "fasta_10x"
)
def test_store_germline_references2(
    create_testfolder, processed_files, database_paths, fasta_10x
):
    """test_store_germline_references2"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0
    out_file = create_testfolder / "test_airr_reannotated.ddl"
    vdj.write(out_file)
    tmp = ddl.read(out_file)
    assert len(tmp.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"],
        corrected=create_testfolder / "filtered_contig.fasta",
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


@pytest.mark.usefixtures("create_testfolder", "processed_files")
def test_store_germline_reference_fail(create_testfolder, processed_files):
    """test_store_germline_reference_fail"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    with pytest.raises(KeyError):
        vdj.store_germline_reference()


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "database_paths", "fasta_10x"
)
def test_store_germline_reference2(
    create_testfolder, processed_files, database_paths, fasta_10x
):
    """test_store_germline_references2"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    vdj = ddl.Dandelion(f)
    vdj.store_germline_reference(germline=database_paths["germline"])
    assert len(vdj.germline) > 0
    out_file = create_testfolder / "test_airr_reannotated.ddl"
    vdj.write(out_file)
    tmp = ddl.read(out_file)
    assert len(tmp.germline) > 0
    vdj.store_germline_reference(
        germline=database_paths["germline"],
        corrected=create_testfolder / "filtered_contig.fasta",
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
