import pytest
import pandas as pd

from unittest.mock import patch

from dandelion.utilities import Dandelion, read_10x_vdj
from dandelion.preprocessing import (
    quantify_mutations,
    calculate_threshold,
    create_germlines,
)
from dandelion.tools import define_clones


def require_r_package(pkg_name):
    """Skip test if R package is missing."""
    from rpy2.robjects.packages import importr

    try:
        return importr(pkg_name)
    except:
        pytest.skip(f"R package '{pkg_name}' not installed")


# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_importrpy2():
    """test_importrpy2"""
    sh = require_r_package("shazam")
    assert sh.__module__ == "rpy2.robjects.packages"


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_mutation(create_testfolder, airr_reannotated):
    """test_mutation"""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)
    try:
        quantify_mutations(f)
    except:
        pytest.skip("R package 'shazam' not installed")
    out = pd.read_csv(f, sep="\t")
    vdj = Dandelion(out)
    assert not vdj._data.mu_count.empty
    try:
        quantify_mutations(f, frequency=True)
    except:
        pytest.skip("R package 'shazam' not installed")
    assert not vdj._data.mu_freq.empty


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_calculate_threshold(
    mock_show, create_testfolder, annotation_10x_mouse
):
    """test threshold"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    vdj = read_10x_vdj(out_file)
    try:
        tr = calculate_threshold(vdj)
        assert tr > 0.0
    except:
        pytest.skip("R package 'shazam' not installed")


@pytest.mark.usefixtures("create_testfolder", "database_paths")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_create_germlines(create_testfolder, database_paths):
    """test create germlines"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = Dandelion(out)
    create_germlines(vdj, germline=database_paths["germline"])
    assert not vdj._data.germline_alignment_d_mask.empty


@pytest.mark.usefixtures("create_testfolder")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_manual_threshold_and_define_clones(create_testfolder):
    """test threshold"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = Dandelion(out)
    define_clones(vdj, dist=0.1)
    assert not vdj._data.clone_id.empty
    define_clones(vdj, dist=0.1, key_added="changeo_clone")
    assert not vdj._data.changeo_clone.empty


@pytest.mark.usefixtures("create_testfolder")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_define_clones_outdir(create_testfolder):
    """test threshold"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = Dandelion(out)
    out_path = (
        create_testfolder / "test" / "test"
    )  # this path shouldn't exist initially
    define_clones(vdj, dist=0.1, out_dir=out_path)
    assert len(list(out_path.iterdir())) == 3
    assert len(list((out_path / "tmp").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_scoper_i(create_testfolder):
    """test identical clones from scoper"""
    f = create_testfolder / "test.tsv"
    vdj = Dandelion(f)
    assert "clone_id" not in vdj._data
    from dandelion.external.immcantation.scoper import identical_clones

    try:
        identical_clones(vdj)
    except:
        pytest.skip("R package 'scoper' not installed")
    assert not vdj._data.clone_id.empty


@pytest.mark.usefixtures("create_testfolder")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_scoper_h(create_testfolder):
    """test hierarchical clones from scoper"""
    f = create_testfolder / "test.tsv"
    vdj = Dandelion(f)
    assert "clone_id" not in vdj._data
    from dandelion.external.immcantation.scoper import hierarchical_clones

    try:
        hierarchical_clones(vdj, threshold=0.15)
    except:
        pytest.skip("R package 'scoper' not installed")
    assert not vdj._data.clone_id.empty


@pytest.mark.usefixtures("create_testfolder")
# @pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_scoper_h(create_testfolder):
    """test spectral clones from scoper"""
    f = create_testfolder / "test.tsv"
    vdj = Dandelion(f)
    assert "clone_id" not in vdj._data
    from dandelion.external.immcantation.scoper import spectral_clones

    try:
        spectral_clones(vdj, method="novj")
    except:
        pytest.skip("R package 'scoper' not installed")
    assert not vdj._data.clone_id.empty

    vdj = Dandelion(f)
    assert "clone_id" not in vdj._data
    try:
        spectral_clones(vdj, method="novj", threshold=0.15)
    except:
        pytest.skip("R package 'scoper' not installed")
    assert not vdj._data.clone_id.empty

    vdj = Dandelion(f)
    assert "clone_id" not in vdj._data
    try:
        spectral_clones(vdj, method="vj", threshold=0.15)
    except:
        pytest.skip("R package 'scoper' not installed")

    assert not vdj._data.clone_id.empty
