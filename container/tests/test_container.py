#!/opt/conda/envs/sc-dandelion-container/bin/python
import os
import pandas as pd
import dandelion as ddl

from unittest.mock import patch
from subprocess import run


def test_callscript():
    """Test script to run preprocessing."""
    p = run(
        ["python", "/share/dandelion_preprocess.py", "-h"],
        capture_output=True,
        encoding="utf8",
    )
    assert p.returncode == 0
    assert p.stdout != ""


def test_container():
    """Test script to run container."""
    os.system(
        "cd /tests; python /share/dandelion_preprocess.py --meta test.csv --file_prefix filtered;"
    )
    dat = pd.read_csv(
        "/tests/sample_test_10x/dandelion/filtered_contig_dandelion.tsv",
        sep="\t",
    )
    assert not dat["c_call"].empty
    assert not dat["v_call_genotyped"].empty
    assert not dat["mu_count"].empty
    assert not dat["mu_freq"].empty
    vdj = None
    try:
        vdj = ddl.Dandelion(dat)
    except:
        pass
    assert vdj is not None


def test_container_skip_tigger():
    """Test script to run container but skip tigger."""
    os.system(
        "cd /tests; python /share/dandelion_preprocess.py --meta test.csv --file_prefix filtered --skip_tigger;"
    )
    dat = pd.read_csv(
        "/tests/sample_test_10x/dandelion/filtered_contig_dandelion.tsv",
        sep="\t",
    )
    assert not dat["c_call"].empty
    assert not dat["mu_count"].empty
    assert not dat["mu_freq"].empty
    vdj = None
    try:
        vdj = ddl.Dandelion(dat)
    except:
        pass
    assert vdj is not None


@patch("matplotlib.pyplot.show")
def test_threshold(mock_show):
    """Test script to run container."""
    os.system(
        "cd /tests; python /share/changeo_clonotypes.py --h5ddl sample_test_10x/demo-vdj.h5ddl;"
    )
    dat = ddl.read_h5ddl("/tests/sample_test_10x/demo-vdj_changeo.h5ddl")
    assert not dat._data.changeo_clone_id.empty
