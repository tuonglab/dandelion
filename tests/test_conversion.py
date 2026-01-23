#!/usr/bin/env python
import pytest

import scirpy as ir

from dandelion.base.core import Dandelion
from dandelion.base.io import read_h5ddl
from dandelion.base.preprocessing import check_contigs
from dandelion.base.tools._tools import to_ak, from_ak, to_scirpy, from_scirpy


@pytest.mark.usefixtures(
    "create_testfolder",
    "airr_reannotated",
    "airr_reannotated2",
    "dummy_adata",
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """test setup"""
    awk, _ = to_ak(airr_reannotated)
    airr_reannotated = from_ak(awk)
    vdj, adata = check_contigs(airr_reannotated, dummy_adata)
    awk, _ = to_ak(airr_reannotated2)
    airr_reannotated2 = from_ak(awk)
    vdj2 = check_contigs(airr_reannotated2)
    assert airr_reannotated.shape[0] == 8
    assert airr_reannotated2.shape[0] == 15
    assert vdj._data.shape[0] == 8
    assert vdj2._data.shape[0] == 14
    assert vdj._metadata.shape[0] == 5
    assert vdj2._metadata.shape[0] == 8
    assert adata.n_obs == 5
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj.write_h5ddl(f)
    vdj2.write_h5ddl(f2)
    assert len(list(create_testfolder.iterdir())) == 2
    vdj3 = read_h5ddl(f)
    assert vdj3.metadata is not None


@pytest.mark.usefixtures(
    "airr_reannotated",
    "airr_reannotated2",
)
def test_chain_qc(
    airr_reannotated,
    airr_reannotated2,
):
    """test chain qc"""
    vdj = Dandelion(airr_reannotated)
    adata = to_scirpy(vdj, to_mudata=False)
    vdj = from_scirpy(adata)
    ir.tl.chain_qc(adata)

    vdj2 = Dandelion(airr_reannotated2)
    adata2 = to_scirpy(vdj2, to_mudata=False)
    vdj2 = from_scirpy(adata2)
    ir.tl.chain_qc(adata2)


@pytest.mark.usefixtures(
    "airr_reannotated",
    "airr_reannotated2",
)
def test_chain_qc_mudata(
    airr_reannotated,
    airr_reannotated2,
):
    """test chain qc on mudata"""
    vdj = Dandelion(airr_reannotated)
    mdata = to_scirpy(vdj, to_mudata=True)
    vdj = from_scirpy(mdata)
    ir.tl.chain_qc(mdata)
    vdj2 = Dandelion(airr_reannotated2)
    mdata2 = to_scirpy(vdj2, to_mudata=True)
    vdj2 = from_scirpy(mdata2)
    ir.tl.chain_qc(mdata2)


@pytest.mark.usefixtures(
    "airr_reannotated",
    "dummy_adata",
)
@pytest.mark.parametrize(
    "mudata_mode",
    [
        False,
        True,
    ],
)
def test_chain_qc_with_gex_adata(
    airr_reannotated,
    dummy_adata,
    mudata_mode,
):
    vdj = Dandelion(airr_reannotated)
    mdata = to_scirpy(vdj, to_mudata=mudata_mode, gex_adata=dummy_adata)
    vdj = from_scirpy(mdata)
    ir.tl.chain_qc(mdata)
