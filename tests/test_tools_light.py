#!/usr/bin/env python
import pytest

from dandelion.base.preprocessing import check_contigs
from dandelion.base.io import read_h5ddl
from dandelion.base.tools import find_clones


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated", "dummy_adata")
def test_setup(create_testfolder, airr_reannotated, dummy_adata):
    """test setup"""
    vdj, adata = check_contigs(airr_reannotated, dummy_adata)
    assert airr_reannotated.shape[0] == 8
    assert vdj._data.shape[0] == 8
    assert vdj._metadata.shape[0] == 5
    assert adata.n_obs == 5
    f = create_testfolder / "test.h5ddl"
    vdj.write_h5ddl(f)
    assert len(list(create_testfolder.iterdir())) == 1
    vdj2 = read_h5ddl(f)
    assert vdj2.metadata is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    """test find clones"""
    f = create_testfolder / "test.h5ddl"
    vdj = read_h5ddl(f)
    find_clones(vdj)
    assert not vdj._data.clone_id.empty
    assert not vdj._metadata.clone_id.empty
    vdj.write_h5ddl(f)
    with pytest.raises(KeyError):
        find_clones(vdj, key="random_column")


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clonesfromfile(airr_reannotated):
    """test find clones"""
    vdj = find_clones(airr_reannotated)
    assert not vdj._data.clone_id.empty
    assert not vdj._metadata.clone_id.empty
    vdj2 = find_clones(airr_reannotated, by_alleles=True)
    assert not vdj2._data.clone_id.empty
    assert not vdj2._metadata.clone_id.empty
