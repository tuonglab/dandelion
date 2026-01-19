import pytest

import polars as pl

from dandelion.utilities._polars import (
    DandelionPolars,
    check_contigs,
    read_zipddl,
)
from dandelion.tools._tools_polars import (
    find_clones,
    to_scirpy,
)
from dandelion.tools._network_polars import generate_network


@pytest.fixture
@pytest.mark.usefixtures("airr_reannotated")
def airr_polars(airr_reannotated):
    """Helper fixture to create DandelionPolars object from airr_reannotated fixture."""
    return DandelionPolars(airr_reannotated)


@pytest.fixture
@pytest.mark.usefixtures("airr_reannotated2")
def airr_polars2(airr_reannotated2):
    """Helper fixture to create DandelionPolars object from airr_reannotated2 fixture."""
    return DandelionPolars(airr_reannotated2)


@pytest.mark.parametrize(
    "resample,expected", [pytest.param(None, 8), pytest.param(16, 16)]
)
@pytest.mark.usefixtures("create_testfolder", "dummy_adata")
def test_setup(
    create_testfolder,
    airr_polars,
    airr_polars2,
    dummy_adata,
    resample,
    expected,
):
    """test setup"""
    vdj, adata = check_contigs(airr_polars, dummy_adata)
    vdj2 = check_contigs(airr_polars2)
    assert vdj.n_contigs == 8
    assert vdj2.n_contigs == 14
    assert vdj.n_obs == 5
    assert vdj2.n_obs == 8
    assert adata.n_obs == 5
    f = create_testfolder / "test.zipddl"
    f2 = create_testfolder / "test2.zipddl"
    vdj.write_zipddl(f)
    vdj2.write_zipddl(f2)
    assert len(list(create_testfolder.iterdir())) == 2
    vdj = read_zipddl(f)
    vdj2 = read_zipddl(f2)
    find_clones(vdj)
    find_clones(vdj2)
    assert not vdj._data.collect().get_column("clone_id").is_empty()
    assert not vdj._metadata.collect().get_column("clone_id").is_empty()
    assert not vdj2._data.collect().get_column("clone_id").is_empty()
    assert not vdj2._metadata.collect().get_column("clone_id").is_empty()
    assert (
        vdj._metadata.collect().get_column("clone_id").drop_nulls().n_unique()
        == 5
    )
    assert (
        vdj2._metadata.collect().get_column("clone_id").drop_nulls().n_unique()
        == 5
    )
    vdj.write_zipddl(f)
    vdj2.write_zipddl(f2)
    vdj = read_zipddl(f)
    vdj2 = read_zipddl(f2)
    # create anndata from here
    adata = to_scirpy(vdj, to_mudata=False)
    if resample is not None:
        vdj, adata = generate_network(
            vdj, adata=adata, sample=resample, layout_method="mod_fr"
        )
        assert vdj.n_obs == expected
        assert vdj.layout is not None
        assert vdj.graph is not None
    else:
        generate_network(vdj2, layout_method="mod_fr")
        assert vdj2.n_obs == expected
        assert vdj2.layout is not None
        assert vdj2.graph is not None
    _data = vdj._data.with_columns(pl.lit("1").alias("clone_id"))
    _data = _data.collect() if isinstance(_data, pl.LazyFrame) else _data
    vdj = DandelionPolars(_data)
    assert vdj._data.collect_schema()["clone_id"] == pl.String
    generate_network(vdj, layout_method="mod_fr")
    assert vdj.layout is not None
