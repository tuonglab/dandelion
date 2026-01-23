#!/usr/bin/env python
import pytest

from dandelion.base.core import Dandelion
from dandelion.base.tools import find_clones, generate_network
from dandelion.base.preprocessing import check_contigs


@pytest.mark.usefixtures("airr_reannotated")
def test_load_data(airr_reannotated):
    """test load_data"""
    vdj = Dandelion(airr_reannotated)
    assert all(
        [x != y for x, y in zip(vdj._data["cell_id"], vdj._data["sequence_id"])]
    )
    cell_ids = list(vdj._data["cell_id"])
    tmp = vdj._data.drop("cell_id", axis=1)
    vdj = Dandelion(tmp)
    assert all([x == y for x, y in zip(vdj._data["cell_id"], cell_ids)])


@pytest.mark.usefixtures("airr_generic")
def test_slice_data(airr_generic):
    """test load_data"""
    vdj = Dandelion(airr_generic)
    assert vdj._data.shape[0] == 130
    assert vdj._metadata.shape[0] == 43
    vdj2 = vdj[vdj.data["productive"] == "T"]
    assert vdj2._data.shape[0] == 119
    assert vdj2._metadata.shape[0] == 43
    vdj2 = vdj[vdj.metadata["productive_VDJ"] == "T"]
    assert vdj2._data.shape[0] == 49
    assert vdj2._metadata.shape[0] == 23
    vdj2 = vdj[
        vdj.metadata_names.isin(
            [
                "IGHA+IGHM+IGHD+IGLv2",
                "IGHA+IGHM+IGHD+IGLv3",
                "IGHM+IGHD+IGL+IGHA",
                "IGHM+IGHD+IGL+IGHAnp",
                "IGHM+IGHD+IGL+IGHM",
            ]
        )
    ]
    assert vdj2._data.shape[0] == 20
    assert vdj2._metadata.shape[0] == 5
    vdj2 = vdj[
        vdj.data_names.isin(
            [
                "IGHM+IGHD+IGL+IGHAnp_contig_1",
                "IGHM+IGHD+IGL+IGHAnp_contig_2",
                "IGHM+IGHD+IGL+IGHAnp_contig_4",
                "IGHM+IGHD+IGL+IGHAnp_contig_3",
                "IGHM+IGHD+IGL+IGHM_contig_1",
                "IGHM+IGHD+IGL+IGHM_contig_2",
                "IGHM+IGHD+IGL+IGHM_contig_4",
                "IGHM+IGHD+IGL+IGHM_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_1",
                "IGHM+IGHD+IGL+IGK_contig_2",
                "IGHM+IGHD+IGL+IGK_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_4",
                "IGHM+IGHM+IGL_contig_3",
                "IGHM+IGHM+IGL_contig_2",
                "IGHM+IGHM+IGL_contig_1",
                "IGHM+TRA_contig_1",
                "IGHM+TRA_contig_2",
                "IGHM+TRG_contig_2",
                "IGHM+TRG_contig_1",
                "IGK+IGL_contig_1",
                "IGK+IGL_contig_2",
                "TRA+TRG_contig_2",
                "TRA+TRG_contig_1",
                "TRB+IGL_contig_1",
                "TRB+IGL_contig_2",
                "TRB+TRG_contig_1",
                "TRB+TRG_contig_2",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_1",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_2",
                "TRBV+TRAJ+TRBC__TRAV+TRAJ_contig_1",
            ]
        )
    ]
    assert vdj2._data.shape[0] == 30
    assert vdj2._metadata.shape[0] == 12


@pytest.mark.usefixtures("airr_generic")
def test_names(airr_generic):
    """test load_data"""
    vdj = Dandelion(airr_generic)
    assert all(i == j for i, j in zip(vdj.data_names, vdj._data.index))
    assert all(i == j for i, j in zip(vdj.metadata_names, vdj._metadata.index))


@pytest.mark.usefixtures("airr_generic")
def test_slice_data_with_graph(airr_generic):
    """Test slicing data with graph"""
    vdj = Dandelion(airr_generic)
    vdj = check_contigs(vdj, productive_only=False)
    find_clones(vdj)
    generate_network(vdj, key="junction", layout_method="mod_fr")
    vdj2 = vdj[vdj.data["productive"] == "T"]
    assert vdj2._data.shape[0] == 111  # 116
    assert vdj2._metadata.shape[0] == 43
    vdj2 = vdj[vdj.metadata["productive_VDJ"] == "T"]
    assert vdj2._data.shape[0] == 69  # 50
    assert vdj2._metadata.shape[0] == 30  # 22
    vdj2 = vdj[
        vdj.metadata_names.isin(
            [
                "IGHA+IGHM+IGHD+IGLv2",
                "IGHA+IGHM+IGHD+IGLv3",
                "IGHM+IGHD+IGL+IGHA",
                "IGHM+IGHD+IGL+IGHAnp",
                "IGHM+IGHD+IGL+IGHM",
            ]
        )
    ]
    assert vdj2._data.shape[0] == 16  # 19
    assert vdj2._metadata.shape[0] == 5
    assert len(vdj2.layout[0]) == 5
    assert len(vdj2.layout[1]) == 5
    assert len(vdj2.graph[0]) == 5
    assert len(vdj2.graph[1]) == 5
    vdj2 = vdj[
        vdj.data_names.isin(
            [
                "IGHM+IGHD+IGL+IGHAnp_contig_1",
                "IGHM+IGHD+IGL+IGHAnp_contig_2",
                "IGHM+IGHD+IGL+IGHAnp_contig_4",
                "IGHM+IGHD+IGL+IGHAnp_contig_3",
                "IGHM+IGHD+IGL+IGHM_contig_1",
                "IGHM+IGHD+IGL+IGHM_contig_2",
                "IGHM+IGHD+IGL+IGHM_contig_4",
                "IGHM+IGHD+IGL+IGHM_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_1",
                "IGHM+IGHD+IGL+IGK_contig_2",
                "IGHM+IGHD+IGL+IGK_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_4",
                "IGHM+IGHM+IGL_contig_3",
                "IGHM+IGHM+IGL_contig_2",
                "IGHM+IGHM+IGL_contig_1",
                "IGHM+TRA_contig_1",
                "IGHM+TRA_contig_2",
                "IGHM+TRG_contig_2",
                "IGHM+TRG_contig_1",
                "IGK+IGL_contig_1",
                "IGK+IGL_contig_2",
                "TRA+TRG_contig_2",
                "TRA+TRG_contig_1",
                "TRB+IGL_contig_1",
                "TRB+IGL_contig_2",
                "TRB+TRG_contig_1",
                "TRB+TRG_contig_2",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_1",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_2",
                "TRBV+TRAJ+TRBC__TRAV+TRAJ_contig_1",
            ]
        )
    ]
    assert vdj2._data.shape[0] == 28  # 30
    assert vdj2._metadata.shape[0] == 12
    assert len(vdj2.layout[0]) == 12
    # assert len(vdj2.layout[1]) == 4
    assert len(vdj2.layout[1]) == 11
    assert len(vdj2.graph[0]) == 12
    # assert len(vdj2.graph[1]) == 4
    assert len(vdj2.graph[1]) == 11


@pytest.mark.usefixtures("airr_generic")
def test_isotype(airr_generic):
    """test load_data"""
    vdj = Dandelion(airr_generic, custom_isotype_dict={"IGHC": "IGC"})


@pytest.mark.usefixtures("airr_generic")
def test_change_ids(airr_generic):
    """test load_data"""
    vdj = Dandelion(airr_generic)
    vdj.add_sequence_prefix("test")
    vdj = Dandelion(airr_generic)
    vdj.add_sequence_suffix("test")
    vdj = Dandelion(airr_generic)
    vdj.add_cell_prefix("test")
    vdj = Dandelion(airr_generic)
    vdj.add_cell_suffix("test")
    vdj = Dandelion(airr_generic)
