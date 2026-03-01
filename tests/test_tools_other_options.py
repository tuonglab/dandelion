#!/usr/bin/env python
import pytest

from pathlib import Path

from dandelion.base.preprocessing import check_contigs
from dandelion.base.tools import find_clones, generate_network


@pytest.mark.usefixtures("airr_generic")
def test_find_clones_other_options(airr_generic):
    """Test find clones."""
    vdj = check_contigs(airr_generic, productive_only=False)
    with pytest.raises(ValueError):
        vdj = find_clones(vdj, recalculate_length=False)
    vdj._data["junction_aa_length"] = 10
    with pytest.raises(ValueError):
        find_clones(vdj, recalculate_length=False)
    vdj = vdj[vdj._data.junction != ""]  # remove empty junctions
    find_clones(vdj, recalculate_length=False)
    assert not vdj._data.clone_id.empty
    assert not vdj._metadata.clone_id.empty


@pytest.mark.usefixtures("create_testfolder", "airr_generic")
def test_find_clones_file(create_testfolder, airr_generic):
    """Test find clones from file."""
    in_file = create_testfolder / "test_airr.tsv"
    out_file = create_testfolder / "test_airr_clone.tsv"
    airr_generic.to_csv(in_file, sep="\t", index=False)
    find_clones(in_file)
    assert Path(out_file) in list(create_testfolder.iterdir())


@pytest.mark.usefixtures("airr_generic")
def test_find_clones_after_network(airr_generic):
    """Test find clones."""
    vdj = check_contigs(airr_generic)
    find_clones(vdj)
    generate_network(vdj, key="junction_aa", layout_method="mod_fr")
    vdj2 = vdj.copy()
    vdj2.germline = {"dummy": "something"}
    find_clones(vdj2)
    assert not vdj2._data.clone_id.empty
    assert not vdj2._metadata.clone_id.empty
    find_clones(vdj2, key_added="cloned_idx")
    assert not vdj2._data.cloned_idx.empty
    assert not vdj2._metadata.cloned_idx.empty
