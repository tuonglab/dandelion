#!/usr/bin/env python
import pytest

from dandelion.base.core import Dandelion
from dandelion.base.preprocessing import check_contigs
from dandelion.base.tools import find_clones, generate_network


@pytest.mark.usefixtures("airr_generic")
def test_generic_check(airr_generic):
    """test data loading and filtering"""
    tmp = Dandelion(airr_generic)
    assert tmp._metadata.shape[0] == 43
    assert tmp._data.shape[0] == airr_generic.shape[0]

    tmp2 = check_contigs(tmp, productive_only=False)
    assert tmp2._metadata.shape[0] == 43
    assert (
        tmp2._data.shape[0] != tmp._data.shape[0]
    )  # this is because filter_extra is True by default

    tmp2 = check_contigs(tmp, productive_only=False, library_type="ig")
    assert tmp2._metadata.shape[0] == 25
    assert tmp2._data.shape[0] != tmp._data.shape[0]
    assert tmp2._data.shape[0] == 55  # 65

    tmp2 = check_contigs(tmp)
    assert tmp2._metadata.shape[0] == 43
    assert tmp2._data.shape[0] != tmp._data.shape[0]

    find_clones(tmp2, identity={"tr-ab": 1})
    assert "clone_id" in tmp2._data
    assert "clone_id" in tmp2._metadata
    assert not tmp2._metadata.clone_id.empty

    generate_network(tmp2, key="junction_aa", compute_layout=False)
    assert tmp2.graph is not None


@pytest.mark.usefixtures("airr_generic")
def test_check_keep_extra(airr_generic):
    """test keep extra option"""
    tmp = Dandelion(airr_generic)
    assert tmp._metadata.shape[0] == 43
    assert tmp._data.shape[0] == airr_generic.shape[0]
    tmp2 = check_contigs(tmp)
    tmp3 = check_contigs(tmp, filter_extra=False)
    assert tmp3._metadata.shape[0] == 43
    assert tmp2._data.shape[0] != tmp._data.shape[0]
    assert tmp3._data.shape[0] == tmp._data.shape[0]
    assert tmp3._data.extra.value_counts()["T"] == 19  # 14


@pytest.mark.usefixtures("airr_generic")
def test_check_remove_ambiguous(airr_generic):
    """test remove ambiguous option"""
    tmp = Dandelion(airr_generic)
    assert tmp._metadata.shape[0] == 43
    assert tmp._data.shape[0] == airr_generic.shape[0]
    tmp2 = check_contigs(tmp)
    tmp3 = check_contigs(tmp, filter_ambiguous=True)
    assert tmp3._metadata.shape[0] == 43
    assert tmp2._data.shape[0] != tmp._data.shape[0]
    assert tmp3._data.shape[0] != tmp2._data.shape[0]
    assert tmp2._data.ambiguous.value_counts()["T"] == 37
    assert tmp2._data.ambiguous.value_counts()["F"] == 74  # 79
    assert all(tmp3._data.ambiguous == "F")
