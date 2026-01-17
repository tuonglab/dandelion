import pytest

import dandelion as ddl
import numpy as np


@pytest.mark.usefixtures(
    "vdj_small",
    "create_testfolder",
)
@pytest.mark.parametrize(
    "sequential_chain,n_cpus,lazy,pad_to_max,expected",
    [
        pytest.param(True, 1, False, False, 34),  # original
        pytest.param(True, 2, False, False, 34),  # original parallel
        pytest.param(False, 1, False, False, 34),  # long
        pytest.param(False, 2, False, False, 34),  # long parallel
        pytest.param(False, 1, True, False, 34),  # lazy
        pytest.param(False, 2, True, False, 34),  # lazy parallel
        pytest.param(True, 1, False, True, 34),  # original
        pytest.param(True, 2, False, True, 34),  # original parallel
        pytest.param(False, 1, False, True, 34),  # long
        pytest.param(False, 2, False, True, 34),  # long parallel
        pytest.param(False, 1, True, True, 34),  # lazy
        pytest.param(False, 2, True, True, 34),  # lazy parallel
    ],
)
def test_generate_network_clone(
    create_testfolder,
    vdj_small,
    sequential_chain,
    n_cpus,
    lazy,
    pad_to_max,
    expected,
):
    """clonal membership"""
    ddl.tl.generate_network(
        vdj_small,
        key="junction",
        distance_mode="clone",
        compute_graph=False,
        use_existing_graph=False,
        sequential_chain=sequential_chain,
        n_cpus=n_cpus,
        pad_to_max=pad_to_max,
        lazy=lazy,
        zarr_path=create_testfolder / "test.zarr",
    )
    if lazy:
        assert np.nan_to_num(vdj_small.distances.compute()).sum() == expected
        vdj_small.compute()
        assert np.nan_to_num(vdj_small.distances.toarray()).sum() == expected
    else:
        assert np.nan_to_num(vdj_small.distances.toarray()).sum() == expected


@pytest.mark.usefixtures(
    "vdj_smaller",
    "create_testfolder",
)
@pytest.mark.parametrize(
    "sequential_chain,n_cpus,lazy,pad_to_max,expected",
    [
        pytest.param(True, 1, False, False, 10398758),  # original
        pytest.param(True, 2, False, False, 10398758),  # original parallel
        pytest.param(False, 1, False, False, 10398758),  # long
        pytest.param(False, 2, False, False, 10398758),  # long parallel
        pytest.param(False, 1, True, False, 10398758),  # lazy
        pytest.param(False, 2, True, False, 10398758),  # lazy parallel
        pytest.param(True, 1, False, True, 12328492),  # original
        pytest.param(True, 2, False, True, 12328492),  # original parallel
        pytest.param(False, 1, False, True, 12186804),  # long
        pytest.param(False, 2, False, True, 12186804),  # long parallel
        pytest.param(False, 1, True, True, 12186804),  # lazy
        pytest.param(False, 2, True, True, 12186804),  # lazy parallel
    ],
)
def test_generate_network_full(
    create_testfolder,
    vdj_smaller,
    sequential_chain,
    n_cpus,
    lazy,
    pad_to_max,
    expected,
):
    """clonal membership"""
    ddl.tl.generate_network(
        vdj_smaller,
        key="junction",
        distance_mode="full",
        compute_graph=False,
        use_existing_graph=False,
        sequential_chain=sequential_chain,
        n_cpus=n_cpus,
        pad_to_max=pad_to_max,
        lazy=lazy,
        zarr_path=create_testfolder / "test.zarr",
    )
    if lazy:
        assert np.nan_to_num(vdj_smaller.distances.compute()).sum() == expected
        vdj_smaller.compute()
        assert np.nan_to_num(vdj_smaller.distances.toarray()).sum() == expected
    else:
        assert np.nan_to_num(vdj_smaller.distances.toarray()).sum() == expected
