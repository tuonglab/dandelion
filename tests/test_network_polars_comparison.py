"""
Test to compare pandas and Polars implementations of generate_network.
Ensures both implementations produce identical results.
"""

import pytest
import numpy as np
import dandelion as ddl
from dandelion.utilities._polars import DandelionPolars
from dandelion.tools._network_polars import generate_network


@pytest.mark.usefixtures("vdj_smaller")
def test_generate_network_polars_vs_pandas_clone(vdj_smaller):
    """
    Compare generate_network results between pandas and Polars implementations.
    Test with distance_mode='clone' and compute_graph=True.
    """
    # Create two identical copies for comparison
    vdj_pandas = vdj_smaller.copy()
    vdj_polars = DandelionPolars(vdj_smaller.data)

    # Run generate_network on pandas version
    ddl.tl.generate_network(
        vdj_pandas,
        key="junction",
        distance_mode="clone",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=False,
        verbose=False,
    )

    # Run generate_network on Polars version
    generate_network(
        vdj_polars,
        key="junction",
        distance_mode="clone",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=False,
        verbose=False,
    )

    # Compare distance matrices
    distances_pandas = vdj_pandas.distances.toarray()
    distances_polars = vdj_polars.distances.toarray()
    print(distances_pandas), print(distances_polars)
    assert np.array_equal(
        distances_pandas, distances_polars, equal_nan=True
    ), "Distance matrices differ between pandas and Polars implementations"
    assert (
        np.nan_to_num(distances_pandas).sum()
        == np.nan_to_num(distances_polars).sum()
    ), "Distance matrix sums differ"

    # Compare graph structure (both full and MST graphs in tuple)
    assert vdj_pandas.graph is not None, "Pandas: graph not computed"
    assert vdj_polars.graph is not None, "Polars: graph not computed"

    # Handle graph as tuple (full_graph, mst_graph)
    pandas_graphs = (
        vdj_pandas.graph
        if isinstance(vdj_pandas.graph, tuple)
        else (vdj_pandas.graph,)
    )
    polars_graphs = (
        vdj_polars.graph
        if isinstance(vdj_polars.graph, tuple)
        else (vdj_polars.graph,)
    )

    assert len(pandas_graphs) == len(polars_graphs), "Number of graphs differ"

    # Compare each graph in the tuple
    for g_idx, (pandas_g, polars_g) in enumerate(
        zip(pandas_graphs, polars_graphs)
    ):
        # Compare number of nodes and edges
        assert len(pandas_g.nodes()) == len(
            polars_g.nodes()
        ), f"Graph {g_idx}: Number of nodes differ"
        assert len(pandas_g.edges()) == len(
            polars_g.edges()
        ), f"Graph {g_idx}: Number of edges differ"

        # Compare edge weights
        pandas_edge_dict = {
            (u, v): w for u, v, w in pandas_g.edges(data="weight")
        }
        polars_edge_dict = {
            (u, v): w for u, v, w in polars_g.edges(data="weight")
        }

        assert set(pandas_edge_dict.keys()) == set(
            polars_edge_dict.keys()
        ), f"Graph {g_idx}: Edge sets differ between implementations"

        for edge in pandas_edge_dict.keys():
            assert (
                pandas_edge_dict[edge] == polars_edge_dict[edge]
            ), f"Graph {g_idx}: Weight for edge {edge} differs"

    for edge in pandas_edge_dict.keys():
        assert (
            pandas_edge_dict[edge] == polars_edge_dict[edge]
        ), f"Weight for edge {edge} differs"


@pytest.mark.usefixtures("vdj_smaller")
def test_generate_network_polars_vs_pandas_full(vdj_smaller):
    """
    Compare generate_network results between pandas and Polars implementations.
    Test with distance_mode='full' and compute_graph=True.
    """
    # Create two identical copies for comparison
    vdj_pandas = vdj_smaller.copy()
    vdj_polars = DandelionPolars(vdj_smaller.data)

    # Run generate_network on pandas version
    ddl.tl.generate_network(
        vdj_pandas,
        key="junction",
        distance_mode="full",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=False,
        verbose=False,
    )

    # Run generate_network on Polars version
    generate_network(
        vdj_polars,
        key="junction",
        distance_mode="full",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=False,
        verbose=False,
    )

    # Compare distance matrices
    distances_pandas = vdj_pandas.distances.toarray()
    distances_polars = vdj_polars.distances.toarray()

    assert np.array_equal(
        distances_pandas, distances_polars, equal_nan=True
    ), "Distance matrices differ between pandas and Polars implementations"
    assert (
        np.nan_to_num(distances_pandas).sum()
        == np.nan_to_num(distances_polars).sum()
    ), "Distance matrix sums differ"

    # Compare graph structure (both full and MST graphs in tuple)
    assert vdj_pandas.graph is not None, "Pandas: graph not computed"
    assert vdj_polars.graph is not None, "Polars: graph not computed"

    # Handle graph as tuple (full_graph, mst_graph)
    pandas_graphs = (
        vdj_pandas.graph
        if isinstance(vdj_pandas.graph, tuple)
        else (vdj_pandas.graph,)
    )
    polars_graphs = (
        vdj_polars.graph
        if isinstance(vdj_polars.graph, tuple)
        else (vdj_polars.graph,)
    )

    assert len(pandas_graphs) == len(polars_graphs), "Number of graphs differ"

    # Compare each graph in the tuple
    for g_idx, (pandas_g, polars_g) in enumerate(
        zip(pandas_graphs, polars_graphs)
    ):
        assert len(pandas_g.nodes()) == len(
            polars_g.nodes()
        ), f"Graph {g_idx}: Number of nodes differ"
        assert len(pandas_g.edges()) == len(
            polars_g.edges()
        ), f"Graph {g_idx}: Number of edges differ"


@pytest.mark.usefixtures("vdj_smaller", "create_testfolder")
def test_generate_network_polars_lazy_vs_eager(create_testfolder, vdj_smaller):
    """
    Compare lazy and eager modes in Polars implementation.
    Ensures both modes produce identical results.
    """
    # Create two identical copies for comparison
    vdj_lazy = DandelionPolars(vdj_smaller.data)
    vdj_eager = DandelionPolars(vdj_smaller.data, lazy=False)

    # Run lazy mode
    generate_network(
        vdj_eager,
        key="junction",
        distance_mode="clone",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=False,
        verbose=False,
    )

    # Run lazy mode
    generate_network(
        vdj_lazy,
        key="junction",
        distance_mode="clone",
        compute_graph=True,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=False,
        lazy=True,
        zarr_path=str(create_testfolder / "test_lazy.zarr"),
        verbose=False,
    )

    # Compute lazy distances if needed
    if hasattr(vdj_lazy.distances, "compute"):
        lazy_distances = vdj_lazy.distances.compute()
    else:
        lazy_distances = vdj_lazy.distances.toarray()

    eager_distances = vdj_eager.distances.toarray()
    # Compare results
    assert np.array_equal(
        eager_distances, lazy_distances, equal_nan=True
    ), "Distance matrices differ between eager and lazy modes"
    assert (
        np.nan_to_num(eager_distances).sum()
        == np.nan_to_num(lazy_distances).sum()
    ), "Distance matrix sums differ between modes"


@pytest.mark.usefixtures("vdj_smaller")
@pytest.mark.parametrize("pad_to_max", [False, True])
def test_generate_network_polars_vs_pandas_padded(vdj_smaller, pad_to_max):
    """
    Compare generate_network with and without padding between implementations.
    """
    # Create two identical copies for comparison
    vdj_pandas = vdj_smaller.copy()
    vdj_polars = DandelionPolars(vdj_smaller.data)

    # Run on pandas version
    ddl.tl.generate_network(
        vdj_pandas,
        key="junction",
        distance_mode="full",
        compute_graph=False,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=pad_to_max,
        lazy=False,
        verbose=False,
    )

    # Run on Polars version
    generate_network(
        vdj_polars,
        key="junction",
        distance_mode="full",
        compute_graph=False,
        use_existing_graph=False,
        sequential_chain=False,
        n_cpus=1,
        pad_to_max=pad_to_max,
        lazy=False,
        verbose=False,
    )

    # Compare distance matrices
    distances_pandas = vdj_pandas.distances.toarray()
    distances_polars = vdj_polars.distances.toarray()

    assert np.array_equal(
        distances_pandas, distances_polars, equal_nan=True
    ), f"Distance matrices differ with pad_to_max={pad_to_max}"
    assert (
        np.nan_to_num(distances_pandas).sum()
        == np.nan_to_num(distances_polars).sum()
    ), f"Distance matrix sums differ with pad_to_max={pad_to_max}"
