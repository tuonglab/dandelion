"""Tests for dandelion.utilities._layout module (CPU paths only)."""

import numpy as np
import pandas as pd
import pytest

import networkx as nx

from dandelion.utilities._layout import (
    _fruchterman_reingold,
    _fruchterman_reingold_barnes_hut_numba,
    _fruchterman_reingold_layout,
    _fruchterman_reingold_layout_bh,
    _fruchterman_reingold_layout_v2,
    _fruchterman_reingold_numba,
    _get_numba_bh_kernels,
    _get_numba_fr_kernel,
    _process_params,
    _rescale_layout,
    _sparse_fruchterman_reingold,
    generate_layout,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _small_graph():
    """Create a 4-node graph with a chain of edges."""
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_weighted_edges_from(
        [("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0)]
    )
    return G


def _small_edges():
    """Create a small edge DataFrame for A-B-C-D chain."""
    return pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "weight": [1.0, 1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# _process_params
# ---------------------------------------------------------------------------
class TestProcessParams:
    """Test _process_params validation."""

    def test_graph_passthrough(self):
        """Test that a Graph is passed through unchanged."""
        G = _small_graph()
        G_out, center = _process_params(G, None, 2)
        assert isinstance(G_out, nx.Graph)
        np.testing.assert_array_equal(center, [0, 0])

    def test_non_graph_converted(self):
        """Test that a node list is converted to a Graph."""
        G_out, center = _process_params(["A", "B"], None, 2)
        assert isinstance(G_out, nx.Graph)
        assert set(G_out.nodes()) == {"A", "B"}

    def test_custom_center(self):
        """Test that a custom center is accepted."""
        G = _small_graph()
        _, center = _process_params(G, [1.0, 2.0], 2)
        np.testing.assert_array_equal(center, [1.0, 2.0])

    def test_center_dim_mismatch_raises(self):
        """Test that mismatched center dimension raises ValueError."""
        with pytest.raises(ValueError, match="length of center"):
            _process_params(_small_graph(), [1.0], 2)


# ---------------------------------------------------------------------------
# _rescale_layout
# ---------------------------------------------------------------------------
class TestRescaleLayout:
    """Test _rescale_layout."""

    def test_rescale_basic(self):
        """Test that positions are rescaled to [-scale, scale]."""
        pos = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
        result = _rescale_layout(pos, scale=1.0)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_rescale_constant(self):
        """Test rescaling when all positions are identical."""
        pos = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = _rescale_layout(pos, scale=1.0)
        np.testing.assert_array_equal(result, [[0.0, 0.0], [0.0, 0.0]])


# ---------------------------------------------------------------------------
# _fruchterman_reingold (dense core)
# ---------------------------------------------------------------------------
class TestFruchtermanReingoldDense:
    """Test dense Fruchterman-Reingold algorithm."""

    def test_basic(self):
        """Test basic layout computation."""
        A = nx.to_numpy_array(_small_graph())
        pos = _fruchterman_reingold(A, iterations=10, seed=42)
        assert pos.shape == (4, 2)

    def test_with_initial_pos(self):
        """Test layout with initial positions."""
        A = nx.to_numpy_array(_small_graph())
        init = np.random.RandomState(0).rand(4, 2)
        pos = _fruchterman_reingold(A, pos=init, iterations=10, seed=42)
        assert pos.shape == (4, 2)

    def test_with_fixed(self):
        """Test that fixed nodes remain in place."""
        A = nx.to_numpy_array(_small_graph())
        init = np.random.RandomState(0).rand(4, 2)
        fixed_pos = init[0].copy()
        pos = _fruchterman_reingold(
            A, pos=init, fixed=[0], iterations=10, seed=42
        )
        np.testing.assert_array_equal(pos[0], fixed_pos)


# ---------------------------------------------------------------------------
# _sparse_fruchterman_reingold
# ---------------------------------------------------------------------------
class TestFruchtermanReingoldSparse:
    """Test sparse Fruchterman-Reingold algorithm."""

    def test_basic(self):
        """Test basic sparse layout computation."""
        from scipy.sparse import csr_matrix

        A = csr_matrix(nx.to_numpy_array(_small_graph()))
        pos = _sparse_fruchterman_reingold(A, iterations=10, seed=42)
        assert pos.shape == (4, 2)


# ---------------------------------------------------------------------------
# _fruchterman_reingold_layout (v1 wrapper)
# ---------------------------------------------------------------------------
class TestFRLayout:
    """Test _fruchterman_reingold_layout wrapper."""

    def test_basic(self):
        """Test basic layout returns positions for all nodes."""
        G = _small_graph()
        pos = _fruchterman_reingold_layout(G, iterations=10, seed=42)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(G.nodes())

    def test_empty_graph(self):
        """Test empty graph returns empty dict."""
        pos = _fruchterman_reingold_layout(nx.Graph(), seed=42)
        assert pos == {}

    def test_single_node(self):
        """Test single-node graph returns that node."""
        G = nx.Graph()
        G.add_node("X")
        pos = _fruchterman_reingold_layout(G, seed=42)
        assert "X" in pos

    def test_fixed_without_pos_raises(self):
        """Test that fixed nodes without positions raises ValueError."""
        G = _small_graph()
        with pytest.raises(ValueError, match="fixed without positions"):
            _fruchterman_reingold_layout(G, fixed=["A"], seed=42)

    def test_with_initial_pos(self):
        """Test layout with user-supplied initial positions."""
        G = _small_graph()
        init_pos = {
            n: (float(i), float(i + 1)) for i, n in enumerate(G.nodes())
        }
        pos = _fruchterman_reingold_layout(
            G, pos=init_pos, iterations=10, seed=np.random.RandomState(42)
        )
        assert isinstance(pos, dict)

    def test_large_graph_uses_sparse(self):
        """Test that graphs >500 nodes trigger the sparse path."""
        G = nx.watts_strogatz_graph(600, 4, 0.3, seed=42)
        pos = _fruchterman_reingold_layout(G, iterations=5, seed=42)
        assert len(pos) == 600


# ---------------------------------------------------------------------------
# _get_numba_fr_kernel / _fruchterman_reingold_numba
# ---------------------------------------------------------------------------
class TestNumbaFR:
    """Test Numba-accelerated Fruchterman-Reingold."""

    def test_kernel_cached(self):
        """Test that the JIT kernel is compiled once and cached."""
        k1 = _get_numba_fr_kernel()
        k2 = _get_numba_fr_kernel()
        assert k1 is k2

    def test_numba_fr_basic(self):
        """Test basic Numba FR layout computation."""
        A = nx.to_numpy_array(_small_graph())
        pos = _fruchterman_reingold_numba(A, iterations=10, seed=42)
        assert pos.shape == (4, 2)

    def test_numba_fr_with_fixed(self):
        """Test that fixed nodes remain in place with Numba FR."""
        A = nx.to_numpy_array(_small_graph())
        init = np.random.RandomState(0).rand(4, 2).astype(np.float32)
        fixed_pos = init[0].copy()
        pos = _fruchterman_reingold_numba(
            A, pos=init, fixed=[0], iterations=10, seed=42
        )
        np.testing.assert_array_almost_equal(pos[0], fixed_pos)


# ---------------------------------------------------------------------------
# _fruchterman_reingold_layout_v2 (numba wrapper)
# ---------------------------------------------------------------------------
class TestFRLayoutV2:
    """Test _fruchterman_reingold_layout_v2 Numba wrapper."""

    def test_basic(self):
        """Test basic v2 layout returns positions for all nodes."""
        G = _small_graph()
        pos = _fruchterman_reingold_layout_v2(G, iterations=10, seed=42)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(G.nodes())

    def test_empty_graph(self):
        """Test empty graph returns empty dict."""
        pos = _fruchterman_reingold_layout_v2(nx.Graph(), seed=42)
        assert pos == {}

    def test_single_node(self):
        """Test single-node graph returns that node."""
        G = nx.Graph()
        G.add_node("X")
        pos = _fruchterman_reingold_layout_v2(G, seed=42)
        assert "X" in pos

    def test_fixed_without_pos_raises(self):
        """Test that fixed nodes without positions raises ValueError."""
        G = _small_graph()
        with pytest.raises(ValueError, match="fixed without positions"):
            _fruchterman_reingold_layout_v2(G, fixed=["A"], seed=42)


# ---------------------------------------------------------------------------
# Barnes-Hut CPU (_get_numba_bh_kernels / _fruchterman_reingold_barnes_hut_numba)
# ---------------------------------------------------------------------------
class TestBarnesHutNumba:
    """Test Barnes-Hut CPU layout with Numba."""

    def test_bh_kernels_cached(self):
        """Test that BH kernels are compiled once and cached."""
        k1 = _get_numba_bh_kernels()
        k2 = _get_numba_bh_kernels()
        assert k1 is k2
        assert len(k1) == 4

    def test_bh_basic(self):
        """Test basic Barnes-Hut layout computation."""
        A = nx.to_numpy_array(_small_graph())
        pos = _fruchterman_reingold_barnes_hut_numba(A, iterations=10, seed=42)
        assert pos.shape == (4, 2)

    def test_bh_dim_not_2_raises(self):
        """Test that dim != 2 raises ValueError."""
        A = nx.to_numpy_array(_small_graph())
        with pytest.raises(ValueError, match="only supports 2D"):
            _fruchterman_reingold_barnes_hut_numba(A, dim=3, seed=42)


# ---------------------------------------------------------------------------
# _fruchterman_reingold_layout_bh (BH wrapper)
# ---------------------------------------------------------------------------
class TestFRLayoutBH:
    """Test Barnes-Hut layout wrapper."""

    def test_basic(self):
        """Test basic BH layout returns positions for all nodes."""
        G = _small_graph()
        pos = _fruchterman_reingold_layout_bh(G, iterations=10, seed=42)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(G.nodes())

    def test_empty_graph(self):
        """Test empty graph returns empty dict."""
        pos = _fruchterman_reingold_layout_bh(nx.Graph(), seed=42)
        assert pos == {}

    def test_single_node(self):
        """Test single-node graph returns that node."""
        G = nx.Graph()
        G.add_node("X")
        pos = _fruchterman_reingold_layout_bh(G, seed=42)
        assert "X" in pos

    def test_fixed_without_pos_raises(self):
        """Test that fixed nodes without positions raises ValueError."""
        G = _small_graph()
        with pytest.raises(ValueError, match="fixed without positions"):
            _fruchterman_reingold_layout_bh(G, fixed=["A"], seed=42)

    def test_singleton_mass(self):
        """Test layout with isolated node and custom singleton mass."""
        G = _small_graph()
        G.add_node("E")  # isolated node
        pos = _fruchterman_reingold_layout_bh(
            G, iterations=10, seed=42, singleton_mass=0.1
        )
        assert "E" in pos


# ---------------------------------------------------------------------------
# generate_layout
# ---------------------------------------------------------------------------
class TestGenerateLayout:
    """Test generate_layout entry point."""

    def test_mod_fr(self):
        """Test mod_fr layout method."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="mod_fr",
            iterations=10,
            seed=42,
            verbose=False,
        )
        assert isinstance(G, nx.Graph)
        assert isinstance(pos, dict)
        assert isinstance(pos_, dict)

    def test_mod_fr2(self):
        """Test mod_fr2 layout method."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="mod_fr2",
            iterations=10,
            seed=42,
            verbose=False,
        )
        assert isinstance(pos, dict)

    def test_fa2(self):
        """Test fa2 (ForceAtlas2) layout method."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="fa2",
            verbose=False,
        )
        assert isinstance(pos, dict)
        assert isinstance(pos_, dict)

    def test_fa2_expanded_only(self):
        """Test fa2 layout with expanded_only."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="fa2",
            expanded_only=True,
            verbose=False,
        )
        # When expanded_only, pos is set to pos_ and G to G_
        assert pos is pos_

    def test_mod_fr_bh(self):
        """Test mod_fr_bh (Barnes-Hut CPU) layout method."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="mod_fr_bh",
            iterations=10,
            seed=42,
            verbose=False,
        )
        assert isinstance(pos, dict)

    def test_no_compute_layout(self):
        """Test that compute_layout=False returns None positions."""
        verts = ["A", "B", "C"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts, edges=edges, compute_layout=False
        )
        assert pos is None
        assert pos_ is None

    def test_expanded_only(self):
        """Test expanded_only with mod_fr layout."""
        verts = ["A", "B", "C", "D"]
        edges = _small_edges()
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            layout_method="mod_fr",
            expanded_only=True,
            iterations=10,
            seed=42,
            verbose=False,
        )
        # When expanded_only, pos is set to pos_ and G to G_
        assert pos is pos_

    def test_min_size_filtering(self):
        """Test that min_size=2 removes isolates from expanded graph."""
        verts = ["A", "B", "C", "D", "E"]
        edges = _small_edges()  # A-B-C-D chain, E is isolated
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            min_size=2,
            layout_method="mod_fr",
            iterations=10,
            seed=42,
            verbose=False,
        )
        # min_size=2 removes isolates from G_
        assert "E" not in G_.nodes()

    def test_min_size_gt2(self):
        """Test that min_size=3 removes small components."""
        verts = ["A", "B", "C", "D", "E", "F"]
        edges = pd.DataFrame(
            {
                "source": ["A", "B", "C", "E"],
                "target": ["B", "C", "D", "F"],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=edges,
            min_size=3,
            layout_method="mod_fr",
            iterations=10,
            seed=42,
            verbose=False,
        )
        # E-F component has size 2, should be removed from G_
        assert "E" not in G_.nodes()
        assert "F" not in G_.nodes()
        # A-B-C-D component has size 4, should remain
        assert "A" in G_.nodes()

    def test_no_edges(self):
        """Test layout with no edges."""
        verts = ["A", "B", "C"]
        G, G_, pos, pos_ = generate_layout(
            vertices=verts,
            edges=None,
            layout_method="mod_fr",
            iterations=10,
            seed=42,
            verbose=False,
        )
        assert isinstance(G, nx.Graph)

    def test_prebuilt_graphs(self):
        """Test layout with pre-built graph tuple."""
        G = _small_graph()
        G_ = G.copy()
        G_out, G_out_, pos, pos_ = generate_layout(
            graphs=(G, G_),
            layout_method="mod_fr",
            iterations=10,
            seed=42,
            verbose=False,
        )
        assert G_out is G
        assert G_out_ is G_
