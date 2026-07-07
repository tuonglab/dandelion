from __future__ import annotations

import math

import igraph as ig
import numpy as np
import pandas as pd
import networkx as nx

from numba import njit, prange, cuda
from scipy.sparse import issparse, coo_matrix, csr_matrix
from scanpy import logging as logg
from typing import Any, Literal

from networkx.utils import np_random_state as random_state


def generate_layout(
    vertices: list | None = None,
    edges: pd.DataFrame | None = None,
    min_size: int = 2,
    weight: str | None = None,
    verbose: bool = True,
    compute_layout: bool = True,
    layout_method: Literal[
        "mod_fr",
        "mod_fr2",
        "mod_fr2_gpu",
        "mod_fr_bh",
        "mod_fr_bh_gpu",
        "fa2",
    ] = "mod_fr2",
    expanded_only: bool = False,
    graphs: tuple[nx.Graph, nx.Graph] | tuple[ig.Graph, ig.Graph] | None = None,
    singleton_mass: float = 0.5,
    backend: Literal["networkx", "igraph"] = "networkx",
    **kwargs,
) -> (
    tuple[nx.Graph, nx.Graph, dict, dict]
    | tuple[ig.Graph, ig.Graph, dict, dict]
):
    """Generate network layout for a given graph or set of vertices and edges."""
    if graphs is None:
        if vertices is not None:
            if backend == "igraph":
                G, G_, Gi, Gi_ = _build_graphs_igraph(vertices, edges, min_size)
            else:
                G, G_ = _build_graphs_networkx(vertices, edges, min_size)
    else:
        G, G_ = graphs
    if isinstance(G, ig.Graph):
        # Convert igraph to networkx for layout computation
        Gi = G
        G = Gi.to_networkx()
    if isinstance(G_, ig.Graph):
        Gi_ = G_
        G_ = Gi_.to_networkx()
    if compute_layout:
        if layout_method == "mod_fr":
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout")
                pos = _fruchterman_reingold_layout(G, weight=weight, **kwargs)
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout")
            pos_ = _fruchterman_reingold_layout(G_, weight=weight, **kwargs)
        elif layout_method == "mod_fr2":
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout")
                pos = _fruchterman_reingold_layout_v2(
                    G, weight=weight, **kwargs
                )
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout")
            pos_ = _fruchterman_reingold_layout_v2(G_, weight=weight, **kwargs)
        elif layout_method == "mod_fr2_gpu":  # pragma: no cover
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout")
                pos = _fruchterman_reingold_layout_gpu(
                    G, weight=weight, **kwargs
                )
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout")
            pos_ = _fruchterman_reingold_layout_gpu(G_, weight=weight, **kwargs)
        elif layout_method == "mod_fr_bh":
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout (Barnes-Hut CPU)")
                pos = _fruchterman_reingold_layout_bh(
                    G, weight=weight, singleton_mass=singleton_mass, **kwargs
                )
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout (Barnes-Hut CPU)")
            pos_ = _fruchterman_reingold_layout_bh(
                G_, weight=weight, singleton_mass=singleton_mass, **kwargs
            )
        elif layout_method == "mod_fr_bh_gpu":  # pragma: no cover
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout (Barnes-Hut GPU)")
                pos = _fruchterman_reingold_layout_bh_gpu(
                    G, weight=weight, singleton_mass=singleton_mass, **kwargs
                )
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout (Barnes-Hut GPU)")
            pos_ = _fruchterman_reingold_layout_bh_gpu(
                G_, weight=weight, singleton_mass=singleton_mass, **kwargs
            )
        elif layout_method == "fa2":
            try:
                from fa2_modified import ForceAtlas2
            except ImportError:  # pragma: no cover
                logg.info(
                    "Please install ForceAtlas2 to use fa2 layout: "
                    "pip install fa2-modified"
                )
            fa2_layout = ForceAtlas2(**kwargs)
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout")
                pos = fa2_layout.forceatlas2_networkx_layout(
                    G, weight_attr=weight
                )
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout")
            pos_ = fa2_layout.forceatlas2_networkx_layout(
                G_, weight_attr=weight
            )
        if pos is None:
            G = G_
            pos = pos_

        if backend == "igraph":
            return (Gi, Gi_, pos, pos_)
        return (G, G_, pos, pos_)
    else:
        if backend == "igraph":
            return (Gi, Gi_, None, None)
        return (G, G_, None, None)


# when dealing with a lot of unconnected vertices, the pieces fly out to infinity and the original fr layout can't be
# used
# workaround from https://stackoverflow.com/questions/14283341/how-to-increase-node-spacing-for-networkx-spring-layout
# code chunk from networkx's layout.py https://github.com/networkx/networkx/blob/master/networkx/drawing/layout.py
def _process_params(
    G: nx.Graph, center: np.ndarray | None, dim: int
) -> tuple[nx.Graph, np.ndarray]:
    """Validate graph input and center coordinates for layout computation.

    Parameters
    ----------
    G : nx.Graph
        Input graph. If not a Graph instance, nodes are extracted into one.
    center : np.ndarray | None
        Coordinate pair around which to center the layout. If None, uses origin.
    dim : int
        Dimension of the layout space.

    Returns
    -------
    tuple[nx.Graph, np.ndarray]
        Validated graph and center array.
    """
    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def _fruchterman_reingold_layout(
    G: nx.Graph,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    weight: str = "weight",
    scale: float = 1,
    center: np.ndarray | None = None,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
) -> dict:
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.
    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.
    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. A position will be assigned to every node in G.
    k : float | None, optional
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.
    pos : dict | None, optional
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.
    fixed : list | None, optional
        Nodes to keep fixed at initial position.
        ValueError raised if `fixed` specified and `pos` not.
    iterations : int, optional
        Maximum number of iterations taken
    threshold: float, optional
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.
    weight : str | None, optional
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.
    scale : float | None, optional
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.
    center : np.ndarray | None, optional
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.
    dim : int, optional
        Dimension of layout.
    seed : int | np.random.RandomState | None, optional
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)
    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)
    """
    G, center = _process_params(G, center, dim)

    if fixed is not None:  # pragma: no cover
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    try:
        # Sparse matrix
        if len(G) < 500:  # sparse solver for large graphs
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
        if k is None and fixed is not None:  # pragma: no cover
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    except ValueError:
        A = nx.to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:  # pragma: no cover
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    if fixed is None and scale is not None:
        pos = _rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


@random_state(7)
def _fruchterman_reingold(
    A: np.ndarray,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    nnodes, _ = A.shape
    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for _ in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.001, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k)
        )
        displacement = displacement - pos / (k * np.sqrt(nnodes))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.01, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


@random_state(7)
def _sparse_fruchterman_reingold(
    A: np.ndarray,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Sparse Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    nnodes, _ = A.shape
    # make sure we have a List of Lists representation
    A = A.tolil()
    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:  # pragma: no cover
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue  # pragma: no cover
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            displacement[:, i] += (
                delta * (k * k / distance**2 - Ai * distance / k)
            ).sum(axis=1)
        displacement = displacement - pos.T / (k * np.sqrt(nnodes))
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.01, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


def _rescale_layout(pos: np.ndarray, scale: float = 1) -> np.ndarray:
    """
    Return scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.
    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : np.ndarray
        positions to be scaled. Each row is a position.
    scale : float, optional
        The size of the resulting extent in all directions.

    Returns
    -------
    np.ndarray
        scaled positions. Each row is a position.
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def _detect_torch_device() -> tuple[Any, Any]:  # pragma: no cover
    """Detect best available PyTorch device for layout computation."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for mod_fr2_gpu layout. "
            "Install it with: pip install torch"
        )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logg.info("Using PyTorch with CUDA GPU for layout")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logg.info("Using PyTorch with Apple Metal GPU for layout")
    else:
        device = torch.device("cpu")
        logg.info("Using PyTorch on CPU for layout")
    return torch, device


_numba_fr_kernel_cache = None


def _get_numba_fr_kernel() -> Any:
    """Lazily compile the Numba-accelerated FR force computation kernel.

    Returns
    -------
    Any
        Numba JIT-compiled kernel function for FR force computation.
    """
    global _numba_fr_kernel_cache
    if _numba_fr_kernel_cache is not None:
        return _numba_fr_kernel_cache

    @njit(parallel=True, cache=True, fastmath=True)
    def _kernel(
        pos,
        A_data,
        A_indices,
        A_indptr,
        k,
        nnodes,
        dim,
        iterations,
        threshold,
        t,
        dt,
        fixed_mask,
    ):
        k2 = k * k
        inv_k = 1.0 / k
        gravity = 1.0 / (k * np.sqrt(float(nnodes)))

        for _iter in range(iterations):
            displacement = np.zeros((nnodes, dim), dtype=pos.dtype)

            for i in prange(nnodes):
                if fixed_mask[i]:
                    continue

                # Repulsive forces from all other nodes
                for j in range(nnodes):
                    if i == j:
                        continue
                    dist_sq = 0.0
                    for d in range(dim):
                        diff = pos[i, d] - pos[j, d]
                        dist_sq += diff * diff
                    if dist_sq < 1e-6:
                        dist_sq = 1e-6
                    factor = k2 / dist_sq
                    for d in range(dim):
                        displacement[i, d] += (pos[i, d] - pos[j, d]) * factor

                # Attractive forces from edges (sparse CSR)
                for idx in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[idx]
                    w = A_data[idx]
                    dist_sq = 0.0
                    for d in range(dim):
                        diff = pos[i, d] - pos[j, d]
                        dist_sq += diff * diff
                    dist = np.sqrt(max(dist_sq, 1e-6))
                    factor = -w * dist * inv_k
                    for d in range(dim):
                        displacement[i, d] += (pos[i, d] - pos[j, d]) * factor

                # Gravity toward center
                for d in range(dim):
                    displacement[i, d] -= pos[i, d] * gravity

            # Update positions (sequential to avoid race conditions)
            err_sum = 0.0
            for i in range(nnodes):
                if fixed_mask[i]:
                    continue
                length_sq = 0.0
                for d in range(dim):
                    length_sq += displacement[i, d] ** 2
                length = np.sqrt(length_sq)
                if length < 0.01:
                    length = 0.01
                scale = t / length
                for d in range(dim):
                    dp = displacement[i, d] * scale
                    pos[i, d] += dp
                    err_sum += dp * dp

            t -= dt
            if np.sqrt(err_sum) / nnodes < threshold:
                break

        return pos

    _numba_fr_kernel_cache = _kernel
    return _kernel


@random_state(7)
def _fruchterman_reingold_numba(
    A: np.ndarray | csr_matrix,
    k: float | None = None,
    pos: np.ndarray | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Numba-accelerated Fruchterman-Reingold algorithm.

    Uses parallel CPU execution via Numba JIT compilation.
    Separates repulsive (O(N^2)) and attractive (O(E)) forces
    with CSR sparse format for efficient edge traversal.

    Parameters
    ----------
    A : np.ndarray | csr_matrix
        Adjacency matrix (dense or sparse).
    k : float | None, optional
        Optimal distance between nodes. If None, defaults to ``sqrt(1/N)``.
    pos : np.ndarray | None, optional
        Initial positions as (N, dim) array. Random if None.
    fixed : list | None, optional
        Indices of nodes whose positions should not change.
    iterations : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative position change.
    dim : int, optional
        Dimension of layout.
    seed : int | np.random.RandomState | None, optional
        Random state for deterministic layouts.

    Returns
    -------
    np.ndarray
        (N, dim) array of node positions.
    """
    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    # Convert to CSR for efficient row access in Numba
    if issparse(A):
        A_csr = A.tocsr().astype(np.float32)
    else:
        A_csr = csr_matrix(A.astype(np.float32))

    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=np.float32)
    else:
        pos = pos.astype(np.float32)

    if k is None:
        k = np.sqrt(1.0 / nnodes)

    t = max(float(pos[:, d].max() - pos[:, d].min()) for d in range(dim)) * 0.1
    dt = t / float(iterations + 1)

    fixed_mask = np.zeros(nnodes, dtype=np.bool_)
    if fixed is not None:
        fixed_mask[np.asarray(fixed)] = True

    kernel = _get_numba_fr_kernel()
    pos = kernel(
        pos,
        np.ascontiguousarray(A_csr.data),
        np.ascontiguousarray(A_csr.indices.astype(np.int64)),
        np.ascontiguousarray(A_csr.indptr.astype(np.int64)),
        float(k),
        nnodes,
        dim,
        iterations,
        float(threshold),
        float(t),
        float(dt),
        fixed_mask,
    )

    return pos


@random_state(9)
def _fruchterman_reingold_torch(
    A: np.ndarray,
    k: float | None = None,
    pos: np.ndarray | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    torch_module: Any = None,
    device: Any = None,
    seed: int | np.random.RandomState | None = None,
) -> np.ndarray:  # pragma: no cover
    """PyTorch GPU-accelerated Fruchterman-Reingold algorithm.

    Uses dense tensor operations on GPU (CUDA/MPS) or CPU.
    The O(N^2) pairwise computation maps naturally to GPU parallelism.

    WARNING: Creates N x N tensors - not suitable for large graphs (>30K nodes).
    Use ``_fruchterman_reingold_torch_tiled`` for larger graphs.

    Parameters
    ----------
    A : np.ndarray
        Dense adjacency matrix.
    k : float | None, optional
        Optimal distance between nodes. If None, defaults to ``sqrt(1/N)``.
    pos : np.ndarray | None, optional
        Initial positions as (N, dim) array. Random if None.
    fixed : list | None, optional
        Indices of nodes whose positions should not change.
    iterations : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative position change.
    dim : int, optional
        Dimension of layout.
    torch_module : Any, optional
        The ``torch`` module (passed to avoid top-level import).
    device : Any, optional
        PyTorch device (e.g. ``torch.device("cuda")``).
    seed : int | np.random.RandomState | None, optional
        Random state for deterministic layouts.

    Returns
    -------
    np.ndarray
        (N, dim) array of node positions.
    """
    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    torch = torch_module

    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=np.float32)
    else:
        pos = pos.astype(np.float32)

    if k is None:
        k = float(np.sqrt(1.0 / nnodes))

    A_t = torch.from_numpy(A.astype(np.float32)).to(device)
    pos_t = torch.from_numpy(pos).to(device)

    k2 = k * k
    inv_k = 1.0 / k
    gravity = 1.0 / (k * float(np.sqrt(nnodes)))

    ranges = pos_t.max(dim=0).values - pos_t.min(dim=0).values
    t = float(ranges.max().item()) * 0.1
    dt = t / (iterations + 1)

    fixed_mask = None
    if fixed is not None:
        fixed_mask = torch.zeros(nnodes, dtype=torch.bool, device=device)
        fixed_mask[torch.tensor(fixed, dtype=torch.long, device=device)] = True

    for _ in range(iterations):
        # Pairwise differences: N x N x dim
        delta = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)
        # Pairwise distances: N x N
        distance = torch.sqrt((delta**2).sum(dim=-1)).clamp(min=0.001)
        # Combined force magnitudes: repulsive + attractive
        force_mag = k2 / (distance * distance) - A_t * distance * inv_k
        # Displacement: sum of directed forces
        displacement = (delta * force_mag.unsqueeze(-1)).sum(dim=1)
        # Gravity toward center
        displacement = displacement - pos_t * gravity

        # Limit step size by temperature
        length = torch.sqrt((displacement**2).sum(dim=-1)).clamp(min=0.01)
        delta_pos = displacement * (t / length).unsqueeze(-1)

        if fixed_mask is not None:
            delta_pos[fixed_mask] = 0.0

        pos_t = pos_t + delta_pos
        t -= dt

        err = float(torch.sqrt((delta_pos**2).sum()).item()) / nnodes
        if err < threshold:
            break

    return pos_t.cpu().numpy()


@random_state(9)
def _fruchterman_reingold_torch_tiled(
    A: np.ndarray | csr_matrix,
    k: float | None = None,
    pos: np.ndarray | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    torch_module: Any = None,
    device: Any = None,
    seed: int | np.random.RandomState | None = None,
    tile_size: int = 4096,
) -> np.ndarray:  # pragma: no cover
    """PyTorch GPU layout with tiled repulsive force computation.

    Uses tiled matrix operations to compute O(N^2) repulsive forces
    without allocating full N x N tensor. Memory: O(N x tile_size).

    For 1M nodes with tile_size=4096:
    - Full N x N would need: 1M x 1M x 4 bytes = 4TB
    - Tiled needs: 1M x 4096 x 4 bytes x 2 = ~32GB (fits in GPU)

    Parameters
    ----------
    A : np.ndarray | csr_matrix
        Adjacency matrix (dense or sparse).
    k : float | None, optional
        Optimal distance between nodes. If None, defaults to ``sqrt(1/N)``.
    pos : np.ndarray | None, optional
        Initial positions as (N, dim) array. Random if None.
    fixed : list | None, optional
        Indices of nodes whose positions should not change.
    iterations : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative position change.
    dim : int, optional
        Dimension of layout.
    torch_module : Any, optional
        The ``torch`` module (passed to avoid top-level import).
    device : Any, optional
        PyTorch device (e.g. ``torch.device("cuda")``).
    seed : int | np.random.RandomState | None, optional
        Random state for deterministic layouts.
    tile_size : int, optional
        Number of nodes per tile. Larger = faster but more memory.
        Default 4096 (uses ~64MB per tile for 2D).

    Returns
    -------
    np.ndarray
        (N, dim) array of node positions.
    """
    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    torch = torch_module

    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=np.float32)
    else:
        pos = pos.astype(np.float32)

    if k is None:
        k = float(np.sqrt(1.0 / nnodes))

    pos_t = torch.from_numpy(pos).to(device)

    # Convert sparse adjacency to COO for efficient edge iteration
    if issparse(A):
        A_coo = A.tocoo()
    else:
        A_coo = coo_matrix(A)

    # Only keep non-zero edges
    edge_src = torch.from_numpy(A_coo.row.astype(np.int64)).to(device)
    edge_dst = torch.from_numpy(A_coo.col.astype(np.int64)).to(device)
    edge_weight = torch.from_numpy(A_coo.data.astype(np.float32)).to(device)

    k2 = k * k
    inv_k = 1.0 / k
    gravity = 1.0 / (k * float(np.sqrt(nnodes)))

    ranges = pos_t.max(dim=0).values - pos_t.min(dim=0).values
    t = float(ranges.max().item()) * 0.1
    dt = t / (iterations + 1)

    fixed_mask = None
    if fixed is not None:
        fixed_mask = torch.zeros(nnodes, dtype=torch.bool, device=device)
        fixed_mask[torch.tensor(fixed, dtype=torch.long, device=device)] = True

    for _ in range(iterations):
        displacement = torch.zeros(
            (nnodes, dim), dtype=torch.float32, device=device
        )

        # Tiled repulsive forces: process tile_size nodes at a time
        for tile_start in range(0, nnodes, tile_size):
            tile_end = min(tile_start + tile_size, nnodes)

            # Positions of nodes in this tile: (tile_size, dim)
            pos_tile = pos_t[tile_start:tile_end]

            # Differences: delta[i,j] = pos_tile[i] - pos_all[j]
            # This is the key: we only allocate tile_size × N, not N × N
            delta = pos_tile.unsqueeze(1) - pos_t.unsqueeze(0)

            # Distances: (tile_size, N)
            distance = torch.sqrt((delta**2).sum(dim=-1)).clamp(min=0.001)

            # Repulsive force magnitude: k²/d² (positive = repel)
            # Same formula as original: k2 / (distance * distance)
            force_mag = k2 / (distance * distance)

            # Displacement from repulsive forces
            # delta points from j to i, force_mag is positive, so this pushes i away from j
            displacement[tile_start:tile_end] += (
                delta * force_mag.unsqueeze(-1)
            ).sum(dim=1)

        # Attractive forces via sparse edges (vectorized)
        if len(edge_src) > 0:
            # Get positions of edge endpoints
            src_pos = pos_t[edge_src]  # (E, dim)
            dst_pos = pos_t[edge_dst]  # (E, dim)

            # Edge vectors: same convention as repulsive delta[i,j] = pos[i] - pos[j]
            # So for edge (src, dst): edge_delta = pos[src] - pos[dst]
            edge_delta = src_pos - dst_pos  # (E, dim)
            edge_dist = torch.sqrt(
                (edge_delta**2).sum(dim=-1).clamp(min=0.001)
            )  # (E,)

            # Attractive force magnitude: -w * d / k
            # Original: force_mag = k2/(d*d) - A*d/k
            # The attractive part is: -A*d/k (negative, so it SUBTRACTS from repulsive)
            # This means: displacement += delta * (-A*d/k)
            # Which pulls nodes together (delta points away, negative flips it)
            attr_mag = (
                -edge_weight * edge_dist * inv_k
            )  # (E,) - negative values

            # Apply to edge_delta: negative * (src-dst) pulls src toward dst
            attr_force = attr_mag.unsqueeze(-1) * edge_delta  # (E, dim)

            # Scatter-add attractive forces to source nodes
            displacement.scatter_add_(
                0, edge_src.unsqueeze(-1).expand(-1, dim), attr_force
            )

        # Gravity toward center
        displacement = displacement - pos_t * gravity

        # Limit step size by temperature
        length = torch.sqrt((displacement**2).sum(dim=-1)).clamp(min=0.01)
        delta_pos = displacement * (t / length).unsqueeze(-1)

        if fixed_mask is not None:
            delta_pos[fixed_mask] = 0.0

        pos_t = pos_t + delta_pos
        t -= dt

        err = float(torch.sqrt((delta_pos**2).sum()).item()) / nnodes
        if err < threshold:
            break

    return pos_t.cpu().numpy()


# ============================================================================
# Barnes-Hut O(N log N) implementations for scalable force-directed layout
# ============================================================================

_numba_bh_kernels_cache = None


def _get_numba_bh_kernels() -> tuple:
    """Lazily compile Numba CPU kernels for Barnes-Hut algorithm.

    Returns
    -------
    tuple
        (_build_quadtree, _barnes_hut_forces, _attractive_forces, _gravity_and_update)
    """
    global _numba_bh_kernels_cache
    if _numba_bh_kernels_cache is not None:
        return _numba_bh_kernels_cache

    @njit(cache=True)
    def _build_quadtree(
        pos,  # (nnodes, 2) float64 particle positions
        particle_mass,  # (nnodes,) float64 mass per particle
        nnodes,  # int, number of particles
        center_x,  # float, x center of root cell
        center_y,  # float, y center of root cell
        half_size,  # float, half-width of root cell
        max_depth,  # int, maximum tree depth
        # Output arrays (pre-allocated, max_tree_nodes each)
        node_center_x,  # (max_tree_nodes,) float64
        node_center_y,  # (max_tree_nodes,) float64
        node_half_size,  # (max_tree_nodes,) float64
        node_mass,  # (max_tree_nodes,) float64
        node_com_x,  # (max_tree_nodes,) float64
        node_com_y,  # (max_tree_nodes,) float64
        node_children,  # (max_tree_nodes, 4) int64, child indices (-1 = empty)
        node_is_leaf,  # (max_tree_nodes,) bool
        node_particle,  # (max_tree_nodes,) int64, particle index (-1 = none)
    ):
        """Build a quadtree from 2D positions using flat arrays.

        Inserts each particle into the tree, subdividing leaf cells when they
        already contain a particle. Center-of-mass is computed using
        particle_mass weights.

        Parameters
        ----------
        pos : np.ndarray
            (nnodes, 2) particle positions.
        particle_mass : np.ndarray
            (nnodes,) mass per particle (degree-based; singletons use singleton_mass).

        Returns
        -------
        int
            Number of tree nodes used.
        """
        # Initialize root node
        node_center_x[0] = center_x
        node_center_y[0] = center_y
        node_half_size[0] = half_size
        node_mass[0] = 0.0
        node_com_x[0] = 0.0
        node_com_y[0] = 0.0
        node_is_leaf[0] = True
        node_particle[0] = -1
        for c in range(4):
            node_children[0, c] = -1

        next_node = 1  # Next available node index

        # Insert each particle
        for p in range(nnodes):
            px, py = pos[p, 0], pos[p, 1]
            pmass = particle_mass[p]

            # Start at root
            current = 0
            depth = 0

            while depth < max_depth:
                # Update center of mass (weighted by particle mass)
                old_mass = node_mass[current]
                new_mass = old_mass + pmass
                node_mass[current] = new_mass
                if old_mass == 0:
                    node_com_x[current] = px
                    node_com_y[current] = py
                elif new_mass > 0:
                    node_com_x[current] = (
                        node_com_x[current] * old_mass + px * pmass
                    ) / new_mass
                    node_com_y[current] = (
                        node_com_y[current] * old_mass + py * pmass
                    ) / new_mass

                if node_is_leaf[current]:
                    if node_particle[current] == -1:
                        # Empty leaf - just add particle
                        node_particle[current] = p
                        break
                    else:
                        # Leaf with existing particle - need to subdivide
                        old_p = node_particle[current]
                        old_px, old_py = pos[old_p, 0], pos[old_p, 1]
                        old_pmass = particle_mass[old_p]

                        # Create 4 children
                        h = node_half_size[current] / 2.0
                        cx, cy = node_center_x[current], node_center_y[current]

                        for c in range(4):
                            child_idx = next_node + c
                            # Quadrant centers: 0=SW, 1=SE, 2=NW, 3=NE
                            if c == 0:
                                node_center_x[child_idx] = cx - h
                                node_center_y[child_idx] = cy - h
                            elif c == 1:
                                node_center_x[child_idx] = cx + h
                                node_center_y[child_idx] = cy - h
                            elif c == 2:
                                node_center_x[child_idx] = cx - h
                                node_center_y[child_idx] = cy + h
                            else:
                                node_center_x[child_idx] = cx + h
                                node_center_y[child_idx] = cy + h

                            node_half_size[child_idx] = h
                            node_mass[child_idx] = 0.0
                            node_com_x[child_idx] = 0.0
                            node_com_y[child_idx] = 0.0
                            node_is_leaf[child_idx] = True
                            node_particle[child_idx] = -1
                            for cc in range(4):
                                node_children[child_idx, cc] = -1
                            node_children[current, c] = child_idx

                        next_node += 4
                        node_is_leaf[current] = False
                        node_particle[current] = -1

                        # Re-insert old particle into appropriate child
                        old_quad = 0
                        if old_px >= cx:
                            old_quad += 1
                        if old_py >= cy:
                            old_quad += 2
                        old_child = node_children[current, old_quad]
                        node_mass[old_child] = old_pmass
                        node_com_x[old_child] = old_px
                        node_com_y[old_child] = old_py
                        node_particle[old_child] = old_p

                        # Continue with current particle
                        # (fall through to find quadrant below)

                # Find quadrant for current particle
                cx, cy = node_center_x[current], node_center_y[current]
                quad = 0
                if px >= cx:
                    quad += 1
                if py >= cy:
                    quad += 2

                child = node_children[current, quad]
                if child == -1:
                    # No child yet - shouldn't happen after subdivision
                    break
                current = child
                depth += 1

        return next_node

    @njit(parallel=True, cache=True, fastmath=True)
    def _barnes_hut_forces(
        pos,  # (nnodes, 2) float64 particle positions
        nnodes,  # int, number of particles
        theta_sq,  # float, theta² (opening angle squared)
        k2,  # float, k² (optimal distance squared)
        num_tree_nodes,  # int, number of active tree nodes
        node_half_size,  # (num_tree_nodes,) float64
        node_mass,  # (num_tree_nodes,) float64
        node_com_x,  # (num_tree_nodes,) float64
        node_com_y,  # (num_tree_nodes,) float64
        node_children,  # (num_tree_nodes, 4) int64
        node_is_leaf,  # (num_tree_nodes,) bool
        displacement,  # (nnodes, 2) float64, output repulsive forces
    ):
        """Compute repulsive forces via Barnes-Hut tree traversal (CPU).

        Each particle traverses the quadtree with a stack. If a cell is
        far enough (size²/dist² < theta²), its center-of-mass is used
        as an approximation. Otherwise, children are expanded.

        Repulsive force: k² * mass / d, direction away from center-of-mass.
        """
        for i in prange(nnodes):
            px = pos[i, 0]
            py = pos[i, 1]
            fx = 0.0
            fy = 0.0

            # Stack-based tree traversal (avoid recursion)
            stack = np.zeros(64, dtype=np.int64)
            stack[0] = 0  # Start at root
            stack_ptr = 1

            while stack_ptr > 0:
                stack_ptr -= 1
                node = stack[stack_ptr]

                if node < 0 or node >= num_tree_nodes:
                    continue

                mass = node_mass[node]
                if mass == 0:
                    continue

                dx = px - node_com_x[node]
                dy = py - node_com_y[node]
                dist_sq = dx * dx + dy * dy

                if dist_sq < 1e-9:
                    continue

                size = 2.0 * node_half_size[node]
                size_sq = size * size

                # Barnes-Hut criterion: size²/dist² < theta²
                if node_is_leaf[node] or size_sq < theta_sq * dist_sq:
                    force = k2 * mass / dist_sq
                    dist = np.sqrt(dist_sq)
                    fx += force * dx / dist
                    fy += force * dy / dist
                else:
                    for c in range(4):
                        child = node_children[node, c]
                        if child >= 0 and stack_ptr < 64:
                            stack[stack_ptr] = child
                            stack_ptr += 1

            displacement[i, 0] = fx
            displacement[i, 1] = fy

    @njit(parallel=True, cache=True, fastmath=True)
    def _attractive_forces(
        pos,  # (nnodes, 2) float64 particle positions
        A_data,  # (nnz,) float64 edge weights
        A_indices,  # (nnz,) int64 column indices (CSR)
        A_indptr,  # (nnodes+1,) int64 row pointers (CSR)
        nnodes,  # int, number of particles
        inv_k,  # float, 1/k
        displacement,  # (nnodes, 2) float64, in/out accumulated forces
    ):
        """Add attractive forces from sparse edges (CSR, CPU).

        Attractive force: w * d / k toward neighbour.
        Uses CSR with prange over nodes (race-free) instead of COO with atomics.
        """
        for i in prange(nnodes):
            for idx in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[idx]
                w = A_data[idx]

                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist_sq = dx * dx + dy * dy

                if dist_sq < 1e-9:
                    continue

                dist = np.sqrt(dist_sq)
                force = w * dist * inv_k
                displacement[i, 0] += force * dx / dist
                displacement[i, 1] += force * dy / dist

    @njit(parallel=True, cache=True, fastmath=True)
    def _gravity_and_update(
        pos,  # (nnodes, 2) float64 positions, modified in-place
        displacement,  # (nnodes, 2) float64 accumulated forces
        gravity,  # float, gravity strength toward origin
        t,  # float, current temperature (max step size)
        fixed_mask,  # (nnodes,) bool, True for fixed nodes
        nnodes,  # int, number of particles
    ):
        """Apply gravity toward origin and update positions (CPU).

        For each non-fixed node: subtract gravity pull from displacement,
        clamp step length, and move the node.
        """
        for i in prange(nnodes):
            if fixed_mask[i]:
                continue

            dx = displacement[i, 0] - pos[i, 0] * gravity
            dy = displacement[i, 1] - pos[i, 1] * gravity

            length = np.sqrt(dx * dx + dy * dy)
            if length < 0.01:
                length = 0.01

            scale = t / length
            pos[i, 0] += dx * scale
            pos[i, 1] += dy * scale

    _numba_bh_kernels_cache = (
        _build_quadtree,
        _barnes_hut_forces,
        _attractive_forces,
        _gravity_and_update,
    )
    return _numba_bh_kernels_cache


@random_state(7)
def _fruchterman_reingold_barnes_hut_numba(
    A: np.ndarray,
    k: float | None = None,
    pos: np.ndarray | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    seed: int | None = None,
    theta: float = 0.8,
    singleton_mass: float = 0.5,
) -> np.ndarray:
    """Barnes-Hut accelerated Fruchterman-Reingold layout (CPU/Numba).

    O(N log N) per iteration. Mirrors the CUDA version exactly but runs
    on CPU with Numba parallel JIT.

    Parameters
    ----------
    A : np.ndarray or sparse matrix
        Adjacency matrix.
    k : float, optional
        Optimal node distance. Default ``sqrt(1/N)``.
    pos : np.ndarray, optional
        (N, dim) initial positions. Random if None.
    fixed : list, optional
        Indices of nodes whose positions should not change.
    iterations : int
        Maximum number of iterations.
    threshold : float
        Convergence threshold on displacement norm.
    dim : int
        Must be 2 (only 2-D supported).
    seed : int, optional
        Random state for reproducibility.
    theta : float
        Barnes-Hut opening angle. Smaller = more accurate, larger = faster.
    singleton_mass : float
        Mass assigned to degree-0 nodes. Lower values reduce their
        repulsive impact on connected components.

    Returns
    -------
    np.ndarray
        (N, 2) float32 node positions.
    """
    if dim != 2:
        raise ValueError("Barnes-Hut currently only supports 2D layouts")

    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    # Get compiled kernels
    (
        _build_quadtree,
        _barnes_hut_forces,
        _attractive_forces,
        _gravity_and_update,
    ) = _get_numba_bh_kernels()

    # Convert to COO (same as CUDA) then to CSR for CPU iteration
    if issparse(A):
        A_coo = A.tocoo().astype(np.float32)
    else:
        A_coo = coo_matrix(A.astype(np.float32))
    A_csr = A_coo.tocsr()

    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=np.float64)
    else:
        pos = pos.astype(np.float64)

    if k is None:
        k = np.sqrt(1.0 / nnodes)

    k2 = k * k
    inv_k = 1.0 / k
    gravity = 1.0 / (k * np.sqrt(float(nnodes)))
    theta_sq = theta * theta

    t = max(float(pos[:, d].max() - pos[:, d].min()) for d in range(dim)) * 0.1
    dt = t / float(iterations + 1)

    fixed_mask = np.zeros(nnodes, dtype=np.bool_)
    if fixed is not None:
        fixed_mask[np.asarray(fixed)] = True

    # Pre-allocate tree arrays
    max_tree_nodes = 4 * nnodes + 4
    node_center_x = np.zeros(max_tree_nodes, dtype=np.float64)
    node_center_y = np.zeros(max_tree_nodes, dtype=np.float64)
    node_half_size = np.zeros(max_tree_nodes, dtype=np.float64)
    node_mass = np.zeros(max_tree_nodes, dtype=np.float64)
    node_com_x = np.zeros(max_tree_nodes, dtype=np.float64)
    node_com_y = np.zeros(max_tree_nodes, dtype=np.float64)
    node_children = np.full((max_tree_nodes, 4), -1, dtype=np.int64)
    node_is_leaf = np.ones(max_tree_nodes, dtype=np.bool_)
    node_particle = np.full(max_tree_nodes, -1, dtype=np.int64)

    # Compute node degrees for mass weighting (from COO)
    # Singletons get small mass so they minimally affect layout
    degrees = np.zeros(nnodes, dtype=np.float64)
    np.add.at(degrees, A_coo.row, 1)
    particle_mass = np.maximum(singleton_mass, degrees)

    max_depth = int(np.ceil(np.log2(nnodes + 1))) + 4

    # Pre-extract CSR arrays
    A_data = np.ascontiguousarray(A_csr.data.astype(np.float64))
    A_indices = np.ascontiguousarray(A_csr.indices.astype(np.int64))
    A_indptr = np.ascontiguousarray(A_csr.indptr.astype(np.int64))

    displacement = np.zeros((nnodes, dim), dtype=np.float64)

    for _iter in range(iterations):
        # Reset tree
        node_mass[:] = 0.0
        node_is_leaf[:] = True
        node_particle[:] = -1
        node_children[:] = -1

        # Compute bounding box
        min_x, min_y = pos[:, 0].min(), pos[:, 1].min()
        max_x, max_y = pos[:, 0].max(), pos[:, 1].max()
        margin = max(max_x - min_x, max_y - min_y) * 0.1 + 1e-6
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_size = max(max_x - min_x, max_y - min_y) / 2.0 + margin

        # Build quadtree on CPU with degree-weighted mass
        num_tree_nodes = _build_quadtree(
            pos,
            particle_mass,
            nnodes,
            center_x,
            center_y,
            half_size,
            max_depth,
            node_center_x,
            node_center_y,
            node_half_size,
            node_mass,
            node_com_x,
            node_com_y,
            node_children,
            node_is_leaf,
            node_particle,
        )

        # Reset displacement
        displacement[:] = 0.0

        # Compute repulsive forces via Barnes-Hut
        _barnes_hut_forces(
            pos,
            nnodes,
            theta_sq,
            k2,
            num_tree_nodes,
            node_half_size[:num_tree_nodes],
            node_mass[:num_tree_nodes],
            node_com_x[:num_tree_nodes],
            node_com_y[:num_tree_nodes],
            node_children[:num_tree_nodes],
            node_is_leaf[:num_tree_nodes],
            displacement,
        )

        # Compute attractive forces
        _attractive_forces(
            pos, A_data, A_indices, A_indptr, nnodes, inv_k, displacement
        )

        # Apply gravity and update positions
        _gravity_and_update(pos, displacement, gravity, t, fixed_mask, nnodes)

        t -= dt

        # Check convergence periodically
        if _iter % 10 == 0:
            err = np.sqrt((displacement**2).sum()) / nnodes
            if err < threshold:
                break

    return pos.astype(np.float32)


# ============================================================================
# Numba CUDA Barnes-Hut for GPU acceleration
# ============================================================================

_numba_cuda_bh_kernels_cache = None


def _get_numba_cuda_bh_kernels() -> tuple:  # pragma: no cover
    """Lazily compile Numba CUDA kernels for Barnes-Hut algorithm.

    Returns
    -------
    tuple
        (_barnes_hut_forces_cuda, _attractive_forces_cuda,
         _gravity_and_update_cuda, cuda)
    """
    global _numba_cuda_bh_kernels_cache
    if _numba_cuda_bh_kernels_cache is not None:
        return _numba_cuda_bh_kernels_cache

    # Check if CUDA is available
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Numba CUDA requires an NVIDIA GPU "
            "with CUDA toolkit installed."
        )

    @cuda.jit
    def _barnes_hut_forces_cuda(
        pos,  # (nnodes, 2) float64 particle positions
        nnodes,  # int, number of particles
        theta_sq,  # float, theta² (opening angle squared)
        k2,  # float, k² (optimal distance squared)
        num_tree_nodes,  # int, number of active tree nodes
        node_half_size,  # (num_tree_nodes,) float64
        node_mass,  # (num_tree_nodes,) float64
        node_com_x,  # (num_tree_nodes,) float64
        node_com_y,  # (num_tree_nodes,) float64
        node_children,  # (num_tree_nodes, 4) int64
        node_is_leaf,  # (num_tree_nodes,) bool
        displacement,  # (nnodes, 2) float64, output repulsive forces
    ):
        """Compute repulsive forces via Barnes-Hut tree traversal (CUDA).

        One thread per particle. Each traverses the quadtree with a local
        stack. Repulsive force: k² * mass / d, direction away from COM.
        """
        i = cuda.grid(1)
        if i >= nnodes:
            return

        px = pos[i, 0]
        py = pos[i, 1]
        fx = 0.0
        fy = 0.0

        # Local stack for tree traversal (in thread-local registers)
        # Max depth is log2(N) + constant, 64 is plenty
        stack = cuda.local.array(64, dtype=np.int64)
        stack[0] = 0  # Start at root
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node = stack[stack_ptr]

            if node < 0 or node >= num_tree_nodes:
                continue

            mass = node_mass[node]
            if mass == 0:
                continue

            # Vector from node's COM to particle
            dx = px - node_com_x[node]
            dy = py - node_com_y[node]
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-9:
                continue

            # Barnes-Hut criterion: size²/dist² < theta²
            size = 2.0 * node_half_size[node]
            size_sq = size * size

            if node_is_leaf[node] or size_sq < theta_sq * dist_sq:
                # Use this node's center of mass
                force = k2 * mass / dist_sq
                dist = math.sqrt(dist_sq)
                fx += force * dx / dist
                fy += force * dy / dist
            else:
                # Traverse children
                for c in range(4):
                    child = node_children[node, c]
                    if child >= 0 and stack_ptr < 64:
                        stack[stack_ptr] = child
                        stack_ptr += 1

        displacement[i, 0] = fx
        displacement[i, 1] = fy

    @cuda.jit
    def _attractive_forces_cuda(
        pos,  # (nnodes, 2) float64 particle positions
        edge_src,  # (num_edges,) int64 source node indices (COO)
        edge_dst,  # (num_edges,) int64 destination node indices (COO)
        edge_weight,  # (num_edges,) float64 edge weights
        num_edges,  # int, number of edges
        inv_k,  # float, 1/k
        displacement,  # (nnodes, 2) float64, in/out accumulated forces
    ):
        """Add attractive forces from edges (COO format, CUDA).

        One thread per edge. Attractive force: w * d / k toward neighbour.
        Uses atomic adds since multiple edges may target the same node.
        """
        e = cuda.grid(1)
        if e >= num_edges:
            return

        i = edge_src[e]
        j = edge_dst[e]
        w = edge_weight[e]

        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1e-9:
            return

        dist = math.sqrt(dist_sq)
        force = w * dist * inv_k

        # Atomic add to handle multiple edges per node
        cuda.atomic.add(displacement, (i, 0), force * dx / dist)
        cuda.atomic.add(displacement, (i, 1), force * dy / dist)

    @cuda.jit
    def _gravity_and_update_cuda(
        pos,  # (nnodes, 2) float64 positions, modified in-place
        displacement,  # (nnodes, 2) float64 accumulated forces
        gravity,  # float, gravity strength toward origin
        t,  # float, current temperature (max step size)
        fixed_mask,  # (nnodes,) bool, True for fixed nodes
        nnodes,  # int, number of particles
    ):
        """Apply gravity toward origin and update positions (CUDA).

        One thread per node. For each non-fixed node: subtract gravity pull
        from displacement, clamp step length, and move the node.
        """
        i = cuda.grid(1)
        if i >= nnodes:
            return

        if fixed_mask[i]:
            return

        # Apply gravity
        dx = displacement[i, 0] - pos[i, 0] * gravity
        dy = displacement[i, 1] - pos[i, 1] * gravity

        # Limit displacement by temperature
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.01:
            length = 0.01

        scale = t / length
        pos[i, 0] += dx * scale
        pos[i, 1] += dy * scale

    _numba_cuda_bh_kernels_cache = (
        _barnes_hut_forces_cuda,
        _attractive_forces_cuda,
        _gravity_and_update_cuda,
        cuda,
    )
    return _numba_cuda_bh_kernels_cache


@random_state(7)
def _fruchterman_reingold_barnes_hut_cuda(
    A: np.ndarray,
    k: float | None = None,
    pos: np.ndarray | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    dim: int = 2,
    seed: int | None = None,
    theta: float = 0.8,
    singleton_mass: float = 0.5,
) -> np.ndarray:  # pragma: no cover
    """Barnes-Hut accelerated Fruchterman-Reingold layout (GPU/Numba CUDA).

    O(N log N) per iteration. Quadtree is built on CPU each iteration;
    force computation and position updates run on GPU via Numba CUDA kernels.

    Parameters
    ----------
    A : np.ndarray or sparse matrix
        Adjacency matrix.
    k : float, optional
        Optimal node distance. Default ``sqrt(1/N)``.
    pos : np.ndarray, optional
        (N, dim) initial positions. Random if None.
    fixed : list, optional
        Indices of nodes whose positions should not change.
    iterations : int
        Maximum number of iterations.
    threshold : float
        Convergence threshold on displacement norm.
    dim : int
        Must be 2 (only 2-D supported).
    seed : int, optional
        Random state for reproducibility.
    theta : float
        Barnes-Hut opening angle. Smaller = more accurate, larger = faster.
    singleton_mass : float
        Mass assigned to degree-0 nodes. Lower values reduce their
        repulsive impact on connected components.

    Returns
    -------
    np.ndarray
        (N, 2) float32 node positions.
    """
    if dim != 2:
        raise ValueError("Barnes-Hut currently only supports 2D layouts")

    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    # Get CUDA kernels (also validates CUDA availability)
    (
        _barnes_hut_forces_cuda,
        _attractive_forces_cuda,
        _gravity_and_update_cuda,
        cuda,
    ) = _get_numba_cuda_bh_kernels()

    # Get CPU tree-building kernel (shared with CPU BH version)
    _build_quadtree, _, _, _ = _get_numba_bh_kernels()

    # Convert to COO for edges
    if issparse(A):
        A_coo = A.tocoo().astype(np.float32)
    else:
        A_coo = coo_matrix(A.astype(np.float32))

    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=np.float64)
    else:
        pos = pos.astype(np.float64)

    if k is None:
        k = np.sqrt(1.0 / nnodes)

    k2 = k * k
    inv_k = 1.0 / k
    gravity = 1.0 / (k * np.sqrt(float(nnodes)))
    theta_sq = theta * theta

    t = max(float(pos[:, d].max() - pos[:, d].min()) for d in range(dim)) * 0.1
    dt = t / float(iterations + 1)

    fixed_mask = np.zeros(nnodes, dtype=np.bool_)
    if fixed is not None:
        fixed_mask[np.asarray(fixed)] = True

    # Pre-allocate tree arrays (CPU)
    max_tree_nodes = 4 * nnodes + 4
    node_center_x = np.zeros(max_tree_nodes, dtype=np.float64)
    node_center_y = np.zeros(max_tree_nodes, dtype=np.float64)
    node_half_size = np.zeros(max_tree_nodes, dtype=np.float64)
    node_mass = np.zeros(max_tree_nodes, dtype=np.float64)
    node_com_x = np.zeros(max_tree_nodes, dtype=np.float64)
    node_com_y = np.zeros(max_tree_nodes, dtype=np.float64)
    node_children = np.full((max_tree_nodes, 4), -1, dtype=np.int64)
    node_is_leaf = np.ones(max_tree_nodes, dtype=np.bool_)
    node_particle = np.full(max_tree_nodes, -1, dtype=np.int64)

    # Compute node degrees for mass weighting (from COO)
    # Singletons get small mass so they minimally affect layout
    degrees = np.zeros(nnodes, dtype=np.float64)
    np.add.at(degrees, A_coo.row, 1)
    particle_mass = np.maximum(singleton_mass, degrees)

    max_depth = int(np.ceil(np.log2(nnodes + 1))) + 4

    # Allocate GPU arrays that persist across iterations
    d_pos = cuda.to_device(pos)
    d_displacement = cuda.device_array((nnodes, dim), dtype=np.float64)
    d_fixed_mask = cuda.to_device(fixed_mask)

    # Edge data (constant)
    d_edge_src = cuda.to_device(A_coo.row.astype(np.int64))
    d_edge_dst = cuda.to_device(A_coo.col.astype(np.int64))
    d_edge_weight = cuda.to_device(A_coo.data.astype(np.float64))
    num_edges = len(A_coo.data)

    # CUDA launch configuration
    threads_per_block = 256
    blocks_nodes = (nnodes + threads_per_block - 1) // threads_per_block
    blocks_edges = (num_edges + threads_per_block - 1) // threads_per_block

    for _iter in range(iterations):
        # Copy positions back to CPU for tree building
        pos = d_pos.copy_to_host()

        # Reset tree
        node_mass[:] = 0.0
        node_is_leaf[:] = True
        node_particle[:] = -1
        node_children[:] = -1

        # Compute bounding box
        min_x, min_y = pos[:, 0].min(), pos[:, 1].min()
        max_x, max_y = pos[:, 0].max(), pos[:, 1].max()
        margin = max(max_x - min_x, max_y - min_y) * 0.1 + 1e-6
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_size = max(max_x - min_x, max_y - min_y) / 2.0 + margin

        # Build quadtree on CPU with degree-weighted mass
        num_tree_nodes = _build_quadtree(
            pos,
            particle_mass,
            nnodes,
            center_x,
            center_y,
            half_size,
            max_depth,
            node_center_x,
            node_center_y,
            node_half_size,
            node_mass,
            node_com_x,
            node_com_y,
            node_children,
            node_is_leaf,
            node_particle,
        )

        # Transfer tree to GPU
        d_node_half_size = cuda.to_device(node_half_size[:num_tree_nodes])
        d_node_mass = cuda.to_device(node_mass[:num_tree_nodes])
        d_node_com_x = cuda.to_device(node_com_x[:num_tree_nodes])
        d_node_com_y = cuda.to_device(node_com_y[:num_tree_nodes])
        d_node_children = cuda.to_device(node_children[:num_tree_nodes])
        d_node_is_leaf = cuda.to_device(node_is_leaf[:num_tree_nodes])

        # Reset displacement
        d_displacement[:] = 0.0

        # Compute repulsive forces on GPU
        _barnes_hut_forces_cuda[blocks_nodes, threads_per_block](
            d_pos,
            nnodes,
            theta_sq,
            k2,
            num_tree_nodes,
            d_node_half_size,
            d_node_mass,
            d_node_com_x,
            d_node_com_y,
            d_node_children,
            d_node_is_leaf,
            d_displacement,
        )

        # Compute attractive forces on GPU
        if num_edges > 0:
            _attractive_forces_cuda[blocks_edges, threads_per_block](
                d_pos,
                d_edge_src,
                d_edge_dst,
                d_edge_weight,
                num_edges,
                inv_k,
                d_displacement,
            )

        # Apply gravity and update positions on GPU
        _gravity_and_update_cuda[blocks_nodes, threads_per_block](
            d_pos,
            d_displacement,
            gravity,
            t,
            d_fixed_mask,
            nnodes,
        )

        t -= dt

        # Check convergence periodically
        if _iter % 10 == 0:
            displacement = d_displacement.copy_to_host()
            err = np.sqrt((displacement**2).sum()) / nnodes
            if err < threshold:
                break

    return d_pos.copy_to_host().astype(np.float32)


def _fruchterman_reingold_layout_bh(
    G: nx.Graph,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    weight: str = "weight",
    scale: float = 1,
    center: np.ndarray | None = None,
    dim: int = 2,
    seed: int | None = None,
    theta: float = 0.8,
    singleton_mass: float = 0.5,
) -> dict:
    """Barnes-Hut accelerated Fruchterman-Reingold layout (CPU).

    O(N log N) per iteration via quadtree approximation. Scales to 1M+ nodes.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : float, optional
        Optimal node distance. Default ``sqrt(1/N)``.
    pos : dict, optional
        ``{node: (x, y)}`` initial positions.
    fixed : list, optional
        Nodes whose positions should not change.
    iterations : int
        Maximum number of iterations.
    threshold : float
        Convergence threshold.
    weight : str
        Edge attribute key for weights.
    scale : float
        Scale factor for final positions.
    center : np.ndarray, optional
        Center of the layout.
    dim : int
        Must be 2.
    seed : int, optional
        Random state for reproducibility.
    theta : float
        Barnes-Hut opening angle (0.3 = accurate, 1.2 = fast).
    singleton_mass : float
        Mass for degree-0 nodes. Lower = less repulsive impact.

    Returns
    -------
    dict
        ``{node: np.ndarray}`` position mapping.
    """
    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed])

    if pos is not None:
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")

    if k is None and fixed is not None:  # pragma: no cover
        nnodes, _ = A.shape
        k = dom_size / np.sqrt(nnodes)

    nnodes = len(G)
    pos = _fruchterman_reingold_barnes_hut_numba(
        A,
        k,
        pos_arr,
        fixed,
        iterations,
        threshold,
        dim,
        seed,
        theta,
        singleton_mass,
    )

    if fixed is None and scale is not None:
        pos = _rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


def _fruchterman_reingold_layout_bh_gpu(
    G: nx.Graph,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    weight: str = "weight",
    scale: float = 1,
    center: np.ndarray | None = None,
    dim: int = 2,
    seed: int | None = None,
    theta: float = 0.8,
    singleton_mass: float = 0.5,
) -> dict:  # pragma: no cover
    """Barnes-Hut accelerated Fruchterman-Reingold layout (GPU).

    O(N log N) per iteration. Uses Numba CUDA on NVIDIA GPUs for best
    performance. Falls back to CPU Barnes-Hut if CUDA is not available.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : float, optional
        Optimal node distance. Default ``sqrt(1/N)``.
    pos : dict, optional
        ``{node: (x, y)}`` initial positions.
    fixed : list, optional
        Nodes whose positions should not change.
    iterations : int
        Maximum number of iterations.
    threshold : float
        Convergence threshold.
    weight : str
        Edge attribute key for weights.
    scale : float
        Scale factor for final positions.
    center : np.ndarray, optional
        Center of the layout.
    dim : int
        Must be 2.
    seed : int, optional
        Random state for reproducibility.
    theta : float
        Barnes-Hut opening angle (0.3 = accurate, 1.2 = fast).
    singleton_mass : float
        Mass for degree-0 nodes. Lower = less repulsive impact.

    Returns
    -------
    dict
        ``{node: np.ndarray}`` position mapping.
    """
    # Try Numba CUDA first (fastest for NVIDIA GPUs)
    cuda_available = False
    cuda_available = cuda.is_available()

    if cuda_available:
        logg.info(f"Using Barnes-Hut with Numba CUDA for {len(G)} nodes")
        G, center = _process_params(G, center, dim)

        if fixed is not None:
            if pos is None:
                raise ValueError("nodes are fixed without positions given")
            for node in fixed:  # pragma: no cover
                if node not in pos:
                    raise ValueError("nodes are fixed without positions given")
            nfixed = {node: i for i, node in enumerate(G)}  # pragma: no cover
            fixed = np.asarray(
                [nfixed[node] for node in fixed]
            )  # pragma: no cover

        if pos is not None:  # pragma: no cover
            dom_size = max(
                coord for pos_tup in pos.values() for coord in pos_tup
            )
            if dom_size == 0:
                dom_size = 1
            pos_arr = seed.rand(len(G), dim) * dom_size + center

            for i, n in enumerate(G):
                if n in pos:
                    pos_arr[i] = np.asarray(pos[n])
        else:
            pos_arr = None
            dom_size = 1

        if len(G) == 0:
            return {}
        if len(G) == 1:
            return {nx.utils.arbitrary_element(G.nodes()): center}

        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")

        if k is None and fixed is not None:
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)

        pos = _fruchterman_reingold_barnes_hut_cuda(
            A,
            k,
            pos_arr,
            fixed,
            iterations,
            threshold,
            dim,
            seed,
            theta,
            singleton_mass,
        )

        if fixed is None and scale is not None:
            pos = _rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))
        return pos

    # Fall back to CPU Barnes-Hut (still fast with Numba parallel)
    logg.info(
        f"CUDA not available, using CPU Barnes-Hut for {len(G)} nodes. "
        "For GPU acceleration, use NVIDIA GPU with CUDA toolkit."
    )
    return _fruchterman_reingold_layout_bh(
        G,
        k=k,
        pos=pos,
        fixed=fixed,
        iterations=iterations,
        threshold=threshold,
        weight=weight,
        scale=scale,
        center=center,
        dim=dim,
        seed=seed,
        theta=theta,
        singleton_mass=singleton_mass,
    )


def _fruchterman_reingold_layout_v2(
    G: nx.Graph,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    weight: str = "weight",
    scale: float = 1,
    center: np.ndarray | None = None,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
) -> dict:
    """Numba-accelerated Fruchterman-Reingold layout (mod_fr2).

    Drop-in replacement for ``_fruchterman_reingold_layout`` with
    Numba JIT-compiled force computation and parallel CPU execution.
    First call incurs ~1-2s JIT compilation overhead; subsequent
    calls use the cached compiled kernel.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : float | None, optional
        Optimal distance between nodes. If None, defaults to ``sqrt(1/N)``.
    pos : dict | None, optional
        ``{node: (x, y)}`` initial positions.
    fixed : list | None, optional
        Nodes whose positions should not change.
    iterations : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative position change.
    weight : str, optional
        Edge attribute key for weights.
    scale : float, optional
        Scale factor for final positions.
    center : np.ndarray | None, optional
        Center of the layout.
    dim : int, optional
        Dimension of layout.
    seed : int | np.random.RandomState | None, optional
        Random state for deterministic layouts.

    Returns
    -------
    dict
        ``{node: np.ndarray}`` position mapping.
    """
    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:  # pragma: no cover
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}  # pragma: no cover
        fixed = np.asarray([nfixed[node] for node in fixed])  # pragma: no cover

    if pos is not None:  # pragma: no cover
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    # Always use sparse CSR for Numba kernel
    A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")

    if k is None and fixed is not None:  # pragma: no cover
        nnodes, _ = A.shape
        k = dom_size / np.sqrt(nnodes)

    pos = _fruchterman_reingold_numba(
        A,
        k,
        pos_arr,
        fixed,
        iterations,
        threshold,
        dim,
        seed,
    )

    if fixed is None and scale is not None:
        pos = _rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


def _fruchterman_reingold_layout_gpu(
    G: nx.Graph,
    k: float | None = None,
    pos: dict | None = None,
    fixed: list | None = None,
    iterations: int = 50,
    threshold: float = 1e-4,
    weight: str = "weight",
    scale: float = 1,
    center: np.ndarray | None = None,
    dim: int = 2,
    seed: int | np.random.RandomState | None = None,
    tile_size: int = 4096,
) -> dict:  # pragma: no cover
    """PyTorch GPU-accelerated Fruchterman-Reingold layout (mod_fr2_gpu).

    Automatically selects between dense (fast) and tiled (memory-efficient)
    based on graph size. Uses tiled mode for graphs with >100K nodes.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : float | None, optional
        Optimal distance between nodes. If None, defaults to ``sqrt(1/N)``.
    pos : dict | None, optional
        ``{node: (x, y)}`` initial positions.
    fixed : list | None, optional
        Nodes whose positions should not change.
    iterations : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative position change.
    weight : str, optional
        Edge attribute key for weights.
    scale : float, optional
        Scale factor for final positions.
    center : np.ndarray | None, optional
        Center of the layout.
    dim : int, optional
        Dimension of layout.
    seed : int | np.random.RandomState | None, optional
        Random state for deterministic layouts.
    tile_size : int, optional
        Tile size for tiled mode. Default 4096.

    Returns
    -------
    dict
        ``{node: np.ndarray}`` position mapping.
    """
    torch, device = _detect_torch_device()

    G, center = _process_params(G, center, dim)
    nnodes = len(G)

    if nnodes == 0:
        return {}
    if nnodes == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed])

    if pos is not None:
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(nnodes, dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    # Use tiled mode for large graphs (>100K nodes)
    use_tiled = nnodes > 100_000

    if use_tiled:
        logg.info(
            f"Using tiled GPU layout for {nnodes} nodes (tile_size={tile_size})"
        )
        # Get sparse matrix for tiled mode
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
    else:
        # Dense mode for smaller graphs
        A = nx.to_numpy_array(G, weight=weight).astype(np.float32)

    if k is None and fixed is not None:
        k = dom_size / np.sqrt(nnodes)

    if use_tiled:
        pos = _fruchterman_reingold_torch_tiled(
            A,
            k,
            pos_arr,
            fixed,
            iterations,
            threshold,
            dim,
            torch_module=torch,
            device=device,
            seed=seed,
            tile_size=tile_size,
        )
    else:
        pos = _fruchterman_reingold_torch(
            A,
            k,
            pos_arr,
            fixed,
            iterations,
            threshold,
            dim,
            torch_module=torch,
            device=device,
            seed=seed,
        )

    if fixed is None and scale is not None:
        pos = _rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


def extract_edge_weights(
    vdj: Dandelion | DandelionPolars, expanded_only: bool = False  # noqa: F821
) -> list:
    """
    Retrieve edge weights from graph.

    Parameters
    ----------
    vdj : Dandelion | DandelionPolars
        Dandelion object after `tl.generate_network` has been run.
    expanded_only : bool, optional
        whether to retrieve the edge weights from the expanded only graph or entire graph.

    Returns
    -------
    list
        list of edge weights.
    """
    graph = vdj.graph[1] if expanded_only else vdj.graph[0]

    try:
        if isinstance(graph, ig.Graph):
            weights = graph.es["weight"]
        else:
            _, weights = zip(*nx.get_edge_attributes(graph, "weight").items())
    except (ValueError, KeyError) as e:
        logg.info(
            "{} i.e. the graph does not contain edges. Therefore, edge weights not returned.".format(
                e
            )
        )

    if "weights" in locals():
        return weights


def _build_graphs_networkx(
    vertices: list, edges: pd.DataFrame, min_size: int
) -> tuple[nx.Graph, nx.Graph]:
    """Helper function to build networkx graphs from vertices and edges."""
    G = nx.Graph()
    G.add_nodes_from(vertices)
    if edges is not None:
        src = edges["source"].to_list()
        tgt = edges["target"].to_list()
        wgt = edges["weight"].to_list()
        G.add_weighted_edges_from(zip(src, tgt, wgt))
        if min_size == 2:
            nodes_with_edges = set(src) | set(tgt)
            keep = [v for v in vertices if v in nodes_with_edges]
        elif min_size > 2:
            components = nx.connected_components(G)
            keep = [
                node
                for component in components
                if len(component) >= min_size
                for node in component
            ]
        else:
            keep = vertices
        G_ = G.subgraph(keep).copy()
    else:
        G_ = G.copy()
    return G, G_


def _build_graphs_igraph(
    vertices: list, edges: pd.DataFrame, min_size: int
) -> tuple[nx.Graph, nx.Graph, ig.Graph, ig.Graph]:
    """Helper function to build igraph graphs from vertices and edges."""
    vertex_list = list(vertices)
    vidx = {v: i for i, v in enumerate(vertex_list)}

    if edges is not None:
        src = edges["source"].to_list()
        tgt = edges["target"].to_list()
        wgt = edges["weight"].to_list()
        edge_tuples = [(vidx[s], vidx[t]) for s, t in zip(src, tgt)]
    else:
        edge_tuples, wgt = [], []

    Gi = ig.Graph(n=len(vertex_list), edges=edge_tuples, directed=False)
    Gi.vs["name"] = vertex_list
    if wgt:
        Gi.es["weight"] = wgt
    if edges is not None:
        if min_size == 2:
            degrees = Gi.degree()
            keep_idx = [i for i, d in enumerate(degrees) if d > 0]
        elif min_size > 2:
            components = Gi.connected_components()
            keep_idx = [
                idx
                for comp in components
                if len(comp) >= min_size
                for idx in comp
            ]
        else:
            keep_idx = list(range(Gi.vcount()))
        Gi_ = Gi.induced_subgraph(keep_idx)
    # Convert back to networkx for downstream layout functions.
    # Note: filtering already happened above on the cheap igraph
    # representation, so these conversions are on the *filtered*
    # (much smaller) G_, and on the full G only once.
    G = _igraph_to_networkx_named(Gi)
    G_ = _igraph_to_networkx_named(Gi_)
    return G, G_, Gi, Gi_


def _igraph_to_networkx_named(Gi: ig.Graph) -> nx.Graph:
    """Convert an igraph.Graph to networkx.Graph, using the 'name' vertex
    attribute as node keys instead of integer indices."""
    Gnx = Gi.to_networkx()
    # to_networkx() typically keys nodes by integer index; relabel using
    # the 'name' attribute stored on each vertex.
    mapping = {i: name for i, name in enumerate(Gi.vs["name"])}
    Gnx = nx.relabel_nodes(Gnx, mapping)
    return Gnx
