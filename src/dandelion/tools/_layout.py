import numpy as np
import pandas as pd
import networkx as nx

from scanpy import logging as logg
from typing import Literal

try:
    from networkx.utils import np_random_state as random_state
except ImportError:
    from networkx.utils import random_state

from dandelion.utilities._core import Dandelion


def generate_layout(
    vertices: list | None = None,
    edges: pd.DataFrame | None = None,
    min_size: int = 2,
    weight: str | None = None,
    verbose: bool = True,
    compute_layout: bool = True,
    layout_method: Literal["mod_fr", "sfdp"] = "mod_fr",
    expanded_only: bool = False,
    graphs: tuple[nx.Graph, nx.Graph] = None,
    **kwargs,
) -> tuple[nx.Graph, nx.Graph, dict, dict]:
    """Generate layout.

    Parameters
    ----------
    vertices : list
        list of vertices
    edges : pd.DataFrame, optional
        edge list in a pandas data frame.
    min_size : int, optional
        minimum clone size.
    weight : str | None, optional
        name of weight column.
    verbose : bool, optional
        whether or not to print status
    compute_layout : bool, optional
        whether or not to compute layout.
    layout_method : Literal["mod_fr", "sfdp"], optional
        layout method.
    expanded_only : bool, optional
        whether or not to only compute layout on expanded clones.
    graphs: tuple[nx.Graph, nx.Graph], optional
        tuple of graphs.
    dist_mat : pd.DataFrame | None, optional
        distance matrix.
    **kwargs
        passed to fruchterman_reingold_layout.

    Returns
    -------
    tuple[nx.Graph, nx.Graph, dict, dict]
        graphs and layout positions.
    """
    logg.info("Generating network layout")
    if graphs is None:
        if vertices is not None:
            G = nx.Graph()
            G.add_nodes_from(vertices)
            if edges is not None:
                G.add_weighted_edges_from(
                    [
                        (x, y, z)
                        for x, y, z in zip(
                            edges["source"], edges["target"], edges["weight"]
                        )
                    ]
                )
        G_ = G.copy()
    else:
        G = graphs[0]
        G_ = graphs[1]
    if min_size == 2:
        if edges is not None:
            G_.remove_nodes_from(nx.isolates(G))
        else:
            pass
    elif min_size > 2:
        if edges is not None:
            for component in list(nx.connected_components(G_)):
                if len(component) < min_size:
                    for node in component:
                        G_.remove_node(node)
        else:
            pass
    if compute_layout:
        if layout_method == "mod_fr":
            if not hasattr(generate_layout, "_has_printed"):
                logg.info(
                    "To benefit from faster layout computation, please install graph-tool and use layout_method='sfdp': "
                    "conda install -c conda-forge graph-tool\n"
                    "This message will only be shown once per session."
                )
                generate_layout._has_printed = True
            if not expanded_only:
                if verbose:
                    logg.info("Computing network layout")
                pos = _fruchterman_reingold_layout(G, weight=weight, **kwargs)
            else:
                pos = None
            if verbose:
                logg.info("Computing expanded network layout")
            pos_ = _fruchterman_reingold_layout(G_, weight=weight, **kwargs)
        elif layout_method == "sfdp":
            try:
                from graph_tool.all import sfdp_layout
            except ImportError:
                logg.info(
                    "Please install graph-tool to use sfdp layout:"
                    "conda install -c conda-forge graph-tool"
                )
                nographtool = True
            if "nographtool" in locals():
                if not expanded_only:
                    if verbose:
                        logg.info("Computing network layout")
                    pos = _fruchterman_reingold_layout(
                        G, weight=weight, **kwargs
                    )
                else:
                    pos = None
                if verbose:
                    logg.info("Computing expanded network layout")
                pos_ = _fruchterman_reingold_layout(G_, weight=weight, **kwargs)
            else:
                if not expanded_only:
                    gtg = nx2gt(G)
                    if verbose:
                        logg.info("Computing network layout")
                    posx = sfdp_layout(gtg, **kwargs)
                    pos = dict(
                        zip(list(gtg.vertex_properties["id"]), list(posx))
                    )
                else:
                    pos = None
                gtg_ = nx2gt(G_)
                if verbose:
                    logg.info("Computing expanded network layout")
                posx_ = sfdp_layout(gtg_, **kwargs)
                pos_ = dict(
                    zip(list(gtg_.vertex_properties["id"]), list(posx_))
                )
        if pos is None:
            G = G_
            pos = pos_

        return (G, G_, pos, pos_)
    else:
        return (G, G_, None, None)


# when dealing with a lot of unconnected vertices, the pieces fly out to infinity and the original fr layout can't be
# used
# work around from https://stackoverflow.com/questions/14283341/how-to-increase-node-spacing-for-networkx-spring-layout
# code chunk from networkx's layout.py https://github.com/networkx/networkx/blob/master/networkx/drawing/layout.py
def _process_params(
    G: nx.Graph, center: np.ndarray | None, dim: int
) -> tuple[nx.Graph, np.ndarray]:
    """Some boilerplate code."""
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

    if fixed is not None:
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
        if int(nx.__version__[0]) > 2:
            A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
        else:
            A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype="f")
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    except ValueError:
        A = nx.to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:
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
):
    """Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

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
        length = np.where(length < 0.01, 0.1, length)
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
):
    """Sparse Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e
    try:
        from scipy.sparse import coo_matrix
    except ImportError as e:
        msg = "_sparse_fruchterman_reingold() scipy numpy: http://scipy.org/ "
        raise ImportError(msg) from e
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except AttributeError:
        A = (coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
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
                continue
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
        displacement = displacement - pos / (k * np.sqrt(nnodes))
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
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


def extract_edge_weights(vdj: Dandelion, expanded_only: bool = False) -> list:
    """
    Retrieve edge weights (BCR levenshtein distance) from graph.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object after `tl.generate_network` has been run.
    expanded_only : bool, optional
        whether to retrieve the edge weights from the expanded only graph or entire graph.

    Returns
    -------
    list
        list of edge weights.
    """
    if expanded_only:
        try:
            edges, weights = zip(
                *nx.get_edge_attributes(vdj.graph[1], "weight").items()
            )
        except ValueError as e:
            logg.info(
                "{} i.e. the graph does not contain edges. Therefore, edge weights not returned.".format(
                    e
                )
            )
    else:
        try:
            edges, weights = zip(
                *nx.get_edge_attributes(vdj.graph[0], "weight").items()
            )
        except ValueError as e:
            logg.info(
                "{} i.e. the graph does not contain edges. Therefore, edge weights not returned.".format(
                    e
                )
            )
    if "weights" in locals():
        return weights


# from https://bbengfort.github.io/2016/06/graph-tool-from-networkx/
def nx2gt(nxG: nx.Graph) -> "gt.Graph":
    """Convert a networkx graph to a graph-tool graph."""
    try:
        import graph_tool as gt
    except ImportError:
        raise ImportError(
            "Please install graph-tool: conda install -c conda-forge graph-tool"
        )
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop  # Set the PropertyMap
        gtG.graph_properties[key] = value  # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node, data in list(nxG.nodes(data=True)):
        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
            gtG.vertex_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties["id"] = gtG.new_vertex_property("string")

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst, data in list(nxG.edges(data=True)):
        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)  # Create the PropertyMap
            gtG.edge_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node, data in list(nxG.nodes(data=True)):
        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data["id"] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value  # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in list(nxG.edges(data=True)):
        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value  # ep is short for edge_properties

    # Done, finally!
    return gtG


def get_prop_type(value, key=None):
    """
    Perform typing and value conversion for the graph_tool PropertyMap class.

    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Deal with the value
    if isinstance(value, bool):
        tname = "bool"

    elif isinstance(value, int):
        tname = "float"
        value = float(value)

    elif isinstance(value, float):
        tname = "float"

    elif isinstance(value, dict):
        tname = "object"

    else:
        tname = "string"
        value = str(value)

    return tname, value, key
