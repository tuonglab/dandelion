from __future__ import annotations
import math
import os
import re
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from collections import defaultdict, Counter
from distance import hamming
from itertools import product
from pathlib import Path
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mudata import MuData
    from awkward import Array

from dandelion.utilities._core import Dandelion, load_data
from dandelion.utilities._io import write_airr
from dandelion.utilities._utilities import (
    FALSES,
    VCALL,
    JCALL,
    VCALLG,
    STRIPALLELENUM,
    EMPTIES,
    flatten,
    is_categorical,
    type_check,
    present,
    check_same_celltype,
    Tree,
    sanitize_column,
)


def find_clones(
    vdj: Dandelion | pd.DataFrame,
    identity: dict[str, float] | float = 0.85,
    key: dict[str, str] | str | None = None,
    by_alleles: bool = False,
    key_added: str | None = None,
    recalculate_length: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Dandelion:
    """
    Find clones based on VDJ chain and VJ chain CDR3 junction hamming distance.

    Parameters
    ----------
    vdj : Dandelion | pd.DataFrame
        Dandelion object, pandas DataFrame in changeo/airr format, or file path to changeo/airr file
        after clones have been determined.
    identity : dict[str, float] | float, optional
        junction similarity parameter. Default 0.85. If provided as a dictionary, please use the following
        keys:'ig', 'tr-ab', 'tr-gd'.
    key : dict[str, str] | str | None, optional
        column name for performing clone clustering. `None` defaults to a dictionary where:
            {'ig': 'junction_aa', 'tr-ab': 'junction', 'tr-gd': 'junction'}
        If provided as a string, this key will be used for all loci.
    by_alleles : bool, optional
        whether or not to collapse alleles to genes. `None` defaults to False.
    key_added : str | None, optional
        If specified, this will be the column name for clones. `None` defaults to 'clone_id'
    recalculate_length : bool, optional
        whether or not to re-calculate junction length, rather than rely on parsed assignment (which occasionally is
        wrong). Default is True
    verbose : bool, optional
        whether or not to print progress.
    **kwargs
        Additional arguments to pass to `Dandelion.update_metadata`.

    Returns
    -------
    Dandelion
        Dandelion object with clone_id annotated in `.data` slot and `.metadata` initialized.

    Raises
    ------
    ValueError
        if `key` not found in Dandelion.data.
    """
    start = logg.info("Finding clonotypes")
    pd.set_option("mode.chained_assignment", None)
    if isinstance(vdj, Dandelion):
        dat_ = load_data(vdj._data)
    else:
        dat_ = load_data(vdj)

    clone_key = key_added if key_added is not None else "clone_id"
    dat_[clone_key] = ""

    dat = dat_.copy()
    if "ambiguous" in dat_:
        dat = dat_[dat_["ambiguous"] == "F"].copy()

    locus_log = {"ig": "B", "tr-ab": "abT", "tr-gd": "gdT"}
    locus_dict1 = {"ig": ["IGH"], "tr-ab": ["TRB"], "tr-gd": ["TRD"]}
    locus_dict2 = {"ig": ["IGK", "IGL"], "tr-ab": ["TRA"], "tr-gd": ["TRG"]}
    DEFAULTIDENTITY = {"ig": 0.85, "tr-ab": 1, "tr-gd": 1}

    # key_ = key if key is not None else "junction_aa"  # default

    # if key_ not in dat.columns:
    #     raise ValueError("key {} not found in input table.".format(key_))

    locuses = ["ig", "tr-ab", "tr-gd"]

    # create a default key with ig="junction_aa", tr-ab="junction", tr-gd="junction"
    key_ = (
        {"ig": "junction_aa", "tr-ab": "junction", "tr-gd": "junction"}
        if key is None
        else key
    )

    # quick check
    for locus in locuses:
        locus_1 = locus_dict1[locus]
        locus_2 = locus_dict2[locus]

        dat_vj = dat[dat["locus"].isin(locus_2)].copy()
        dat_vdj = dat[dat["locus"].isin(locus_1)].copy()

        chain_check = check_chains(dat_vdj=dat_vdj, dat_vj=dat_vj)
        if all(~chain_check["All VDJ"]) and all(~chain_check["All VJ"]):
            locuses.remove(locus)

    if len(locuses) > 0:
        for locusx in locuses:
            locus_1 = locus_dict1[locusx]
            locus_2 = locus_dict2[locusx]
            if isinstance(identity, dict):
                if locusx not in identity:
                    identity.update({locusx: DEFAULTIDENTITY[locusx]})
                    warnings.warn(
                        UserWarning(
                            "Identity value for {} chains ".format(locusx)
                            + "not specified in provided dictionary. "
                            + "Defaulting to {} for {} chains.".format(
                                DEFAULTIDENTITY[locusx], locusx
                            )
                        )
                    )
                identity_ = identity[locusx]
            else:
                identity_ = identity

            dat_vj = dat[dat["locus"].isin(locus_2)].copy()
            dat_vdj = dat[dat["locus"].isin(locus_1)].copy()
            chain_check = check_chains(dat_vdj=dat_vdj, dat_vj=dat_vj)
            if dat_vdj.shape[0] > 0:
                vj_len_grp_vdj, seq_grp_vdj = group_sequences(
                    dat_vdj,
                    junction_key=(
                        key_[locusx] if isinstance(key_, dict) else key_
                    ),
                    recalculate_length=recalculate_length,
                    by_alleles=by_alleles,
                    locus=locusx,
                )
                cid_vdj = group_pairwise_hamming_distance(
                    clonotype_vj_len_group=vj_len_grp_vdj,
                    clonotype_sequence_group=seq_grp_vdj,
                    identity=identity_,
                    locus=locus_log[locusx],
                    chain="VDJ",
                    junction_key=(
                        key_[locusx] if isinstance(key_, dict) else key_
                    ),
                    verbose=verbose,
                )
                clone_dict_vdj = rename_clonotype_ids(
                    clonotype_groups=cid_vdj,
                    prefix=locus_log[locusx] + "_VDJ_",
                )
                # add it to the original dataframes
                dat_vdj[clone_key] = pd.Series(clone_dict_vdj)
                # dat[clone_key].update(pd.Series(dat_vdj[clone_key]))
                for i, row in dat_vdj.iterrows():
                    if i in dat.index:
                        dat.at[i, clone_key] = row[clone_key]
            if dat_vj.shape[0] > 0:
                vj_len_grp_vj, seq_grp_vj = group_sequences(
                    dat_vj,
                    junction_key=(
                        key_[locusx] if isinstance(key_, dict) else key_
                    ),
                    recalculate_length=recalculate_length,
                    by_alleles=by_alleles,
                    locus=locusx,
                )
                cid_vj = group_pairwise_hamming_distance(
                    clonotype_vj_len_group=vj_len_grp_vj,
                    clonotype_sequence_group=seq_grp_vj,
                    identity=identity_,
                    locus=locus_log[locusx],
                    chain="VJ",
                    junction_key=(
                        key_[locusx] if isinstance(key_, dict) else key_
                    ),
                    verbose=verbose,
                )
                clone_dict_vj = rename_clonotype_ids(
                    clonotype_groups=cid_vj,
                    prefix=locus_log[locusx] + "_VJ_",
                )
                refine_clone_assignment(
                    dat=dat,
                    clone_key=clone_key,
                    clone_dict_vj=clone_dict_vj,
                    verbose=verbose,
                )
                # dat_[clone_key].update(pd.Series(dat[clone_key]))
            for i, row in dat.iterrows():
                if i in dat_.index:
                    dat_.at[i, clone_key] = row[clone_key]

    # dat_[clone_key].replace('', 'unassigned')
    if os.path.isfile(str(vdj)):
        data_path = Path(vdj)
        write_airr(dat_, data_path.parent / (data_path.stem + "_clone.tsv"))
    if verbose:
        logg.info("Initialising Dandelion object")
    if isinstance(vdj, Dandelion):
        vdj._data[str(clone_key)] = dat_[str(clone_key)]
        vdj.update_metadata(clone_key=str(clone_key))
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'data', contig-indexed AIRR table\n"
                "   'metadata', cell-indexed observations table\n"
            ),
        )
        return vdj
    else:
        out = Dandelion(
            data=dat_,
            clone_key=clone_key,
            verbose=False,
            **kwargs,
        )
        logg.info(
            " finished",
            time=start,
            deep=(
                "Returning Dandelion object: \n"
                "   'data', contig-indexed AIRR table\n"
                "   'metadata', cell-indexed observations table\n"
            ),
        )
        return out


def transfer(
    adata: AnnData | MuData,
    vdj: Dandelion,
    expanded: bool = False,
    gex_key: str | None = None,
    vdj_key: str | None = None,
    clone_key: str | None = None,
    collapse_nodes: bool = False,
    overwrite: bool | list[str] | str | None = None,
    obs: bool = True,
    obsm: bool = True,
    uns: bool = True,
    obsp: bool = True,
) -> None:
    """
    Transfer data in Dandelion slots to AnnData, updating `.obs`, `.uns`, `.obsm`, and `.obsp`.
    Transfers both graphs:
      - graph[0] -> adata.obsm['X_vdj_all']
      - graph[1] -> adata.obsm['X_vdj_expanded']
    The `expanded` flag controls which graph becomes the *main* adjacency written to
    adata.obsp['connectivities'] / ['distances'] (but both graphs are stored).

    Parameters
    ----------
    adata : AnnData | MuData
        AnnData object or `MuData` object.
    vdj : Dandelion
        Dandelion object.
    expanded : bool, optional
        Whether or not to transfer the embedding with all cells with BCR (False) or only for expanded clones (True).
    gex_key : str | None, optional
        prefix for stashed RNA connectivities and distances.
    vdj_key : str | None, optional
        prefix for stashed VDJ connectivities and distances.
    clone_key : str | None, optional
        column name of clone/clonotype ids. Only used for integration with scirpy.
    collapse_nodes : bool, optional
        Whether or not to transfer a cell x cell or clone x clone connectivity matrix into `.uns`. Only used for
        integration with scirpy.
    overwrite : bool | list[str] | str | None, optional
        Whether or not to overwrite existing anndata columns. Specifying a string indicating column name or
        list of column names will overwrite that specific column(s).
    """
    start = logg.info("Transferring network")

    # if the provide adata is an MuData, we need to transfer to mudata.mod['gex']
    # but we don’t want to add mudata as a dependency here, so we do a duck-typing check
    if hasattr(adata, "mod"):
        if "airr" in adata.mod:
            recipient = adata.mod["airr"]
        else:
            raise ValueError(
                "Provided AnnData is a MuData object without 'airr' modality."
            )
    # we just associate recipient to adata directly
    else:
        recipient = adata
    # --- 1) metadata -> adata.obs (preserve original overwrite semantics) ---
    if obs:
        for x in vdj._metadata.columns:
            if x not in recipient.obs.columns:
                recipient.obs[x] = pd.Series(vdj._metadata[x])
            elif overwrite is True:
                recipient.obs[x] = pd.Series(vdj._metadata[x])
            if type_check(vdj._metadata, x):
                recipient.obs[x] = recipient.obs[x].replace(np.nan, "No_contig")
            if recipient.obs[x].dtype == "bool":
                recipient.obs[x] = recipient.obs[x].astype(str)

        # explicit overwrite list/string handling (matches original)
        if (overwrite is not None) and (overwrite is not True):
            if not isinstance(overwrite, list):
                overwrite = [overwrite]
            for ow in overwrite:
                recipient.obs[ow] = pd.Series(vdj._metadata[ow])
                if type_check(vdj._metadata, ow):
                    recipient.obs[ow] = recipient.obs[ow].replace(
                        np.nan, "No_contig"
                    )

    # also check that all the cells in vdj are in recipient
    common_cells = recipient.obs_names.intersection(vdj._metadata.index)
    # subset to common cells only
    vdj = vdj[vdj._metadata.index.isin(common_cells)].copy()

    # If there's no graph, we're done with metadata only
    if vdj.graph is None:
        logg.info(
            " finished", time=start, deep=("updated `.obs` with `.metadata`\n")
        )
        return

    # --- 2) prepare neighbor keys and stash RNA neighbors/connectivities if present ---
    neighbors_key = "neighbors"
    skip_stash = neighbors_key not in recipient.uns
    if obsp:
        gex_key = "gex" if gex_key is None else gex_key
        g_connectivities_key = f"{gex_key}_connectivities"
        g_distances_key = f"{gex_key}_distances"
        vdj_key = "vdj" if vdj_key is None else vdj_key
        v_connectivities_key = f"{vdj_key}_connectivities"
        v_distances_key = f"{vdj_key}_distances"

        # Stash RNA connectivities/distances before we overwrite connectivities/distances
        if not skip_stash:
            # preferred stash from obsp if exist
            recipient.obsp[g_connectivities_key] = recipient.obsp[
                "connectivities"
            ].copy()
            recipient.obsp[g_distances_key] = recipient.obsp["distances"].copy()
            g_neighbors_key = f"{gex_key}_{neighbors_key}"
            recipient.uns[g_neighbors_key] = recipient.uns[neighbors_key].copy()

    # --- 3) Convert both graphs ---
    graph_connectivities, graph_distances = {}, {}
    # handle graph[0] and graph[1]
    for idx in (0, 1):
        G = None
        if vdj.graph is not None:
            try:
                G = vdj.graph[idx]
            except Exception:
                pass

        if G is not None:
            graph_connectivities[idx], graph_distances[idx] = (
                _graph_to_matrices(G, recipient, None)
            )

    # handle precomputed distances (sparse or DataFrame)
    if getattr(vdj, "distances", None) is not None:
        graph_connectivities[2], graph_distances[2] = _graph_to_matrices(
            None, recipient, vdj.distances
        )

    # Determine main graph index
    main_idx = 1 if expanded else 0
    if main_idx not in graph_connectivities:
        main_idx = next(iter(graph_connectivities.keys()))

    if obsp:
        # --- 4) Update recipient.obsp ---
        recipient.obsp["connectivities"] = graph_connectivities[main_idx].copy()
        recipient.obsp["distances"] = graph_distances[main_idx].copy()

        # store the all (graph[0]) and expanded graph (graph[1]) if available
        if 0 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_all"] = (
                graph_connectivities[0].copy()
            )
            recipient.obsp[f"{v_distances_key}_all"] = graph_distances[0].copy()
        if 1 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_expanded"] = (
                graph_connectivities[1].copy()
            )
            recipient.obsp[f"{v_distances_key}_expanded"] = graph_distances[
                1
            ].copy()
        if 2 in graph_connectivities:
            recipient.obsp[f"{v_connectivities_key}_full"] = (
                graph_connectivities[2].copy()
            )
            recipient.obsp[f"{v_distances_key}_full"] = graph_distances[
                2
            ].copy()
        recipient.uns[neighbors_key] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "params": {
                "n_neighbors": 1,
                "method": "custom",
                "metric": "precomputed",
            },
        }

    if uns:
        # --- 5) Clone-level mapping (scirpy compatible) ---
        clone_key = clone_key if clone_key is not None else "clone_id"

        if not collapse_nodes:
            for idx in graph_connectivities:
                graph_connectivities[idx][
                    graph_connectivities[idx].nonzero()
                ] = 1
            cell_indices = {
                str(i): np.array([k])
                for i, k in zip(
                    range(0, len(recipient.obs_names)), recipient.obs_names
                )
            }
            bin_conn = graph_connectivities[main_idx]
        else:
            invalid = [
                "",
                "unassigned",
                "NaN",
                "NA",
                "nan",
                "None",
                "none",
                None,
            ]
            cell_indices = Tree()
            for x, y in recipient.obs[clone_key].items():
                if y not in invalid:
                    cell_indices[y][x].value = 1
            cell_indices = {
                str(x): np.array(list(r))
                for x, r in zip(
                    range(0, len(cell_indices)), cell_indices.values()
                )
            }
            bin_conn = np.zeros([len(cell_indices), len(cell_indices)])
            np.fill_diagonal(bin_conn, 1)
            bin_conn = csr_matrix(bin_conn)

        recipient.uns[clone_key] = {
            # this is a symmetrical, pairwise, sparse distance matrix of clonotypes
            # the matrix is offset by 1, i.e. 0 = no connection, 1 = distance 0
            "distances": bin_conn,
            # '0' refers to the row/col index in the `distances` matrix
            # (numeric index, but needs to be strbecause of h5py)
            # np.array(["cell1", "cell2"]) points to the rows in `recipient.obs`
            "cell_indices": cell_indices,
        }

    if obsm:
        # --- 6) Layouts ---
        if vdj.layout is not None:
            stored_embeddings = {}
            for idx, obsm_name in (
                (0, "X_vdj_all"),
                (1, "X_vdj_expanded"),
            ):
                try:
                    layout = vdj.layout[idx]
                except Exception:
                    continue
                if layout is None:
                    continue
                coord = pd.DataFrame.from_dict(layout, orient="index")
                coord = coord.reindex(index=recipient.obs_names).fillna(np.nan)
                if coord.shape[1] >= 2:
                    embedding = coord.iloc[:, :2].to_numpy(dtype=np.float32)
                else:
                    col0 = (
                        coord.iloc[:, 0]
                        .to_numpy(dtype=np.float32)
                        .reshape(-1, 1)
                    )
                    col1 = np.zeros_like(col0)
                    embedding = np.hstack([col0, col1])

                recipient.obsm[obsm_name] = embedding
                stored_embeddings[idx] = obsm_name

            # Set the "active" embedding safely
            main_idx = 1 if expanded else 0
            active_obsm = stored_embeddings.get(main_idx)
            if active_obsm is not None:
                recipient.obsm["X_vdj"] = recipient.obsm[active_obsm].copy()

    # break up the message depending on which parts were executed
    message_parts = []
    if obs:
        message_parts += [f"updated `.obs` with `.metadata`\n"]
    if obsm:
        message_parts += [
            f"wrote `.obsm['X_vdj']` and `.obsm['X_vdj_expanded']`\n"
        ]
    if obsp:
        message_parts += [
            f"wrote adata.obsp['connectivities'] & ['distances'] from graph[{main_idx}]\n",
            "stored RNA matrices under rna_* keys (stashed)\n",
            f"stored vdj matrices under '{v_connectivities_key}' (+ '_expanded' and + '_full' if available)\n",
        ]
    if uns:
        message_parts += [f"added `.uns['{clone_key}']` clone-level mapping"]

    # --- 7) Done ---
    logg.info(
        " finished",
        time=start,
        deep="".join(message_parts),
    )


tf = transfer  # alias for transfer


def _graph_to_matrices(
    G: nx.Graph | None,
    adata: AnnData,
    distances: csr_matrix | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Convert a graph or provided distances into properly aligned sparse
    connectivities and distances matrices.

    Rules:
    - If G is provided, convert edges → sparse distance matrix.
    - If a CSR distance matrix is provided, must have `._index_names`.
    - If a DataFrame is provided, use its index/columns.
    - Reindex to `adata.obs_names` without dense conversion.
    - Compute connectivities as exp(-d) on non-zero entries.
    - Add tiny self-edge if matrix is entirely empty.
    """

    target_names = list(adata.obs_names)
    n = len(target_names)
    name_to_new = {name: i for i, name in enumerate(target_names)}
    # CASE A: Build distances from a NetworkX graph
    if distances is None and G is not None:
        # Build COO arrays directly
        edges = list(G.edges(data=True))
        if not edges:
            distances = csr_matrix((n, n), dtype=np.float32)
        else:
            u, v, w = zip(*[(u, v, d.get("weight", 1.0)) for u, v, d in edges])
            # Filter edges where both nodes exist in target
            mask_u = np.array([node in name_to_new for node in u])
            mask_v = np.array([node in name_to_new for node in v])
            mask = mask_u & mask_v  # vectorized AND
            u = np.array(u)[mask]
            v = np.array(v)[mask]
            w = np.array(w, dtype=np.float32)[mask]

            # Map names to target indices
            u_idx = np.array([name_to_new[x] for x in u])
            v_idx = np.array([name_to_new[x] for x in v])

            # Make symmetric
            rows = np.concatenate([u_idx, v_idx])
            cols = np.concatenate([v_idx, u_idx])
            vals = np.concatenate([w, w])
            vals += 1.0

            distances = csr_matrix(
                (vals, (rows, cols)), shape=(n, n), dtype=np.float32
            )

    # CASE B: distances provided as a csr_matrix with _index_names
    elif isinstance(distances, csr_matrix):
        old_names = np.array(distances._index_names)
        coo = distances.tocoo()

        # Map old names to target indices, missing names → -1
        old_row_names = old_names[coo.row]
        old_col_names = old_names[coo.col]

        row_idx = np.array(
            [name_to_new.get(name, -1) for name in old_row_names]
        )
        col_idx = np.array(
            [name_to_new.get(name, -1) for name in old_col_names]
        )

        # Keep only edges where both row and col exist in target
        mask = (row_idx >= 0) & (col_idx >= 0)
        rows = row_idx[mask]
        cols = col_idx[mask]
        vals = coo.data[mask]
        vals += 1.0

        distances = csr_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=np.float32
        )
    # Build connectivities = exp(-d) for non-zero entries
    connectivities = distances.copy()
    if connectivities.nnz > 0:
        connectivities.data = np.exp(-connectivities.data)
        connectivities.data = np.clip(connectivities.data, 1e-45, np.inf)

    distances.data -= 1.0

    # Ensure matrix is not completely empty
    if connectivities.nnz == 0:
        connectivities = connectivities.tolil()
        distances = distances.tolil()
        connectivities[0, 0] = 1e-10
        distances[0, 0] = 0.0
        connectivities = connectivities.tocsr()
        distances = distances.tocsr()

    return connectivities, distances


def clone_view(
    adata: AnnData,
    mode: Literal["all", "expanded", "full", "gex"] | None = "expanded",
    connectivities_key: str | None = None,
    distances_key: str | None = None,
    embedding_key: str | None = None,
):
    """
    Swap the 'active' connectivities, distances, and optionally embedding in AnnData.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    mode : Literal["all", "expanded", "full", "gex"] | None, optional
        If specified, set the active connectivities/distances/embedding to one of the preset modes.
    connectivities_key : str | None, optional
        The key in `.obsp` to set as active `.obsp["connectivities"]` if `mode` is None.
    distances_key : str | None, optional
        The key in `.obsp` to set as active `.obsp["distances"]` if `mode` is None.
    embedding_key : str | None, optional
        If specified, set `.obsm["X_vdj"]` to `.obsm[embedding_key]` if `mode` is None.
    """
    if mode is None:
        # use the other key directly
        if connectivities_key in adata.obsp:
            adata.obsp["connectivities"] = adata.obsp[connectivities_key].copy()
        else:
            raise KeyError(f"{connectivities_key} not found in adata.obsp")

        if distances_key in adata.obsp:
            adata.obsp["distances"] = adata.obsp[distances_key].copy()
        else:
            raise KeyError(f"{distances_key} not found in adata.obsp")

        if embedding_key is not None:
            if embedding_key in adata.obsm:
                adata.obsm["X_vdj"] = adata.obsm[embedding_key].copy()
            else:
                raise KeyError(f"{embedding_key} not found in adata.obsm")
    else:
        if mode == "gex":
            conn_key = f"{mode}_connectivities"
            dist_key = f"{mode}_distances"
            neighbors_key = f"{mode}_neighbors"
            emb_key = None
        else:
            conn_key = f"vdj_connectivities_{mode}"
            dist_key = f"vdj_distances_{mode}"
            neighbors_key = None
            emb_key = f"X_vdj_{mode}" if mode != "full" else None
        adata.obsp["connectivities"] = adata.obsp[conn_key].copy()
        adata.obsp["distances"] = adata.obsp[dist_key].copy()
        if emb_key is not None:
            adata.obsm["X_vdj"] = adata.obsm[emb_key].copy()
        if neighbors_key is not None:
            adata.uns["neighbors"] = adata.uns[neighbors_key].copy()
        else:
            adata.uns["neighbors"] = {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
                "params": {
                    "n_neighbors": 1,
                    "method": "custom",
                    "metric": "precomputed",
                },
            }


def tabulate_clone_sizes(
    metadata_: pd.DataFrame, clonesize_dict: dict, clonekey: str
) -> pd.Series:
    """Tabulate clone sizes."""
    return pd.Series(
        dict(
            zip(
                metadata_.index,
                [
                    str(y) if pd.notnull(y) else str(0)
                    for y in [
                        (
                            sorted(
                                list(
                                    {clonesize_dict[c_] for c_ in c.split("|")}
                                ),
                                key=lambda x: (
                                    int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x)
                                ),
                                reverse=True,
                            )[0]
                            if "|" in c
                            else clonesize_dict[c]
                        )
                        for c in metadata_[str(clonekey)]
                    ]
                ],
            )
        )
    )


def clone_size(
    vdj: Dandelion | AnnData | MuData,
    groupby: str | None = None,
    max_size: int | None = None,
    clone_key: str | None = None,
    key_added: str | None = None,
) -> None:
    """
    Quantify clone sizes, globally or per group.

    If `groupby` is specified, clone sizes and proportions are calculated
    within each group separately. Each cell is then annotated with the size,
    proportion, and frequency category based on sizes similar to scRepertoire.
    If a cell belongs to multiple clones (e.g., multiple chains assigned
    to different clones), the largest clone is used for annotation.

    Parameters
    ----------
    vdj : Dandelion | AnnData | MuData
        VDJ data.
    groupby : str | None, optional
        Column in metadata to group by before calculating clone sizes.
        If None, calculates global clone sizes.
    max_size : int | None, optional
        Clip clone size values at this maximum.
    clone_key : str | None, optional
        Column specifying clone identifiers. Defaults to 'clone_id'.
    key_added : str | None, optional
        Prefix for new metadata column names.
    """
    # --- Select metadata
    if hasattr(vdj, "mod"):
        metadata_ = vdj.mod["airr"].obs.copy()
    elif isinstance(vdj, AnnData):
        metadata_ = vdj.obs.copy()
    elif isinstance(vdj, Dandelion):
        metadata_ = vdj._metadata.copy()

    clone_key = "clone_id" if clone_key is None else clone_key
    if clone_key not in metadata_.columns:
        raise KeyError(f"Column '{clone_key}' not found in metadata.")

    # --- Expand multi-clone entries
    tmp = metadata_[clone_key].astype(str).str.split("|", expand=True).stack()
    # drop None/No_contig entries
    tmp = tmp[~tmp.isin(["No_contig", "unassigned"] + EMPTIES)]
    tmp = tmp.reset_index(drop=False)
    tmp.columns = ["cell_id", "tmp", clone_key]

    # --- Compute clone sizes (global or per group)
    if groupby is None:
        clonesize = tmp[clone_key].value_counts()
        prop = clonesize / metadata_.shape[0]
    else:
        # Merge with groupby column using cell_id as key
        # Reset index to make cell_id a regular column for merging
        metadata_with_index = metadata_.reset_index()
        metadata_with_index = metadata_with_index.rename(
            columns={"index": "cell_id"}
        )

        tmp = tmp.merge(
            metadata_with_index[["cell_id", groupby]], on="cell_id", how="left"
        )
        clonesize = tmp.groupby([groupby, clone_key]).size()
        group_sizes = metadata_[groupby].value_counts()

        # Calculate proportion correctly for each group
        prop_dict = {}
        for grp in clonesize.index.get_level_values(0).unique():
            group_clones = clonesize.loc[grp]
            group_total = group_sizes[grp]
            for clone_id, size in group_clones.items():
                prop_dict[(grp, clone_id)] = size / group_total

        # Create Series with MultiIndex
        prop = pd.Series(prop_dict)
        prop.index = pd.MultiIndex.from_tuples(
            prop.index, names=[groupby, clone_key]
        )

    # --- Create max_size categories if specified
    if max_size is not None:

        def categorize_size(size):
            if pd.isna(size):
                return np.nan
            if size < max_size:
                return str(int(size))
            else:
                return f">= {max_size}"

        clonesize_cat = clonesize.apply(categorize_size)
        clonesize_cat_map = clonesize_cat.to_dict()

    # --- Define clone frequency bins
    bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    labels = ["Rare", "Small", "Medium", "Large", "Hyperexpanded"]
    if groupby is None:
        prop_bins = pd.cut(prop, bins=bins, labels=labels, include_lowest=True)
    else:
        # Apply pd.cut to the entire Series at once, preserving the MultiIndex
        prop_bins = pd.cut(prop, bins=bins, labels=labels, include_lowest=True)

    # --- Build lookup maps
    size_map = clonesize.to_dict()
    prop_map = prop.to_dict()
    cat_map = prop_bins.to_dict()

    # --- Assign to each cell
    cell_sizes = []
    cell_props = []
    cell_cats = []
    cell_size_cats = [] if max_size is not None else None

    for i, row in metadata_.iterrows():
        clone_ids = str(row[clone_key])
        # Check for empty/invalid entries
        if pd.isna(clone_ids) or clone_ids in [
            "No_contig",
            "unassigned",
            "None",
            "nan",
        ]:
            cell_sizes.append(np.nan)
            cell_props.append(np.nan)
            cell_cats.append(np.nan)
            if max_size is not None:
                cell_size_cats.append(np.nan)
            continue

        clones = clone_ids.split("|")

        if groupby is None:
            # look up sizes directly
            sizes = [size_map.get(c, np.nan) for c in clones]
            props = [prop_map.get(c, np.nan) for c in clones]
            cats = [cat_map.get(c, np.nan) for c in clones]
            if max_size is not None:
                size_cats = [clonesize_cat_map.get(c, np.nan) for c in clones]
        else:
            grp = row[groupby]
            # Use tuple keys for grouped lookups
            sizes = [size_map.get((grp, c), np.nan) for c in clones]
            props = [prop_map.get((grp, c), np.nan) for c in clones]
            cats = [cat_map.get((grp, c), np.nan) for c in clones]
            if max_size is not None:
                size_cats = [
                    clonesize_cat_map.get((grp, c), np.nan) for c in clones
                ]

        # take the largest available clone (by numeric size)
        if len(sizes) == 0 or all(pd.isna(sizes)):
            cell_sizes.append(np.nan)
            cell_props.append(np.nan)
            cell_cats.append(np.nan)
            if max_size is not None:
                cell_size_cats.append(np.nan)
        else:
            max_idx = np.nanargmax(sizes)
            cell_sizes.append(sizes[max_idx])
            cell_props.append(props[max_idx])
            cell_cats.append(cats[max_idx])
            if max_size is not None:
                cell_size_cats.append(size_cats[max_idx])

    metadata_[f"{clone_key}_size"] = cell_sizes
    metadata_[f"{clone_key}_size_prop"] = cell_props
    metadata_[f"{clone_key}_size_category"] = cell_cats
    if max_size is not None:
        metadata_[f"{clone_key}_size_max_{max_size}"] = cell_size_cats

    # --- Write results back to object
    col_key = key_added if key_added is not None else clone_key

    if isinstance(vdj, Dandelion):
        vdj._metadata[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj._metadata[f"{col_key}_size_prop"] = metadata_[
            f"{clone_key}_size_prop"
        ]
        vdj._metadata[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj._metadata[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]
    elif isinstance(vdj, AnnData):
        vdj.obs[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj.obs[f"{col_key}_size_prop"] = metadata_[f"{clone_key}_size_prop"]
        vdj.obs[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj.obs[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]
    elif hasattr(vdj, "mod"):
        vdj.mod["airr"].obs[f"{col_key}_size"] = metadata_[f"{clone_key}_size"]
        vdj.mod["airr"].obs[f"{col_key}_size_prop"] = metadata_[
            f"{clone_key}_size_prop"
        ]
        vdj.mod["airr"].obs[f"{col_key}_size_category"] = metadata_[
            f"{clone_key}_size_category"
        ]
        if max_size is not None:
            vdj.mod["airr"].obs[f"{col_key}_size_max_{max_size}"] = metadata_[
                f"{clone_key}_size_max_{max_size}"
            ]


def clone_overlap(
    vdj: Dandelion | AnnData,
    groupby: str,
    min_clone_size: int | None = None,
    weighted_overlap: bool = False,
    clone_key: str | None = None,
) -> pd.DataFrame:
    """
    A function to tabulate clonal overlap for input as a circos-style plot.

    Parameters
    ----------
    vdj : Dandelion | AnnData
        Dandelion or AnnData object.
    groupby : str
        column name in obs/metadata for collapsing to columns in the clone_id x groupby data frame.
    min_clone_size : int | None, optional
        minimum size of clone for plotting connections. Defaults to 2 if left as None.
    weighted_overlap : bool, optional
        if True, instead of collapsing to overlap to binary, overlap will be returned as the number of cells.
        In the future, there will be the option to use something like a jaccard index.
    clone_key : str | None, optional
        column name for clones. `None` defaults to 'clone_id'.

    Returns
    -------
    pd.DataFrame
        clone_id x groupby overlap :class:`pandas.core.frame.DataFrame'.

    Raises
    ------
    ValueError
        if min_clone_size is 0.
    """
    start = logg.info("Calculating clone overlap")
    if isinstance(vdj, Dandelion):
        data = vdj._metadata.copy()
    elif isinstance(vdj, AnnData):
        data = vdj.obs.copy()
    elif isinstance(vdj, MuData):
        data = vdj.mod["airr"].obs.copy()

    if min_clone_size is None:
        min_size = 2
    else:
        min_size = int(min_clone_size)

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    # get rid of problematic rows that appear because of category conversion?
    allgroups = list(data[groupby].unique())
    data = data[
        ~(
            data[clone_].isin(
                [np.nan, "nan", "NaN", "No_contig", "unassigned", "None", None]
            )
        )
    ]

    # prepare a summary table
    datc_ = data[clone_].str.split("|", expand=True).stack()
    datc_ = pd.DataFrame(datc_)
    datc_.reset_index(drop=False, inplace=True)
    datc_.columns = ["cell_id", "tmp", clone_]
    datc_.drop("tmp", inplace=True, axis=1)
    datc_ = datc_[
        ~(
            datc_[clone_].isin(
                [
                    "",
                    np.nan,
                    "nan",
                    "NaN",
                    "No_contig",
                    "unassigned",
                    "None",
                    None,
                ]
            )
        )
    ]
    dictg_ = dict(data[groupby])
    datc_[groupby] = [dictg_[l] for l in datc_["cell_id"]]

    overlap = pd.crosstab(datc_[clone_], datc_[groupby])
    for x in allgroups:
        if x not in overlap:
            overlap[x] = 0

    if min_size == 0:
        raise ValueError("min_size must be greater than 0.")
    if not weighted_overlap:
        if min_size > 2:
            overlap[overlap < min_size] = 0
            overlap[overlap >= min_size] = 1
        elif min_size == 2:
            overlap[overlap >= min_size] = 1

    overlap.index.name = None
    overlap.columns.name = None

    if isinstance(vdj, AnnData):
        vdj.uns["clone_overlap"] = overlap.copy()
        logg.info(
            " finished",
            time=start,
            deep=("Updated AnnData: \n" "   'uns', clone overlap table"),
        )
    else:
        return overlap


def clustering(
    distance_dict: dict, threshold: float, sequences_dict: dict[str, str]
) -> dict:
    """Clustering the sequences."""
    out_dict = {}
    # find out the unique indices in this subset
    i_unique = list(set(flatten(distance_dict)))
    # for every pair of i1,i2 is their dictance smaller than the thresholdeshold?
    i_pair_d = {
        (i1, i2): (
            distance_dict[(i1, i2)] <= threshold
            if (i1, i2) in distance_dict
            else False
        )
        for i1, i2 in product(i_unique, repeat=2)
    }
    i_pair_d.update(
        {
            (i2, i1): (
                distance_dict[(i2, i1)] <= threshold
                if (i2, i1) in distance_dict
                else False
            )
            for i1, i2 in product(i_unique, repeat=2)
        }
    )
    # so which indices should not be part of a clone?
    canbetogether = defaultdict(list)
    for ii1, ii2 in product(i_unique, repeat=2):
        if i_pair_d[(ii1, ii2)] or i_pair_d[(ii2, ii1)]:
            if (ii1, ii2) in distance_dict:
                canbetogether[ii1].append((ii1, ii2))
                canbetogether[ii2].append((ii1, ii2))
            elif (ii2, ii1) in distance_dict:
                canbetogether[ii2].append((ii2, ii1))
                canbetogether[ii1].append((ii2, ii1))
        else:
            if (ii1, ii2) or (ii2, ii1) in distance_dict:
                canbetogether[ii1].append(())
                canbetogether[ii2].append(())
    for x in canbetogether:
        canbetogether[x] = list({y for y in canbetogether[x] if len(y) > 0})
    # convert the indices to sequences
    for x in canbetogether:
        if len(canbetogether[x]) > 0:
            out_dict[sequences_dict[x]] = tuple(
                sorted(
                    set(
                        list(
                            [
                                sequences_dict[y]
                                for y in flatten(canbetogether[x])
                            ]
                        )
                        + [sequences_dict[x]]
                    )
                )
            )
        else:
            out_dict[sequences_dict[x]] = tuple([sequences_dict[x]])
    return out_dict


def productive_ratio(
    adata: AnnData,
    vdj: Dandelion,
    groupby: str,
    groups: list[str] | None = None,
    locus: Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"] = "TRB",
):
    """
    Compute the cell-level productive/non-productive contig ratio.

    Only the contig with the highest umi count in a cell will be used for this
    tabulation.

    Returns inplace AnnData with `.uns['productive_ratio']`.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata (`.obs`).
    vdj : Dandelion
        Dandelion object holding the repertoire data (`.data`).
    groupby : str
        Name of column in `AnnData.obs` to return the row tabulations.
    groups : list[str] | None, optional
        Optional list of categories to return.
    locus : Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"], optional
        One of the accepted locuses to perform the tabulation
    """
    start = logg.info("Tabulating productive ratio")
    vdjx = vdj[(vdj._data.cell_id.isin(adata.obs_names))].copy()
    if "ambiguous" in vdjx._data:
        tmp = vdjx[
            (vdjx._data.locus == locus) & (vdjx._data.ambiguous == "F")
        ].copy()
    else:
        tmp = vdjx[(vdjx._data.locus == locus)].copy()

    if groups is None:
        if is_categorical(adata.obs[groupby]):
            groups = list(adata.obs[groupby].cat.categories)
        else:
            groups = list(set(adata.obs[groupby]))
    df = tmp._data.drop_duplicates(subset="cell_id")
    dict_df = dict(zip(df.cell_id, df.productive))
    res = pd.DataFrame(
        columns=["productive", "non-productive", "total"],
        index=groups,
    )

    adata.obs[locus + "_productive"] = pd.Series(dict_df)
    for i in range(res.shape[0]):
        cell = res.index[i]
        res.loc[cell, "total"] = sum(adata.obs[groupby] == cell)
        if res.loc[cell, "total"] > 0:
            res.loc[cell, "productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(["T"])
                )
                / res.loc[cell, "total"]
                * 100
            )
            res.loc[cell, "non-productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(
                        [
                            "F",
                        ]
                    )
                )
                / res.loc[cell, "total"]
                * 100
            )
    res[groupby] = res.index
    res["productive+non-productive"] = res["productive"] + res["non-productive"]
    out = {"results": res, "locus": locus, "groupby": groupby}
    adata.uns["productive_ratio"] = out
    logg.info(
        " finished",
        time=start,
        deep=("Updated AnnData: \n" "   'uns', productive_ratio"),
    )


def vj_usage_pca(
    adata: AnnData,
    groupby: str,
    min_size: int = 20,
    mode: Literal["B", "abT", "gdT"] = "abT",
    use_vdj_v: bool = True,
    use_vdj_j: bool = True,
    use_vj_v: bool = True,
    use_vj_j: bool = True,
    transfer_mapping=None,
    n_comps: int = 30,
    groups: list[str] | None = None,
    allowed_chain_status: list[str] | None = [
        "Single pair",
        "Extra pair",
        "Extra pair-exception",
        "Orphan VDJ-exception",
    ],
    verbose=False,
    **kwargs,
) -> AnnData:
    """
    Extract productive V/J gene usage from single cell data and compute PCA.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata with Dandelion VDJ info transferred.
    groupby : str
        Column name in `adata.obs` to groupby as observations for PCA.
    min_size : int, optional
        Minimum cell size numbers to keep for computing the final matrix. Defaults to 20.
    mode : Literal["B", "abT", "gdT"], optional
        Mode for extract the V/J genes.
    use_vdj_v : bool, optional
        Whether to use V gene from VDJ contigs for tabulation. Defaults to True.
    use_vdj_j : bool, optional
        Whether to use J gene from VDJ contigs for tabulation. Defaults to True.
    use_vj_v : bool, optional
        Whether to use V genes from VJ contigs for tabulation. Defaults to True.
    use_vj_j : bool, optional
        Whether to use J genes from VJ contigs for tabulation. Defaults to True.
    transfer_mapping : None, optional
        If provided, the columns will be mapped to the output AnnData from the original AnnData.
    n_comps : int, optional
        Number of principal components to compute. Defaults to 30.
    groups : list[str] | None, optional
        If provided, only the following groups/categories will be used for computing the PCA.
    allowed_chain_status : list[str] | None, optional
        If provided, only the ones in this list are kept from the `chain_status` column.
        Defaults to ["Single pair", "Extra pair", "Extra pair-exception", "Orphan VDJ-exception"].
    verbose : bool, optional
        Whether to display progress
    **kwargs
        Additional keyword arguments passed to `scanpy.pp.pca`.

    Returns
    -------
    AnnData
        AnnData object with obs as groups and V/J genes as features.
    """
    start = logg.info("Computing PCA for V/J gene usage")
    # filtering
    if allowed_chain_status is not None:
        adata_ = adata[
            adata.obs["chain_status"].isin(allowed_chain_status)
        ].copy()

    if groups is not None:
        adata_ = adata_[adata_.obs[groupby].isin(groups)].copy()
    # build config
    gene_config = {
        "vdj_v": dict(
            enabled=use_vdj_v,
            main=f"v_call_{mode}_VDJ_main",
            full=f"v_call_{mode}_VDJ",
        ),
        "vdj_j": dict(
            enabled=use_vdj_j,
            main=f"j_call_{mode}_VDJ_main",
            full=f"j_call_{mode}_VDJ",
        ),
        "vj_v": dict(
            enabled=use_vj_v,
            main=f"v_call_{mode}_VJ_main",
            full=f"v_call_{mode}_VJ",
        ),
        "vj_j": dict(
            enabled=use_vj_j,
            main=f"j_call_{mode}_VJ_main",
            full=f"j_call_{mode}_VJ",
        ),
    }
    if not any(cfg["enabled"] for cfg in gene_config.values()):
        raise ValueError("At least one of the use_vj/vdj_v/j must be True.")

    # Determine which groups to keep
    cell_counts = adata_.obs[groupby].value_counts()
    keep_groups = cell_counts[cell_counts >= min_size].index

    # collect gene lists
    gene_lists = {}
    for key, cfg in gene_config.items():
        if cfg["enabled"]:
            uniq = adata_.obs[cfg["main"]].unique().tolist()
            gene_lists[key] = [
                g for g in uniq if g not in ("None", "No_contig")
            ]
        else:
            gene_lists[key] = []

    all_genes = [g for genes in gene_lists.values() for g in genes]

    # initialise results df
    vdj_df = pd.DataFrame(
        index=keep_groups, columns=all_genes, dtype=float
    ).fillna(0)

    # count genes per group
    for group in tqdm(
        vdj_df.index,
        desc="Tabulating V/J gene usage",
        disable=not verbose,
    ):
        group_mask = adata_.obs[groupby] == group
        obs_group = adata_.obs.loc[group_mask]

        for key, cfg in gene_config.items():
            if not cfg["enabled"]:
                continue

            counts = Counter(obs_group[cfg["full"]])
            for gene in gene_lists[key]:
                vdj_df.loc[group, gene] = counts.get(gene, 0)

    # normalize each chain separately
    for key, cfg in gene_config.items():
        if not cfg["enabled"]:
            continue

        cols = gene_lists[key]
        colsum = vdj_df[cols].sum(axis=1)
        vdj_df.loc[:, cols] = vdj_df[cols].div(colsum, axis=0) * 100

    # Create new AnnData + PCA
    obs_df = pd.DataFrame(index=vdj_df.index)
    obs_df["cell_type"] = vdj_df.index
    obs_df["cell_count"] = cell_counts.loc[vdj_df.index]

    vdj_adata = AnnData(
        X=vdj_df.values,
        obs=obs_df,
        var=pd.DataFrame(index=vdj_df.columns),
    )

    sc.pp.pca(vdj_adata, n_comps=n_comps, use_highly_variable=False, **kwargs)

    # Transfer old obs columns to new AnnData
    if transfer_mapping is not None:
        collapsed = adata_.obs.drop_duplicates(subset=groupby)
        for to in transfer_mapping:
            mapping = dict(zip(collapsed[groupby], collapsed[to]))
            vdj_adata.obs[to] = vdj_adata.obs.index.map(mapping)

    logg.info(
        " finished",
        time=start,
        deep=("Returned AnnData: \n" "   'obsm', X_pca for V/J gene usage"),
    )
    return vdj_adata


def group_sequences(
    input_vdj: pd.DataFrame,
    junction_key: str,
    recalculate_length: bool = True,
    by_alleles: bool = False,
    locus: Literal["ig", "tr-ab", "tr-gd"] = "ig",
):
    """
    Groups sequence IDs with the same V and J genes, and then splits the groups based on the lengths of the sequences.

    Parameters
    ----------
    input_vdj : pd.DataFrame
        The input data frame.
    junction_key : str
        The name of the column in the input data that contains the junction sequences.
    recalculate_length : bool, optional
        Whether to recalculate the lengths of the junction sequences.
    by_alleles : bool, optional
        Whether to group sequence IDs by their V and J gene alleles, rather than by only the V and J genes.
    locus : Literal["ig", "tr-ab", "tr-gd"], optional
        The locus of the input data. One of "ig", "tr-ab", or "tr-gd".

    Returns
    -------
    vj_len_grp : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence length.
    seq_grp : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence.

    Raises
    ------
    ValueError
        Raised when the sequence length column is not found in the input table.
    """
    locus_log1_dict = {"ig": "IGH", "tr-ab": "TRB", "tr-gd": "TRD"}
    # retrieve the J genes and J genes
    if not by_alleles:
        if VCALLG in input_vdj.columns:
            V = [re.sub(STRIPALLELENUM, "", str(v)) for v in input_vdj[VCALLG]]
        else:
            V = [re.sub(STRIPALLELENUM, "", str(v)) for v in input_vdj[VCALL]]
        J = [re.sub(STRIPALLELENUM, "", str(j)) for j in input_vdj[JCALL]]
    else:
        if VCALLG in input_vdj.columns:
            V = [str(v) for v in input_vdj[VCALLG]]
        else:
            V = [str(v) for v in input_vdj[VCALL]]
        J = [str(j) for j in input_vdj[JCALL]]
    # collapse the alleles to just genes
    V = [",".join(list(set(v.split(",")))) for v in V]
    J = [",".join(list(set(j.split(",")))) for j in J]
    seq = dict(zip(input_vdj.index, input_vdj[junction_key]))
    if recalculate_length:
        # Set length to 0 for null/NaN values, otherwise calculate from sequence
        seq_length = [
            len(str(l)) if pd.notnull(l) else 0 for l in input_vdj[junction_key]
        ]
    else:
        try:
            seq_length = [l for l in input_vdj[junction_key + "_length"]]
        except:
            raise ValueError(
                "{} not found in {} input table.".format(
                    junction_key + "_length", locus_log1_dict[locus]
                )
            )
    seq_length_dict = dict(zip(input_vdj.index, seq_length))
    # Create a dictionary and group sequence ids with same V and J genes
    V_J = dict(zip(input_vdj.index, zip(V, J)))
    vj_grp = defaultdict(list)
    for key, val in sorted(V_J.items()):
        vj_grp[val].append(key)
    # and now we split the groups based on lengths of the seqs
    vj_len_grp = Tree()
    seq_grp = Tree()
    for g in vj_grp:
        # first, obtain what's the unique lengths
        jlen = []
        for contig_id in vj_grp[g]:
            jlen.append(seq_length_dict[contig_id])
        setjlen = list(set(jlen))
        # then for each unique length, we add the contigs to a new tree if it matches the length
        # and also the actual seq sequences into a another one
        for s in setjlen:
            for contig_id in vj_grp[g]:
                jlen_ = seq_length_dict[contig_id]
                if jlen_ == s:
                    vj_len_grp[g][s][contig_id].value = 1
                    seq_grp[g][s][seq[contig_id]].value = 1
                    for c in [contig_id]:
                        vj_len_grp[g][s][c] = seq[c]
    return vj_len_grp, seq_grp


def group_pairwise_hamming_distance(
    clonotype_vj_len_group: Tree,
    clonotype_sequence_group: Tree,
    identity: float,
    locus: str,
    chain: Literal["VDJ", "VJ"],
    junction_key: str,
    verbose: bool = True,
) -> Tree:
    """
    Group clonotypes by pairwise hamming distance and generate a nested dictionary of clonotype groups.

    Parameters
    ----------
    clonotype_vj_len_group : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence length.
    clonotype_sequence_group : Tree
        A nested dictionary that groups sequence IDs by V and J gene, and then by sequence.
    identity : float
        The identity threshold for grouping sequences.
    locus : str
        The locus of the chain.
    chain : Literal["VDJ", "VJ"]
        The chain type.
    verbose : bool, optional
        Whether to show the progress bar, by default True.

    Returns
    -------
    Tree
        A nested dictionary containing contigs assigned to clonotype groups.
    """
    clones = Tree()

    # for each seq group, calculate the hamming distance matrix
    for g in tqdm(
        clonotype_sequence_group,
        desc=f"Finding clones based on {locus} cell {chain} chains using {junction_key}".format(),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        for l in clonotype_sequence_group[g]:
            seq_ = list(clonotype_sequence_group[g][l])
            tdarray = np.array(seq_).reshape(-1, 1)
            d_mat = squareform(pdist(tdarray, lambda x, y: hamming(x[0], y[0])))
            # then calculate what the acceptable threshold is for each length of sequence
            tr = math.floor(int(l) * (1 - identity))
            # convert diagonal and upper triangle to zeroes
            d_mat = np.tril(d_mat)
            np.fill_diagonal(d_mat, 0)
            # get the coordinates/indices of seqs to match against the threshold later
            indices_temp = []
            indices = []
            indices_temp = [list(x) for x in np.tril_indices_from(d_mat)]
            indices = list(zip(indices_temp[0], indices_temp[1]))
            # if there's more than 1 contig, remove the diagonal
            if len(indices) > 1:
                for pairs in indices:
                    # remove diagonals
                    if pairs[0] == pairs[1]:
                        indices.remove(pairs)
            indices_j = []
            # use the coordinates/indices to retrieve the seq sequences
            for p in range(0, len(indices)):
                a1, b1 = indices[p]
                indices_j.append(seq_[a1])
                indices_j.append(seq_[b1])
            # convert the distance matrix to coordinate (source) and distance (target) and create it
            # as a dictionary
            source, target = d_mat.nonzero()
            source_target = list(zip(source.tolist(), target.tolist()))
            if len(source) == 0 & len(target) == 0:
                source_target = list([(0, 0)])
            dist = {}
            for st in source_target:
                dist.update({st: d_mat[st]})
            if d_mat.shape[0] > 1:
                seq_tmp_dict = clustering(dist, tr, seq_)
            else:
                seq_tmp_dict = {seq_[0]: tuple([seq_[0]])}
            # sort the list so that clones that are larger have a smaller number
            clones_tmp = sorted(
                list(set(seq_tmp_dict.values())),
                key=len,
                reverse=True,
            )
            for x in range(0, len(clones_tmp)):
                clones[g][l][x + 1] = clones_tmp[x]
    # now to retrieve the contig ids that are grouped together
    cid = Tree()
    for g in clones:
        for l in clones[g]:
            # retrieve the clone 'numbers'
            for c in clones[g][l]:
                grp_seq = clones[g][l][c]
                for key, value in clonotype_vj_len_group[g][l].items():
                    if value in grp_seq:
                        cid[g][l][c][key].value = 1

    return cid


def rename_clonotype_ids(
    clonotype_groups: Tree,
    prefix: str = "",
) -> dict[str, str]:
    """
    Renames clonotype IDs to numerical barcode system.

    Parameters
    ----------
    clonotype_groups : Tree
        A nested dictionary that containing clonotype groups of contigs.
    prefix : str, optional
        Prefix to append to front, if necessary.
    Returns
    -------
    dict[str, str]
        A dictionary that maps sequence IDs to clonotype IDs.
    """
    clone_dict = {}
    first_key = []
    for k1 in clonotype_groups.keys():
        first_key.append(k1)
    first_key = list(set(first_key))
    first_key_dict = dict(zip(first_key, range(1, len(first_key) + 1)))
    for g in clonotype_groups:
        second_key = []
        for k2 in clonotype_groups[g].keys():
            second_key.append(k2)
        second_key = list(set(second_key))
        second_key_dict = dict(zip(second_key, range(1, len(second_key) + 1)))
        for l in clonotype_groups[g]:
            third_key = []
            for k3 in clonotype_groups[g][l].keys():
                third_key.append(k3)
            third_key = list(set(third_key))
            third_key_dict = dict(zip(third_key, range(1, len(third_key) + 1)))
            for key, value in dict(clonotype_groups[g][l]).items():
                for v in value:
                    if type(v) is int:
                        break
                    clone_dict[v] = (
                        prefix
                        + str(first_key_dict[g])
                        + "_"
                        + str(second_key_dict[l])
                        + "_"
                        + str(third_key_dict[key])
                    )

    return clone_dict


def refine_clone_assignment(
    dat: pd.DataFrame,
    clone_key: str,
    clone_dict_vj: dict,
    verbose: bool = True,
) -> None:
    """
    Refines the clone assignment of VDJ sequences based on their VJ chain pairing.

    Parameters
    ----------
    dat : pd.DataFrame
        The input data frame containing the full VDJ/VJ data.
    clone_key : str
        The name of the column in the input data that contains the clone IDs.
    clone_dict_vj : dict
        A dictionary for the VJ chains for us to map sequence IDs to clone IDs.
    verbose : bool, optional
        Whether to print progress messages.
    """
    cellclonetree = Tree()
    seqcellclonetree = Tree()
    for c, s, z in zip(dat["cell_id"], dat["sequence_id"], dat[clone_key]):
        seqcellclonetree[c][s].value = 1
        if pd.notnull(z):
            cellclonetree[c][z].value = 1

    for c in cellclonetree:
        cellclonetree[c] = list(cellclonetree[c])

    fintree = Tree()
    for c in tqdm(
        cellclonetree,
        desc="Refining clone assignment based on VJ chain pairing ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        suffix = [
            clone_dict_vj[x] for x in seqcellclonetree[c] if x in clone_dict_vj
        ]
        fintree[c] = []
        if len(suffix) > 0:
            for cl in cellclonetree[c]:
                if present(cl):
                    for s in suffix:
                        if check_same_celltype(cl, s):
                            fintree[c].append(
                                cl + "_" + "".join(s.split("_", 1)[1])
                            )
                        else:
                            fintree[c].append(cl + "|" + s)
        else:
            for cl in cellclonetree[c]:
                if present(cl):
                    fintree[c].append(cl)
        fintree[c] = "|".join(fintree[c])
    dat[clone_key] = [fintree[x] for x in dat["cell_id"]]
    for i, row in dat.iterrows():  # is this going to be slow...?
        if not present(row[clone_key]):
            if i in clone_dict_vj and i in dat.index:
                dat.at[i, clone_key] = clone_dict_vj[i]


def check_chains(dat_vdj: pd.DataFrame, dat_vj: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of whether the chain is orphan or not.

    Parameters
    ----------
    dat_vdj : pd.DataFrame
        Input dataframe containing VDJ chains of a particular locus.
    dat_vj : pd.DataFrame
        Input dataframe containing VJ chains of a particular locus.

    Returns
    -------
    pd.DataFrame
        Output dataframe containing chain status.
    """

    vj_check = pd.crosstab(dat_vj.cell_id, dat_vj.locus).apply(sum, axis=1)
    vj_check[vj_check > 1] = 1
    vdj_check = pd.crosstab(dat_vdj.cell_id, dat_vdj.locus).apply(sum, axis=1)
    vdj_check[vdj_check > 1] = 1
    chain_check = pd.concat([vdj_check, vj_check], axis=1)
    chain_check.columns = ["VDJ", "VJ"]
    chain_check["Orphan VDJ"] = pd.notnull(chain_check["VDJ"]) & (
        pd.isnull(chain_check["VJ"])
    )
    chain_check["Orphan VJ"] = pd.notnull(chain_check["VJ"]) & (
        pd.isnull(chain_check["VDJ"])
    )
    chain_check["All VDJ"] = pd.notnull(chain_check["VDJ"])
    chain_check["All VJ"] = pd.notnull(chain_check["VJ"])
    chain_check["VDJ"] = (
        (~chain_check["Orphan VDJ"])
        & (~chain_check["Orphan VJ"])
        & (pd.notnull(chain_check["VDJ"]))
    )
    chain_check["VJ"] = (
        (~chain_check["Orphan VDJ"])
        & (~chain_check["Orphan VJ"])
        & (pd.notnull(chain_check["VJ"]))
    )
    return chain_check


def vdj_sample(
    vdj: Dandelion,
    size: int,
    adata: AnnData | MuData | None = None,
    p: list[float] | np.ndarray[float] | None = None,
    force_replace: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[Dandelion, AnnData] | Dandelion:
    """
    Resample vdj data and corresponding AnnData to a specified size.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object containing VDJ data.
    size : int
        Desired size for resampling.
    adata : AnnData | MuData | None, optional
        AnnData or MuData object corresponding to the gene expression data.
    p : list[float] | np.ndarray[float] | None, optional
        Drawing probabilities for each cell, must sum to 1. If None, uniform probabilities are used.
    force_replace : bool, optional
        Whether to force sampling with replacement, by default False.
    random_state : int | np.random.RandomState | None, optional
        Random state for reproducibility, by default None.


    Returns
    -------
    tuple[Dandelion, AnnData] | Dandelion
        Resampled Dandelion and AnnData objects if adata is provided, otherwise only Dandelion.
    """
    logg.info("Resampling to {} cells.".format(str(size)))
    if adata is None:
        replace = True if size > vdj._metadata.shape[0] else False
        if force_replace:
            replace = True
        keep_cells = vdj._metadata.sample(
            size, replace=replace, random_state=random_state, weights=p
        )
        keep_cells = list(keep_cells.index)
    else:
        # check if MuData and extract the gex modality
        if hasattr(adata, "mod"):
            adata = adata.mod["gex"].copy()
        else:
            adata = adata.copy()
        # ensure only cells present in both vdj and adata are sampled
        common_cells = list(
            set(vdj._metadata.index).intersection(set(adata.obs_names))
        )
        adata = adata[adata.obs_names.isin(common_cells)].copy()
        vdj = vdj[vdj._metadata.index.isin(common_cells)].copy()
        replace = True if size > vdj._metadata.shape[0] else False
        if force_replace:
            replace = True
        # use scanpy to sample
        sc.pp.sample(adata, n=size, replace=replace, rng=random_state, p=p)
        keep_cells = list(adata.obs_names)

    # get the .data without ambiguous assignments
    if "ambiguous" in vdj._data:
        vdj_dat = vdj._data[vdj._data["ambiguous"].isin(FALSES)].copy()
    else:
        vdj_dat = vdj._data.copy()

    vdj_dat = vdj_dat[vdj_dat["cell_id"].isin(keep_cells)].copy()

    if replace:
        # sample with replacement
        cell_counts = Counter(keep_cells)

        # Only process cells that appear more than once
        duplicated_cells = {
            cell: count for cell, count in cell_counts.items() if count > 1
        }

        if duplicated_cells:
            # Separate data for duplication
            vdj_dat_to_duplicate = vdj_dat[
                vdj_dat["cell_id"].isin(duplicated_cells.keys())
            ].copy()
            vdj_dat_to_keep = vdj_dat[
                ~vdj_dat["cell_id"].isin(duplicated_cells.keys())
            ].copy()

            # Create duplicates for both dat and adata in one loop
            all_duplicated_vdj = []

            for cell_id, count in duplicated_cells.items():
                # Duplicate dat rows
                cell_rows = vdj_dat_to_duplicate[
                    vdj_dat_to_duplicate["cell_id"] == cell_id
                ].copy()

                for i in range(count):
                    suffix = f"-{str(i)}" if i > 0 else ""

                    # Add dat rows
                    temp_rows = cell_rows.copy()
                    if suffix:
                        temp_rows["cell_id"] = temp_rows["cell_id"] + suffix
                        temp_rows["sequence_id"] = (
                            temp_rows["sequence_id"] + suffix
                        )
                    all_duplicated_vdj.append(temp_rows)

            # Combine everything back together
            vdj_dat = pd.concat(
                [vdj_dat_to_keep] + all_duplicated_vdj, ignore_index=True
            )

    # reinitialise a copy of the sampled dandelion object using vdj_dat
    vdj = Dandelion(vdj_dat)
    if adata is not None:
        adata.obs_names_make_unique()
        if hasattr(adata, "mod"):
            # if MuData, update the gex modality
            return vdj, to_scirpy(vdj, gex_adata=adata)
        else:
            return vdj, adata
    else:
        return vdj


def to_scirpy(
    vdj: Dandelion,
    transfer: bool = False,
    to_mudata: bool = True,
    gex_adata: AnnData | None = None,
    key: tuple[str, str] = ("gex", "airr"),
    **kwargs,
) -> AnnData | MuData:
    """
    Convert Dandelion data to scirpy-compatible format.

    Parameters
    ----------
    vdj : Dandelion
        The Dandelion object containing the data to be converted.
    transfer : bool, optional
        Whether to transfer additional information from Dandelion to the converted data. Defaults to False.
    to_mudata : bool, optional
        Whether to convert the data to MuData format instead of AnnData. Defaults to True.
        If converting to AnnData, it will assert that the same cell_ids and .obs_names are present in the `gex_adata` provided.
    gex_adata : AnnData, optional
        An existing AnnData object to be used as the base for the converted data if provided.
    key : tuple[str, str], optional
        A tuple specifying the keys for the 'gex' and 'airr' fields in the converted data. Defaults to ("gex", "airr").
    **kwargs
        Additional keyword arguments passed to `scirpy.io.read_airr`.

    Returns
    -------
    AnnData | MuData
        The converted data in either AnnData or MuData format.
    """
    # if gex_adata is provided, make sure to only transfer cells that are present in both
    # we will only filter the vdj data to match gex_adata
    if gex_adata is not None:
        vdj = vdj[vdj.metadata_names.isin(gex_adata.obs_names)].copy()
        tmp_gex = gex_adata.copy()
        if not to_mudata:
            tf(
                tmp_gex, vdj, obs=False, uns=True, obsp=False, obsm=False
            )  # so that the slots are properly filled
    else:
        tmp_gex = None

    if "umi_count" not in vdj._data and "duplicate_count" in vdj._data:
        vdj._data["umi_count"] = vdj._data["duplicate_count"]
    for h in [
        "sequence",
        "rev_comp",
        "sequence_alignment",
        "germline_alignment",
        "v_cigar",
        "d_cigar",
        "j_cigar",
    ]:
        if h not in vdj._data:
            vdj._data[h] = None

    airr, obs = to_ak(vdj._data, **kwargs)
    if to_mudata:
        airr_adata = _create_anndata(airr, obs)
        if tmp_gex is not None:
            tf(airr_adata, vdj, obs=False, uns=True, obsp=False, obsm=False)
        mdata = _create_mudata(tmp_gex, airr_adata, key)
        if transfer:
            tf(mdata, vdj)
        return mdata
    else:
        adata = _create_anndata(airr, obs, tmp_gex)
        if transfer:
            tf(adata, vdj)
        return adata


def from_scirpy(data: AnnData | MuData) -> Dandelion:
    """
    Convert data from scirpy format to Dandelion format.

    Parameters
    ----------
    data : AnnData | MuData
        The input data in scirpy format.

    Returns
    -------
    Dandelion
        The converted data in Dandelion format.
    """
    if not isinstance(data, AnnData):
        data = data.mod["airr"]
    data = data.copy()
    data.obsm["airr"]["cell_id"] = data.obs.index
    df = from_ak(data.obsm["airr"])
    vdj = Dandelion(df, verbose=False)
    # Reverse transfer (recover metadata + clone graph)
    _reverse_transfer(data, vdj)
    return vdj


def _reverse_transfer(
    data: AnnData | MuData,
    dandelion: Dandelion,
    clone_key: str = "clone_id",
) -> None:
    """
    Reverse-transfer scirpy data (AnnData/MuData) into a Dandelion object.

    Pulls metadata, clone mappings, graphs, and embeddings from scirpy's structure.

    Parameters
    ----------
    data : AnnData | MuData
        Input scirpy object (AnnData or MuData with .mod['airr']).
    dandelion : Dandelion
        The Dandelion object to update in place.
    clone_key : str, optional
        Key under .uns containing scirpy clone-level mapping (default: 'clone_id').
    """
    # --- Handle MuData case ---
    if hasattr(data, "mod"):
        if "airr" not in data.mod:
            raise ValueError(
                "MuData object must contain an 'airr' modality for scirpy data."
            )
        adata = data.mod["airr"]
    else:
        adata = data

    # --- Copy metadata ---
    for col in adata.obs:
        if col not in dandelion._metadata.columns:
            dandelion._metadata[col] = adata.obs[col]

    # --- Extract clone-level connection info ---
    if clone_key in adata.uns:
        clone_uns = adata.uns[clone_key]
        distances = clone_uns["distances"]
        cell_indices = clone_uns["cell_indices"]
        # --- Rebuild graph ---
        G = nx.from_scipy_sparse_array(distances)
        # Relabel nodes: scirpy stores numeric keys ("0", "1", ...) mapped to arrays of cell_ids
        mapping = {}
        for k, v in cell_indices.items():
            k_int = int(k)
            if isinstance(v, (list, np.ndarray)):
                # If clone node has multiple cells, store them all in node attribute
                mapping[k_int] = str(v[0]) if len(v) > 0 else str(k)
                G.nodes[k_int]["cells"] = list(v)
            else:
                mapping[k_int] = str(v)
                G.nodes[k_int]["cells"] = [v]
        G = nx.relabel_nodes(G, mapping)

        # Store the graph
        dandelion.graph = [G, None]

    # map the obs back to data as well
    dandelion.update_data()


def from_ak(airr: Array) -> pd.DataFrame:
    """
    Convert an AIRR-formatted array to a pandas DataFrame.

    Parameters
    ----------
    airr : Array
        The AIRR-formatted array to be converted.

    Returns
    -------
    pd.DataFrame
        The converted pandas DataFrame.

    Raises
    ------
    KeyError
        If `sequence_id` not found in the data.
    """
    import awkward as ak

    df = ak.to_dataframe(airr)
    # check if 'sequence_id' column does not exist or if any value in 'sequence_id' is NaN
    if "sequence_id" not in df.columns or df["sequence_id"].isnull().any():
        df_reset = df.reset_index()

        # create a new 'sequence_id' column
        df_reset["sequence_id"] = df_reset.apply(
            lambda row: f"{row['cell_id']}_contig_{row['subentry'] + 1}", axis=1
        )

        # set 'entry' and 'subentry' back as the index
        df = df_reset.set_index(["entry", "subentry"])

    if "sequence_id" in df.columns:
        df.set_index("sequence_id", drop=False, inplace=True)
    if "cell_id" not in df.columns:
        df["cell_id"] = [c.split("_contig")[0] for c in df["sequence_id"]]

    return df


def to_ak(
    data: pd.DataFrame,
    **kwargs,
) -> tuple[Array, pd.DataFrame]:
    """
    Convert data from a DataFrame to an AnnData object with AIRR format.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    **kwargs
        Additional keyword arguments passed to `scirpy.io.read_airr`.

    Returns
    -------
    tuple[Array, pd.DataFrame]
        A tuple containing the AIRR-formatted data as an ak.Array and the cell-level attributes as a pd.DataFrame.
    """

    try:
        import scirpy as ir
    except:
        raise ImportError("Please install scirpy to use this function.")

    adata = ir.io.read_airr(data, **kwargs)

    return adata.obsm["airr"], adata.obs


def _create_anndata(
    airr: Array,
    obs: pd.DataFrame,
    adata: AnnData | None = None,
) -> AnnData:
    """
    Create an AnnData object with the given AIRR array and observation data.

    Parameters
    ----------
    airr : Array
        The AIRR array.
    obs : pd.DataFrame
        The observation data.
    adata : AnnData | None, optional
        An existing AnnData object to update. If None, a new AnnData object will be created.

    Returns
    -------
    AnnData
        The AnnData object with the AIRR array and observation data.
    """
    obsm = {"airr": airr}
    temp = AnnData(X=None, obs=obs, obsm=obsm)

    if adata is None:
        adata = temp
    else:
        cell_names = adata.obs_names.intersection(temp.obs_names)
        adata = adata[adata.obs_names.isin(cell_names)].copy()
        temp = temp[temp.obs_names.isin(cell_names)].copy()
        adata.obsm = dict() if adata.obsm is None else adata.obsm
        adata.obsm.update(temp.obsm)

    return adata


def _create_mudata(
    gex: AnnData,
    adata: AnnData,
    key: tuple[str, str] = ("gex", "airr"),
) -> MuData:
    """
    Create a MuData object from the given AnnData objects.

    Parameters
    ----------
    gex : AnnData
        The AnnData object containing gene expression data.
    adata : AnnData
        The AnnData object containing additional data.
    key : tuple[str, str], optional
        The keys to use for the gene expression and additional data in the MuData object. Defaults to ("gex", "airr").

    Returns
    -------
    MuData
        The created MuData object.

    Raises
    ------
    ImportError
        If the mudata package is not installed.
    """

    try:
        import mudata
    except ImportError:
        raise ImportError("Please install mudata. pip install mudata")
    if gex is not None:
        return mudata.MuData({key[0]: gex, key[1]: adata})
    return mudata.MuData({key[1]: adata})


def concat(
    arrays: list[pd.DataFrame | Dandelion] | dict[pd.DataFrame | Dandelion],
    check_unique: bool = True,
    collapse_cells: bool = True,
    sep: str = "_",
    suffixes: list[str] | None = None,
    prefixes: list[str] | None = None,
    remove_trailing_hyphen_number: bool = False,
    verbose: bool = True,
) -> Dandelion:
    """
    Concatenate data frames and return as Dandelion object.

    If both suffixes and prefixes are `None` and check_unique is True, then a sequential number suffix will be appended.

    Parameters
    ----------
    arrays : list[pd.DataFrame | Dandelion] | dict[pd.DataFrame | Dandelion]
        List or dictionary of Dandelion objects or pandas DataFrames to concatenate.
    check_unique : bool, optional
        Check the new index for duplicates. Otherwise defer the check until necessary.
        Setting to False will improve the performance of this method.
    collapse_cells : bool, optional
        whether or not to collapse multiple contigs per cell into one row in the
        metadata. By default True.
    sep : str, optional
        the separator to append suffix/prefix.
    suffixes : list[str] | None, optional
        List of suffixes to append to sequence_id and cell_id.
    prefixes : list[str] | None, optional
        List of prefixes to append to sequence_id and cell_id.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    verbose : bool, optional
        Whether to print the messages, by default True.

    Returns
    -------
    Dandelion
        concatenated Dandelion object

    Raises
    ------
    ValueError
        if both prefixes and suffixes are provided.
    """
    if (suffixes is not None) and (prefixes is not None):
        raise ValueError("Please provide only prefixes or suffixes, not both.")

    if suffixes is not None:
        if len(arrays) != len(suffixes):
            raise ValueError(
                "Please provide the same number of suffixes as the number of objects to concatenate."
            )

    if prefixes is not None:
        if len(arrays) != len(prefixes):
            raise ValueError(
                "Please provide the same number of prefixes as the number of objects to concatenate."
            )
    # first convert dict to list if necessary
    if isinstance(arrays, dict):
        arrays = [arrays[x].copy() for x in arrays]
    # first, check if all input are dandelion instances
    ddl_check = [True if isinstance(x, Dandelion) else False for x in arrays]
    if all(ddl_check):
        vdjs_ = [vdj.copy() for vdj in arrays]
    else:
        # first check if any of the input arrays are compatible
        ddl_check2 = [
            (
                True
                if isinstance(x, Dandelion)
                else True if isinstance(x, pd.DataFrame) else False
            )
            for x in arrays
        ]
        if all(ddl_check2):
            vdjs_ = [
                (
                    x.copy()
                    if isinstance(x, Dandelion)
                    else Dandelion(x, verbose=False)
                )
                for x in arrays
            ]
        else:
            raise ValueError(
                "All input arrays must be either Dandelion instances or pandas DataFrames."
            )

        # create a check here if v_call_genotyped is only in some of the data
    # if it's not uniformly present, create "v_call_genotyped" column with values from "v_call" for the missing ones.

    # let's do a check now of all the metadata indices that there are no duplicates
    # this list will double up as the metadata index order after concatenation
    tmp_meta_names, tmp_data_names = [], []
    for tmp in vdjs_:
        tmp_meta_names.extend(list(tmp.metadata_names))
        tmp_data_names.extend(list(tmp.data_names))

    if collapse_cells:
        # we should preserve the order but remove duplicates
        tmp_meta_names = list(dict.fromkeys(tmp_meta_names))

    if len(tmp_meta_names) != len(set(tmp_meta_names)):
        metadata_index_order = None
    else:
        # we will use this list to reindex the metadata after concatenation
        metadata_index_order = tmp_meta_names

    if len(tmp_data_names) != len(set(tmp_data_names)):
        data_index_order = None
    else:
        # we will use this list to reindex the data after concatenation
        data_index_order = tmp_data_names

    # now, if check_unique is True, we will append suffixes/prefixes to both metadata and data indices
    if check_unique:
        if metadata_index_order is None and data_index_order is None:
            # store the modified names as well
            metadata_index_order, data_index_order = [], []
            for i in range(0, len(vdjs_)):
                # this will always sync to the sequence_id in .data
                if (suffixes is None) and (prefixes is None):
                    vdjs_[i].add_cell_suffix(
                        str(i),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif suffixes is not None:
                    vdjs_[i].add_cell_suffix(
                        str(suffixes[i]),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif prefixes is not None:
                    vdjs_[i].add_cell_prefix(
                        str(prefixes[i]),
                        sep=sep,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                metadata_index_order.extend(list(vdjs_[i].metadata_names))
                data_index_order.extend(list(vdjs_[i].data_names))
        elif data_index_order is None:
            data_index_order = []
            for i in range(0, len(vdjs_)):
                # this will always sync to the sequence_id in .data
                if (suffixes is None) and (prefixes is None):
                    vdjs_[i].add_sequence_suffix(
                        str(i),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif suffixes is not None:
                    vdjs_[i].add_sequence_suffix(
                        str(suffixes[i]),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                elif prefixes is not None:
                    vdjs_[i].add_sequence_prefix(
                        str(prefixes[i]),
                        sep=sep,
                        sync=False,
                        remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    )
                data_index_order.extend(list(vdjs_[i].data_names))
    else:
        # don't add suffixes/prefixes, but check if indices are unique
        if metadata_index_order is None or data_index_order is None:
            raise ValueError(
                "Cell/contig indices are not unique. Please set check_unique=True to append suffixes/prefixes or ensure unique indices before concatenation."
            )

    # now, instead of just concatenating the metadata as is, we should use dandelion to reinitialise the metadata from scratch, and then add the missing columns, filling out any missing values appropriately
    # first, obtain all the metadata column names
    all_meta_cols = set()
    for vdj_ in vdjs_:
        all_meta_cols.update(set(vdj_._metadata.columns))

    genotyped_v_call = [
        True for vdj in vdjs_ if "v_call_genotyped" in vdj._data
    ]
    if len(genotyped_v_call) > 0:
        if len(genotyped_v_call) != len(vdjs_):
            if verbose:
                # print a warning
                logg.info(
                    "For consistency, 'v_call_genotyped' will be used where available. Filling missing values from 'v_call'."
                )
            for i in range(0, len(vdjs_)):
                if "v_call_genotyped" not in vdjs_[i]._data:
                    vdjs_[i]._data["v_call_genotyped"] = vdjs_[i]._data[
                        "v_call"
                    ]

    arrays_ = [vdj._data for vdj in vdjs_]
    vdj_concat = Dandelion(pd.concat(arrays_), verbose=False)
    # find out if there are missing metadata in the initialised vdj_concat.metadata
    vdj_meta_cols = set(vdj_concat._metadata.columns)
    # find out what are the missing columns from all_meta_cols
    missing_meta_cols = all_meta_cols - vdj_meta_cols
    if len(missing_meta_cols) > 0:
        # print(missing_meta_cols)
        # now, for each missing column, we need to fill in the values from the original vdjs_ using .at to avoid alignment issues
        for col in missing_meta_cols:
            # create an empty series with None first - sanitisation later will take care of the dtypes
            vdj_concat._metadata[col] = pd.Series(
                [None] * vdj_concat._metadata.shape[0],
                index=vdj_concat._metadata.index,
            )
            for vdj_ in vdjs_:
                for idx in vdj_._metadata.index:
                    if idx in vdj_concat._metadata.index:
                        if col in vdj_._metadata.columns:
                            vdj_concat._metadata.at[idx, col] = (
                                vdj_._metadata.at[idx, col]
                            )
    # now check the dtype of each of the vdj_concat._metadata columns. if it's object, change pd.null to ""
    for col in vdj_concat._metadata:
        if vdj_concat._metadata[col].dtype == object:
            vdj_concat._metadata[col] = sanitize_column(
                vdj_concat._metadata[col], "string"
            )
    # finally, reorder the metadata and data according to the original order
    vdj_concat._metadata = vdj_concat._metadata.loc[metadata_index_order]

    return vdj_concat
