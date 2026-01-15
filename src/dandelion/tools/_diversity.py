import numpy as np
import networkx as nx
import pandas as pd

from anndata import AnnData
from collections import defaultdict
from joblib import Parallel, delayed
from plotnine import (
    aes,
    geom_line,
    ggplot,
    ggtitle,
    labs,
    options,
    scale_color_manual,
    scale_linetype_manual,
    theme_classic,
    xlab,
    ylab,
)
from tqdm import tqdm
from scanpy import logging as logg
from scipy.special import gammaln
from scipy.optimize import curve_fit
from typing import Literal

from dandelion.external.skbio._chao1 import chao1
from dandelion.external.skbio._gini import gini_index
from dandelion.external.skbio._shannon import shannon
from dandelion.tools._network import (
    clone_centrality,
    clone_degree,
    generate_network,
)
from dandelion.utilities._core import Dandelion
from dandelion.utilities._utilities import flatten


def clone_rarefaction(
    data: Dandelion | AnnData,
    groupby: str,
    clone_key: str | None = None,
    palette: list[str] | None = None,
    figsize: tuple[float, float] = (5, 3),
    chain_status_include=[
        "Single pair",
        "Orphan VDJ",
        "Orphan VDJ-exception",
        "Orphan VJ",
        "Orphan VJ-exception",
        "Extra pair",
        "Extra pair-exception",
    ],
    plot: bool = False,
    plateau_fraction: float = 0.95,  # fraction of asymptote to stop extrapolation
    step: int = 1,
) -> pd.DataFrame | ggplot:
    """
    Compute sample-based rarefaction curves with asymptotic extrapolation
    and optional plotting.

    This function calculates rarefaction curves per group, fits an
    asymptotic model (Michaelis–Menten) to estimate the expected plateau
    of clone richness, and extrapolates each curve until the predicted
    value reaches a specified fraction of the asymptote. It supports both
    tabular output and ggplot-style visualization.

    Parameters
    ----------
    data : AnnData or Dandelion
        Object containing V(D)J metadata. Clone IDs must be stored in
        `.obs` (AnnData) or `.metadata` (Dandelion).
    groupby : str
        Column in metadata specifying the grouping variable (e.g., sample,
        donor, condition).
    clone_key : str, optional
        Column containing clone identifiers. Defaults to `"clone_id"` if
        not provided.
    palette : list of str, optional
        List of colors to use for plotting. If `None`, the function tries
        to use `data.uns[f"{groupby}_colors"]` when available.
    figsize : tuple of float, optional
        Width and height of the plot (in inches). Defaults to `(5, 3)`.
    chain_status_include : list of str, optional
        List of chain-status categories to retain. All other chain-status
        entries are excluded. Defaults to a set of productive/orphan chain
        categories commonly used in V(D)J QC.
    plot : bool, optional
        If `True`, returns a ggplot object. If `False`, returns a tidy
        DataFrame with observed and extrapolated rarefaction values.
    plateau_fraction : float, optional
        Fraction of the estimated asymptote at which extrapolation stops.
        For example, `0.95` stops when the curve reaches 95% of the fitted
        asymptotic clone richness.
    step : int, optional
        Increment for generating extrapolated sampling depths. Smaller
        values produce smoother curves but increase computation time.

    Returns
    -------
    pandas.DataFrame or ggplot
        If `plot=False`:
            A tidy DataFrame with the following columns:
                - `cells`: number of sampled cells
                - `yhat`: predicted clone richness
                - `group`: group label
                - `type`: "observed" or "extrapolated"
                - `plateau`: plateau threshold for that group
        If `plot=True`:
            A ggplot object showing observed and extrapolated rarefaction
            curves with solid and dashed line types, respectively.

    Notes
    -----
    * Rarefaction for observed values is computed using rarefun adapted
      from the `vegan` R package.
    * Asymptotic extrapolation is performed using a Michaelis–Menten
      saturation curve:
        ``y = a * x / (b + x)``
    * If the nonlinear fit fails, the function falls back to extending the
      observed rarefaction curve without asymptotic modeling.
    * The function automatically filters out unused categories in the
      clone column and removes `"No_contig"` clones if present.
    """
    # --------------------------
    # Extract metadata
    # --------------------------
    if isinstance(data, AnnData):
        metadata = data.obs.copy()
    elif isinstance(data, Dandelion):
        metadata = data._metadata.copy()
    elif hasattr(data, "mod"):
        metadata = data.mod["airr"].copy()

    clonekey = "clone_id" if clone_key is None else clone_key

    groups = list(set(metadata[groupby]))
    if "chain_status" in metadata:
        metadata = metadata[metadata["chain_status"].isin(chain_status_include)]

    if pd.api.types.is_categorical_dtype(metadata[clonekey]):
        metadata[clonekey] = metadata[clonekey].cat.remove_unused_categories()

    # Count clone sizes per group
    res = {}
    for g in groups:
        _metadata = metadata[metadata[groupby] == g]
        res[g] = _metadata[clonekey].value_counts()
    res_ = pd.DataFrame.from_dict(res, orient="index")

    # Remove empty and no-contig
    res_ = res_[~(res_.sum(axis=1) == 0)]
    res_ = res_.drop("No_contig", axis=1, errors="ignore")
    tot = res_.sum(axis=1)
    nr = res_.shape[0]
    all_results = []
    # --------------------------
    # Compute curves per group
    # --------------------------
    for i in tqdm(
        range(nr),
        desc="Calculating rarefaction + extrapolation",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        clone_sizes = np.array(res_.iloc[i,])
        total_cells = tot.iloc[i]

        # Observed curve
        n_obs = np.arange(1, total_cells, 10)
        if n_obs[-1] != total_cells:
            n_obs = np.append(n_obs, total_cells)
        y_obs = [rarefun(clone_sizes, z) for z in n_obs]

        # Fit asymptotic model
        try:
            popt, _ = curve_fit(
                michaelis_menten_curve,
                n_obs,
                y_obs,
                maxfev=5000,
                bounds=(0, np.inf),
            )
        except Exception:
            # fallback: no fitting
            popt = None

        # Determine plateau target
        if popt is not None:
            a = popt[0]
            target = plateau_fraction * a
        else:
            target = max(y_obs)

        z_pred = np.arange(1, total_cells * 20, step)
        if popt is not None:
            y_pred = michaelis_menten_curve(z_pred, *popt)
        else:
            y_pred = [rarefun(clone_sizes, z) for z in z_pred]

        # Stop at plateau
        plateau_idx = np.argmax(np.array(y_pred) >= target)
        if plateau_idx == 0 and y_pred[0] < target:
            plateau_idx = len(y_pred) - 1
        z_pred = z_pred[: plateau_idx + 1]
        y_pred = y_pred[: plateau_idx + 1]

        # Separate observed vs extrapolated for plotting
        type_labels = [
            "observed" if z <= n_obs[-1] else "extrapolated" for z in z_pred
        ]

        df_i = pd.DataFrame(
            {
                "cells": z_pred,
                "yhat": y_pred,
                "group": res_.index[i],
                "type": type_labels,
                "plateau": [target] * len(z_pred),
            }
        )
        all_results.append(df_i)

    pred = pd.concat(all_results, ignore_index=True)

    if not plot:
        return pred

    # --------------------------
    # Plotting
    # --------------------------
    options.figure_size = figsize

    if palette is None:
        if (
            isinstance(data, AnnData)
            and (str(groupby) + "_colors") in data.uns
        ):
            palette = data.uns[str(groupby) + "_colors"]

    p = (
        ggplot(pred, aes(x="cells", y="yhat", color="group", linetype="type"))
        + theme_classic()
        + xlab("number of cells")
        + ylab("number of clones")
        + ggtitle("rarefaction curve with plateau extrapolation")
        + labs(color=groupby, linetype="data type")
        + geom_line()
        + scale_linetype_manual(
            values={"observed": "solid", "extrapolated": "dashed"}
        )
    )

    if palette is not None:
        p += scale_color_manual(values=palette)

    return p


def clone_diversity(
    data: Dandelion | AnnData,
    groupby: str,
    method: Literal["gini", "chao1", "shannon"] = "gini",
    use_network: bool = True,
    network_metric: Literal[
        "clone_network", "clone_degree", "clone_centrality"
    ] = "clone_network",
    clone_key: str | None = None,
    min_size: int | None = None,
    n_boot: int = 200,
    n_cpus: int = -1,
    normalize: bool = True,
    expanded_only: bool = False,
    use_contracted: bool = False,
    verbose: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, dict[list[float]]]:
    """
    Compute clonal diversity with bootstrapping.

    Parameters
    ----------
    data : Dandelion | AnnData
        Dandelion or AnnData object.
    groupby : str
        Column name to calculate the gini indices on, for e.g. sample id, patient etc.
    method : Literal["gini", "chao1", "shannon"], optional
        Method for diversity estimation. Either one of ['gini', 'chao1', 'shannon'].
    use_network : bool, optional
        Whether or not to use network-based Gini index calculation. Default is True.
    network_metric : Literal["clone_network", "clone_degree", "clone_centrality"], optional
        Metric to use for calculating Gini indices of clones if `use_network` is True.
        Accepts one of ['clone_network', 'clone_degree', 'clone_centrality'].
    clone_key : str | None, optional
        Column name specifying the clone_id column in metadata.
    min_size : int | None, optional
        Minimum cell numbers to keep for diversity calculation. If None, defaults to size of smallest sample.
        Beware that this may lead to very small sample sizes and unreliable estimates if left as None.
    n_boot : int, optional
        Number of times to perform resampling. Default is 200.
    n_cpus : int, optional
        Number of CPUs to use for parallel processing. Default is -1 (use all available cores).
    normalize : bool, optional
        Whether or not to return normalized Shannon Entropy according to https://math.stackexchange.com/a/945172.
        Default is True.
    expanded_only : bool, optional
        Whether or not to calculate gini indices using expanded clones only. Default is False i.e. use all cells/clones.
    use_contracted : bool, optional
        Whether or not to perform the gini calculation after contraction of clone network.
        Only applies to calculation of clone size gini index. Default is False.
        This is to try and preserve the single-cell properties of the network.
    verbose : bool, optional
        whether to print progress.
    **kwargs
        Additional keyword arguments passed to ddl.tl.generate_network if using network-based gini.

    Returns
    -------
    tuple[pd.DataFrame, dict[list[float]]]
        pandas DataFrame holding summarised diversity estimation and the raw bootstrap results.
    """
    # if AnnData, cannot use network-based gini
    use_network = False if isinstance(data, AnnData) else use_network
    if (method == "gini") and use_network:
        return diversity_gini(
            data,
            groupby=groupby,
            metric=network_metric,
            clone_key=clone_key,
            min_size=min_size,
            n_boot=n_boot,
            n_cpus=n_cpus,
            expanded_only=expanded_only,
            use_contracted=use_contracted,
            verbose=verbose,
            **kwargs,
        )
    else:
        return diversity_estimates(
            data,
            method=method,
            groupby=groupby,
            clone_key=clone_key,
            normalize=normalize,
            min_size=min_size,
            n_boot=n_boot,
            n_cpus=n_cpus,
            verbose=verbose,
        )


def diversity_gini(
    data: Dandelion | AnnData,
    groupby: str,
    metric: str | None = None,
    clone_key: str | None = None,
    min_size: int | None = None,
    n_boot: int = 200,
    n_cpus: int = -1,
    expanded_only: bool = False,
    use_contracted: bool = False,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[pd.DataFrame], dict[np.ndarray]]:
    """
    Compute clones Gini indices.

    Parameters
    ----------
    data : Dandelion | AnnData
        Dandelion or AnnData object.
    groupby : str
        Column name to calculate the Gini indices on, for e.g. sample id, patient etc.
    metric : str | None, optional
        Metric to use for calculating Gini indices of clones.
        Accepts one of ['clone_network', 'clone_degree', 'clone_centrality'].
        Defaults to 'clone_centrality'.
    clone_key : str | None, optional
        Column name specifying the clone_id column in metadata.
    min_size : int | None, optional
        Minimum cell numbers to keep for diversity calculation. If None, defaults to size of smallest sample.
        Beware that this may lead to very small sample sizes and unreliable estimates if left as None.
    n_boot : int, optional
        Bootstrap iterations for calculations. Default is 200.
    n_cpus : int, optional
        Number of CPUs to use for parallel processing. Default is -1 (use all available cores).
    expanded_only : bool, optional
        Whether or not to calculate gini indices using expanded clones only. Default is False i.e. use all cells/clones.
    use_contracted : bool, optional
        Whether or not to perform the gini calculation after contraction of clone network.
        Only applies to calculation of clone size gini index. Default is False.
        This is to try and preserve the single-cell properties of the network.
    verbose : bool, optional
        whether to print progress.
    **kwargs
        Additional keyword arguments passed to ddl.tl.generate_network

    Returns
    -------
    tuple[dict[pd.DataFrame], dict[np.ndarray]]
        Dictionaries of pandas DataFrame holding summarised diversity estimation and the raw bootstrap results.
    """
    logg.info("Calculating Gini indices")

    cluster_size, vertex_size, cluster_raw, vertex_raw = gini_indices(
        data,
        groupby=groupby,
        clone_key=clone_key,
        metric=metric,
        min_size=min_size,
        n_boot=n_boot,
        n_cpus=n_cpus,
        expanded_only=expanded_only,
        contracted=use_contracted,
        verbose=verbose,
        **kwargs,
    )

    return {
        "cluster_size_gini": cluster_size,
        "vertex_size_gini": vertex_size,
    }, {
        "cluster_size_gini": cluster_raw,
        "vertex_size_gini": vertex_raw,
    }


def diversity_estimates(
    data: Dandelion | AnnData,
    groupby: str,
    method: Literal["chao1", "shannon", "gini"] = "chao1",
    clone_key: str | None = None,
    normalize: bool = True,
    min_size: int | None = None,
    n_boot: int = 200,
    n_cpus: int = -1,
    verbose: bool = False,
) -> tuple[dict[pd.DataFrame], dict[np.ndarray]]:
    """
    Compute clones Chao1 estimates.

    Parameters
    ----------
    data : Dandelion | AnnData
        Dandelionr AnnData object.
    groupby : str
        Column name to calculate the Chao1 estimates on, for e.g. sample id, patient etc.
    method : Literal["chao1", "shannon", "gini"], optional
        Diversity metric to compute.
    clone_key : str | None, optional
        Column name specifying the clone_id column in metadata
    normalize : bool, optional
        Whether or not to return normalized Shannon Entropy according to https://math.stackexchange.com/a/945172. Default is True.
    min_size : int | None, optional
        Minimum cell numbers to keep for diversity calculation. If None, defaults to size of smallest sample.
        Beware that this may lead to very small sample sizes and unreliable estimates if left as None.
    n_boot : int, optional
        Bootstrap iterations for calculations. Default is 200.
    n_cpus: int, optional
        Number of CPUs to use for parallel processing. Default is -1 (use all available CPUs).
    verbose : bool, optional
        whether to print progress.

    Returns
    -------
    tuple[dict[pd.DataFrame], dict[np.ndarray]]
        Dictionaries of pandas DataFrame holding diversity information and the raw bootstrap results.
    """
    res, res_raw = estimate_diversity(
        data,
        groupby=groupby,
        clone_key=clone_key,
        metric=method,
        normalize=normalize,
        min_size=min_size,
        n_boot=n_boot,
        n_cpus=n_cpus,
        verbose=verbose,
    )

    return {f"{method}": res}, {f"{method}": res_raw}


def gini_indices(
    data: Dandelion,
    groupby: str,
    metric: str | None = None,
    clone_key: str | None = None,
    min_size: int | None = None,
    n_boot: int = 200,
    n_cpus: int = -1,
    expanded_only: bool = False,
    contracted: bool = False,
    verbose: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, dict[list[float]], dict[list[float]]]:
    """Gini indices."""
    if isinstance(data, AnnData):
        raise TypeError("Only Dandelion class object accepted.")
    clonekey = clone_key if clone_key is not None else "clone_id"

    # --- Prepare data and filter groups
    data, _, groups, min_size = _prepare_diversity_groups(
        data, groupby, min_size
    )
    # filter the vdj object as well
    if isinstance(data, Dandelion):
        data = data[data.metadata[groupby].isin(groups)].copy()

    res1, res2, cluster_raw, vertex_raw = {}, {}, {}, {}
    for g in groups:
        # clone size distribution
        ddl_dat = data[data.metadata[groupby] == g].copy()

        # --- parallel bootstrap
        iterator = tqdm(
            range(n_boot),
            desc=f"Bootstrapping... {g}",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        )
        results = Parallel(n_jobs=n_cpus)(
            delayed(_bootstrap_network)(
                ddl_dat,
                clonekey,
                metric,
                min_size,
                expanded_only,
                contracted,
                **kwargs,
            )
            for _ in iterator
        )
        # unpack
        sizelist = np.array([c for c, _ in results], dtype=float)
        graphlist = np.array([v for _, v in results], dtype=float)

        cluster_raw[g] = sizelist
        vertex_raw[g] = graphlist

        safe_bootstrap_summary(res1, g, sizelist)
        safe_bootstrap_summary(res2, g, graphlist)
    res_df1 = pd.DataFrame.from_dict(res1).T
    res_df2 = pd.DataFrame.from_dict(res2).T
    res_df1.reset_index(inplace=True)
    res_df2.reset_index(inplace=True)
    res_df1.rename(columns={"index": groupby}, inplace=True)
    res_df2.rename(columns={"index": groupby}, inplace=True)

    return res_df1, res_df2, cluster_raw, vertex_raw


def estimate_diversity(
    data: Dandelion | AnnData,
    groupby: str,
    clone_key: str | None = None,
    metric: Literal["chao1", "shannon", "gini"] = "chao1",
    normalize: bool = True,  # used only if metric == "shannon"
    min_size: int | None = None,
    n_boot: int = 200,
    n_cpus: int = -1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[list[float]]]:
    """
    Estimate immune repertoire diversity metrics (Chao1 or Shannon).

    Parameters
    ----------
    data : Dandelion | AnnData
        Input repertoire or annotated single-cell data.
    groupby : str
        Column name used to group cells/samples for diversity estimation.
    clone_key : str, optional
        Column containing clone identifiers. Defaults to "clone_id".
    metric : Literal["chao1", "shannon", "gini"], optional
        Diversity metric to compute.
    normalize : bool, default=True
        If True, returns normalized Shannon entropy (ignored for Chao1).
    min_size : int, optional
        Minimum group size required for diversity calculation.
    n_boot : int, default=200
        Number of bootstrap iterations.
    n_cpus: int = -1,
        Number of CPUs to use for parallel processing. Default is -1 (use all available CPUs).
    verbose : bool, default=False
        If True, show progress bars.

    Returns
    -------
    tuple[pd.DataFrame, dict[list[float]]]
        pandas DataFrame holding summarised diversity estimation and the raw bootstrap results.

    """
    clonekey = clone_key if clone_key is not None else "clone_id"

    # --- Prepare data and filter groups
    data, _metadata, groups, min_size = _prepare_diversity_groups(
        data, groupby, min_size
    )

    # --- Subset the actual data
    if isinstance(data, Dandelion):
        data = data[data.metadata[groupby].isin(groups)].copy()
    elif isinstance(data, AnnData):
        data = data[data.obs[groupby].isin(groups)].copy()

    # --- Compute diversity estimates via bootstrapping
    res, res_raw = {}, {}
    for g in groups:
        _dat = _metadata[_metadata[groupby] == g]

        iterator = tqdm(
            range(n_boot),
            desc=f"Bootstrapping… {g}",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        )
        values = Parallel(n_jobs=n_cpus)(
            delayed(_bootstrap_diversity_iteration)(
                _dat,
                clonekey,
                metric,
                normalize,
                min_size,
            )
            for _ in iterator
        )

        values = np.array(values, dtype=float)
        res_raw[g] = values.copy()
        safe_bootstrap_summary(res, g, values)

    # --- Format result
    res_df = pd.DataFrame.from_dict(res).T
    res_df.reset_index(inplace=True)
    res_df.rename(columns={"index": groupby}, inplace=True)

    return res_df, res_raw


def _prepare_diversity_groups(
    data: Dandelion | AnnData,
    groupby: str,
    min_size: int | None = None,
):
    """Shared logic for preparing metadata and filtering valid groups."""
    if isinstance(data, AnnData):
        _metadata = data.obs.copy()
    elif isinstance(data, Dandelion):
        _metadata = data._metadata.copy()
    else:
        raise TypeError("data must be an AnnData or Dandelion object")

    _metadata[groupby] = _metadata[groupby].astype("category")
    _metadata[groupby] = _metadata[groupby].cat.remove_unused_categories()
    groups = list(set(_metadata[groupby]))

    if min_size is None:
        min_size = min_sample_size(df=_metadata, col=groupby)
    group_counts = _metadata[groupby].value_counts()
    valid_groups = group_counts[group_counts >= min_size].index.tolist()
    if len(valid_groups) < 1:
        raise ValueError(
            f"No groups have sufficient size for diversity calculation. "
            f"Please choose a smaller min_size than {min_size}."
        )
    if len(valid_groups) < len(groups):
        logg.warning(
            f"The following groups were excluded (<{min_size} cells): "
            + ", ".join(set(groups) - set(valid_groups))
        )
    groups = valid_groups
    _metadata = _metadata[_metadata[groupby].isin(groups)].copy()

    return data, _metadata, groups, min_size


def _bootstrap_network(
    ddl_dat: Dandelion,
    clonekey: str,
    met: str,
    min_size: int,
    expanded_only: bool,
    contracted: bool,
    **kwargs,
):
    """
    Runs a single bootstrap iteration and returns:
    (cluster_gini, vertex_gini)
    """
    resample_sized = generate_network(
        ddl_dat,
        clone_key=clonekey,
        sample=min_size,
        force_replace=True,
        verbose=False,
        compute_layout=False,
        **kwargs,
    )
    # ---- Choose metric mode
    if met == "clone_network":
        g_c_v_res, g_c_c_res = process_clone_network_stats(
            resample_sized,
            expanded_only=expanded_only,
            contracted=contracted,
        )
        resample_sized._metadata["clone_network_vertex_size_gini"] = pd.Series(
            g_c_v_res
        )
        resample_sized._metadata["clone_network_cluster_size_gini"] = pd.Series(
            g_c_c_res
        )
    elif met == "clone_centrality":
        clone_centrality(resample_sized)
    elif met == "clone_degree":
        clone_degree(resample_sized)
    else:
        raise ValueError("Unknown metric.")
    # ---- Clone size gini
    _dat = resample_sized._metadata.copy()
    _tab = _dat[clonekey].value_counts()
    drop_nan_values(_tab)
    # cluster gini
    if met == "clone_network":
        cluster_gini = _dat[met + "_cluster_size_gini"].mean()
    else:
        clonesizecounts = np.array(_tab)
        clonesizecounts = clonesizecounts[clonesizecounts > 0]
        if len(clonesizecounts) > 1:
            clonesizecounts = np.append(clonesizecounts, 0)
        cluster_gini = (
            calculate_gini_index(clonesizecounts)
            if len(clonesizecounts) > 0
            else 0
        )
    # vertex gini
    if met == "clone_network":
        vertex_gini = _dat[met + "_vertex_size_gini"].mean()
    else:
        connected = resample_sized._metadata[met][
            resample_sized._metadata[met] > 0
        ]
        graphcounts = np.array(connected.value_counts())
        vertex_gini = (
            calculate_gini_index(graphcounts) if len(graphcounts) > 0 else 0
        )
    return cluster_gini, vertex_gini


def _bootstrap_diversity_iteration(
    dat: pd.DataFrame,
    clonekey: str,
    metric: str,
    normalize: bool,
    min_size: int,
):
    """
    One bootstrap iteration for diversity:
    Returns a single diversity value.
    """

    _sample = dat.sample(min_size, replace=True)
    _tab = _sample[clonekey].value_counts()
    drop_nan_values(_tab)

    clone_sizes = np.array(_tab)
    clone_sizes = clone_sizes[clone_sizes > 0]

    if len(clone_sizes) == 0:
        return 0

    metric = metric.lower()
    if metric == "chao1":
        return calculate_chao1(clone_sizes)
    elif metric == "shannon":
        return calculate_shannon_entropy(clone_sizes, normalize)
    elif metric == "gini":
        return calculate_gini_index(clone_sizes)


def calculate_gini_index(
    values: np.ndarray, method: str = "trapezoids"
) -> float:
    """
    Calculate the Gini index for a given array of values.

    Parameters
    ----------
    values : np.ndarray
        Array of values to calculate the Gini index for.
    method : str, optional
        Method to use for Gini index calculation, by default "trapezoids".

    Returns
    -------
    float
        The calculated Gini index, or 0 if the index is negative or NaN.
    """
    gini = gini_index(values, method=method)
    return 0 if gini < 0 or np.isnan(gini) else gini


def calculate_chao1(values: np.ndarray) -> float:
    """
    Calculate the Chao1 estimate for a given array of values

    Parameters
    ----------
    values : np.ndarray
        Array of values to calculate the Chao1 estimates for.

    Returns
    -------
    float
        The calculated Chao1 estimates, or 0 if the index is negative or NaN.
    """
    chao1e = chao1(values)
    return 0 if chao1e < 0 or np.isnan(chao1e) else chao1e


def calculate_shannon_entropy(values: np.ndarray, normalize: bool) -> float:
    """
    Calculate the Shannon entropy for a given array of values

    Parameters
    ----------
    values : np.ndarray
        Array of values to calculate the Shannon entropys for.
    normalize : bool, optional
        Whether or not to return normalized Shannon Entropy according to https://math.stackexchange.com/a/945172. Default is True.
    Returns
    -------
    float
        The calculated Shannon entropys, or 0 if the index is negative or NaN.
    """
    if normalize:
        if len(values) == 1:
            return 0
        else:
            values_freqs = values / np.sum(values)
            return -np.sum(
                (values_freqs * np.log(values_freqs))
                / np.log(len(values_freqs))
            )
    else:
        return shannon(values)


def min_sample_size(
    df: pd.DataFrame,
    col: str,
):
    """
    Calculate the minimum group size for downsampling and log related information.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame containing the grouping information.
    col : str
        Column name to group by.

    Returns
    -------
    int
        The determined minimum group size (`min_size`).
    """
    # Determine minimum group size
    min_size = df[col].value_counts().min()
    return min_size


def chooseln(N: int, k: int) -> float:
    """
    R's lchoose in python
    from https://stackoverflow.com/questions/21767690/python-log-n-choose-k
    """
    return gammaln(N + 1) - gammaln(N - k + 1) - gammaln(k + 1)


def rarefun(y: np.ndarray, sample_size: int) -> float:
    """
    Adapted from rarefun from vegan:
    https://github.com/vegandevs/vegan/blob/master/R/rarefy.R
    """
    res = []
    y = y[y > 0]
    J = np.sum(y)
    ldiv = chooseln(J, sample_size)
    for d in J - y:
        if d < sample_size:
            res.append(0)
        else:
            res.append(np.exp(chooseln(d, sample_size) - ldiv))
    out = np.sum(1 - np.array(res))
    return out


def michaelis_menten_curve(x: float, a: float, b: float) -> float:
    """
    Michaelis-Menten curve function. Used for extrapolation in rarefaction.
    """
    return (a * x) / (b + x)


def drop_nan_values(df: pd.DataFrame) -> None:
    """
    Drop NaN values from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to drop NaN values.
    """
    if "nan" in df.index or np.nan in df.index:
        try:
            df.drop("nan", inplace=True)
        except:
            df.drop(np.nan, inplace=True)


def safe_bootstrap_summary(
    data_dict: dict,
    key: str,
    values: np.ndarray,
    ci: float = 95,
) -> None:
    """
    Safely compute mean, standard deviation, and confidence interval
    from bootstrap values and update the dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary to update.
    key : str
        Key identifying the group or sample.
    values : list of float
        Bootstrap values.
    ci : float, default=95
        Confidence interval width (percentile-based).
    """
    values = values[np.isfinite(values)]  # remove NaN/inf

    if len(values) == 0:
        mean, std, lower, upper = (np.nan, np.nan, np.nan, np.nan)
    else:
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        alpha = (100 - ci) / 2
        lower, upper = np.percentile(values, [alpha, 100 - alpha])

    data_dict[key] = {
        "mean": mean,
        "std": std,
        f"lower_{ci}": lower,
        f"upper_{ci}": upper,
    }


def process_clone_network_stats(
    ddl_dat: Dandelion, expanded_only: bool, contracted: bool
) -> tuple[dict, dict, dict]:
    """
    Process clone network statistics and calculate Gini indices.

    Parameters
    ----------
    ddl_dat : Any
        The Dandelion object to process.
    expanded_only : bool
        Whether to consider only expanded clones.
    contracted : bool
        Whether to use contracted network clusters.

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing dictionaries for node names, vertex sizes, and cluster sizes.
    """
    n_n, v_s, c_s = clone_networkstats(
        ddl_dat,
        expanded_only=expanded_only,
        network_clustersize=contracted,
    )
    g_c_v = defaultdict(dict)
    g_c_v_res, g_c_c_res = {}, {}

    for vs in v_s:
        v_sizes = np.array(v_s[vs])
        if len(v_sizes) > 1:
            v_sizes = np.append(v_sizes, 0)
        g_c_v[vs] = calculate_gini_index(v_sizes)
        for cell in n_n:
            g_c_v_res.update({cell: g_c_v[n_n[cell]]})

    c_sizes = np.array(sorted(list(flatten(c_s.values())), reverse=True))
    if len(c_sizes) > 1:
        c_sizes = np.append(c_sizes, 0)
    g_c_c = calculate_gini_index(c_sizes)
    for cell in n_n:
        g_c_c_res.update({cell: g_c_c})

    return g_c_v_res, g_c_c_res


def clone_networkstats(
    vdj: Dandelion,
    expanded_only: bool = False,
    network_clustersize: bool = False,
) -> tuple[defaultdict, defaultdict, defaultdict]:
    """Retrieve network stats.

    Parameters
    ----------
    vdj : Dandelion
        input object
    expanded_only : bool, optional
        whether or not to calculate only on expanded clones.
    network_clustersize : bool, optional
        depends on metric.
    verbose : bool, optional
        whether to print progress.

    Returns
    -------
    tuple[defaultdict, defaultdict, defaultdict]
        output nodes names, vertex sizes and cluster sizes.

    Raises
    ------
    AttributeError
        if graph not found.
    """
    logg.info("Calculating vertex size of nodes after contraction")
    if vdj.graph is None:
        raise AttributeError("Graph not found. Please run tl.generate_network.")
    else:
        if expanded_only:
            G = vdj.graph[1]
        else:
            G = vdj.graph[0]
        remove_edges = defaultdict(list)
        vertexsizes = defaultdict(list)
        clustersizes = defaultdict(list)
        nodes_names = defaultdict(list)
        for subg in nx.connected_components(G):
            nodes = sorted(list(subg))
            # just assign the value in a single cell, because this will be representative of the clone
            tmp = nodes[0]
            for n in nodes:
                nodes_names[n] = tmp  # keep so i can reference later
            if len(nodes) > 1:
                G_ = G.subgraph(nodes).copy()
                remove_edges[tmp] = [
                    (e[0], e[1])
                    for e in G_.edges(data=True)
                    if e[2]["weight"] > 0
                ]
                if len(remove_edges[tmp]) > 0:
                    G_.remove_edges_from(remove_edges[tmp])
                    for connected in nx.connected_components(G_):
                        vertexsizes[tmp].append(len(connected))
                    vertexsizes[tmp] = sorted(vertexsizes[tmp], reverse=True)
                else:
                    vertexsizes[tmp] = [
                        1 for i in range(len(G_.edges(data=True)))
                    ]
                if network_clustersize:
                    clustersizes[tmp] = len(vertexsizes[tmp])
                else:
                    clustersizes[tmp] = len(nodes)
            else:
                vertexsizes[tmp] = [1]
                clustersizes[tmp] = [1]
        return (nodes_names, vertexsizes, clustersizes)
