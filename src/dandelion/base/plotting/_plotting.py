from __future__ import annotations

import circlify
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nxviz as nxv
import pandas as pd
import seaborn as sns

from anndata import AnnData
from collections import defaultdict
from contextlib import contextmanager
from itertools import product, cycle
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from nxviz import annotate

from scanpy.plotting import palettes
from scanpy.plotting._tools.scatterplots import embedding
from typing import Callable, Literal, TYPE_CHECKING

from dandelion.base.core._core import Dandelion

if TYPE_CHECKING:
    from mudata import MuData


def clone_network(
    adata: AnnData | MuData, basis: str = "vdj", edges: bool = True, **kwargs
) -> None:
    """
    Using scanpy's plotting module to plot the network.

    Only thing that is changed is the default options:
    `basis = 'vdj'` and `edges = True`.

    Parameters
    ----------
    adata : AnnData | MuData
        AnnData or scirpy-formatted MuData object.
    basis : str, optional
        key for embedding.
    edges : bool, optional
        whether or not to plot edges.
    **kwargs
        passed `sc.pl.embedding`.
    """
    is_mudata = hasattr(adata, "mod")
    base_adata = adata.mod["airr"] if is_mudata else adata

    with _temporary_obs_columns(
        base_adata, adata if is_mudata else None, **kwargs
    ) as kw:
        embedding(base_adata, basis=basis, edges=edges, **kw)


def barplot(
    data: AnnData | Dandelion,
    color: str,
    palette: str = "Set1",
    figsize: tuple[float, float] = (8, 3),
    normalize: bool = True,
    sort_descending: bool = True,
    title: str | None = None,
    xtick_fontsize: int | None = None,
    xtick_rotation: int | float | None = None,
    min_clone_size: int = 1,
    clone_key: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    A barplot function to plot usage of V/J genes in the data.

    Parameters
    ----------
    data : AnnData | Dandelion
        Dandelion or AnnData object.
    color : str
        column name in metadata for plotting in bar plot.
    palette : str, optional
        Colors to use for the different levels of the color variable.
        Should be something that can be interpreted by
        [color_palette](https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette),
        or a dictionary mapping hue levels to matplotlib colors.
        See [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html).
    figsize : tuple[float, float], optional
        figure size.
    normalize : bool, optional
        if True, will return as proportion out of 1.
        Otherwise False will return counts.
    sort_descending : bool, optional
        whether or not to sort the order of the plot.
    title : str | None, optional
        title of plot.
    xtick_fontsize : int | None, optional
        size of x tick labels
    xtick_rotation : int | float | None, optional
        rotation of x tick labels.
    min_clone_size : int, optional
        minimum clone size to keep.
    clone_key : str | None, optional
        column name for clones. None defaults to 'clone_id'.
    **kwargs
        passed to `sns.barplot`.

    Returns
    -------
    tuple[Figure, Axes]
        bar plot.

    """
    if isinstance(data, Dandelion):
        data = data._metadata.copy()
    elif isinstance(data, AnnData):
        data = data.obs.copy()

    min_size = min_clone_size

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    size = data[clone_].value_counts()
    keep = list(size[size >= min_size].index)
    data_ = data[data[clone_].isin(keep)]

    sns.set_style("whitegrid", {"axes.grid": False})
    res = pd.DataFrame(data_[color].value_counts(normalize=normalize))
    if not sort_descending:
        res = res.sort_index()
    res.reset_index(drop=False, inplace=True)

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot
    try:
        sns.barplot(x="index", y=color, data=res, palette=palette, **kwargs)
    except ValueError:
        yname = "proportion" if normalize else "count"
        sns.barplot(x=color, y=yname, data=res, palette=palette, **kwargs)
    # change some parts
    if title is None:
        ax.set_title(color.replace("_", " ") + " usage")
    else:
        ax.set_title(title)
    if normalize:
        ax.set_ylabel("proportion")
    else:
        ax.set_ylabel("count")
    ax.set_xlabel("")
    # modify the x ticks accordingly
    xtick_params = {}
    if xtick_rotation is None:
        xtick_params["rotation"] = 90
    else:
        xtick_params["rotation"] = xtick_rotation
    if xtick_fontsize is not None:
        xtick_params["fontsize"] = xtick_fontsize
    plt.xticks(**xtick_params)
    return fig, ax


def stackedbarplot(
    data: AnnData | Dandelion,
    color: str,
    group_by: str | None,
    figsize: tuple[float, float] = (8, 3),
    normalize: bool = False,
    title: str | None = None,
    sort_descending: bool = True,
    xtick_fontsize: int | None = None,
    xtick_rotation: int | float | None = None,
    hide_legend: bool = False,
    legend_options: tuple[str, tuple[float, float], int] = (
        "upper left",
        (1, 1),
        1,
    ),
    labels: list[str] | None = None,
    min_clone_size: int = 1,
    clone_key: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    A stacked bar plot function to plot usage of V/J genes in the data split by groups.

    Parameters
    ----------
    data : AnnData | Dandelion
        Dandelion or AnnData object.
    color : str
        column name in metadata for plotting in bar plot.
    group_by : str | None
        column name in metadata to split by during plotting.
    figsize : tuple[float, float], optional
        figure size.
    normalize : bool, optional
        if True, will return as proportion out of 1, otherwise False will return counts.
    sort_descending : bool, optional
        whether or not to sort the order of the plot.
    xtick_fontsize : int | None, optional
        size of x tick labels
    xtick_rotation : int | float | None, optional
        rotation of x tick labels.
    hide_legend : bool, optional
        whether or not to hide the legend.
    legend_options : tuple[str, tuple[float, float], int], optional
        a tuple holding 3 options for specify legend options: 1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
    labels : list[str] | None, optional
        Names of objects will be used for the legend if list of multiple data frames supplied.
    min_clone_size : int, optional
        minimum clone size to keep.
    clone_key : str | None, optional
        column name for clones. None defaults to 'clone_id'.
    **kwargs
        other kwargs passed to `matplotlib.plt`.

    Returns
    -------
    tuple[Figure, Axes]
        stacked barplot.
    """
    if isinstance(data, Dandelion):
        data = data._metadata.copy()
    elif isinstance(data, AnnData):
        data = data.obs.copy()
    # quick fix to prevent dropping of nan
    data[group_by] = [str(l) for l in data[group_by]]

    min_size = min_clone_size

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    size = data[clone_].value_counts()
    keep = list(size[size >= min_size].index)
    data_ = data[data[clone_].isin(keep)]

    dat_ = pd.DataFrame(
        data_.groupby(color)[group_by]
        .value_counts(normalize=normalize)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_order = pd.DataFrame(data[color].value_counts(normalize=normalize))
    dat_ = dat_.pivot(index=color, columns=group_by, values="value")
    if sort_descending is True:
        dat_ = dat_.reindex(dat_order.index)
    elif sort_descending is False:
        dat_ = dat_.reindex(dat_order.index[::-1])
    elif sort_descending is None:
        dat_ = dat_.sort_index()

    def _plot_bar_stacked(
        dfall: pd.DataFrame,
        labels: list[str] | None = None,
        figsize: tuple[float, float] = (8, 3),
        title: str = "multiple stacked bar plot",
        xtick_fontsize: int | None = None,
        xtick_rotation: int | float | None = None,
        legend_options: tuple[str, tuple[float, float], int] = None,
        hide_legend: bool = False,
        H: Literal["/"] = "/",
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Given a list of data frames, with identical columns and index, create a clustered stacked bar plot.

        Parameters
        ----------
        dfall : pd.DataFrame
            data frame for plotting.
        labels : list[str] | None, optional
            a list of the data frame objects. Names of objects will be used for the legend.
        figsize : tuple[float, float], optional
            size of figure.
        title : str, optional
            string for the title of the plot
        xtick_fontsize : int | None, optional
            xtick fontsize.
        xtick_rotation : int | float | None, optional
            rotation of xtick labels
        legend_options : tuple[str, tuple[float, float], int], optional
            legend options.
        hide_legend : bool, optional
            whether to show legend.
        H : Literal["/"], optional
            is the hatch used for identification of the different data frames
        **kwargs
            other kwargs passed to matplotlib.plt

        Returns
        -------
        tuple[Figure, Axes]
            stacked barplot.
        """
        if type(dfall) is not list:
            dfall = [dfall]
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)
        # Initialize the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        for df in dfall:  # for each data frame
            ax = df.plot(
                kind="bar",
                linewidth=0,
                stacked=True,
                ax=ax,
                legend=False,
                grid=False,
                **kwargs,
            )  # make bar plots
        (
            h,
            l,
        ) = ax.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(h[i : i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(
                        rect.get_x() + 1 / float(n_df + 1) * i / float(n_col)
                    )
                    rect.set_hatch(H * int(i / n_col))  # edited part
                    rect.set_width(1 / float(n_df + 1))
        ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
        ax.set_xticklabels(df.index, rotation=0)
        ax.set_title(title)
        if normalize:
            ax.set_ylabel("proportion")
        else:
            ax.set_ylabel("count")
        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(ax.bar(0, 0, color="grey", hatch=H * i))
        if legend_options is None:
            Legend = ("center right", (1.15, 0.5), 1)
        else:
            Legend = legend_options
        if hide_legend is False:
            l1 = ax.legend(
                h[:n_col],
                l[:n_col],
                loc=Legend[0],
                bbox_to_anchor=Legend[1],
                ncol=Legend[2],
                frameon=False,
            )
            if labels is not None:
                l2 = plt.legend(
                    n,
                    labels,
                    loc=Legend[0],
                    bbox_to_anchor=Legend[1],
                    ncol=Legend[2],
                    frameon=False,
                )
                ax.add_artist(l2)
            ax.add_artist(l1)
        # modify the x ticks accordingly
        xtick_params = {}
        if xtick_rotation is None:
            xtick_params["rotation"] = 90
        else:
            xtick_params["rotation"] = xtick_rotation
        if xtick_fontsize is not None:
            xtick_params["fontsize"] = xtick_fontsize
        plt.xticks(**xtick_params)

        return fig, ax

    if title is None:
        title = (
            "multiple stacked bar plot : " + color.replace("_", " ") + " usage"
        )
    else:
        title = title

    return _plot_bar_stacked(
        dat_,
        labels=labels,
        figsize=figsize,
        title=title,
        xtick_fontsize=xtick_fontsize,
        xtick_rotation=xtick_rotation,
        legend_options=legend_options,
        hide_legend=hide_legend,
        **kwargs,
    )


def spectratype(
    vdj: Dandelion,
    color: str,
    group_by: str,
    locus: str,
    figsize: tuple[float, float] = (5, 3),
    width: int | float | None = None,
    title: str | None = None,
    xtick_fontsize: int | None = None,
    xtick_rotation: int | float | None = None,
    hide_legend: bool = False,
    legend_options: tuple[str, tuple[float, float], int] = (
        "upper left",
        (1, 1),
        1,
    ),
    labels: list[str] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    A spectratype function to plot usage of CDR3 length.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object.
    color : str
        column name in metadata for plotting in bar plot.
    group_by : str
        column name in metadata to split by during plotting.
    locus : str
        either IGH or IGL.
    figsize : tuple[float, float], optional
        figure size.
    width : int | float | None, optional
        width of bars.
    title : str | None, optional
        title of plot.
    xtick_fontsize : int | None, optional
        size of x tick labels
    xtick_rotation : int | float | None, optional
        rotation of x tick labels.
    hide_legend : bool, optional
        whether or not to hide the legend.
    legend_options : tuple[str, tuple[float, float], int], optional
        a tuple holding 3 options for specify legend options:
        1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
    labels : list[str] | None, optional
        Names of objects will be used for the legend if list of
        multiple data frames supplied.
    **kwargs
        other kwargs passed to matplotlib.pyplot.plot

    Returns
    -------
    tuple[Figure, Axes]
        spectratype plot.
    """
    data = vdj._data.copy()
    if "ambiguous" in data:
        data = data[data["ambiguous"] == "F"].copy()

    if type(locus) is not list:
        locus = [locus]
    data = data[data["locus"].isin(locus)].copy()
    data[group_by] = [str(l) for l in data[group_by]]
    dat_ = pd.DataFrame(
        data.groupby(color)[group_by]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_[color] = pd.to_numeric(dat_[color], errors="coerce")
    dat_.sort_values(by=color)
    dat_2 = dat_.pivot(index=color, columns=group_by, values="value")
    new_index = range(0, int(dat_[color].max()) + 1)
    dat_2 = dat_2.reindex(new_index, fill_value=0)

    def _plot_spectra_stacked(
        dfall: pd.DataFrame,
        labels: list[str] | None = None,
        figsize: tuple[float, float] = (5, 3),
        title: str = "multiple stacked bar plot",
        width: int | float | None = None,
        xtick_fontsize: int | None = None,
        xtick_rotation: int | float | None = None,
        legend_options: tuple[str, tuple[float, float], int] = None,
        hide_legend: bool = False,
        H: Literal["/"] = "/",
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Stacked spectratype plots.

        Parameters
        ----------
        dfall : pd.DataFrame
            data frame for plotting.
        labels : list[str] | None, optional
            a list of the data frame objects. Names of objects will be used for the legend.
        figsize : tuple[float, float], optional
            size of figure.
        title : str, optional
            string for the title of the plot.
        width : int | float | None, optional
            width of bars.
        xtick_fontsize : int | None, optional
            size of x tick labels
        xtick_rotation : int | float | None, optional
            rotation of x tick labels.
        legend_options : tuple[str, tuple[float, float], int], optional
            a tuple holding 3 options for specify legend options:
            1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
        hide_legend : bool, optional
            whether or not to hide the legend.
        H : Literal["/"], optional
            not sure.
        **kwargs
            other kwargs passed to matplotlib.plt

        Returns
        -------
        tuple[Figure, Axes]
            spectratype plot.
        """
        if type(dfall) is not list:
            dfall = [dfall]
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)
        if width is None:
            wdth = 0.1 * n_ind / 60 + 0.8
        else:
            wdth = width
        # Initialize the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        for df in dfall:  # for each data frame
            ax = df.plot(
                kind="bar",
                linewidth=0,
                stacked=True,
                ax=ax,
                legend=False,
                grid=False,
                **kwargs,
            )  # make bar plots
        (
            h,
            l,
        ) = ax.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(h[i : i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(
                        rect.get_x() + 1 / float(n_df + 1) * i / float(n_col)
                    )
                    rect.set_hatch(H * int(i / n_col))  # edited part
                    # need to see if there's a better way to toggle this.
                    rect.set_width(wdth)

        n = 5  # Keeps every 5th label visible and hides the rest
        [
            l.set_visible(False)
            for (i, l) in enumerate(ax.xaxis.get_ticklabels())
            if i % n != 0
        ]
        ax.set_title(title)
        ax.set_ylabel("count")
        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(ax.bar(0, 0, color="gray", hatch=H * i))
        if legend_options is None:
            Legend = ("center right", (1.25, 0.5), 1)
        else:
            Legend = legend_options
        if hide_legend is False:
            l1 = ax.legend(
                h[:n_col],
                l[:n_col],
                loc=Legend[0],
                bbox_to_anchor=Legend[1],
                ncol=Legend[2],
                frameon=False,
            )
            if labels is not None:
                l2 = plt.legend(
                    n,
                    labels,
                    loc=Legend[0],
                    bbox_to_anchor=Legend[1],
                    ncol=Legend[2],
                    frameon=False,
                )
            ax.add_artist(l1)
        # modify the x ticks accordingly
        xtick_params = {}
        if xtick_rotation is None:
            xtick_params["rotation"] = 90
        else:
            xtick_params["rotation"] = xtick_rotation
        if xtick_fontsize is not None:
            xtick_params["fontsize"] = xtick_fontsize
        plt.xticks(**xtick_params)

        return fig, ax

    return _plot_spectra_stacked(
        dat_2,
        labels=labels,
        figsize=figsize,
        title=title,
        width=width,
        xtick_fontsize=xtick_fontsize,
        xtick_rotation=xtick_rotation,
        legend_options=legend_options,
        hide_legend=hide_legend,
        **kwargs,
    )


def clone_overlap(
    adata: AnnData,
    group_by: str,
    color_by: str | None = None,
    weighted_overlap: bool = False,
    clone_key: str | None = None,
    color_mapping: list | dict | None = None,
    node_labels: bool = True,
    return_graph: bool = False,
    save: str | None = None,
    legend_kwargs: dict = {
        "ncol": 2,
        "bbox_to_anchor": (1, 0.5),
        "frameon": False,
        "loc": "center left",
    },
    node_label_size: int = 10,
    as_heatmap: bool = False,
    return_heatmap_data: bool = False,
    scale_edge_lambda: Callable | None = None,
    **kwargs,
) -> nxv.CircosPlot:
    """
    A plot function to visualise clonal overlap as a circos-style plot.

    Originally written with nxviz < 0.7.3. Ported from https://github.com/zktuong/nxviz/tree/custom_color_mapping_circos_nodes_and_edges

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    group_by : str
        column name in obs for collapsing to nodes in circos plot.
    color_by : str | None, optional
        column name in obs for grouping and color of nodes in plot. Must be a same or subcategory of the `group_by` categories e.g. `group_by="group_tissue", color_by="tissue"`.
    weighted_overlap : bool, optional
        if True, instead of collapsing to overlap to binary, edge thickness will reflect the number of
        cells found in the overlap. In the future, there will be the option to use something like a jaccard
        index instead.
    clone_key : str | None, optional
        column name for clones. None defaults to 'clone_id'.
    color_mapping : list | dict | None, optional
        custom color mapping provided as a sequence (correpsonding to order of categories or
        alpha-numeric order ifdtype is not category), or dictionary containing custom {category:color} mapping.
    node_labels : bool, optional
        whether to use node objects as labels or not
    return_graph : bool, optional
        whether or not to return the graph for fine tuning.
    save : str | None, optional
        file path for saving plot
    legend_kwargs : dict, optional
        options for adjusting legend placement
    node_label_size : int, optional
        size of labels if node_labels = True
    as_heatmap : bool, optional
        whether to return plot as heatmap.
    return_heatmap_data : bool, optional
        whether to return heatmap data as a pandas dataframe.
    scale_edge_lambda : Callable | None, optional
        a lambda function to scale the edge thickness. If None, will not scale.
    **kwargs
        passed to `matplotlib.pyplot.savefig`.

    Returns
    -------
    nxv.CircosPlot
        a `nxviz.CircosPlot` object.

    Raises
    ------
    KeyError
        if `clone_overlap` not found in `adata.uns`.
    ValueError
        if input is not AnnData.
    """

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    if isinstance(adata, AnnData):
        data = adata.obs.copy()
        # get rid of problematic rows that appear because of category conversion?
        if "clone_overlap" in adata.uns:
            overlap = adata.uns["clone_overlap"].copy()
        else:
            raise KeyError(
                "`clone_overlap` not found in `adata.uns`. Did you run `tl.clone_overlap`?"
            )
    else:
        raise ValueError("Please provide a AnnData object.")

    edges = {}
    if not weighted_overlap:
        for x in overlap.index:
            if overlap.loc[x].sum() > 1:
                edges[x] = [
                    y + ({str(clone_): x},)
                    for y in list(
                        product(
                            [
                                i
                                for i in overlap.loc[x][
                                    overlap.loc[x] > 0
                                ].index
                            ],
                            repeat=2,
                        )
                    )
                ]
    else:
        tmp_overlap = overlap.astype(bool).sum(axis=1)
        combis = {
            x: list(
                product(
                    [i for i in overlap.loc[x][overlap.loc[x] > 0].index],
                    repeat=2,
                )
            )
            for x in tmp_overlap.index
            if tmp_overlap.loc[x] > 1
        }

        tmp_edge_weight_dict = defaultdict(list)
        for k_clone, val_pair in combis.items():
            for pair in val_pair:
                tmp_edge_weight_dict[pair].append(
                    overlap.loc[k_clone, list(pair)].sum()
                )
        for combix in tmp_edge_weight_dict:
            if scale_edge_lambda is not None:
                tmp_edge_weight_dict[combix] = scale_edge_lambda(
                    sum(tmp_edge_weight_dict[combix])
                )
            else:
                tmp_edge_weight_dict[combix] = sum(tmp_edge_weight_dict[combix])
        for x in overlap.index:
            if overlap.loc[x].sum() > 1:
                edges[x] = [
                    y
                    + (
                        {
                            str(clone_): x,
                            "weight": (
                                tmp_edge_weight_dict[y]
                                if not isinstance(tmp_edge_weight_dict[y], list)
                                else 0
                            ),
                        },
                    )
                    for y in list(
                        product(
                            [
                                i
                                for i in overlap.loc[x][
                                    overlap.loc[x] > 0
                                ].index
                            ],
                            repeat=2,
                        )
                    )
                ]

    color_by = group_by if color_by is None else color_by
    # create graph
    G = nx.Graph()
    # add in the nodes
    G.add_nodes_from(
        [
            (p, {str(color_by): d})
            for p, d in zip(data[group_by], data[color_by])
        ]
    )

    # unpack the edgelist and add to the graph
    for edge in edges:
        G.add_edges_from(edges[edge])

    if not weighted_overlap:
        weighted_attr = None
    else:
        weighted_attr = "weight"

    if color_mapping is None:
        if str(color_by) + "_colors" in adata.uns:
            if pd.api.types.is_categorical_dtype(adata.obs[group_by]):
                color_by_dict = dict(
                    zip(
                        list(adata.obs[str(color_by)].cat.categories),
                        adata.uns[str(color_by) + "_colors"],
                    )
                )
            else:
                color_by_dict = dict(
                    zip(
                        list(adata.obs[str(color_by)].unique()),
                        adata.uns[str(color_by) + "_colors"],
                    )
                )
        else:
            if len(adata.obs[str(color_by)].unique()) <= 20:
                pal = cycle(palettes.default_20)
            elif len(adata.obs[str(color_by)].unique()) <= 28:
                pal = cycle(palettes.default_28)
            else:
                pal = cycle(palettes.default_102)
            color_by_dict = dict(
                zip(list(adata.obs[str(color_by)].unique()), pal)
            )
    else:
        if type(color_mapping) is dict:
            color_by_dict = color_mapping
        else:
            if pd.api.types.is_categorical_dtype(data[group_by]):
                color_by_dict = dict(
                    zip(list(data[str(color_by)].cat.categories), color_mapping)
                )
            else:
                color_by_dict = dict(
                    zip(sorted(list(set(data[str(color_by)]))), color_mapping)
                )
    df = data[[group_by, color_by]]
    if group_by == color_by:
        df = data[[group_by]]
        df = (
            df.sort_values(group_by)
            .drop_duplicates(subset=group_by, keep="first")
            .reset_index(drop=True)
        )
    else:
        df = (
            df.sort_values(color_by)
            .drop_duplicates(subset=group_by, keep="first")
            .reset_index(drop=True)
        )

    if as_heatmap:
        hm = nx.to_pandas_adjacency(G)
        sns.clustermap(hm, **kwargs)
        if return_heatmap_data:
            return hm
    else:
        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))
        ax = nxv.circos(
            G,
            group_by=color_by,
            node_color_by=color_by,
            edge_lw_by=weighted_attr,
            node_palette=color_by_dict,
        )  # group_by
        if node_labels:
            annotate.circos_group(
                G,
                group_by=color_by,
                midpoint=False,
                fontdict={"size": node_label_size},
            )
        annotate.node_colormapping(
            G,
            color_by=color_by,
            palette=color_by_dict,
            legend_kwargs=legend_kwargs,
        )
    if save is not None:
        plt.savefig(save, bbox_inches="tight", **kwargs)
    if return_graph:
        return G


def productive_ratio(
    adata: AnnData,
    figsize: tuple[float, float] = (8, 4),
    palette: list[str] = ["lightblue", "darkblue"],
    fontsize: int | float = 8,
    rotation: int | float = 90,
    legend_kwargs: dict = {
        "bbox_to_anchor": (1, 0.5),
        "loc": "center left",
        "frameon": False,
    },
):
    """Plot productive/non-productive contig ratio from AnnData (cell level).

    Parameters
    ----------
    adata : AnnData
        AnnData object with `.uns['productive_ratio']` computed from
        `tl.productive_ratio`.
    figsize : tuple[float, float], optional
        Size of figure.
    palette : list[str], optional
        List of colours to plot non-productive and productive respectively.
    fontsize : int | float, optional
        Font size of x and y tick labels.
    rotation : int | float, optional
        Rotation of x tick labels.
    legend_kwargs : dict, optional
        Any additional kwargs to `plt.legend`
    """
    res = adata.uns["productive_ratio"]["results"]
    locus = adata.uns["productive_ratio"]["locus"]
    group_by = adata.uns["productive_ratio"]["group_by"]

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x=group_by, y="productive+non-productive", data=res, color=palette[0]
    )
    ax = sns.barplot(
        x=group_by, y="productive", data=res, color=palette[1], ax=ax
    )
    legend = [
        mpatches.Patch(
            color=palette[0], label="% with non-productive " + locus
        ),
        mpatches.Patch(color=palette[1], label="% with productive " + locus),
    ]
    plt.xticks(fontsize=fontsize, rotation=rotation)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("")
    plt.ylabel("")
    ax.set(ylim=(0, 100))
    plt.title(locus)
    # add legend
    plt.legend(handles=legend, **legend_kwargs)


def _pack(hierarchy: list[dict], target_enc, packer: str) -> list:
    """Dispatch circle packing to circlify or packcircles."""
    if packer == "circlify":
        return circlify.circlify(
            hierarchy, show_enclosure=False, target_enclosure=target_enc
        )
    if packer == "packcircles":
        return _pack_with_packcircles(hierarchy, target_enc)
    raise ValueError(
        f"Unknown packer '{packer}'. Choose 'circlify' or 'packcircles'."
    )


def _pack_with_packcircles(hierarchy: list[dict], target_enc) -> list:
    """Recursive circle packer using packcircles for leaf levels.

    Groups (nodes with children) are positioned with circlify (small N, fast).
    Leaves are packed with packcircles, which uses an iterative approach that
    scales much better than circlify's O(N²) sequential algorithm.
    """
    import math
    from types import SimpleNamespace

    try:
        import packcircles as pc
    except ImportError:
        raise ImportError(
            "packcircles is required for packer='packcircles'. "
            "Install it with: pip install packcircles"
        )

    all_circles: list = []

    def _do_pack(
        nodes: list[dict], enc_x: float, enc_y: float, enc_r: float, level: int
    ) -> None:
        if not nodes:
            return

        has_children = any("children" in n for n in nodes)

        if not has_children:
            sorted_nodes = sorted(nodes, key=lambda n: n["datum"], reverse=True)
            radii = [math.sqrt(n["datum"]) for n in sorted_nodes]

            if len(radii) < 3:
                # packcircles requires >= 3 circles; fall back to circlify for small N.
                fallback_data = [
                    {"id": n["id"], "datum": n["datum"]} for n in sorted_nodes
                ]
                enc = circlify.Circle(enc_x, enc_y, enc_r)
                fb = circlify.circlify(
                    fallback_data, show_enclosure=False, target_enclosure=enc
                )
                id_to_node = {n["id"]: n for n in sorted_nodes}
                for fc in fb:
                    node = id_to_node.get(fc.ex.get("id", ""))
                    if node is not None:
                        all_circles.append(
                            SimpleNamespace(x=fc.x, y=fc.y, r=fc.r, ex=node)
                        )
                return

            packed = list(pc.pack(radii))

            if not packed:
                return

            total_w = sum(pr * pr for px, py, pr in packed)
            if total_w == 0:
                return
            w_cx = sum(pr * pr * px for px, py, pr in packed) / total_w
            w_cy = sum(pr * pr * py for px, py, pr in packed) / total_w

            max_reach = max(
                math.sqrt((px - w_cx) ** 2 + (py - w_cy) ** 2) + pr
                for px, py, pr in packed
            )
            if max_reach == 0:
                return
            scale = enc_r / max_reach

            for (px, py, pr), node in zip(packed, sorted_nodes):
                all_circles.append(
                    SimpleNamespace(
                        x=enc_x + (px - w_cx) * scale,
                        y=enc_y + (py - w_cy) * scale,
                        r=pr * scale,
                        ex=node,
                    )
                )
        else:
            stripped = [{"id": n["id"], "datum": n["datum"]} for n in nodes]
            group_enc = circlify.Circle(enc_x, enc_y, enc_r)
            group_circles = circlify.circlify(
                stripped, show_enclosure=False, target_enclosure=group_enc
            )
            node_by_id = {n["id"]: n for n in nodes}
            for gc in group_circles:
                node = node_by_id.get(gc.ex.get("id", ""))
                if node is None:
                    continue
                all_circles.append(
                    SimpleNamespace(x=gc.x, y=gc.y, r=gc.r, ex=node)
                )
                if "children" in node:
                    _do_pack(node["children"], gc.x, gc.y, gc.r, level + 1)

    _do_pack(hierarchy, target_enc.x, target_enc.y, target_enc.r, 1)
    return all_circles


def clone_circlepackplot(
    data: AnnData | Dandelion,
    group_by: str | list[str],
    palette: str | dict | None = None,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    min_clone_size: int = 1,
    clone_key: str | None = None,
    show_group_labels: bool = True,
    show_clone_labels: bool = False,
    show_count_labels: bool = False,
    alpha: float = 0.6,
    show_legend: str | list[str] | None = None,
    legend_kwargs: dict = {
        "bbox_to_anchor": (1, 0.5),
        "loc": "center left",
        "frameon": False,
    },
    as_subplots: bool = False,
    n_col: int | None = None,
    n_row: int | None = None,
    scale_subplots: bool = True,
    scale_factor: float | None = None,
    outer_ring_color: str | None = None,
    show_enclosure_label: bool = True,
    max_clones_per_group: int | None = None,
    aggregate_by_size: bool = False,
    packer: Literal["circlify", "packcircles"] = "circlify",
) -> tuple[Figure, Axes] | tuple[Figure, list[Axes]]:
    """
    A bubble plot to visualise clone sizes within groups using circle packing.

    Each group (e.g. sample, celltype) is represented as an enclosing circle,
    with clones within that group shown as packed inner circles sized
    proportionally to clone size.

    When `group_by` is a list the hierarchy follows the list order: the first
    element is the outermost ring, subsequent elements are nested rings, and
    clone circles sit at the innermost level. Each level is coloured
    independently using its own colour map.

    Parameters
    ----------
    data : AnnData | Dandelion
        Dandelion or AnnData object.
    group_by : str | list[str]
        Column name(s) in metadata to group clones by. A single string gives
        one level of nesting; a list gives multi-level nesting in list order
        (e.g. ``['sample_id', 'leiden', 'isotype']``).
    palette : str | dict | None, optional
        Colour specification.

        * ``None`` (default): for Dandelion objects, uses ``"Set2"`` for every
          level; for AnnData objects, each column is looked up in ``adata.uns``
          (e.g. ``uns["leiden_colors"]``) and falls back to scanpy's default
          palette cycle when the key is absent.
        * ``str``: a seaborn palette name applied uniformly to every level.
        * ``dict``: a nested mapping where each key is a column name from
          ``group_by`` and each value is either a ``{category: colour}`` dict or
          a list of colours assigned in category order (respecting
          ``.cat.categories`` for categoricals, numeric sort order, or
          alphabetical otherwise).  Missing columns or values are
          auto-assigned.  Examples for ``group_by=["A", "B"]``::

              palette={"A": {"x": "red", "y": "blue"}, "B": {"x": "green"}}
              palette={"A": ["red", "blue"], "B": ["green", "orange"]}
    figsize : tuple[float, float], optional
        Figure size. When ``as_subplots=True`` this is interpreted as the size
        of each individual subplot panel.
    title : str | None, optional
        Title of plot.
    min_clone_size : int, optional
        Minimum clone size to include. Set to 2 to exclude singletons.
    clone_key : str | None, optional
        Column name for clones. None defaults to 'clone_id'.
    show_group_labels : bool, optional
        Whether to annotate each group enclosure with its label.
    show_clone_labels : bool, optional
        Whether to annotate each clone circle with its clone ID.
    show_count_labels : bool, optional
        Whether to annotate each circle with its cell count.
    alpha : float, optional
        Transparency of clone circles.
    show_legend : str | list[str] | None, optional
        Controls which group_by levels appear in the legend.

        * ``None`` (default): show all levels.
        * ``str``: show only that level (e.g. ``"isotype"``).
        * ``list[str]``: show only those levels.
        * ``False``: hide the legend entirely.
    legend_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.legend``.
    as_subplots : bool, optional
        If ``True``, split the figure into one subplot per top-level group
        (the first element of ``group_by``). The subplots are tiled according
        to ``n_col`` and ``n_row``.
    n_col : int | None, optional
        Number of subplot columns when ``as_subplots=True``. If ``None``,
        defaults to 4 when there are more than 4 subplots, otherwise the
        number of subplots. Ignored when ``as_subplots=False``.
    n_row : int | None, optional
        Number of subplot rows when ``as_subplots=True``. If ``None``,
        computed automatically from ``n_col``. Ignored when
        ``as_subplots=False``.
    scale_subplots : bool, optional
        When ``as_subplots=True``, scale each subplot's enclosing circle so
        that its area is proportional to its total cell count relative to the
        largest subplot. This keeps circle sizes visually comparable across
        panels: a clone of size *n* always occupies the same area regardless
        of which subplot it appears in. Uniform axis limits are applied to all
        subplots. Has no effect when ``as_subplots=False``. Default ``True``.
    scale_factor : float | None, optional
        Direct multiplier on the enclosure radius. Larger values produce
        larger circles; smaller values produce smaller circles. Applies in
        both ``as_subplots=True`` and ``as_subplots=False`` modes.

        * ``None`` (default): in single-panel mode the enclosure fills the
          panel (radius = 1.0); in subplot mode the radius is determined
          by ``scale_subplots`` alone.
        * ``float``: the enclosure radius is set to ``scale_factor`` in
          single-panel mode, or multiplied by ``scale_factor`` on top of
          the ``scale_subplots`` auto-computed radius in subplot mode.
          Axes are fixed to ``[-1.05, 1.05]`` so relative sizes remain
          visually meaningful. Values < 1 shrink circles; values > 1
          enlarge them.
    outer_ring_color : str | None, optional
        If set, all outermost group rings (depth-0 circles in single-panel
        mode and the enclosing ring in each subplot) are drawn in this single
        colour instead of their level-0 palette colour.  Any valid matplotlib
        colour string is accepted (e.g. ``"black"``, ``"#333333"``).
        ``None`` (default) preserves per-group colouring from ``palette``.
    show_enclosure_label : bool, optional
        Whether to display the total cell count below each enclosing group
        ring.  Only has an effect when ``show_count_labels=True``.
        Default ``True``.
    max_clones_per_group : int | None, optional
        If set, only the top-*N* largest clones within each leaf group are
        drawn.  Clones are ranked by size (descending) and any beyond the
        *N*-th are silently dropped before packing.  Useful for keeping
        very large groups readable.  ``None`` (default) includes all clones
        that pass ``min_clone_size``.
    aggregate_by_size : bool, optional
        When ``True``, instead of drawing one circle per clone, clones are
        grouped into buckets by their size.  Each bucket becomes a single
        circle whose area is proportional to ``clone_size × number_of_clones``
        (total cells in that bucket).  The circle label is ``"n=<size>"`` and,
        when ``show_count_labels=True``, the annotation reads
        ``"<size>\\n(<total_cells>)"``.  This greatly reduces the number of
        circles for samples with many small clones and gives a compact
        overview of the size distribution.  Default ``False``.
    packer : {"circlify", "packcircles"}, optional
        Circle-packing backend to use.

        * ``"circlify"`` (default): uses the *circlify* library, which
          produces deterministic, aesthetically balanced layouts.
        * ``"packcircles"``: uses the *packcircles* library, which applies
          an iterative overlap-removal algorithm.  Leaf-level circles are
          packed with *packcircles*; enclosing group circles are still laid
          out by *circlify*.  Requires ``pip install packcircles``.

    Returns
    -------
    tuple[Figure, Axes]
        Circle-packing bubble plot (when ``as_subplots=False``).
    tuple[Figure, list[Axes]]
        Figure and list of per-group axes (when ``as_subplots=True``).

    Raises
    ------
    ValueError
        If no clones remain after filtering by `min_clone_size`.
    ValueError
        If ``packer`` is not ``"circlify"`` or ``"packcircles"``.
    """
    _is_adata = isinstance(data, AnnData)
    _adata_uns = data.uns if _is_adata else {}

    if isinstance(data, Dandelion):
        data = data._metadata.copy()
    elif isinstance(data, AnnData):
        data = data.obs.copy()

    clone_ = clone_key if clone_key is not None else "clone_id"

    # Remove cells with no assigned clone (No_contig, NaN, etc.)
    _no_clone = {"No_contig", "nan", "None", "NA", ""}
    data = data[data[clone_].notna()].copy()
    data = data[~data[clone_].astype(str).isin(_no_clone)].copy()

    size = data[clone_].value_counts()
    keep = list(size[size >= min_clone_size].index)
    data_ = data[data[clone_].isin(keep)]

    if data_.empty:
        raise ValueError(
            f"No clones remaining after filtering with min_clone_size={min_clone_size}."
        )

    group_by_cols = [group_by] if isinstance(group_by, str) else list(group_by)

    # Remove rows where any group_by column contains no-data sentinel values
    for col in group_by_cols:
        data_ = data_[~data_[col].astype(str).isin(_no_clone)].copy()

    def _auto_colors(
        col: str, vals: list[str], offset: int = 0
    ) -> dict[str, tuple]:
        """Return a {value: colour} map for *vals* in *col*.

        For AnnData inputs the column's ``.uns`` entry is consulted first.
        Dandelion inputs always use Set2.

        *offset* shifts the starting index in the palette so that successive
        hierarchy levels draw from distinct colours rather than all cycling
        from position 0.
        """
        if _is_adata:
            uns_key = f"{col}_colors"
            if uns_key in _adata_uns:
                obs_col = data[col] if col in data.columns else None
                if obs_col is not None and pd.api.types.is_categorical_dtype(
                    obs_col
                ):
                    cats = list(obs_col.cat.categories)
                else:
                    cats = (
                        sorted(data[col].dropna().unique().tolist())
                        if col in data.columns
                        else []
                    )
                color_dict = {
                    str(c): clr for c, clr in zip(cats, _adata_uns[uns_key])
                }
                return {v: color_dict.get(v, "gray") for v in vals}
            # Fallback: scanpy default palette cycle with per-level offset
            n = len(vals)
            total_needed = offset + n
            if total_needed <= 20:
                pal = list(palettes.default_20)
            elif total_needed <= 28:
                pal = list(palettes.default_28)
            else:
                pal = list(palettes.default_102)
            colors = [pal[i % len(pal)] for i in range(offset, offset + n)]
            return dict(zip(vals, colors))
        # Dandelion / non-AnnData: use Set2 with per-level offset
        full_pal = sns.color_palette("Set2", max(8, offset + len(vals)))
        return dict(zip(vals, full_pal[offset : offset + len(vals)]))

    def _ordered_vals(col: str) -> list[str]:
        """Return unique string values for col in the correct display order.

        Categorical columns respect their ``.cat.categories`` order (matching
        scanpy). Purely numeric columns sort numerically (1, 2, 10 not 1, 10,
        2). Everything else sorts alphabetically.
        """
        col_data = data_[col]
        if pd.api.types.is_categorical_dtype(col_data):
            present = set(col_data.dropna().astype(str))
            return [
                str(c) for c in col_data.cat.categories if str(c) in present
            ]
        raw_vals = col_data.dropna().unique()
        as_numeric = pd.to_numeric(pd.Series(raw_vals), errors="coerce")
        if as_numeric.notna().all():
            return [str(v) for v in sorted(raw_vals, key=float)]
        return sorted(str(v) for v in raw_vals)

    # Build one colour map per group_by level.
    # _palette_offset ensures each level starts from a fresh position in the
    # auto-assigned palette so that outer-ring colours (level 0) are always
    # visually distinct from inner-ring colours (level 1+).
    level_color_maps: list[dict[str, tuple]] = []
    level_ordered_vals: list[list[str]] = []
    _palette_offset = 0
    for col in group_by_cols:
        unique_vals = _ordered_vals(col)
        level_ordered_vals.append(unique_vals)
        if isinstance(palette, str):
            full_pal = sns.color_palette(
                palette, _palette_offset + len(unique_vals)
            )
            cmap: dict[str, tuple] = dict(
                zip(unique_vals, full_pal[_palette_offset:])
            )
        elif isinstance(palette, dict):
            col_palette = palette.get(col, {})
            if isinstance(col_palette, list):
                cmap = dict(zip(unique_vals, col_palette))
                missing = [v for v in unique_vals if v not in cmap]
                if missing:
                    cmap.update(_auto_colors(col, missing, _palette_offset))
            else:
                cmap = {}
                missing = []
                for v in unique_vals:
                    if v in col_palette:
                        cmap[v] = col_palette[v]
                    else:
                        missing.append(v)
                if missing:
                    cmap.update(_auto_colors(col, missing, _palette_offset))
        else:
            cmap = _auto_colors(col, unique_vals, _palette_offset)
        _palette_offset += len(unique_vals)
        level_color_maps.append(cmap)

    # Side-lookup: id(node) → (level_index, group_value)
    # Avoids embedding extra keys in the dicts that circlify would warn about.
    color_group_lookup: dict[int, tuple[int, str]] = {}

    def _build_hierarchy(
        df: pd.DataFrame,
        levels: list[str],
        depth: int,
        parent_info: tuple[int, str] | None,
    ) -> list[dict]:
        """Recursively build a circlify-compatible hierarchy.

        Each node dict is registered by its object identity so the renderer
        can retrieve the correct per-level colour without polluting the dicts.
        """
        if not levels:
            # Leaves inherit the deepest group_by level's colour.
            clone_sizes = df[clone_].value_counts()
            # Re-apply min_clone_size per group: a clone whose cells are spread
            # across groups can pass the global filter yet show up with size 1
            # inside a group (e.g. shared clonotypes split across samples).
            clone_sizes = clone_sizes[clone_sizes >= min_clone_size]
            if (
                max_clones_per_group is not None
                and len(clone_sizes) > max_clones_per_group
            ):
                clone_sizes = clone_sizes.head(max_clones_per_group)
            leaf_info: tuple[int, str] = (
                parent_info if parent_info is not None else (0, "")
            )
            result = []
            if aggregate_by_size:
                # One circle per distinct clone size; datum = size × count (total cells).
                size_dist = clone_sizes.value_counts().sort_index(
                    ascending=False
                )
                for clone_size, n_clones in size_dist.items():
                    node: dict = {
                        "id": f"n={int(clone_size)}",
                        "datum": int(clone_size) * int(n_clones),
                    }
                    color_group_lookup[id(node)] = leaf_info
                    result.append(node)
            else:
                for cid, cnt in clone_sizes.items():
                    node = {"id": str(cid), "datum": int(cnt)}
                    color_group_lookup[id(node)] = leaf_info
                    result.append(node)
            return result
        current, rest = levels[0], levels[1:]
        result = []
        for grp, gdata in df.groupby(current, observed=True):
            grp_str = str(grp)
            my_info: tuple[int, str] = (depth, grp_str)
            children = _build_hierarchy(gdata, rest, depth + 1, my_info)
            node = {
                "id": grp_str,
                "datum": sum(c["datum"] for c in children),
                "children": children,
            }
            color_group_lookup[id(node)] = my_info
            result.append(node)
        return result

    def _render_circles_on_ax(
        ax: Axes,
        circles: list,
        lookup: dict[int, tuple[int, str]],
        count_groups: bool = True,
    ):
        filled_patches = []
        filled_facecolors = []
        ring_patches = []
        ring_edgecolors = []
        texts = []

        for circle in circles:
            if circle.ex is None:
                continue
            x, y, r = circle.x, circle.y, circle.r
            label = circle.ex.get("id", "")
            level_idx, group_val = lookup.get(id(circle.ex), (0, label))
            level_idx = min(level_idx, len(level_color_maps) - 1)
            color = level_color_maps[level_idx].get(group_val, "gray")
            is_group = "children" in circle.ex

            if is_group:
                _ring_color = (
                    outer_ring_color
                    if outer_ring_color is not None and level_idx == 0
                    else color
                )
                ring_patches.append(mpatches.Circle((x, y), r))
                ring_edgecolors.append(_ring_color)
                if show_group_labels:
                    texts.append(
                        (
                            x,
                            y + r,
                            label,
                            dict(
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontweight="bold",
                                color=_ring_color,
                            ),
                        )
                    )
                if show_count_labels and count_groups and show_enclosure_label:
                    texts.append(
                        (
                            x,
                            y - r - 0.05,
                            str(circle.ex["datum"]),
                            dict(
                                ha="center",
                                va="top",
                                fontsize=7,
                                color=_ring_color,
                            ),
                        )
                    )
            else:
                filled_patches.append(mpatches.Circle((x, y), r))
                filled_facecolors.append(color)
                # For aggregate_by_size nodes (id="n=<size>"), show "size\n(total_cells)"
                _is_agg = label.startswith("n=") and label[2:].isdigit()
                _count_str = (
                    f"{label[2:]}\n({circle.ex['datum']})"
                    if _is_agg
                    else str(circle.ex["datum"])
                )
                if show_clone_labels and show_count_labels:
                    texts.append(
                        (
                            x,
                            y + r * 0.25,
                            label,
                            dict(ha="center", va="center", fontsize=7),
                        )
                    )
                    texts.append(
                        (
                            x,
                            y - r * 0.25,
                            _count_str,
                            dict(ha="center", va="center", fontsize=7),
                        )
                    )
                elif show_clone_labels:
                    texts.append(
                        (
                            x,
                            y,
                            label,
                            dict(ha="center", va="center", fontsize=7),
                        )
                    )
                elif show_count_labels:
                    texts.append(
                        (
                            x,
                            y,
                            _count_str,
                            dict(ha="center", va="center", fontsize=7),
                        )
                    )

        if filled_patches:
            pc = PatchCollection(
                filled_patches,
                facecolors=filled_facecolors,
                edgecolors="white",
                linewidths=0.5,
                alpha=alpha,
                zorder=1,
            )
            ax.add_collection(pc)

        if ring_patches:
            rc = PatchCollection(
                ring_patches,
                facecolors="none",
                edgecolors=ring_edgecolors,
                linewidths=2,
                zorder=2,
            )
            ax.add_collection(rc)

        for tx, ty, ts, kw in texts:
            ax.text(tx, ty, ts, **kw)

    def _set_ax_limits(ax: Axes, circles) -> None:
        xs = [c.x for c in circles if c.ex is not None]
        ys = [c.y for c in circles if c.ex is not None]
        rs = [c.r for c in circles if c.ex is not None]
        if xs:
            margin = 0.05
            ax.set_xlim(
                min(cx - cr for cx, cr in zip(xs, rs)) - margin,
                max(cx + cr for cx, cr in zip(xs, rs)) + margin,
            )
            ax.set_ylim(
                min(cy - cr for cy, cr in zip(ys, rs)) - margin,
                max(cy + cr for cy, cr in zip(ys, rs)) + margin,
            )
        else:
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

    def _build_legend_handles() -> list:
        if show_legend is None or show_legend is True:
            _legend_levels = None
        elif isinstance(show_legend, str):
            _legend_levels = {show_legend}
        else:
            _legend_levels = set(show_legend)
        handles = []
        first_added = True
        for col, cmap, ordered_vals in zip(
            group_by_cols, level_color_maps, level_ordered_vals
        ):
            if _legend_levels is not None and col not in _legend_levels:
                continue
            if not first_added:
                handles.append(
                    mpatches.Patch(
                        facecolor="none", edgecolor="none", label=" "
                    )
                )
            first_added = False
            handles.append(
                mpatches.Patch(facecolor="none", edgecolor="none", label=col)
            )
            handles += [
                mpatches.Patch(color=cmap[v], label=v) for v in ordered_vals
            ]
        return handles

    if as_subplots:
        _top_col_str = data_[group_by_cols[0]].astype(str)
        top_groups = level_ordered_vals[0]
        n_subplots = len(top_groups)
        _n_col = (
            n_col
            if n_col is not None
            else (4 if n_subplots > 4 else n_subplots)
        )
        _n_row = n_row if n_row is not None else -(-n_subplots // _n_col)

        # Pre-build all sub-hierarchies so totals are available for scaling
        _sub_builds: list[tuple[str, list[dict], dict]] = []
        for grp_val in top_groups:
            # sub_data = data_[data_[group_by_cols[0]].astype(str) == grp_val]
            sub_data = data_[_top_col_str == grp_val]
            color_group_lookup.clear()
            sub_hier = _build_hierarchy(
                sub_data, group_by_cols[1:], 1, (0, grp_val)
            )
            _sub_builds.append((grp_val, sub_hier, dict(color_group_lookup)))

        if scale_subplots:
            _sub_totals = [
                sum(c["datum"] for c in hier) if hier else 0
                for _, hier, _ in _sub_builds
            ]
            _max_total = max(_sub_totals) if any(_sub_totals) else 1

        fig, axes = plt.subplots(
            _n_row,
            _n_col,
            figsize=(figsize[0] * _n_col, figsize[1] * _n_row),
            squeeze=False,
        )
        axes_flat: np.ndarray = axes.flatten()

        for idx, (grp_val, sub_hierarchy, lookup) in enumerate(_sub_builds):
            ax = axes_flat[idx]
            if not sub_hierarchy:
                ax.axis("off")
                continue
            if scale_subplots:
                _base_r = (_sub_totals[idx] / _max_total) ** 0.5
                _enc_r = (
                    _base_r * scale_factor
                    if scale_factor is not None
                    else _base_r
                )
            else:
                _enc_r = scale_factor if scale_factor is not None else 1.0
            _target_enc = circlify.Circle(0, 0, _enc_r)
            sub_circles = _pack(sub_hierarchy, _target_enc, packer)
            ax.set_aspect("equal")
            _render_circles_on_ax(ax, sub_circles, lookup, count_groups=False)
            # Outer ring for the top-level group (mirrors single-panel behaviour)
            _outer_color = (
                outer_ring_color
                if outer_ring_color is not None
                else level_color_maps[0].get(grp_val, "gray")
            )
            ax.add_patch(
                mpatches.Circle(
                    (0, 0),
                    _enc_r,
                    fill=False,
                    edgecolor=_outer_color,
                    linewidth=2,
                )
            )
            if show_count_labels and show_enclosure_label:
                _total = sum(c["datum"] for c in sub_hierarchy)
                ax.text(
                    0,
                    -_enc_r - 0.05,
                    str(_total),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=_outer_color,
                )
            if scale_subplots or scale_factor is not None:
                _margin = 0.05
                ax.set_xlim(-1.0 - _margin, 1.0 + _margin)
                ax.set_ylim(-1.0 - _margin, 1.0 + _margin)
            else:
                _set_ax_limits(ax, sub_circles)
            ax.set_title(grp_val, fontsize=10, fontweight="bold")
            ax.axis("off")

        for idx in range(n_subplots, _n_row * _n_col):
            axes_flat[idx].axis("off")

        if show_legend is not False:
            handles = _build_legend_handles()
            if handles:
                axes_flat[n_subplots - 1].legend(
                    handles=handles, **legend_kwargs
                )

        if title is not None:
            fig.suptitle(title)

        return fig, list(axes_flat[:n_subplots])

    hierarchy = _build_hierarchy(data_, group_by_cols, 0, None)

    if scale_factor is not None:
        _target_enc = circlify.Circle(0, 0, scale_factor)
    else:
        _target_enc = circlify.Circle(0, 0, 1)

    circles = _pack(hierarchy, _target_enc, packer)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    _render_circles_on_ax(ax, circles, color_group_lookup)
    if scale_factor is not None:
        _margin = 0.05
        ax.set_xlim(-1.0 - _margin, 1.0 + _margin)
        ax.set_ylim(-1.0 - _margin, 1.0 + _margin)
    else:
        _set_ax_limits(ax, circles)

    if show_legend is not False:
        handles = _build_legend_handles()
        ax.legend(handles=handles, **legend_kwargs)

    if title is not None:
        ax.set_title(title)
    ax.axis("off")

    return fig, ax


@contextmanager
def _temporary_obs_columns(adata: AnnData, mudata: MuData | None, **kwargs):
    """Temporarily add columns from submodalities or shared obs to adata.obs."""

    if mudata is None:
        # plain AnnData, nothing to do
        yield kwargs
        return

    added = {}
    try:
        for key, value in kwargs.items():
            if key in {"color", "size", "shape"} and isinstance(value, str):
                if ":" in value:
                    # case: "mod:col"
                    mod, col = value.split(":", 1)
                    if mod not in mudata.mod:
                        raise KeyError(f"MuData has no modality '{mod}'")
                    sub = mudata.mod[mod]
                    if col not in sub.obs.columns:
                        raise KeyError(
                            f"'{col}' not found in mudata.mod['{mod}'].obs"
                        )
                    temp_col = f"{mod}:{col}"
                    adata.obs[temp_col] = sub.obs[col].reindex(adata.obs.index)
                    kwargs[key] = temp_col
                    added[temp_col] = None
                else:
                    # case: shared obs in mudata.obs
                    if value not in mudata.obs:
                        raise KeyError(f"'{value}' not found in mudata.obs")
                    adata.obs[value] = mudata.obs[value].reindex(
                        adata.obs.index
                    )
                    kwargs[key] = value
                    added[value] = None

        yield kwargs
    finally:
        # cleanup temporary columns
        for temp_col in added:
            adata.obs.drop(columns=temp_col, inplace=True, errors="ignore")
