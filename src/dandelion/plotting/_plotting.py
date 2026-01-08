from __future__ import annotations

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
from matplotlib.figure import Figure
from nxviz import annotate

from scanpy.plotting import palettes
from scanpy.plotting._tools.scatterplots import embedding
from typing import Callable, Literal, TYPE_CHECKING

from dandelion.utilities._core import Dandelion

if TYPE_CHECKING:
    from anndata import AnnData
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
    vdj_data: AnnData | Dandelion,
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
    vdj_data : AnnData | Dandelion
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
    if isinstance(vdj_data, Dandelion):
        data = vdj_data._metadata.copy()
    elif isinstance(vdj_data, AnnData):
        data = vdj_data.obs.copy()

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
    vdj_data: AnnData | Dandelion,
    color: str,
    groupby: str | None,
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
    vdj_data : AnnData | Dandelion
        Dandelion or AnnData object.
    color : str
        column name in metadata for plotting in bar plot.
    groupby : str | None
        column name in metadata to split by during plotting.
    figsize : tuple[float, float], optional
        figure size.
    normalize : bool, optional
        if True, will return as proportion out of 1, otherwise False will return counts.
    title : str | None, optional
        title of plot.
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
    if isinstance(vdj_data, Dandelion):
        data = vdj_data._metadata.copy()
    elif isinstance(vdj_data, AnnData):
        data = vdj_data.obs.copy()
    # quick fix to prevent dropping of nan
    data[groupby] = [str(l) for l in data[groupby]]

    min_size = min_clone_size

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    size = data[clone_].value_counts()
    keep = list(size[size >= min_size].index)
    data_ = data[data[clone_].isin(keep)]

    dat_ = pd.DataFrame(
        data_.groupby(color)[groupby]
        .value_counts(normalize=normalize)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_order = pd.DataFrame(data[color].value_counts(normalize=normalize))
    dat_ = dat_.pivot(index=color, columns=groupby, values="value")
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
    vdj_data: Dandelion,
    color: str,
    groupby: str,
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
    vdj_data : Dandelion
        Dandelion object.
    color : str
        column name in metadata for plotting in bar plot.
    groupby : str
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
    data = vdj_data._data.copy()
    if "ambiguous" in data:
        data = data[data["ambiguous"] == "F"].copy()

    if type(locus) is not list:
        locus = [locus]
    data = data[data["locus"].isin(locus)].copy()
    data[groupby] = [str(l) for l in data[groupby]]
    dat_ = pd.DataFrame(
        data.groupby(color)[groupby]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_[color] = pd.to_numeric(dat_[color], errors="coerce")
    dat_.sort_values(by=color)
    dat_2 = dat_.pivot(index=color, columns=groupby, values="value")
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
    gex_data: AnnData,
    groupby: str,
    colorby: str | None = None,
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
    gex_data : AnnData
        AnnData object.
    groupby : str
        column name in obs for collapsing to nodes in circos plot.
    colorby : str | None, optional
        column name in obs for grouping and color of nodes in plot. Must be a same or subcategory of the `groupby` categories e.g. `groupby="group_tissue", colorby="tissue"`.
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

    if isinstance(gex_data, AnnData):
        data = gex_data.obs.copy()
        # get rid of problematic rows that appear because of category conversion?
        if "clone_overlap" in gex_data.uns:
            overlap = gex_data.uns["clone_overlap"].copy()
        else:
            raise KeyError(
                "`clone_overlap` not found in `gex_data.uns`. Did you run `tl.clone_overlap`?"
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

    colorby = groupby if colorby is None else colorby
    # create graph
    G = nx.Graph()
    # add in the nodes
    G.add_nodes_from(
        [(p, {str(colorby): d}) for p, d in zip(data[groupby], data[colorby])]
    )

    # unpack the edgelist and add to the graph
    for edge in edges:
        G.add_edges_from(edges[edge])

    if not weighted_overlap:
        weighted_attr = None
    else:
        weighted_attr = "weight"

    if color_mapping is None:
        if str(colorby) + "_colors" in adata.uns:
            if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
                colorby_dict = dict(
                    zip(
                        list(gex_data.obs[str(colorby)].cat.categories),
                        gex_data.uns[str(colorby) + "_colors"],
                    )
                )
            else:
                colorby_dict = dict(
                    zip(
                        list(gex_data.obs[str(colorby)].unique()),
                        gex_data.uns[str(colorby) + "_colors"],
                    )
                )
        else:
            if len(gex_data.obs[str(colorby)].unique()) <= 20:
                pal = cycle(palettes.default_20)
            elif len(gex_data.obs[str(colorby)].unique()) <= 28:
                pal = cycle(palettes.default_28)
            else:
                pal = cycle(palettes.default_102)
            colorby_dict = dict(
                zip(list(gex_data.obs[str(colorby)].unique()), pal)
            )
    else:
        if type(color_mapping) is dict:
            colorby_dict = color_mapping
        else:
            if pd.api.types.is_categorical_dtype(data[groupby]):
                colorby_dict = dict(
                    zip(list(data[str(colorby)].cat.categories), color_mapping)
                )
            else:
                colorby_dict = dict(
                    zip(sorted(list(set(data[str(colorby)]))), color_mapping)
                )
    df = data[[groupby, colorby]]
    if groupby == colorby:
        df = data[[groupby]]
        df = (
            df.sort_values(groupby)
            .drop_duplicates(subset=groupby, keep="first")
            .reset_index(drop=True)
        )
    else:
        df = (
            df.sort_values(colorby)
            .drop_duplicates(subset=groupby, keep="first")
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
            group_by=colorby,
            node_color_by=colorby,
            edge_lw_by=weighted_attr,
            node_palette=colorby_dict,
        )  # group_by
        if node_labels:
            annotate.circos_group(
                G,
                group_by=colorby,
                midpoint=False,
                fontdict={"size": node_label_size},
            )
        annotate.node_colormapping(
            G,
            color_by=colorby,
            palette=colorby_dict,
            legend_kwargs=legend_kwargs,
        )
    if save is not None:
        plt.savefig(save, bbox_inches="tight", **kwargs)
    if return_graph:
        return G


def productive_ratio(
    gex_data: AnnData,
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
    gex_data : AnnData
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
    res = gex_data.uns["productive_ratio"]["results"]
    locus = gex_data.uns["productive_ratio"]["locus"]
    groupby = gex_data.uns["productive_ratio"]["groupby"]

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x=groupby, y="productive+non-productive", data=res, color=palette[0]
    )
    ax = sns.barplot(
        x=groupby, y="productive", data=res, color=palette[1], ax=ax
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
