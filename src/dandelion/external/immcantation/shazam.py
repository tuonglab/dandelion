import os
import re
import functools
import warnings

import pandas as pd
import numpy as np

from scanpy import logging as logg
from typing import Literal

from plotnine import (
    ggplot,
    options,
    aes,
    xlab,
    ylab,
    facet_wrap,
    theme,
    annotate,
    theme_bw,
    geom_histogram,
    geom_vline,
    save_as_pdf_pages,
)

from dandelion.base.core._core import Dandelion, load_data
from dandelion.base.utilities._io import write_airr
from dandelion.utilities._utilities import (
    sanitize_data,
    sanitize_data_for_saving,
)


def quantify_mutations(
    data: Dandelion | str,
    split_locus: bool = False,
    sequence_column: str | None = None,
    germline_column: str | None = None,
    region_definition: str | None = None,
    mutation_definition: str | None = None,
    frequency: bool = False,
    combine: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Run basic mutation load analysis.

    Implemented in `shazam` https://shazam.readthedocs.io/en/stable/vignettes/Mutation-Vignette.

    Parameters
    ----------
    data : Dandelion | str
        Dandelion object, file path to AIRR file.
    split_locus : bool, optional
        whether to return the results for heavy chain and light chain separately.
    sequence_column : str | None, optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/observedMutations
    germline_column : str | None, optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/observedMutations
    region_definition : str | None, optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/IMGT_SCHEMES/
    mutation_definition : str | None, optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/MUTATION_SCHEMES/
    frequency : bool, optional
        whether to return the results a frequency or counts.
    combine : bool, optional
        whether to return the results for replacement and silent mutations separately.
    **kwargs
        passed to shazam::observedMutations.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame holding mutation information.
    """
    start = logg.info("Quantifying mutations")
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    sh = importr("shazam")
    base = importr("base")
    if isinstance(data, Dandelion):
        dat = load_data(data._data)
    else:
        dat = load_data(data)

    warnings.filterwarnings("ignore")

    dat = sanitize_data(dat)

    if "ambiguous" in dat:
        dat_ = dat[dat["ambiguous"] == "F"].copy()
    else:
        dat_ = dat.copy()

    # sanitize before passing to R
    dat_, _ = sanitize_data_for_saving(dat_)

    seq_ = "sequence_alignment" if sequence_column is None else sequence_column
    germline_ = (
        "germline_alignment_d_mask"
        if germline_column is None
        else germline_column
    )
    reg_d = NULL if region_definition is None else base.get(region_definition)
    mut_d = (
        NULL if mutation_definition is None else base.get(mutation_definition)
    )

    if split_locus is False:
        dat_ = dat_.where(dat_.isna(), dat_.astype(str))
        dat_r = safe_py2rpy(dat_)
        results = sh.observedMutations(
            dat_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        pd_df = safe_rpy2py(results)
    else:
        dat_h = dat_[dat_["locus"] == "IGH"].copy()
        dat_l = dat_[dat_["locus"].isin(["IGK", "IGL"])].copy()

        dat_h = dat_h.where(dat_h.isna(), dat_h.astype(str))
        dat_h_r = safe_py2rpy(dat_h)

        dat_l = dat_l.where(dat_l.isna(), dat_l.astype(str))
        dat_l_r = safe_py2rpy(dat_l)

        results_h = sh.observedMutations(
            dat_h_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        results_l = sh.observedMutations(
            dat_l_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        results_h = safe_rpy2py(results_h)
        results_l = safe_rpy2py(results_l)
        pd_df = pd.concat([results_h, results_l])

    pd_df.set_index("sequence_id", inplace=True, drop=False)
    # this doesn't actually catch overwritten columns
    cols_to_return = pd_df.columns.difference(dat.columns)
    if len(cols_to_return) < 1:
        cols_to_return = list(
            filter(re.compile("mu_.*").match, [c for c in pd_df.columns])
        )
    else:
        cols_to_return = cols_to_return

    res = {}
    if isinstance(data, Dandelion):
        for x in cols_to_return:
            res[x] = list(pd_df[x])
            # TODO: str will make it work for the back and forth conversion with rpy2. but maybe can use a better option
            dat_[x] = [str(r) for r in res[x]]
            data._data[x] = pd.Series(dat_[x])
        if split_locus is False:
            metadata_ = data._data[["cell_id"] + list(cols_to_return)]
        else:
            metadata_ = data._data[["locus", "cell_id"] + list(cols_to_return)]
        for x in cols_to_return:
            metadata_[x] = metadata_[x].astype(float)

        if split_locus is False:
            metadata_ = metadata_.groupby("cell_id").sum()
        else:
            metadata_ = metadata_.groupby(["locus", "cell_id"]).sum()
            metadatas = []
            for x in list(set(data._data["locus"])):
                tmp = metadata_.iloc[
                    metadata_.index.isin([x], level="locus"), :
                ]
                tmp.index = tmp.index.droplevel()
                tmp.columns = [c + "_" + str(x) for c in tmp.columns]
                metadatas.append(tmp)
            metadata_ = functools.reduce(
                lambda x, y: pd.merge(
                    x, y, left_index=True, right_index=True, how="outer"
                ),
                metadatas,
            )

        metadata_.index.name = None
        data._data = sanitize_data(data._data)
        if data._metadata is None:
            data._metadata = metadata_
        else:
            for x in metadata_.columns:
                data._metadata[x] = pd.Series(metadata_[x])
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'data', contig-indexed AIRR table\n"
                "   'metadata', cell-indexed observations table\n"
            ),
        )
    else:
        for x in cols_to_return:
            res[x] = list(pd_df[x])
            # TODO: str will make it work for the back and forth conversion with rpy2. but maybe can use a better option
            dat[x] = [str(r) for r in res[x]]
        # dat = sanitize_data(dat)
        if isinstance(data, pd.DataFrame):
            logg.info(" finished", time=start, deep=("Returning DataFrame\n"))
            return dat
        elif os.path.isfile(data):
            logg.info(
                " finished",
                time=start,
                deep=("saving DataFrame at {}\n".format(str(data))),
            )
            write_airr(dat, data)


def calculate_threshold(
    data: Dandelion | pd.DataFrame | str,
    mode: Literal["single-cell", "heavy"] = "single-cell",
    manual_threshold: float | None = None,
    VJthenLen: bool = False,
    model: (
        Literal[
            "ham",
            "aa",
            "hh_s1f",
            "hh_s5f",
            "mk_rs1nf",
            "hs1f_compat",
            "m1n_compat",
        ]
        | None
    ) = None,
    normalize_method: Literal["len"] | None = None,
    threshold_method: Literal["gmm", "density"] | None = None,
    edge: float | None = None,
    cross: list[float] | None = None,
    subsample: int | None = None,
    threshold_model: (
        Literal["norm-norm", "norm-gamma", "gamma-norm", "gamma-gamma"] | None
    ) = None,
    cutoff: Literal["optimal", "intersect", "user"] | None = None,
    sensitivity: float | None = None,
    specificity: float | None = None,
    plot: bool = True,
    plot_group: str | None = None,
    figsize: tuple[float, float] = (4.5, 2.5),
    save_plot: str | None = None,
    n_cpus: int = 1,
    **kwargs,
) -> float:
    """
    Calculating nearest neighbor distances for tuning clonal assignment with `shazam`.

    https://shazam.readthedocs.io/en/stable/vignettes/DistToNearest-Vignette/

    Runs the following:

    distToNearest
        Get non-zero distance of every heavy chain (IGH) sequence (as defined by sequenceColumn) to its nearest sequence
        in a partition of heavy chains sharing the same V gene, J gene, and junction length (VJL), or in a partition of
        single cells with heavy chains sharing the same heavy chain VJL combination, or of single cells with heavy and
        light chains sharing the same heavy chain VJL and light chain VJL combinations.
    findThreshold
        automtically determines an optimal threshold for clonal assignment of Ig sequences using a vector of nearest
        neighbor distances. It provides two alternative methods using either a Gamma/Gaussian Mixture Model fit
        (threshold_method="gmm") or kernel density fit (threshold_method="density").

    Parameters
    ----------
    data : Dandelion | pd.DataFrame | str
        input `Danelion`, AIRR data as pandas DataFrame or path to file.
    mode : Literal["single-cell", "heavy"], optional
        accepts one of "heavy" or "single-cell".
        Refer to https://shazam.readthedocs.io/en/stable/vignettes/DistToNearest-Vignette.
    manual_threshold : float | None, optional
        value to manually plot in histogram.
    VJthenLen : bool, optional
        logical value specifying whether to perform partitioning as a 2-stage process.
        If True, partitions are made first based on V and J gene, and then further split
        based on junction lengths corresponding to sequenceColumn.
        If False, perform partition as a 1-stage process during which V gene, J gene, and junction length
        are used to create partitions simultaneously.
        Defaults to False.
    model : Literal["ham", "aa", "hh_s1f", "hh_s5f", "mk_rs1nf", "hs1f_compat", "m1n_compat", ] | None, optional
        underlying SHM model, which must be one of "ham","aa","hh_s1f","hh_s5f","mk_rs1nf","hs1f_compat","m1n_compat".
    normalize_method : Literal["len"] | None, optional
        method of normalization. The default is "len", which divides the distance by the length of the sequence group.
        If "none" then no normalization if performed.
    threshold_method : Literal["gmm", "density"] | None, optional
        string defining the method to use for determining the optimal threshold. One of "gmm" or "density".
    edge : float | None, optional
        upper range as a fraction of the data density to rule initialization of Gaussian fit parameters.
        Default value is 0.9 (or 90). Applies only when threshold_method="density".
    cross : list[float] | None, optional
        supplementary nearest neighbor distance vector output from distToNearest for initialization of the Gaussian fit
        parameters. Applies only when method="gmm".
    subsample : int | None, optional
        maximum number of distances to subsample to before threshold detection.
    threshold_model : Literal["norm-norm", "norm-gamma", "gamma-norm", "gamma-gamma"] | None, optional
        allows the user to choose among four possible combinations of fitting curves: "norm-norm", "norm-gamma",
        "gamma-norm", and "gamma-gamma". Applies only when method="gmm".
    cutoff : Literal["optimal", "intersect", "user"] | None, optional
        method to use for threshold selection: the optimal threshold "optimal", the intersection point of the two fitted
        curves "intersect", or a value defined by user for one of the sensitivity or specificity "user". Applies only
        when method="gmm".
    sensitivity : float | None, optional
        sensitivity required. Applies only when method="gmm" and cutoff="user".
    specificity : float | None, optional
        specificity required. Applies only when method="gmm" and cutoff="user".
    plot : bool, optional
        whether or not to return plot.
    plot_group : str | None, optional
        determines the fill color and facets.
    figsize : tuple[float, float], optional
        size of plot.
    save_plot : str | None, optional
        if specified, plot will be save with this path.
    n_cpus : int, optional
        number of cpus to run `distToNearest`. defaults to 1.
    **kwargs
        passed to shazam's `distToNearest <https://shazam.readthedocs.io/en/stable/topics/distToNearest/>`__.

    Returns
    -------
    float
        threshold value for clonal assignment in DefineClones.

    Raises
    ------
    ValueError
        if automatic thresholding failed.
    """
    logg.info("Calculating threshold")
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import FloatVector
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    if isinstance(data, Dandelion):
        dat = load_data(data._data)
    elif isinstance(data, pd.DataFrame) or os.path.isfile(str(data)):
        dat = load_data(data)
        warnings.filterwarnings("ignore")

    sh = importr("shazam")
    v_call = (
        "v_call_genotyped" if "v_call_genotyped" in dat.columns else "v_call"
    )
    model_ = "ham" if model is None else model
    norm_ = "len" if normalize_method is None else normalize_method
    threshold_method_ = (
        "density" if threshold_method is None else threshold_method
    )
    subsample_ = NULL if subsample is None else subsample

    # sanitize before passing to R
    dat, _ = sanitize_data_for_saving(dat)
    if mode == "heavy":
        dat_h = dat[dat["locus"].isin(["IGH", "TRB", "TRD"])].copy()
        dat_h_r = safe_py2rpy(dat_h)
        dist_ham = sh.distToNearest(
            dat_h_r, vCallColumn=v_call, model=model_, normalize=norm_, **kwargs
        )
    elif mode == "single-cell":
        dat_r = safe_py2rpy(dat)
        try:
            dist_ham = sh.distToNearest(
                dat_r,
                cellIdColumn="cell_id",
                locusColumn="locus",
                VJthenLen=VJthenLen,
                vCallColumn=v_call,
                normalize=norm_,
                model=model_,
                nproc=n_cpus,
                **kwargs,
            )
        except:
            logg.info(
                "Rerun this after filtering. For now, switching to heavy mode."
            )
            dat_h = dat[dat["locus"].isin(["IGH", "TRB", "TRD"])].copy()
            # drop "cell_id" column as it causes issues
            dat_h = dat_h.drop("cell_id", axis=1)
            dat_h_r = safe_py2rpy(dat_h)

            dist_ham = sh.distToNearest(
                dat_h_r,
                vCallColumn=v_call,
                model=model_,
                normalize=norm_,
                nproc=n_cpus,
                **kwargs,
            )
    dist_ham = safe_rpy2py(dist_ham)
    # Find threshold using density method
    dist = np.array(dist_ham["dist_nearest"])
    if manual_threshold is None:
        if threshold_method_ == "density":
            edge_ = 0.9 if edge is None else edge
            dist_threshold = sh.findThreshold(
                FloatVector(dist[~np.isnan(dist)]),
                method=threshold_method_,
                subsample=subsample_,
                edge=edge_,
            )
            threshold = np.array(dist_threshold.slots["threshold"])[0]
            if np.isnan(threshold):
                logg.info(
                    "      Threshold method 'density' did not return with any values. Switching to method = 'gmm'."
                )
                threshold_method_ = "gmm"
                threshold_model_ = (
                    "gamma-gamma"
                    if threshold_model is None
                    else threshold_model
                )
                cross_ = NULL if cross is None else cross
                cutoff_ = "optimal" if cutoff is None else cutoff
                sen_ = NULL if sensitivity is None else sensitivity
                spc_ = NULL if specificity is None else specificity

                dist_threshold = sh.findThreshold(
                    FloatVector(dist[~np.isnan(dist)]),
                    method=threshold_method_,
                    model=threshold_model_,
                    cross=cross_,
                    subsample=subsample_,
                    cutoff=cutoff_,
                    sen=sen_,
                    spc=spc_,
                )
                dist_threshold = safe_rpy2py(dist_threshold)
                threshold = np.array(dist_threshold.slots["threshold"])[0]
        else:
            threshold_model_ = (
                "gamma-gamma" if threshold_model is None else threshold_model
            )
            cross_ = NULL if cross is None else cross
            cutoff_ = "optimal" if cutoff is None else cutoff
            sen_ = NULL if sensitivity is None else sensitivity
            spc_ = NULL if specificity is None else specificity
            dist_threshold = sh.findThreshold(
                FloatVector(dist[~np.isnan(dist)]),
                method=threshold_method_,
                model=threshold_model_,
                cross=cross_,
                subsample=subsample_,
                cutoff=cutoff_,
                sen=sen_,
                spc=spc_,
            )
            dist_threshold = safe_rpy2py(dist_threshold)
            threshold = np.array(dist_threshold.slots["threshold"])[0]
        if np.isnan(threshold):
            raise ValueError(
                "Automatic thresholding failed. Please visually inspect the resulting distribution fits"
                + " and choose a threshold value manually."
            )
        # dist_ham = pandas2ri.rpy2py_data frame(dist_ham)
        tr = threshold
    else:
        tr = manual_threshold

    if plot:
        options.figure_size = figsize
        if plot_group is None:
            if "sample_id" in dist_ham.columns:
                plot_group = "sample_id"
            else:
                plot_group = None
        else:
            plot_group = plot_group
        if plot_group is None:
            p = (
                ggplot(dist_ham, aes("dist_nearest"))
                + theme_bw()
                + xlab("Grouped Hamming distance")
                + ylab("Count")
                + geom_histogram(binwidth=0.01)
                + geom_vline(
                    xintercept=tr, linetype="dashed", color="blue", size=0.5
                )
                + annotate(
                    "text",
                    x=tr + 0.02,
                    y=10,
                    label="Threshold:\n" + str(np.around(tr, decimals=2)),
                    size=8,
                    color="Blue",
                )
                + theme(legend_position="none")
            )
        else:
            p = (
                ggplot(dist_ham, aes("dist_nearest", fill=str(plot_group)))
                + theme_bw()
                + xlab("Grouped Hamming distance")
                + ylab("Count")
                + geom_histogram(binwidth=0.01)
                + geom_vline(
                    xintercept=tr, linetype="dashed", color="blue", size=0.5
                )
                + annotate(
                    "text",
                    x=tr + 0.02,
                    y=10,
                    label="Threshold:\n" + str(np.around(tr, decimals=2)),
                    size=8,
                    color="Blue",
                )
                + facet_wrap("~" + str(plot_group), scales="free_y")
                + theme(legend_position="none")
            )
        if save_plot is not None:
            save_as_pdf_pages([p], filename=save_plot, verbose=False)
        p.show()

    return tr


def safe_py2rpy(df: pd.DataFrame) -> "rpy2 object":
    """Convert pandas DataFrame to R object safely."""
    try:
        import rpy2
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    try:
        with localconverter(
            rpy2.robjects.default_converter + pandas2ri.converter
        ):
            return pandas2ri.py2rpy(df)
    except:
        df = df.astype(str)
        with localconverter(
            rpy2.robjects.default_converter + pandas2ri.converter
        ):
            return pandas2ri.py2rpy(df)


def safe_rpy2py(r_object) -> pd.DataFrame:
    """Convert R object to pandas DataFrame safely."""
    try:
        import rpy2
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
        return rpy2.robjects.conversion.rpy2py(r_object)
