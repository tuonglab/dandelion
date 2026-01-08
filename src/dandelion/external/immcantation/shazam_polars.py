import os
import re
import functools
import warnings

import pandas as pd  # used only for rpy2 interop
import polars as pl
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

from dandelion.utilities._polars import (
    DandelionPolars,
    load_polars,
    _sanitize_data_polars,
    _write_airr,
)


def quantify_mutations(
    data: DandelionPolars | pl.DataFrame | pl.LazyFrame | str,
    split_locus: bool = False,
    sequence_column: str | None = None,
    germline_column: str | None = None,
    region_definition: str | None = None,
    mutation_definition: str | None = None,
    frequency: bool = False,
    combine: bool = True,
    **kwargs,
) -> pl.DataFrame | None:
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
    except Exception:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    sh = importr("shazam")
    base = importr("base")
    # Load input as Polars
    if isinstance(data, DandelionPolars):
        dat = load_polars(data._data)
    else:
        dat = load_polars(data)

    warnings.filterwarnings("ignore")

    # Sanitize and filter using Polars
    dat = _sanitize_data_polars(dat)
    if isinstance(dat, pl.LazyFrame):
        dat = dat.collect()
    if "ambiguous" in dat.columns:
        dat_ = dat.filter(pl.col("ambiguous") == "F")
    else:
        dat_ = dat

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

    # Convert to pandas only for rpy2 interop
    if split_locus is False:
        dat_r = safe_py2rpy(
            dat_.with_columns(pl.col("*").cast(pl.String)).to_pandas()
        )
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
        dat_h = dat_.filter(pl.col("locus") == "IGH")
        dat_l = dat_.filter(pl.col("locus").is_in(["IGK", "IGL"]))

        dat_h_r = safe_py2rpy(
            dat_h.with_columns(pl.col("*").cast(pl.String)).to_pandas()
        )
        dat_l_r = safe_py2rpy(
            dat_l.with_columns(pl.col("*").cast(pl.String)).to_pandas()
        )

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

    # Identify new columns produced by R
    pd_df.set_index("sequence_id", inplace=True, drop=False)
    dat_cols = dat_.columns
    cols_to_return = [c for c in pd_df.columns if c not in dat_cols]
    if len(cols_to_return) < 1:
        cols_to_return = list(
            filter(re.compile("mu_.*").match, [c for c in pd_df.columns])
        )
    else:
        cols_to_return = cols_to_return
    # Convert R output to Polars and merge back
    # Clean R NA types more thoroughly before converting to Polars
    for col in pd_df.columns:
        if pd_df[col].dtype == object:
            pd_df[col] = pd_df[col].apply(
                lambda x: (
                    None
                    if (
                        hasattr(x, "__class__")
                        and (
                            "NACharacterType" in str(type(x).__name__)
                            or "NAType" in str(type(x).__name__)
                        )
                    )
                    else x
                )
            )
    r_out_pl = pl.from_pandas(pd_df)

    if isinstance(data, DandelionPolars):
        # Append new columns to data._data via sequence_id join
        base_df = data._data
        if isinstance(base_df, pl.LazyFrame):
            base_df = base_df.collect()
        add_df = r_out_pl.select(["sequence_id"] + cols_to_return)
        data._data = base_df.join(add_df, on="sequence_id", how="left")

        # Build metadata in Polars
        if split_locus is False:
            metadata_ = (
                data._data.select(["cell_id"] + cols_to_return)
                .with_columns(
                    [pl.col(c).cast(pl.Float64) for c in cols_to_return]
                )
                .group_by("cell_id")
                .sum()
            )
        else:
            grouped = (
                data._data.select(["locus", "cell_id"] + cols_to_return)
                .with_columns(
                    [pl.col(c).cast(pl.Float64) for c in cols_to_return]
                )
                .group_by(["locus", "cell_id"])
                .sum()
            )
            loci = grouped.select("locus").unique().to_series().to_list()
            metadatas: list[pl.DataFrame] = []
            for loc in loci:
                tmp = grouped.filter(pl.col("locus") == loc).drop("locus")
                tmp = tmp.rename({c: f"{c}_{loc}" for c in cols_to_return})
                metadatas.append(tmp)
            # Outer join across all locus-specific summaries on cell_id
            if len(metadatas) > 0:
                metadata_ = functools.reduce(
                    lambda left, right: left.join(
                        right, on="cell_id", how="outer"
                    ),
                    metadatas,
                )
            else:
                metadata_ = pl.DataFrame({"cell_id": []})

        data._metadata = metadata_
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
        # Merge results back into Polars DataFrame and return/save
        base_df = dat_
        # Drop existing mutation columns if they exist to avoid conflicts
        existing_cols_to_drop = [
            c for c in cols_to_return if c in base_df.columns
        ]
        if existing_cols_to_drop:
            base_df = base_df.drop(existing_cols_to_drop)
        add_df = r_out_pl.select(["sequence_id"] + cols_to_return)
        out_df = base_df.join(add_df, on="sequence_id", how="left")
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            logg.info(
                " finished", time=start, deep=("Returning Polars DataFrame\n")
            )
            return out_df
        elif os.path.isfile(str(data)):
            logg.info(
                " finished",
                time=start,
                deep=("saving DataFrame at {}\n".format(str(data))),
            )
            _write_airr(out_df, data)
            return out_df
    return None


def calculate_threshold(
    data: DandelionPolars | pl.DataFrame | pl.LazyFrame | str,
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
    except Exception:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    # Load input as Polars
    if isinstance(data, DandelionPolars):
        dat = load_polars(data._data)
    else:
        dat = load_polars(data)
    warnings.filterwarnings("ignore")

    sh = importr("shazam")
    # Choose v_call column in Polars
    dat_cols = (
        dat.collect_schema().names()
        if isinstance(dat, pl.LazyFrame)
        else dat.columns
    )
    v_call = "v_call_genotyped" if "v_call_genotyped" in dat_cols else "v_call"
    model_ = "ham" if model is None else model
    norm_ = "len" if normalize_method is None else normalize_method
    threshold_method_ = (
        "density" if threshold_method is None else threshold_method
    )
    subsample_ = NULL if subsample is None else subsample

    # Sanitize using Polars; convert to pandas only for rpy2
    dat_pl = _sanitize_data_polars(dat)
    if isinstance(dat_pl, pl.LazyFrame):
        dat_pl = dat_pl.collect()
    if mode == "heavy":
        dat_h = dat_pl.filter(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
        dat_h_r = safe_py2rpy(
            dat_h.with_columns(pl.col("*").cast(pl.String)).to_pandas()
        )
        dist_ham = sh.distToNearest(
            dat_h_r, vCallColumn=v_call, model=model_, normalize=norm_, **kwargs
        )
    elif mode == "single-cell":
        dat_r = safe_py2rpy(
            dat_pl.with_columns(pl.col("*").cast(pl.String)).to_pandas()
        )
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
        except Exception:
            logg.info(
                "Rerun this after filtering. For now, switching to heavy mode."
            )
            dat_h = dat_pl.filter(pl.col("locus").is_in(["IGH", "TRB", "TRD"]))
            # drop "cell_id" column as it causes issues
            dat_h = dat_h.drop("cell_id")
            dat_h_r = safe_py2rpy(
                dat_h.with_columns(pl.col("*").cast(pl.String)).to_pandas()
            )

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


def safe_py2rpy(df: pd.DataFrame) -> object:
    """Convert pandas DataFrame to R object safely."""
    try:
        import rpy2
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
    except Exception:
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
    except Exception:
        df = df.astype(str)
        with localconverter(
            rpy2.robjects.default_converter + pandas2ri.converter
        ):
            return pandas2ri.py2rpy(df)


def safe_rpy2py(r_object):
    """Convert R object to pandas DataFrame or other Python object safely."""
    try:
        import rpy2
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
    except Exception:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with shazam's tutorials."
            )
        )
    with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
        result = rpy2.robjects.conversion.rpy2py(r_object)
    # Replace R NA types with proper pandas NA/None for downstream Polars compatibility
    # Only apply to DataFrame objects
    if isinstance(result, pd.DataFrame):
        for col in result.columns:
            if result[col].dtype == object:
                # Replace various R NA types
                result[col] = result[col].apply(
                    lambda x: (
                        None
                        if (
                            hasattr(x, "__class__")
                            and "NACharacterType" in str(type(x).__name__)
                        )
                        else x
                    )
                )
    return result
