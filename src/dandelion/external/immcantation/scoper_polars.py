import warnings

import polars as pl

from typing import Literal

from dandelion.utilities._polars import DandelionPolars, _sanitize_data_polars


def identical_clones(
    vdj_data: DandelionPolars,
    method: Literal["nt", "aa"] = "nt",
    junction: str = "junction",
    v_call: str = "v_call",
    j_call: str = "j_call",
    clone_key: str = "clone_id",
    fields: list[str] | None = None,
    cell_id: str | None = "cell_id",
    locus: str = "locus",
    only_heavy: bool = True,
    split_light: bool = True,
    first: bool = False,
    cdr3: bool = False,
    mod3: bool = False,
    max_n: int | None = 0,
    nproc: int = 1,
    verbose: bool = False,
    summarize_clones: bool = True,
    remove_ambiguous: bool = True,
    remove_extra: bool = True,
) -> DandelionPolars:
    """
    Clonal assignment using sequence identity partitioning with Polars.

    https://scoper.readthedocs.io/en/stable/topics/identicalClones/

    This is a wrapper for one of scoper's method to perform clone clustering using Polars
    internally for data manipulation.

    Parameters
    ----------
    vdj_data : DandelionPolars
        a DandelionPolars object containing the airr data.
    method : Literal["nt", "aa"], optional
        one of the "nt" for nucleotide based clustering or "aa" for amino acid based clustering.
    junction : str, optional
        character name of the column containing junction sequences.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone_key : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.

    Returns
    -------
    DandelionPolars
        DandelionPolars object with `.clone_id` column populated.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise ImportError(
            "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
        )

    from dandelion.external.immcantation.scoper import safe_py2rpy, safe_rpy2py

    scp = importr("scoper")

    # Convert to pandas for R interop, then back to polars
    db = (
        vdj_data._data.collect()
        if isinstance(vdj_data._data, pl.LazyFrame)
        else vdj_data._data
    )
    db = _sanitize_data_polars(db)
    db_pandas = db.to_pandas()

    warnings.filterwarnings("ignore")

    if remove_ambiguous:
        if "ambiguous" in db_pandas:
            db_pandas = db_pandas[db_pandas["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db_pandas:
            db_pandas = db_pandas[db_pandas["extra"] == "F"].copy()
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    db_r = safe_py2rpy(db_pandas)
    results = scp.identicalClones(
        db=db_r,
        method=method,
        junction=junction,
        v_call=v_call,
        j_call=j_call,
        clone=clone_key,
        fields=fields,
        cell_id=cell_id,
        locus=locus,
        only_heavy=only_heavy,
        split_light=split_light,
        first=first,
        cdr3=cdr3,
        mod3=mod3,
        max_n=max_n,
        nproc=nproc,
        verbose=verbose,
        summarize_clones=summarize_clones,
    )
    results_dataframe = r["as.data.frame"](results)
    df = safe_rpy2py(results_dataframe)

    # Clean NA_character_ before converting to polars
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: (
                    None
                    if (
                        hasattr(x, "__class__")
                        and "NACharacter" in str(x.__class__.__name__)
                    )
                    else x
                )
            )

    # Convert back to polars
    df_polars = pl.from_pandas(df)
    vdj_data._data = df_polars

    vdj_data.update_metadata(
        clone_key=clone_key,
        retrieve=clone_key,
    )

    return vdj_data


def hierarchical_clones(
    vdj_data: DandelionPolars,
    threshold: float,
    method: Literal["nt", "aa"] = "nt",
    linkage: Literal["single", "average", "complete"] = "single",
    normalize: Literal["len", "none"] = "len",
    junction: str = "junction",
    v_call: str = "v_call",
    j_call: str = "j_call",
    clone_id: str = "clone_id",
    fields: list[str] | None = None,
    cell_id: str | None = "cell_id",
    locus: str = "locus",
    only_heavy: bool = True,
    split_light: bool = True,
    first: bool = False,
    cdr3: bool = False,
    mod3: bool = False,
    max_n: int | None = 0,
    nproc: int = 1,
    verbose: bool = False,
    summarize_clones: bool = True,
    remove_ambiguous: bool = True,
    remove_extra: bool = True,
) -> DandelionPolars:
    """
    Hierarchical clustering approach to clonal assignment with Polars.

    https://scoper.readthedocs.io/en/stable/topics/hierarchicalClones/

    This is a wrapper for one of scoper's method to perform clone clustering using Polars
    internally for data manipulation.

    Parameters
    ----------
    vdj_data : DandelionPolars
        a DandelionPolars object containing the airr data.
    threshold : float
        numeric scalar where the tree should be cut (the distance threshold for clonal grouping).
    method : Literal["nt", "aa"], optional
        one of the "nt" for nucleotide based clustering or "aa" for amino acid based clustering.
    linkage : Literal["single", "average", "complete"], optional
        one of the "single", "average" or "complete" for the hierarchical clustering method.
    normalize : Literal["len", "none"], optional
        method of normalization.
    junction : str, optional
        character name of the column containing junction sequences.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone_id : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.

    Returns
    -------
    DandelionPolars
        DandelionPolars object with `.clone_id` column populated.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise ImportError(
            "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
        )

    from dandelion.external.immcantation.scoper import safe_py2rpy, safe_rpy2py

    scp = importr("scoper")

    # Convert to pandas for R interop, then back to polars
    db = (
        vdj_data._data.collect()
        if isinstance(vdj_data._data, pl.LazyFrame)
        else vdj_data._data
    )
    db_pandas = _sanitize_data_polars(db)
    db_pandas = db.to_pandas()

    warnings.filterwarnings("ignore")
    if remove_ambiguous:
        if "ambiguous" in db_pandas:
            db_pandas = db_pandas[db_pandas["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db_pandas:
            db_pandas = db_pandas[db_pandas["extra"] == "F"].copy()
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    db_r = safe_py2rpy(db_pandas)
    results = scp.hierarchicalClones(
        db=db_r,
        threshold=threshold,
        method=method,
        linkage=linkage,
        normalize=normalize,
        junction=junction,
        v_call=v_call,
        j_call=j_call,
        clone=clone_id,
        fields=fields,
        cell_id=cell_id,
        locus=locus,
        only_heavy=only_heavy,
        split_light=split_light,
        first=first,
        cdr3=cdr3,
        mod3=mod3,
        max_n=max_n,
        nproc=nproc,
        verbose=verbose,
        summarize_clones=summarize_clones,
    )
    results_dataframe = r["as.data.frame"](results)
    df = safe_rpy2py(results_dataframe)

    # Clean NA_character_ before converting to polars
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: (
                    None
                    if (
                        hasattr(x, "__class__")
                        and "NACharacter" in str(x.__class__.__name__)
                    )
                    else x
                )
            )

    # Convert back to polars
    df_polars = pl.from_pandas(df)
    vdj_data._data = df_polars

    vdj_data.update_metadata(
        clone_key=clone_id,
        retrieve=clone_id,
    )

    return vdj_data


def spectral_clones(
    vdj_data: DandelionPolars,
    method: Literal["novj", "vj"] = "novj",
    germline: str = "germline_alignment",
    sequence: str = "sequence_alignment",
    junction: str = "junction",
    v_call: str = "v_call",
    j_call: str = "j_call",
    clone_id: str = "clone_id",
    fields: list[str] | None = None,
    cell_id: str | None = "cell_id",
    locus: str = "locus",
    only_heavy: bool = True,
    split_light: bool = True,
    first: bool = False,
    cdr3: bool = False,
    mod3: bool = False,
    max_n: int | None = 0,
    threshold: float | None = None,
    base_sim: float = 0.95,
    iter_max: int = 1000,
    nstart: int = 1000,
    nproc: int = 1,
    verbose: bool = False,
    summarize_clones: bool = True,
    remove_ambiguous: bool = True,
    remove_extra: bool = True,
) -> DandelionPolars:
    """
    Spectral clustering method for clonal partitioning with Polars.

    https://scoper.readthedocs.io/en/stable/topics/spectralClones/

    This is a wrapper for one of scoper's method to perform clone clustering using Polars
    internally for data manipulation.

    Parameters
    ----------
    vdj_data : DandelionPolars
        a DandelionPolars object containing the airr data.
    method : Literal["novj", "vj"], optional
        one of the "novj" or "vj".
    germline : str, optional
        character name of the column containing the germline or reference sequence.
    sequence : str, optional
        character name of the column containing input sequences.
    junction : str, optional
        character name of the column containing junction sequences.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone_id : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence.
    threshold : float | None, optional
        the supervising cut-off to enforce an upper-limit distance for clonal grouping.
    base_sim : float, optional
        required similarity cut-off for sequences in equal distances from each other.
    iter_max : int, optional
        the maximum number of iterations allowed for kmean clustering step.
    nstart : int, optional
        the number of random sets chosen for kmean clustering initialization.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.

    Returns
    -------
    DandelionPolars
        DandelionPolars object with `.clone_id` column populated.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise ImportError(
            "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
        )

    from dandelion.external.immcantation.scoper import safe_py2rpy, safe_rpy2py

    scp = importr("scoper")

    # Convert to pandas for R interop, then back to polars
    db = (
        vdj_data._data.collect()
        if isinstance(vdj_data._data, pl.LazyFrame)
        else vdj_data._data
    )
    db = _sanitize_data_polars(db)
    db_pandas = db.to_pandas()

    warnings.filterwarnings("ignore")
    if remove_ambiguous:
        if "ambiguous" in db_pandas:
            db_pandas = db_pandas[db_pandas["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db_pandas:
            db_pandas = db_pandas[db_pandas["extra"] == "F"].copy()
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    threshold = NULL if threshold is None else threshold
    db_r = safe_py2rpy(db_pandas)
    results = scp.spectralClones(
        db=db_r,
        method=method,
        germline=germline,
        sequence=sequence,
        junction=junction,
        v_call=v_call,
        j_call=j_call,
        clone=clone_id,
        fields=fields,
        cell_id=cell_id,
        locus=locus,
        only_heavy=only_heavy,
        split_light=split_light,
        targeting_model=NULL,
        len_limit=NULL,
        first=first,
        cdr3=cdr3,
        mod3=mod3,
        max_n=max_n,
        threshold=threshold,
        base_sim=base_sim,
        iter_max=iter_max,
        nstart=nstart,
        nproc=nproc,
        verbose=verbose,
        summarize_clones=summarize_clones,
    )
    results_dataframe = r["as.data.frame"](results)
    df = safe_rpy2py(results_dataframe)

    # Clean NA_character_ before converting to polars
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: (
                    None
                    if (
                        hasattr(x, "__class__")
                        and "NACharacter" in str(x.__class__.__name__)
                    )
                    else x
                )
            )

    # Convert back to polars
    df_polars = pl.from_pandas(df)
    vdj_data._data = df_polars

    vdj_data.update_metadata(
        clone_key=clone_id,
        retrieve=clone_id,
    )

    return vdj_data
