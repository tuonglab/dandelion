import warnings

import pandas as pd

from typing import Literal

from dandelion.utilities._core import Dandelion, load_data
from dandelion.utilities._utilities import (
    sanitize_data,
    sanitize_data_for_saving,
)


def identical_clones(
    vdj: Dandelion,
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
) -> None:
    """
    Clonal assignment using sequence identity partitioning.

    https://scoper.readthedocs.io/en/stable/topics/identicalClones/

    This is a wrapper for one of scoper's method to perform clone clustering. From the original description: identicalClones provides a simple sequence identity based partitioning approach for
    inferring clonal relationships in high-throughput Adaptive Immune Receptor Repertoire sequencing (AIRR-seq) data. This approach partitions B or T cell receptor sequences into clonal groups
    based on junction region sequence identity within partitions that share the same V gene, J gene, and junction length, allowing for ambiguous V or J gene annotations.

    see also https://scoper.readthedocs.io/en/stable/vignettes/Scoper-Vignette/

    Parameters
    ----------
    vdj : Dandelion
        a dandelion object containing the airr data.
    method : Literal["nt", "aa"], optional
        one of the "nt" for nucleotide based clustering or "aa" for amino acid based clustering.
    junction : str, optional
        character name of the column containing junction sequences. Also used to determine sequence length for grouping.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone_key : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping. Sequences with disjoint values in the specified fields will be classified as separate clones.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes. If specified, grouping will be performed in single-cell mode with the behavior governed by the locus and only_heavy arguments. If set to None then the bulk sequencing data is assumed.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping. If True only the first call of the gene assignments is used. If False the union of ambiguous gene assignments is used to group all sequences with any overlapping gene calls.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering (converts IMGT junction to CDR3 region). If True this will also remove records with a junction length less than 7 nucleotides.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3 in nucleotide space.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence before excluding the record from clonal assignment. Default is set to be zero. Set it as "None" for no action.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process. if False (default) process cloning silently.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape and returns a ScoperClones object. If False then a modified input db is returned. When grouping by fields, summarize_clones should be False.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments flagged by `check_contigs`.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
            )
        )
    scp = importr("scoper")

    db = load_data(vdj._data)
    warnings.filterwarnings("ignore")

    # sanitize before passing to R
    db = sanitize_data(db)
    if remove_ambiguous:
        if "ambiguous" in db:
            db = db[db["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db:
            db = db[db["extra"] == "F"].copy()
    # sanitize before passing to R
    db, _ = sanitize_data_for_saving(db)
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    db_r = safe_py2rpy(db)
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
    vdj._data = df.copy()
    vdj.update_metadata(
        clone_key=clone_key,
        retrieve=clone_key,
        retrieve_mode="merge and unique only",
    )


def hierarchical_clones(
    vdj: Dandelion,
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
) -> None:
    """
    Hierarchical clustering approach to clonal assignment.

    https://scoper.readthedocs.io/en/stable/topics/hierarchicalClones/

    This is a wrapper for one of scoper's method to perform clone clustering. From the original description: hierarchicalClones provides a hierarchical agglomerative clustering approach
    to infer clonal relationships in high-throughput Adaptive Immune Receptor Repertoire sequencing (AIRR-seq) data. This approach clusters B or T cell receptor sequences based on junction
    region sequence similarity within partitions that share the same V gene, J gene, and junction length, allowing for ambiguous V or J gene annotations.

    see also https://scoper.readthedocs.io/en/stable/vignettes/Scoper-Vignette/

    Parameters
    ----------
    vdj : Dandelion
        a dandelion object containing the airr data.
    threshold : float
        numeric scalar where the tree should be cut (the distance threshold for clonal grouping).
    method : Literal["nt", "aa"], optional
        one of the "nt" for nucleotide based clustering or "aa" for amino acid based clustering.
    linkage : Literal["single", "average", "complete"], optional
        one of the "single", "average" or "complete" for the hierarchical clustering method.
    normalize : Literal["len", "none"], optional
        method of normalization. The default is "len", which divides the distance by the length of the sequence group. If "none" then no normalization if performed.
    junction : str, optional
        character name of the column containing junction sequences. Also used to determine sequence length for grouping.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping. Sequences with disjoint values in the specified fields will be classified as separate clones.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes. If specified, grouping will be performed in single-cell mode with the behavior governed by the locus and only_heavy arguments. If set to None then the bulk sequencing data is assumed.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping. If True only the first call of the gene assignments is used. If False the union of ambiguous gene assignments is used to group all sequences with any overlapping gene calls.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering (converts IMGT junction to CDR3 region). If True this will also remove records with a junction length less than 7 nucleotides.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3 in nucleotide space.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence before excluding the record from clonal assignment. Default is set to be zero. Set it as "None" for no action.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process. if False (default) process cloning silently.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape and returns a ScoperClones object. If False then a modified input db is returned. When grouping by fields, summarize_clones should be False.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments flagged by `check_contigs`.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
            )
        )
    scp = importr("scoper")

    db = load_data(vdj._data)
    warnings.filterwarnings("ignore")
    db = sanitize_data(db)
    if remove_ambiguous:
        if "ambiguous" in db:
            db = db[db["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db:
            db = db[db["extra"] == "F"].copy()
    # sanitize before passing to R
    db, _ = sanitize_data_for_saving(db)
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    db_r = safe_py2rpy(db)
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
    vdj._data = df.copy()
    vdj.update_metadata(
        clone_key=clone_id,
        retrieve=clone_id,
        retrieve_mode="merge and unique only",
    )


def spectral_clones(
    vdj: Dandelion,
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
) -> None:
    """
    Spectral clustering method for clonal partitioning.

    https://scoper.readthedocs.io/en/stable/topics/spectralClones/

    This is a wrapper for one of scoper's method to perform clone clustering. spectralClones provides an unsupervised spectral clustering approach
    to infer clonal relationships in high-throughput Adaptive Immune Receptor Repertoire sequencing (AIRR-seq) data. This approach clusters B or T
    cell receptor sequences based on junction region sequence similarity and shared mutations within partitions that share the same V gene, J gene,
    and junction length, allowing for ambiguous V or J gene annotations. This is not a full implementation as additional arguments such as
    `targeting_model` and `len_limit` requires access to additional objects that needs to be separately created through other packages e.g. `shazam`.
    As such, we will only implement the default argument were both will be set to `None` (or `NULL` in R). If you want to use this method in its
    full functionality, please run it  separately through R with scoper's tutorial.

    see also https://scoper.readthedocs.io/en/stable/vignettes/Scoper-Vignette/

    Parameters
    ----------
    vdj : Dandelion
        a dandelion object containing the airr data.
    threshold : float
        numeric scalar where the tree should be cut (the distance threshold for clonal grouping).
    method : Literal["novj", "vj"], optional
        one of the "novj" or "vj".
        If method="novj", then clonal relationships are inferred using an adaptive threshold that indicates the level of similarity among junction sequences in a local neighborhood.
        If method="vj", then clonal relationships are inferred not only on junction region homology, but also taking into account the mutation profiles in the V and J segments.
        Mutation counts are determined by comparing the input sequences (in the column specified by sequence) to the effective germline sequence (IUPAC representation of sequences in the column specified by germline).
    germline : str, optional
        character name of the column containing the germline or reference sequence.
    sequence : str, optional
        character name of the column containing input sequences.
    junction : str, optional
        character name of the column containing junction sequences. Also used to determine sequence length for grouping.
    v_call : str, optional
        name of the column containing the V-segment allele calls.
    j_call : str, optional
        name of the column containing the J-segment allele calls.
    clone : str, optional
        output column name containing the clonal cluster identifiers.
    fields : list[str], optional
        character vector of additional columns to use for grouping. Sequences with disjoint values in the specified fields will be classified as separate clones.
    cell_id : str | None, optional
        name of the column containing cell identifiers or barcodes. If specified, grouping will be performed in single-cell mode with the behavior governed by the locus and only_heavy arguments. If set to None then the bulk sequencing data is assumed.
    locus : str, optional
        name of the column containing locus information.
    only_heavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping.
    split_light : bool, optional
        split clones by light chains.
    first : bool, optional
        specifies how to handle multiple V(D)J assignments for initial grouping. If True only the first call of the gene assignments is used. If False the union of ambiguous gene assignments is used to group all sequences with any overlapping gene calls.
    cdr3 : bool, optional
        if True removes 3 nucleotides from both ends of "junction" prior to clustering (converts IMGT junction to CDR3 region). If True this will also remove records with a junction length less than 7 nucleotides.
    mod3 : bool, optional
        if True removes records with a junction length that is not divisible by 3 in nucleotide space.
    max_n : int | None, optional
        The maximum number of degenerate characters to permit in the junction sequence before excluding the record from clonal assignment. Default is set to be zero. Set it as "None" for no action.
    threshold : float | None, optional
        the supervising cut-off to enforce an upper-limit distance for clonal grouping. A numeric value between (0,1).
    base_sim : float, optional
        required similarity cut-off for sequences in equal distances from each other.
    iter_max : int, optional
        the maximum number of iterations allowed for kmean clustering step.
    nstart : int, optional
        the number of random sets chosen for kmean clustering initialization.
    nproc : int, optional
        number of cores to distribute the function over.
    verbose : bool, optional
        if True prints out a summary of each step cloning process. if False (default) process cloning silently.
    summarize_clones : bool, optional
        if True performs a series of analysis to assess the clonal landscape and returns a ScoperClones object. If False then a modified input db is returned. When grouping by fields, summarize_clones should be False.
    remove_ambiguous : bool, optional
        if True removes contigs with ambiguous V(D)J assignments flagged by `check_contigs`.
    remove_extra : bool, optional
        if True removes extra contigs flagged by `check_contigs`.
    """
    try:
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import r
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with scoper's tutorials."
            )
        )
    scp = importr("scoper")

    db = load_data(vdj._data)
    warnings.filterwarnings("ignore")
    db = sanitize_data(db)
    if remove_ambiguous:
        if "ambiguous" in db:
            db = db[db["ambiguous"] == "F"].copy()
    if remove_extra:
        if "extra" in db:
            db = db[db["extra"] == "F"].copy()
    # sanitize before passing to R
    db, _ = sanitize_data_for_saving(db)
    fields = NULL if fields is None else fields
    cell_id = NULL if cell_id is None else cell_id
    threshold = NULL if threshold is None else threshold
    db_r = safe_py2rpy(db)
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
    vdj._data = df.copy()
    vdj.update_metadata(
        clone_key=clone_id,
        retrieve=clone_id,
        retrieve_mode="merge and unique only",
    )


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
        return pandas2ri.rpy2py(r_object)
