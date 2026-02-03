"""
Lazy distance matrix computation for polars DataFrames.

This module provides streaming computation of distance matrices using Zarr storage
and Dask for parallelization, adapted to work with native polars DataFrames.
"""

import functools
import math
import multiprocessing
import os
import shutil
import tempfile

import dask
import dask.array as da
import numpy as np
import polars as pl
import zarr

from dask import compute
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from joblib import Parallel, delayed
from packaging import version
from tqdm import tqdm
from scanpy import logging as logg

from dandelion.utilities._distances import (
    Metric,
    _prepare_sequences_with_separator,
)

ZARR_V3 = version.parse(zarr.__version__) >= version.parse("3.0.0")
if ZARR_V3:
    from zarr.storage import LocalStore
    from zarr.codecs import BloscCodec

    def open_zarr_group(store, mode="a"):
        return zarr.open_group(store=store, mode=mode)

    def create_zarr_array(root, name, **kwargs):
        return root.create_array(name, **kwargs)

else:
    from zarr import DirectoryStore
    from zarr.codecs import Blosc as BloscCodec

    def LocalStore(path):
        return DirectoryStore(path)

    def open_zarr_group(store, mode="a"):
        if mode == "w":
            return zarr.group(store=store, overwrite=True)
        else:
            return zarr.group(store=store)

    def create_zarr_array(root, name, **kwargs):
        return root.create(name, **kwargs)


def calculate_distance_matrix_zarr(
    dat_seq: pl.DataFrame,
    metric: Metric,
    pad_to_max: bool = True,
    membership: dict | None = None,
    zarr_path: str | None = None,
    chunk_size: int | None = None,
    n_cpus: int = 1,
    memory_limit_gb: float | None = None,
    memory_safety_fraction: float = 0.3,
    compress: bool = True,
    verbose: bool = True,
) -> da.Array:
    """
    Calculate distance matrix using Zarr for out-of-core computation with polars DataFrames.

    This function computes a pairwise distance matrix for sequences stored in a polars DataFrame.
    It uses Zarr arrays for efficient on-disk storage and Dask for parallel computation.
    Ideal for large datasets (100k-10M sequences) that don't fit in memory.

    Parameters
    ----------
    dat_seq : pl.DataFrame
        Polars DataFrame containing sequence data. Must include 'cell_id' column for indexing.
    metric : Metric
        Distance metric to use.
    pad_to_max : bool
        Whether to pad sequences to maximum length before distance calculation.
    membership : dict or None, optional
        Dictionary mapping indices to membership groups. If provided, distances are only
        computed within groups (clone mode). If None, computes full pairwise distances.
    zarr_path : str, optional
        Path to save the Zarr array. If None, uses a temporary directory.
    chunk_size : int, optional
        Size of chunks for computation. If None, automatically determined based on memory.
    n_cpus : int
        Number of CPUs to use for parallel computation.
    memory_limit_gb : float, optional
        Memory limit per worker in GB. If None, auto-detected from environment or system.
    memory_safety_fraction : float
        Fraction of available memory to use (conservative for Dask overhead)
    compress : bool
        Whether to compress the Zarr array.
    verbose : bool
        Whether to show progress bars.

    Returns
    -------
    dask.array.Array
        A Dask array view of the computed distance matrix stored in Zarr on disk.
    """
    # Get sequence columns (all except cell_id, locus, junction, junction_aa)
    exclude_cols = {
        "cell_id",
        "locus",
        "junction",
        "junction_aa",
        "_original_order",
    }
    seq_cols = [
        c for c in dat_seq.collect_schema().names() if c not in exclude_cols
    ]

    # Get cell IDs and create index mapping (for membership-based computation)
    cell_id_list = dat_seq["cell_id"].to_list()
    cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(cell_id_list)}

    # Add row index to preserve order, then clean sequences
    dat_seq_indexed = dat_seq.with_row_index("_original_order")

    # Clean sequences: cast to string, replace dots, fill nulls, replace "None"
    dat_seq_clean = dat_seq_indexed.select(
        [
            pl.col("_original_order"),
            *[
                pl.col(c)
                .cast(pl.String)
                .str.replace_all(r"\.", "")
                .fill_null("")
                .str.replace_all("None", "")
                .alias(c)
                for c in seq_cols
            ],
        ]
    )

    # Prepare sequences once with global padding before scattering to workers
    # Convert to list of lists, prepare, then store back as single column
    seq_arrays = dat_seq_clean.select(seq_cols).to_numpy(allow_copy=True)
    seq_lists = seq_arrays.tolist()
    prepared_seqs = _prepare_sequences_with_separator(
        seq_lists, metric=metric, pad_to_max=pad_to_max, sep="#"
    )

    # Reconstruct DataFrame with prepared sequences as single column
    dat_seq_clean = pl.DataFrame(
        {
            "_original_order": dat_seq_clean["_original_order"],
            "_prepared_seq": prepared_seqs,
        }
    )

    # Store cleaned dataframe for lazy chunk extraction - do NOT convert to numpy yet
    m = dat_seq_clean.height
    n_cols = 1  # Now we have single prepared sequence column

    logg.info(
        f"Preparing distance matrix computation for {m} sequences across {n_cols} columns..."
    )

    # Auto-determine chunk size if not provided
    if chunk_size is None:
        chunk_size, _ = _auto_chunk_size(
            m,
            n_cpus=n_cpus,
            memory_limit_gb=memory_limit_gb,
            safety_fraction=memory_safety_fraction,
        )
        logg.info(
            f"Auto-determined chunk size: {chunk_size} (for {m} sequences)"
        )

    if verbose:
        logg.info(f"\nCreated Zarr array at: {zarr_path}")

    # Setup Dask client
    client = _setup_dask_client(n_cpus=n_cpus, memory_limit_gb=memory_limit_gb)

    # Resolve output path, support alias `zarr_path`
    if zarr_path is None:
        zarr_path = zarr_path if zarr_path is not None else tempfile.mkdtemp()

    comp = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    store = LocalStore(zarr_path + "/distance_matrix.zarr")
    root = open_zarr_group(store, mode="w")
    z_array = create_zarr_array(
        root,
        "distance_matrix",
        shape=(m, m),
        chunks=(chunk_size, chunk_size),
        dtype=np.float64,
        fill_value=0.0,
        compressors=[comp] if compress else None,
    )

    logg.info(f"Created Zarr array at: {zarr_path}")

    # Scatter the DataFrame to workers once to avoid graph bloat
    # This sends the data once instead of embedding it in every task
    if client is not None:
        df_future = client.scatter(dat_seq_clean, broadcast=True)
        logg.info("Scattered DataFrame to workers")
    else:
        df_future = dat_seq_clean

    # Prepare delayed computation blocks
    delayed_blocks = []

    if membership is not None:
        # Clone mode: only compute within membership groups
        logg.info(
            "Computing distances in clone mode (within membership groups)"
        )

        n_groups = len(membership)
        logg.info(
            f"Processing {n_groups} membership groups using Polars vectorized approach"
        )

        # Use Polars-native vectorized approach for moderate number of groups
        # This is much faster than Dask for ~100-1000 groups due to lower overhead
        tmp_results = _compute_distances_polars_native(
            dat_seq_clean=dat_seq_clean,
            membership=membership,
            cell_id_to_idx=cell_id_to_idx,
            metric=metric,
            z_array=z_array,
            n_jobs=n_cpus,
            verbose=verbose,
        )

    else:
        # Determine number of chunks
        n_chunks = max(1, math.ceil(m / chunk_size))
        if n_chunks < n_cpus:
            n_chunks = n_cpus

        # Compute chunk boundaries
        chunk_sizes = [chunk_size] * (m // chunk_size)
        if m % chunk_size != 0:
            chunk_sizes.append(m % chunk_size)
        cum_sizes = np.cumsum([0] + chunk_sizes)

        # Process chunks using client.submit for better efficiency with large numbers of tasks
        # This avoids nested delayed objects which cause graph bloat
        futures = []
        zarr_path_str = zarr_path + "/distance_matrix.zarr"

        for i in range(len(chunk_sizes)):
            for j in range(i, len(chunk_sizes)):
                if chunk_sizes[i] == 0 or chunk_sizes[j] == 0:
                    continue
                start_i, end_i = cum_sizes[i], cum_sizes[i + 1]
                start_j, end_j = cum_sizes[j], cum_sizes[j + 1]
                is_diagonal = i == j

                if client is not None:
                    # Use submit() instead of delayed() - much more efficient for 60k+ tasks
                    future = client.submit(
                        _compute_block_and_write,
                        df_future,
                        start_i,
                        end_i,
                        start_j,
                        end_j,
                        metric,
                        zarr_path_str,
                        is_diagonal,
                        compress,
                    )
                    futures.append(future)
                else:
                    # Fallback to delayed for single-threaded execution
                    chunk_i = dask.delayed(_extract_chunk_from_polars)(
                        df_future, start_i, end_i
                    )
                    chunk_j = dask.delayed(_extract_chunk_from_polars)(
                        df_future, start_j, end_j
                    )
                    delayed_blocks.append(
                        dask.delayed(_compute_and_write_block)(
                            chunk_i=chunk_i,
                            chunk_j=chunk_j,
                            metric=metric,
                            zarr_path=zarr_path_str,
                            start_i=start_i,
                            end_i=end_i,
                            start_j=start_j,
                            end_j=end_j,
                            is_diagonal=is_diagonal,
                            compress=compress,
                        )
                    )

        logg.info(
            f"Starting computation of {len(futures) if client else len(delayed_blocks)} chunks...",
        )
        # Compute blocks - they write to Zarr as they complete
        if client is not None:
            progress(futures)
            tmp_results = client.gather(futures)
        else:
            with ProgressBar():
                tmp_results = compute(*delayed_blocks, scheduler="threads")

    logg.info("Merging temporary results into final Zarr array...")
    # Merge all temporary arrays into the main array
    merge_tmp_arrays(z_array, tmp_results, verbose)

    # Set diagonal to NaN
    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        diag_block = z_array[i:end, i:end]
        np.fill_diagonal(diag_block, np.nan)
        z_array[i:end, i:end] = diag_block

    # Return underlying Zarr array; callers may wrap with Dask when needed
    return z_array


def _extract_chunk_from_polars(
    df: pl.DataFrame | object, start: int, end: int
) -> np.ndarray:
    """
    Extract a chunk of prepared sequences from Polars DataFrame lazily.

    This function only loads the requested rows into memory, avoiding
    loading the entire dataset at once. Handles both regular DataFrames
    and Dask futures (from client.scatter()).

    Parameters
    ----------
    df : pl.DataFrame | Future
        Polars DataFrame with prepared sequence data or a Dask future containing one
    start : int
        Starting row index
    end : int
        Ending row index (exclusive)

    Returns
    -------
    np.ndarray
        2D numpy array with prepared sequences (shape: (n_rows, 1))
    """
    # If df is a Dask future, it will be automatically resolved by Dask
    # when this function is called by a worker
    # Extract the prepared sequence column
    return (
        df.slice(start, end - start)
        .select(["_prepared_seq"])
        .to_numpy(allow_copy=True)
    )


def _compute_block_and_write(
    df_future: pl.DataFrame | object,
    start_i: int,
    end_i: int,
    start_j: int,
    end_j: int,
    metric: Metric,
    zarr_path: str,
    is_diagonal: bool,
    compress: bool,
) -> str:
    """
    Combined function to extract chunks, compute distances, and write to Zarr.

    This function combines extraction and computation in a single task to avoid
    nested delayed objects that cause graph bloat.

    Parameters
    ----------
    df_future : pl.DataFrame | Future
        Scattered DataFrame or future reference
    start_i, end_i : int
        Row indices for first chunk
    start_j, end_j : int
        Row indices for second chunk
    metric : Metric
        Distance metric
    zarr_path : str
        Path to Zarr array
    is_diagonal : bool
        Whether this is a diagonal block
    compress : bool
        Whether to compress temporary arrays

    Returns
    -------
    str
        Path to temporary Zarr array
    """
    # Extract chunks
    chunk_i = _extract_chunk_from_polars(df_future, start_i, end_i)
    chunk_j = _extract_chunk_from_polars(df_future, start_j, end_j)

    # Compute and write
    return _compute_and_write_block(
        chunk_i,
        chunk_j,
        metric,
        zarr_path,
        start_i,
        end_i,
        start_j,
        end_j,
        is_diagonal,
        compress,
    )


def _compute_block_multicol(
    seqs_i: np.ndarray,
    seqs_j: np.ndarray,
    metric: Metric,
):
    """
    Compute pairwise distances between two sequence chunks.

    seqs_i and seqs_j are 2D arrays: shape (n_rows, 1)

    Note: Sequences are already prepared (padded, concatenated) before this function,
    so we skip re-preparation and directly compute distances.
    """
    # Extract prepared sequences (already concatenated with separators and padded)
    seqs_i_flat = seqs_i.flatten().tolist()
    seqs_j_flat = seqs_j.flatten().tolist()

    # Deduplicate within each chunk
    unique_i = []
    i_to_unique = {}
    for seq in seqs_i_flat:
        if seq not in i_to_unique:
            i_to_unique[seq] = len(unique_i)
            unique_i.append(seq)

    unique_j = []
    j_to_unique = {}
    for seq in seqs_j_flat:
        if seq not in j_to_unique:
            j_to_unique[seq] = len(unique_j)
            unique_j.append(seq)

    # Check if deduplication provides benefit
    if len(unique_i) < len(seqs_i_flat) or len(unique_j) < len(seqs_j_flat):
        # Compute distances for unique sequences only
        all_unique = unique_i + unique_j
        n_unique_i = len(unique_i)
        full_dist_unique = metric.compute_vectorized(all_unique)
        unique_block = full_dist_unique[:n_unique_i, n_unique_i:]

        # Map back to full i×j block
        n_i = len(seqs_i_flat)
        n_j = len(seqs_j_flat)
        result = np.zeros((n_i, n_j), dtype=np.float64)
        for i, seq_i in enumerate(seqs_i_flat):
            unique_i_idx = i_to_unique[seq_i]
            for j, seq_j in enumerate(seqs_j_flat):
                unique_j_idx = j_to_unique[seq_j]
                result[i, j] = unique_block[unique_i_idx, unique_j_idx]
        return result
    else:
        # No duplicates, compute normally
        all_seqs = seqs_i_flat + seqs_j_flat
        n_i = len(seqs_i_flat)
        full_dist = metric.compute_vectorized(all_seqs)
        return full_dist[:n_i, n_i:]


def dask_safe_slice_square(arr: da.Array, pos: list) -> da.Array:
    """Return arr[pos, pos], Dask-safe."""
    if isinstance(arr, da.Array):
        return arr[pos, :][:, pos]
    else:
        return arr[np.ix_(pos, pos)]


def _compute_and_write_block(
    chunk_i: np.ndarray,
    chunk_j: np.ndarray,
    metric: Metric,
    zarr_path: str,
    start_i: int,
    end_i: int,
    start_j: int,
    end_j: int,
    is_diagonal: bool,
    compress: bool = True,
):
    """
    Compute a single block and write it directly to Zarr.

    Parameters
    ----------
    chunk_i : np.ndarray
        First chunk of sequences
    chunk_j : np.ndarray
        Second chunk of sequences
    metric : Metric
        Distance metric
    pad_to_max : bool
        Whether to pad sequences
    zarr_path : str
        Path to Zarr array (passed as string to reduce graph size)
    start_i : int
        Starting row index
    end_i : int
        Ending row index
    start_j : int
        Starting column index
    end_j : int
        Ending column index
    is_diagonal : bool
        Whether this is a diagonal block
    compress : bool
        Whether to compress temporary arrays

    Returns
    -------
    str
        Path to temporary Zarr array
    """
    # Open Zarr array from path
    store = LocalStore(zarr_path)
    root = open_zarr_group(store, mode="r")
    z_array = root["distance_matrix"]

    block = _compute_block_multicol(chunk_i, chunk_j, metric)
    tmp_array, tmp_dir = create_tmp_zarr(z_array, compress)

    # Write to Zarr immediately
    tmp_array[start_i:end_i, start_j:end_j] = block
    # If not diagonal block, write transpose too
    if not is_diagonal:
        tmp_array[start_j:end_j, start_i:end_i] = block.T

    return tmp_dir


def _process_batch(
    seq_cat: list[np.ndarray],
    idx_cat: list[list[int]],
    seq_lengths: list[int],
    metric: Metric,
    zarr_path: str,
    compress: bool = True,
):
    """
    Process a batch of membership groups.

    Parameters
    ----------
    seq_cat : list[np.ndarray]
        Concatenated sequences for the batch
    idx_cat : list[list[int]]
        Concatenated indices for the batch
    seq_lengths : list[int]
        Lengths of each group in the batch
    metric : Metric
        Distance metric
    zarr_path : str
        Path to Zarr array
    compress : bool
        Whether to compress temporary arrays

    Returns
    -------
    str
        Path to temporary Zarr array
    """
    # Open Zarr array from path
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="r")
    z_array = root["distance_matrix"]

    boundaries = np.cumsum(seq_lengths)
    boundaries = np.insert(boundaries, 0, 0)

    results = _compute_and_write_membership_cat(
        seqs_cat=seq_cat,
        idxs_cat=idx_cat,
        boundaries=boundaries,
        metric=metric,
        z_array=z_array,
        compress=compress,
    )
    return results


def _compute_and_write_membership_cat(
    seqs_cat: np.ndarray,
    idxs_cat: np.ndarray,
    boundaries: np.ndarray,
    metric: Metric,
    z_array: zarr.Array,
    compress: bool = True,
):
    """
    Compute distances within each membership group only.
    """
    # Compute full block once
    block = _compute_block_multicol(seqs_cat, seqs_cat, metric)
    # compress the tmp_array
    n_groups = len(boundaries) - 1
    tmp_array, tmp_dir = create_tmp_zarr(z_array, compress)
    # Only process diagonal blocks (within each group)
    for g in range(n_groups):
        s, e = boundaries[g], boundaries[g + 1]
        idx = idxs_cat[s:e]

        # Extract subblock for this group only
        subblock = block[s:e, s:e]

        # Write to zarr
        tmp_array[np.ix_(idx, idx)] = subblock
    return tmp_dir


def _auto_chunk_size(
    n: int,
    n_cpus: int,
    memory_limit_gb: float | None = None,
    safety_fraction: float = 0.3,
) -> tuple[int, int]:
    """
    Compute dynamic chunk size to stay within memory budget.

    Parameters
    ----------
    n : int
        Total number of sequences
    n_cpus : int
        Number of cores/workers
    memory_limit_gb : float, optional
        Memory limit per worker in GB
    safety_fraction : float
        Fraction of available memory to use (conservative for Dask overhead)

    Returns
    -------
    chunk_size : int
        Recommended chunk size
    n_chunks : int
        Number of chunks along one dimension
    """
    try:
        import psutil
    except ImportError:
        raise ImportError(
            "Please install psutil to enable automatic chunk size calculation: pip install psutil"
        )
    if memory_limit_gb is None:
        # Scheduler environment variables
        if "SLURM_MEM_PER_CPU" in os.environ:
            memory_limit_gb = float(os.environ["SLURM_MEM_PER_CPU"]) / 1024
        elif (
            "SLURM_MEM_PER_NODE" in os.environ and "SLURM_NTASKS" in os.environ
        ):
            memory_limit_gb = (
                float(os.environ["SLURM_MEM_PER_NODE"])
                / float(os.environ["SLURM_NTASKS"])
                / 1024
            )
        elif "PBS_MEM" in os.environ:
            memory_limit_gb = float(os.environ["PBS_MEM"]) / 1024
        elif "LSF_MEM" in os.environ:
            memory_limit_gb = float(os.environ["LSF_MEM"]) / 1024
        # Fallback to psutil
        elif psutil is not None:
            available_mem = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = available_mem
        else:
            raise RuntimeError(
                "Cannot determine memory limit: install psutil or set memory_limit_gb manually."
            )

    # Apply safety fraction regardless of source
    mem_per_core = memory_limit_gb * safety_fraction / n_cpus

    # Each chunk block is chunk_size^2 * 8 bytes
    # Keep blocks small enough that 2-3 can fit in memory per worker
    chunk_size = int(math.sqrt((mem_per_core * (1024**3)) / 8 / 3))

    # Ensure minimum chunk size
    chunk_size = max(100, min(chunk_size, n))

    n_chunks = max(1, math.ceil(n / chunk_size))

    return chunk_size, n_chunks


def _setup_dask_client(
    n_cpus: int, memory_limit_gb: float | None = None
) -> Client | None:
    """
    Setup Dask distributed client.

    Parameters
    ----------
    n_cpus : int
        Number of workers
    memory_limit_gb : float, optional
        Memory limit per worker

    Returns
    -------
    Client or None
        Dask client if n_cpus > 1, else None
    """
    try:
        from dask.distributed import Client
    except ImportError:
        raise ImportError(
            "Please install dask distributed to enable parallel processing: pip install dask distributed"
        )
    if n_cpus <= 1:
        return None

    client_kwargs = {
        "n_workers": n_cpus,
        "threads_per_worker": 1,  # for simplicity and to avoid GIL issues
        "processes": True,  # Critical for memory isolation
    }

    if memory_limit_gb is not None:
        client_kwargs["memory_limit"] = f"{memory_limit_gb}GB"

    client = Client(**client_kwargs)

    logg.info(f"Dask client started: {client.dashboard_link}")

    return client


def merge_tmp_arrays(
    main_array: zarr.Array, tmp_results: list[str], verbose: bool = True
) -> None:
    """
    Function to merge all the temporary zarr arays after computation.

    Parameters
    ----------
    main_array : zarr.Array
        Main zarr array to which temporary arrays will be merged.
    tmp_results : list[str]
        List of temporary directory paths containing zarr arrays to merge.
    verbose : bool
        Whether to show progress bar.
    """
    for tmp_dir in tqdm(
        tmp_results,
        disable=not verbose,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        root = open_zarr_group(tmp_dir + "/tmp.zarr", mode="r")
        tmp_array = root["distance_matrix"]
        # Iterate over chunks
        main_array[:] += tmp_array[:]

        # Clean up
        shutil.rmtree(tmp_dir)


def create_tmp_zarr(z_array: zarr.Array, compress: bool = True) -> zarr.Array:
    """Assist function to create a temporary zarr array with same shape/chunks as input z_array."""
    tmp_dir = tempfile.mkdtemp()
    comp = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    store = LocalStore(tmp_dir + "/tmp.zarr")
    root = open_zarr_group(store, mode="w")
    tmp_array = create_zarr_array(
        root,
        "distance_matrix",
        shape=z_array.shape,
        chunks=z_array.chunks,
        dtype=z_array.dtype,
        fill_value=0.0,
        compressors=[comp] if compress else None,
    )
    return tmp_array, tmp_dir


def _compute_group_distances_sequential(
    group_df: pl.DataFrame, metric: Metric, z_array: zarr.Array
) -> pl.DataFrame:
    """
    Compute pairwise distances for a group and write to Zarr (sequential version).

    Used with Polars map_groups for sequential processing.

    Parameters
    ----------
    group_df : pl.DataFrame
        DataFrame for a single group/clone
    metric : Metric
        Distance metric to use
    z_array : zarr.Array
        Zarr array to write distances to

    Returns
    -------
    pl.DataFrame
        Empty DataFrame (distances written as side effect)
    """
    if group_df.height < 2:
        return pl.DataFrame()

    # Direct column access is faster than select + to_numpy
    array_indices = group_df["_original_order"].to_numpy()
    seqs_flat = group_df["_prepared_seq"].to_list()

    # Deduplicate sequences within this group for faster computation
    unique_seqs = []
    seq_to_unique_idx = {}
    for seq in seqs_flat:
        if seq not in seq_to_unique_idx:
            seq_to_unique_idx[seq] = len(unique_seqs)
            unique_seqs.append(seq)

    # Only compute distances if we have duplicates (speedup opportunity)
    if len(unique_seqs) < len(seqs_flat):
        # Compute distances only for unique sequences
        dist_block_unique = metric.compute_vectorized(unique_seqs)

        # Map back to full distance matrix
        n = len(seqs_flat)
        dist_block = np.zeros((n, n), dtype=np.float64)
        for i, seq_i in enumerate(seqs_flat):
            unique_i = seq_to_unique_idx[seq_i]
            for j, seq_j in enumerate(seqs_flat):
                unique_j = seq_to_unique_idx[seq_j]
                dist_block[i, j] = dist_block_unique[unique_i, unique_j]
    else:
        # No duplicates, compute normally
        dist_block = metric.compute_vectorized(seqs_flat)

    # Write directly to Zarr
    z_array[np.ix_(array_indices, array_indices)] = dist_block

    return pl.DataFrame()


def _process_group_parallel(
    group_df: pl.DataFrame, metric: Metric, z_array: zarr.Array
) -> None:
    """
    Process a single group - write to Zarr array (parallel version).

    Used with joblib for parallel processing.

    Parameters
    ----------
    group_df : pl.DataFrame
        DataFrame for a single group/clone
    metric : Metric
        Distance metric to use
    z_array : zarr.Array
        Zarr array to write distances to (thread-safe)
    """
    if group_df.height < 2:
        return

    array_indices = group_df["_original_order"].to_numpy()
    seqs_flat = group_df["_prepared_seq"].to_list()

    # Deduplicate sequences within this group for faster computation
    unique_seqs = []
    seq_to_unique_idx = {}
    for seq in seqs_flat:
        if seq not in seq_to_unique_idx:
            seq_to_unique_idx[seq] = len(unique_seqs)
            unique_seqs.append(seq)

    # Only compute distances if we have duplicates (speedup opportunity)
    if len(unique_seqs) < len(seqs_flat):
        # Compute distances only for unique sequences
        dist_block_unique = metric.compute_vectorized(unique_seqs)

        # Map back to full distance matrix
        n = len(seqs_flat)
        dist_block = np.zeros((n, n), dtype=np.float64)
        for i, seq_i in enumerate(seqs_flat):
            unique_i = seq_to_unique_idx[seq_i]
            for j, seq_j in enumerate(seqs_flat):
                unique_j = seq_to_unique_idx[seq_j]
                dist_block[i, j] = dist_block_unique[unique_i, unique_j]
    else:
        # No duplicates, compute normally
        dist_block = metric.compute_vectorized(seqs_flat)

    # Write to Zarr (Zarr handles thread-safe writes)
    z_array[np.ix_(array_indices, array_indices)] = dist_block


def _compute_distances_polars_native(
    dat_seq_clean: pl.DataFrame,
    membership: dict,
    cell_id_to_idx: dict,
    metric: Metric,
    z_array: zarr.Array,
    n_jobs: int = 1,
    verbose: bool = True,
) -> list[str]:
    """
    Compute distances using fully vectorized Polars operations.

    Uses group_by to partition data and applies vectorized distance computation
    within each group. Optimized with vectorized joins and optional parallelization.

    Parameters
    ----------
    dat_seq_clean : pl.DataFrame
        Cleaned Polars DataFrame with prepared sequences (already padded and concatenated)
    membership : dict
        Mapping from clone_id -> list of cell_ids
    cell_id_to_idx : dict
        Mapping from cell_id -> array index
    metric : Metric
        Distance metric
    z_array : zarr.Array
        Zarr array to write to
    n_jobs : int, optional
        Number of parallel jobs for group processing. Default 1 (no parallelization).
        Use -1 to use all available CPUs.
    verbose: bool, optional
        Whether to print progress bar.

    Returns
    -------
    list[str]
        List of temporary Zarr array paths
    """
    # Build mapping DataFrames using vectorized joins (much faster than map_elements)
    # 1. Create cell_id -> clone_id mapping as DataFrame
    cell_clone_pairs = [
        (cell_id, clone_id)
        for clone_id, members in membership.items()
        for cell_id in members
        if cell_id in cell_id_to_idx
    ]

    if not cell_clone_pairs:
        return []

    cell_clone_df = pl.DataFrame(
        {
            "cell_id": [pair[0] for pair in cell_clone_pairs],
            "clone_id": [pair[1] for pair in cell_clone_pairs],
        }
    )

    # 2. Create _original_order -> cell_id mapping as DataFrame
    order_cell_df = pl.DataFrame(
        {
            "_original_order": list(cell_id_to_idx.values()),
            "cell_id": list(cell_id_to_idx.keys()),
        }
    )

    # 3. Use vectorized joins instead of map_elements (10-100x faster)
    df_with_clone = (
        dat_seq_clean.join(order_cell_df, on="_original_order", how="left")
        .join(cell_clone_df, on="cell_id", how="left")
        .filter(pl.col("clone_id").is_not_null())
    )

    # Determine parallelization strategy
    if n_jobs == 1:
        # Sequential processing - use Polars map_groups (original approach)
        # Use partial to bind metric and z_array to the helper function
        compute_fn = functools.partial(
            _compute_group_distances_sequential, metric=metric, z_array=z_array
        )

        # Apply sequentially to each group
        _ = (
            df_with_clone.lazy()
            .group_by("clone_id", maintain_order=True)
            .map_groups(compute_fn, schema={})
            .collect(engine="streaming")
        )
    else:
        # Parallel processing using joblib
        # Determine actual number of jobs
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Partition by group and collect groups as list of DataFrames
        groups = df_with_clone.partition_by("clone_id", as_dict=True)

        # Use partial to bind metric and z_array to the helper function
        process_fn = functools.partial(
            _process_group_parallel, metric=metric, z_array=z_array
        )

        # Process groups in parallel
        # Use threading backend since we're doing I/O (Zarr writes)
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_fn)(group_df)
            for group_df in tqdm(
                groups.values(),
                total=len(groups),
                desc="Processing clone groups",
                disable=not verbose,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        )

    # Return empty list since we write directly to z_array
    return []
