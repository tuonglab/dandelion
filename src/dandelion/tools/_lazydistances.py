from __future__ import annotations
import math
import os
import tempfile
import shutil
import zarr
import zarr.storage

import dask.array as da
import numpy as np
import pandas as pd

from dask import compute
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from pathlib import Path
from scanpy import logging as logg
from tqdm import tqdm
from zarr.codecs import BloscCodec

from dandelion.utilities._distances import (
    Metric,
    _prepare_sequences_with_separator,
)


def calculate_distance_matrix_zarr(
    dat_seq: pd.DataFrame,
    metric: Metric,
    pad_to_max: bool = False,
    membership: dict | None = None,
    zarr_path: Path | str | None = None,
    chunk_size: int | None = None,
    max_clones_per_chunk: int | None = None,
    n_cpus: int = 1,
    memory_limit_gb: float | None = None,
    memory_safety_fraction: float = 0.3,
    compress: bool = True,
    lazy: bool = False,
    verbose: bool = True,
) -> np.ndarray | da.Array:
    """
    Compute full pairwise distance matrix writing directly to Zarr.

    This approach:
    - Uses Dask to parallelize distance computation
    - Writes results directly to compressed Zarr storage AS blocks complete
    - Never materializes full matrix in memory
    - Suitable for 100k-10M sequences with compression

    The output matrix will have the same row/column order as dat_seq.index.
    Reorder dat_seq before calling this function if you need a specific order.

    Parameters
    ----------
    dat_seq : pd.DataFrame
        DataFrame of sequences (rows=samples, columns=different sequence types).
        Row order determines the order of the distance matrix.
    metric : Metric
        Distance metric to use.
    pad_to_max : bool, optional
        Whether to pad sequences to maximum length before distance calculation.
    zarr_path : str, optional
        Path to save Zarr array (directory). If None, uses 'distance_matrix.zarr'.
    chunk_size : int | None, optional
        Number of sequences per computation chunk. If None, auto-computed.
    max_clones_per_chunk : int | None, optional
        Maximum number of clones per computation chunk. If None, no limit is applied
        and instead if will be based on memory limits.
    n_cpus : int
        Number of cores/workers to use
    memory_limit_gb : float | None, optional
        Memory limit per worker in GB
    memory_safety_fraction : float
        Fraction of available memory to use (conservative for Dask overhead)
    chunk_size : int | None, optional
        Chunk size for Zarr storage (can differ from computation chunk_size).
        Larger = better compression, smaller = better random access.
        If None, uses same as chunk_size.
    compress : bool, optional
        Whether to compress the Zarr array using Blosc with zstd.
    lazy: bool, optional
        If True, computation will be performed lazily using Dask/Zarr arrays. True will also return a Dask array view of the
        distance matrix stored on disk instead of a numpy array stored in memory.
    verbose : bool
        Whether to show progress

    Returns
    -------
    np.ndarray | da.Array
        Distance matrix in requested format
    """
    # reduce to only clonotypes with >1 member
    if membership is not None:
        membership = {k: v for k, v in membership.items() if len(v) > 1}
        if len(membership) < 1:
            # there's nothing to compute. so just return
            logg.info(
                msg=(
                    "No clonotypes with more than one member found. "
                    "Try using distance_mode = 'full' instead.\n"
                    "Skipping distance computation.\n"
                )
            )
            return
    # Try import the dask dependencies otherwise raise error and tell user to install dask[complete]
    try:
        import dask.array as da
    except ImportError as e:
        raise ImportError(
            "Please install dask with the complete set of dependencies: pip install 'dask[complete]'"
        )
    if zarr_path is None:
        zarr_path = "distance_matrix.zarr"

    n = dat_seq.shape[0]

    # Determine computation chunk size
    if chunk_size is None:
        chunk_size, _ = _auto_chunk_size(
            n,
            n_cpus=n_cpus,
            memory_limit_gb=memory_limit_gb,
            safety_fraction=memory_safety_fraction,
        )
        if verbose:
            logg.info(
                f"Auto computation chunk size: {chunk_size} sequences per block"
            )
            logg.info(
                f"Estimated memory per block: {(chunk_size**2 * 8) / (1024**3):.2f} GB"
            )
    else:
        if verbose:
            logg.info(f"Computation chunk size: {chunk_size}")

    if verbose:
        logg.info(f"Zarr storage chunk size: {chunk_size}")
        logg.info(f"Output size: {(n * n * 8) / (1024**3):.2f} GB uncompressed")

    # Setup compressor
    comp = (
        BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        if compress
        else None
    )

    # Create temporary Zarr array
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")

    # create temp Zarr array without compression for writing first
    z_array = root.create_array(
        "distance_matrix",
        shape=(n, n),
        chunks=(chunk_size, chunk_size),
        dtype="float64",
        fill_value=0.0,
        compressors=[comp] if compress else None,
    )

    # Store metadata
    z_array.attrs["n_sequences"] = n
    z_array.attrs["chunk_size"] = chunk_size
    z_array.attrs["index"] = list(dat_seq.index)
    z_array.attrs["columns"] = list(dat_seq.columns)

    if verbose:
        logg.info(f"\nCreated Zarr array at: {zarr_path}")
        logg.info(f"Compressor: {comp}\n")

    # Setup Dask client
    client = _setup_dask_client(n_cpus=n_cpus, memory_limit_gb=memory_limit_gb)

    try:
        # Compute distances and write blocks as they complete
        _compute_multicol_distances_streaming(
            dat_seq=dat_seq,
            metric=metric,
            pad_to_max=pad_to_max,
            z_array=z_array,
            chunk_size=chunk_size,
            n_cpus=n_cpus,
            chunk_batch_limit=max_clones_per_chunk,
            client=client,
            membership=membership,
            verbose=verbose,
            compress=compress,
        )

        # Set diagonal to NaN
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            diag_block = z_array[i:end, i:end]
            np.fill_diagonal(diag_block, np.nan)
            z_array[i:end, i:end] = diag_block

        if verbose:
            logg.info(f"\n{'='*60}\n")
            logg.info("✓ Distance matrix complete!")
            logg.info("\nFinal array info:")
            logg.info(f"  Shape: {z_array.shape}")
            logg.info(f"  Dtype: {z_array.dtype}")
            logg.info(f"  Chunks: {z_array.chunks}")
            logg.info(
                f"  Size on disk: {z_array.nbytes / (1024**3):.2f} GB uncompressed"
            )

            try:
                actual_size = sum(
                    f.stat().st_size
                    for f in Path(zarr_path).rglob("*")
                    if f.is_file()
                )
                logg.info(
                    f"  Actual size: {actual_size / (1024**3):.2f} GB (with compression)"
                )
                compression_ratio = (
                    z_array.nbytes / actual_size if actual_size > 0 else 0
                )
                logg.info(f"  Compression ratio: {compression_ratio:.1f}x")
            except Exception:
                pass

        # Return based on return_type
        if lazy:
            return da.from_zarr(z_array)
        else:
            return z_array[:]

    finally:
        if client is not None:
            client.close()


def _compute_multicol_distances_streaming(
    dat_seq: pd.DataFrame,
    metric: Metric,
    pad_to_max: bool,
    z_array: zarr.Array,
    chunk_size: int,
    n_cpus: int,
    chunk_batch_limit: int | None = None,
    client: Client | None = None,
    membership: dict | None = None,
    verbose: bool = True,
    compress: bool = True,
):
    """
    Compute distance matrix using concatenation across all columns,
    writing blocks directly to Zarr as they complete.
    """
    try:
        import dask
    except ImportError:
        raise ImportError(
            "Please install dask with the complete set of dependencies: pip install 'dask[complete]'"
        )
    # Clean all sequences
    dat_seq_clean = (
        dat_seq.replace("[.]", "", regex=True).fillna("").replace("None", "")
    )

    # Prepare sequences once with global padding before scattering to workers
    seqs_np = dat_seq_clean.to_numpy(dtype=object)
    seqs_list = seqs_np.tolist()
    prepared_seqs = _prepare_sequences_with_separator(
        seqs_list, metric=metric, pad_to_max=pad_to_max, sep="#"
    )

    # Reconstruct DataFrame with prepared sequences as single column
    dat_seq_clean = pd.DataFrame(
        {"_prepared_seq": prepared_seqs}, index=dat_seq_clean.index
    )

    seqs_np = dat_seq_clean.to_numpy(dtype=object)
    seqs = da.from_delayed(
        dask.delayed(lambda x: x)(seqs_np), shape=seqs_np.shape, dtype=object
    )
    m = len(seqs)
    if m <= 1:
        return None

    delayed_blocks = []
    logg.info(
        msg="Setting up Dask scheduler and task graph...",
    )
    if membership is not None:
        # Build index map
        index_list = list(dat_seq_clean.index)
        index_map = {idx: i for i, idx in enumerate(index_list)}
        # Build reordering (flatten membership)
        group_idx_flat = np.array(
            [index_map[cell] for group in membership.values() for cell in group]
        )

        group_lengths = np.fromiter(
            (len(v) for v in membership.values()), dtype=int
        )
        boundaries = np.cumsum(group_lengths[:-1])

        # Reorder seqs contiguously - convert result to dask array
        seqs_reordered_np = seqs_np[group_idx_flat]
        # Now each membership group is a contiguous slice (a view)
        seqs_m = np.split(seqs_reordered_np, boundaries)
        idx_m = np.split(group_idx_flat, boundaries)

        batched_idx_m = []
        batched_seqs_m = []

        current_batch_idx = []
        current_batch_seqs = []
        current_batch_size = 0  # total number of sequences in current batch
        current_batch_clones = 0  # total number of clones in current batch

        # Calculate soft batch limit if not provided
        if chunk_batch_limit is None and n_cpus > 1:
            total_clonotypes = len(membership)
            # Target: distribute clonotypes across cores with some overhead for load balancing
            # Use 2x cores to allow for better parallelization
            target_batches = n_cpus * 2
            soft_chunk_batch_limit = max(1, total_clonotypes // target_batches)
        else:
            soft_chunk_batch_limit = None

        for idx, seq in zip(idx_m, seqs_m):
            n = len(idx)
            # Determine if the current batch should be finalized
            finalize = False
            if (
                chunk_batch_limit is not None
                and current_batch_clones >= chunk_batch_limit
            ):
                finalize = True
            elif (
                soft_chunk_batch_limit is not None
                and current_batch_clones >= soft_chunk_batch_limit
            ):
                # Use soft limit if no hard limit and batch is getting large
                finalize = True
            elif (
                chunk_batch_limit is None
                and current_batch_size + n > chunk_size
                and current_batch_size > 0
            ):
                finalize = True

            if finalize:
                # finalize current batch
                batched_idx_m.append(current_batch_idx)
                batched_seqs_m.append(current_batch_seqs)
                current_batch_idx, current_batch_seqs = [], []
                current_batch_size, current_batch_clones = 0, 0

            # Add current clonotype to batch
            current_batch_idx.append(idx)
            current_batch_seqs.append(seq)
            current_batch_size += n
            current_batch_clones += 1

        # Add the last batch
        if current_batch_idx:
            batched_idx_m.append(current_batch_idx)
            batched_seqs_m.append(current_batch_seqs)

        # average number of clonotypes per batch
        avg_clonotypes_per_batch = sum(len(b) for b in batched_seqs_m) / len(
            batched_idx_m
        )
        seq_lengths = [[len(s) for s in seq] for seq in batched_seqs_m]
        logg.info(
            msg=f"Created {len(batched_seqs_m)} chunks for distance computation of ~{math.ceil(avg_clonotypes_per_batch)} clonotypes per batch...",
        )
        concatenated_batches_seqs = []
        concatenated_batches_idx = []
        for batch_seqs, batch_idx in zip(batched_seqs_m, batched_idx_m):
            total_rows = sum(s.shape[0] for s in batch_seqs)
            seq_width = batch_seqs[0].shape[1]
            total_idx = sum(i.shape[0] for i in batch_idx)
            if client is not None:
                # 1. Submit concatenations to workers
                fut_seqs = client.submit(np.concatenate, batch_seqs, axis=0)
                fut_idx = client.submit(np.concatenate, batch_idx, axis=0)
                # 2. Create dask arrays from futures
                darr_seqs = da.from_delayed(
                    fut_seqs,
                    shape=(total_rows, seq_width),
                    dtype=object,
                )
                total_idx = sum(x.shape[0] for x in batch_idx)
                darr_idx = da.from_delayed(
                    fut_idx,
                    shape=(total_idx,),
                    dtype=object,
                )
                # 3. persist
                # concatenated_batches_seqs.append(client.persist(darr_seqs))
                # concatenated_batches_idx.append(client.persist(darr_idx))
                concatenated_batches_seqs.append(darr_seqs)
                concatenated_batches_idx.append(darr_idx)
            else:
                concatenated_batches_seqs.append(
                    np.concatenate(batch_seqs, axis=0)
                )
                concatenated_batches_idx.append(
                    np.concatenate(batch_idx, axis=0)
                )
        for seq_cat, idx_cat, seq_len in zip(
            concatenated_batches_seqs,
            concatenated_batches_idx,
            seq_lengths,
        ):
            delayed_blocks.append(
                dask.delayed(_process_batch)(
                    seq_cat=seq_cat,
                    idx_cat=idx_cat,
                    seq_lengths=seq_len,
                    metric=metric,
                    z_array=z_array,
                    compress=compress,
                )
            )
    else:
        # Determine number of chunks
        n_chunks = max(1, math.ceil(m / chunk_size))
        if n_chunks < n_cpus:
            n_chunks = n_cpus
        # Work with numpy for splitting, then convert chunks to dask arrays
        seqs_computed = seqs.compute() if hasattr(seqs, "compute") else seqs_np
        chunks_list = np.array_split(seqs_computed, n_chunks)
        # Convert each chunk to a dask array if client is provided
        if client is not None:
            chunks_list = [
                # client.persist(
                da.from_delayed(
                    dask.delayed(lambda x: x)(chunk),
                    shape=chunk.shape,
                    dtype=object,
                )
                # )
                for chunk in chunks_list
            ]

        chunk_sizes = [len(c) for c in chunks_list]
        cum_sizes = np.cumsum([0] + chunk_sizes)
        for i in range(len(chunks_list)):
            for j in range(i, len(chunks_list)):
                if chunk_sizes[i] == 0 or chunk_sizes[j] == 0:
                    continue
                start_i, end_i = cum_sizes[i], cum_sizes[i + 1]
                start_j, end_j = cum_sizes[j], cum_sizes[j + 1]
                is_diagonal = i == j
                delayed_blocks.append(
                    dask.delayed(_compute_and_write_block)(
                        chunk_i=chunks_list[i],
                        chunk_j=chunks_list[j],
                        metric=metric,
                        z_array=z_array,
                        start_i=start_i,
                        end_i=end_i,
                        start_j=start_j,
                        end_j=end_j,
                        is_diagonal=is_diagonal,
                        compress=compress,
                    )
                )

    logg.info(
        f"Starting computation of {len(delayed_blocks)} chunks...",
    )
    # Compute blocks - they write to Zarr as they complete
    if client is not None:
        futures = client.compute(delayed_blocks)
        progress(futures)
        tmp_results = client.gather(futures)
    else:
        with ProgressBar():
            tmp_results = compute(*delayed_blocks, scheduler="threads")

    logg.info("Merging temporary results into final Zarr array...")
    # Merge all temporary arrays into the main array
    merge_tmp_arrays(z_array, tmp_results, verbose)
    return True


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
    all_seqs = seqs_i_flat + seqs_j_flat

    # Vectorized computation on already-prepared sequences
    n_i = len(seqs_i_flat)
    full_dist = metric.compute_vectorized(all_seqs)

    # Extract the i×j block (cross-distances between chunks)
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
    z_array: zarr.Array,
    start_i: int,
    end_i: int,
    start_j: int,
    end_j: int,
    is_diagonal: bool,
    compress: bool = True,
):
    """
    Compute a single block and write it directly to Zarr.
    """
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
    z_array: zarr.Array,
    compress: bool = True,
):
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
        root = zarr.open_group(tmp_dir + "/tmp.zarr", mode="r")
        tmp_array = root["distance_matrix"]
        # Iterate over chunks
        main_array[:] += tmp_array[:]

        # Clean up
        shutil.rmtree(tmp_dir)


def create_tmp_zarr(z_array: zarr.Array, compress: bool = True) -> zarr.Array:
    """Assist function to create a temporary zarr array with same shape/chunks as input z_array."""
    tmp_dir = tempfile.mkdtemp()
    comp = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(tmp_dir + "/tmp.zarr")
    root = zarr.open_group(store=store, mode="w")
    # create temp Zarr array without compression for writing first
    tmp_array = root.create_array(
        "distance_matrix",
        shape=z_array.shape,
        chunks=z_array.chunks,
        dtype=z_array.dtype,
        fill_value=0.0,
        compressors=[comp] if compress else None,
    )
    return tmp_array, tmp_dir
