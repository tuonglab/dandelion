from __future__ import annotations
import hashlib

import numpy as np

from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process
from typing import Callable, Protocol, runtime_checkable


def dist_func_long_sep(
    x: list[str],
    y: list[str],
    metric: Metric,
    pad_to_max: bool = False,
    sep: str = "#",
) -> float:
    """
    Concatenate column-wise with long separators and apply the metric.
    """
    # If the metric is a substitution matrix, skip sep padding
    if isinstance(metric, SubstitutionMatrixMetric):
        # Just concatenate without adding separators
        s1 = "".join(x)
        s2 = "".join(y)
    elif pad_to_max:
        max_len = [max(len(a), len(b)) for a, b in zip(x, y)]
        s1_parts, s2_parts = [], []

        for s1, s2, le in zip(x, y, max_len):
            # Pad each element
            s1_parts.append(s1.ljust(le + 1, sep))
            s2_parts.append(s2.ljust(le + 1, sep))

        # Join with per-column separators
        s1_result, s2_result = [], []
        for i, (s1_part, s2_part, le) in enumerate(
            zip(s1_parts, s2_parts, max_len)
        ):
            s1_result.append(s1_part)
            s2_result.append(s2_part)

            # Add separator between columns (but not after the last one)
            if i < len(max_len) - 1:
                col_sep = sep * (le + 2)  # Longer than padded length (le + 1)
                s1_result.append(col_sep)
                s2_result.append(col_sep)

        s1 = "".join(s1_result)
        s2 = "".join(s2_result)
    else:
        # Dynamically choose separator length: longer than the max column
        try:
            max_x = max(len(s) for s in x)
        except ValueError:
            max_x = 0
        try:
            max_y = max(len(s) for s in y)
        except ValueError:
            max_y = 0
        max_len = max(max_x, max_y)
        long_sep = sep * (max_len + 1)
        s1 = long_sep.join(x)
        s2 = long_sep.join(y)

    return metric.compute(s1, s2)


@runtime_checkable
class Metric(Protocol):
    """
    Protocol for metric objects used to compute distance between two strings.

    Implementors must provide `compute(s1: str, s2: str) -> float`.
    Optionally can provide `compute_vectorized(seqs: list[str]) -> np.ndarray` for performance.
    """

    def compute(self, s1: str, s2: str) -> float: ...

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray: ...


class CallableMetric:
    """Wraps a callable f(s1, s2) or f(seqs) into a Metric object."""

    def __init__(
        self,
        func: Callable[[str, str], float] | None = None,
        vectorized_func: Callable[[list[str]], np.ndarray] | None = None,
    ):
        if func is None and vectorized_func is None:
            raise ValueError(
                "Must provide at least one of 'func' or 'vectorized_func'"
            )

        self.func = func
        self._vectorized_func = vectorized_func

    def compute(self, s1: str, s2: str) -> float:
        """Use provided function to compute distance between two strings."""
        return float(self.func(s1, s2))

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray:
        """Use provided vectorized function."""
        # Use custom vectorized implementation if provided
        if self._vectorized_func is not None:
            return self._vectorized_func(seqs)

        # Fall back to loop using pairwise function
        n = len(seqs)
        if n == 0:
            return np.empty((0, 0), dtype=float)

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute(seqs[i], seqs[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix


class LevenshteinMetric:
    """Metric that computes Levenshtein edit distance."""

    def compute(self, s1: str, s2: str) -> float:
        return float(Levenshtein.distance(s1, s2))

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray:
        if not seqs:
            return np.empty((0, 0), dtype=float)

        return process.cdist(
            seqs,
            seqs,
            scorer=Levenshtein.distance,
            dtype=np.int32,
        ).astype(float)


class HammingMetric:
    """Metric that computes Hamming distance between two strings."""

    def __init__(self, verbose: bool = True):
        """
        Initialize with automatic or manual backend selection.

        Parameters
        ----------
        backend : str, optional
            Force a specific backend: "cupy", "torch", or "numpy".
            If None, auto-detects best available option.
        verbose : bool
            Print which backend is being used.
        """
        self.verbose = verbose
        self._auto_detect_backend()

    def _auto_detect_backend(self):
        """Auto-detect best available backend."""
        # Try CuPy first (CUDA)
        try:
            import cupy as cp

            cp.cuda.Device(0).compute_capability  # Test if GPU is accessible
            self.backend_name = "cupy"
            self.cp = cp
            if self.verbose:
                print(f"Using CuPy backend with CUDA GPU")
            return
        except (ImportError, Exception):
            pass

        # Try PyTorch with GPU (CUDA or MPS)
        try:
            import torch

            if torch.cuda.is_available():
                self.backend_name = "torch"
                self.torch = torch
                self.device = torch.device("cuda")
                if self.verbose:
                    print(f"Using PyTorch backend with CUDA GPU")
                return
            elif torch.backends.mps.is_available():
                self.backend_name = "torch"
                self.torch = torch
                self.device = torch.device("mps")
                if self.verbose:
                    print(f"Using PyTorch backend with Apple Metal GPU")
                return
        except ImportError:
            pass

        # Fall back to NumPy CPU
        self.backend_name = "numpy"
        if self.verbose:
            print(f"Using NumPy backend (CPU only)")

    def compute(self, s1: str, s2: str) -> float:
        """
        Compute Hamming distance between two strings.

        Raises ValueError if strings are not of equal length.
        """
        if len(s1) != len(s2):
            raise ValueError(
                f"Hamming distance requires equal length strings. "
                f"Got lengths {len(s1)} and {len(s2)}"
            )
        return float(sum(c1 != c2 for c1, c2 in zip(s1, s2)))

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray:
        """
        Compute pairwise Hamming distances using the best available backend.

        Parameters
        ----------
        seqs : list[str]
            List of sequences to compare.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n, n).
        """
        n = len(seqs)

        if n == 0:
            return np.array([[]])

        # Route to appropriate backend
        if self.backend_name == "cupy":
            return self._compute_cupy(seqs, n)
        elif self.backend_name == "torch":
            return self._compute_torch(seqs, n)
        else:  # numpy
            return self._compute_numpy(seqs, n)

    def _compute_cupy(self, seqs: list[str], n: int) -> np.ndarray:
        """CuPy implementation."""

        # Convert sequences to integer arrays (ASCII codes)
        try:
            seqs_array = self.cp.array(
                [[ord(c) for c in s] for s in seqs], dtype=self.cp.int16
            )
        except ValueError:
            max_len = max(len(s) for s in seqs)
            seqs_array = self.cp.zeros((n, max_len), dtype=self.cp.int16)
            for i, s in enumerate(seqs):
                seqs_array[i, : len(s)] = self.cp.array(
                    [ord(c) for c in s], dtype=self.cp.int16
                )

        dist_matrix = (
            (seqs_array[:, None, :] != seqs_array[None, :, :])
            .sum(axis=2)
            .astype(self.cp.float32)
        )

        return self.cp.asnumpy(dist_matrix)

    def _compute_torch(self, seqs: list[str], n: int) -> np.ndarray:
        """PyTorch implementation."""
        seqs_tensor = self.torch.tensor(
            np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(
                n, -1
            ),
            device=self.device,
        )

        dist_matrix = (
            (seqs_tensor[:, None, :] != seqs_tensor[None, :, :])
            .sum(dim=2)
            .float()
        )

        return dist_matrix.cpu().numpy()

    def _compute_numpy(self, seqs: list[str], n: int) -> np.ndarray:
        """NumPy implementation."""
        seqs_array = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(
            n, -1
        )
        return (
            (seqs_array[:, None, :] != seqs_array[None, :, :])
            .sum(axis=2)
            .astype(np.float32)
        )


class IdentityMetric:
    """Metric that computes identity distance between strings using stable hash-based comparison."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._auto_detect_backend()

    def _auto_detect_backend(self):
        """Auto-detect best available backend."""
        # Try CuPy first (CUDA)
        try:
            import cupy as cp

            cp.cuda.Device(0).compute_capability
            self.backend_name = "cupy"
            self.cp = cp
            if self.verbose:
                print("Using CuPy backend with CUDA GPU for identity")
            return
        except Exception:
            pass

        # Try PyTorch with GPU
        try:
            import torch

            if torch.cuda.is_available():
                self.backend_name = "torch"
                self.torch = torch
                self.device = torch.device("cuda")
                if self.verbose:
                    print("Using PyTorch backend with CUDA GPU for identity")
                return
            elif torch.backends.mps.is_available():
                self.backend_name = "torch"
                self.torch = torch
                self.device = torch.device("mps")
                if self.verbose:
                    print(
                        "Using PyTorch backend with Apple Metal GPU for identity"
                    )
                return
        except ImportError:
            pass

        # Fall back to NumPy CPU
        self.backend_name = "numpy"
        if self.verbose:
            print("Using NumPy backend (CPU only) for identity")

    @staticmethod
    def _stable_hash(s: str) -> np.uint64:
        """Compute a stable 64-bit hash for a string."""
        # blake2b is faster than sha256 and designed for hashing
        return np.uint64(
            int.from_bytes(
                hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                byteorder="little",
                signed=False,
            )
        )

    def _hash_sequences(self, seqs: list[str]) -> np.ndarray:
        """Hash all sequences into a NumPy int64 array."""
        # Pre-allocate array for better performance
        hashes = np.empty(len(seqs), dtype=np.uint64)
        for i, s in enumerate(seqs):
            hashes[i] = self._stable_hash(s)
        return hashes

    def compute(self, s1: str, s2: str) -> float:
        """Compute identity distance between two strings (0 = same, 1 = different)."""
        return 0.0 if s1 == s2 else 1.0

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray:
        """
        Compute pairwise identity distance matrix.

        Parameters
        ----------
        seqs : list[str]
            List of sequences to compare.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n, n), where:
            - 0.0 = sequences are identical
            - 1.0 = sequences are different
        """
        n = len(seqs)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        hashes = self._hash_sequences(seqs)

        if self.backend_name == "cupy":
            return self._compute_cupy(hashes)
        elif self.backend_name == "torch":
            return self._compute_torch(hashes)
        else:
            return self._compute_numpy(hashes)

    def _compute_numpy(self, hashes: np.ndarray) -> np.ndarray:
        """NumPy backend."""
        identity = hashes[:, None] == hashes[None, :]
        return (~identity).astype(np.float32)

    def _compute_cupy(self, hashes: np.ndarray) -> np.ndarray:
        """CuPy backend."""
        h = self.cp.asarray(hashes)
        identity = h[:, None] == h[None, :]
        return self.cp.asnumpy((~identity).astype(self.cp.float32))

    def _compute_torch(self, hashes: np.ndarray) -> np.ndarray:
        """PyTorch backend."""
        h = self.torch.as_tensor(hashes, device=self.device)
        identity = h[:, None] == h[None, :]
        return (~identity).float().cpu().numpy()


class SubstitutionMatrixMetric:
    """
    Metric that uses a substitution matrix (e.g. BLOSUM62),
    following the approach used by Scirpy's AlignmentDistanceCalculator.

    Distance between sequences s1 and s2 is defined as:
        distance = max_achievable_score - alignment_score
    where max_achievable_score = min(self_score(s1), self_score(s2)).

    Parameters
    ----------
    matrix_name : str
        Name of the Biopython substitution matrix (e.g., "BLOSUM62").
    gap_open : float, default=-11
        Gap opening penalty for alignments.
    gap_extend : float, default=-11
        Gap extension penalty for alignments.
    """

    def __init__(self, matrix_name="BLOSUM62", gap_open=-11, gap_extend=-11):
        self.aligner = PairwiseAligner()
        self.aligner.mode = "global"

        # Load substitution matrix
        sub_mat = load(matrix_name)
        self.aligner.substitution_matrix = sub_mat

        # Set affine gap penalties
        self.aligner.open_gap_score = float(gap_open)
        self.aligner.extend_gap_score = float(gap_extend)

    def _self_score(self, seq: str) -> float:
        """
        Compute the self-alignment score of a sequence using the configured aligner.

        Parameters
        ----------
        seq : str
            Amino acid sequence.

        Returns
        -------
        float
            Self-alignment score.
        """
        return self.aligner.score(seq, seq)

    def compute(self, s1: str, s2: str) -> float:
        """
        Compute the BLOSUM-based distance between two sequences.

        Parameters
        ----------
        s1, s2 : str
            Amino acid sequences to compare.

        Returns
        -------
        float
            Non-negative distance between s1 and s2.
        """
        score_12 = self.aligner.score(s1, s2)
        s1_self = self._self_score(s1)
        s2_self = self._self_score(s2)
        max_score = min(s1_self, s2_self)
        return max(0.0, max_score - score_12)

    def compute_vectorized(self, seqs: list[str]) -> np.ndarray:
        """
        Vectorized pairwise distance computation using substitution matrices.

        Uses Biopython's PairwiseAligner to score all pairs and
        computes distances as min(self_score_i, self_score_j) - pair_score.

        Parameters
        ----------
        seqs : list[str]
            List of sequences to compare (can have different lengths).

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n, n).
        """
        n = len(seqs)

        if n == 0:
            return np.empty((0, 0), dtype=float)

        # Precompute self-alignment scores once
        self_scores = np.array([self._self_score(s) for s in seqs], dtype=float)

        lengths = np.array([len(s) for s in seqs], dtype=int)
        all_equal_len = np.all(lengths == lengths[0])
        assert (
            all_equal_len
        ), "Currently only equal-length sequences are supported for vectorized scoring."

        # Build character index and score lookup from the configured substitution matrix
        sub_mat = self.aligner.substitution_matrix
        alphabet = list(sub_mat.alphabet)
        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        k = len(alphabet)
        score_lookup = np.zeros((k, k), dtype=float)
        for a in alphabet:
            for b in alphabet:
                score_lookup[char_to_idx[a], char_to_idx[b]] = float(
                    sub_mat[a, b]
                )
        # Encode sequences to indices (n, L)
        seq_indices = np.array(
            [[char_to_idx[c] for c in s] for s in seqs], dtype=int
        )
        # Broadcast lookup to get per-position pair scores and sum over positions
        pair_scores = score_lookup[
            seq_indices[:, None, :], seq_indices[None, :, :]
        ].sum(axis=2)
        # Distance matrix from min of self scores
        max_scores = np.minimum(self_scores[:, None], self_scores[None, :])
        dist_matrix = np.maximum(max_scores - pair_scores, 0.0)
        # Ensure diagonal is exactly zero
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix


# -------------------------
# Metric resolver
# -------------------------
def resolve_metric(
    dist_func: Callable[[str, str], float] | str | Metric,
) -> Metric:
    """
    Convert user-supplied dist_func into a Metric object.

    Parameters
    ----------
    dist_func : Callable[[str, str], float] | str | Metric | None
        Distance function specification. Can be:
        - "levenshtein": uses Levenshtein distance
        - "hamming": uses Hamming distance
        - "identity": uses 100% identity metric (based on stable hashing)
        - str (other): treated as substitution matrix name (e.g., "BLOSUM62")
        - Callable: wrapped in CallableMetric
        - Metric: returned as-is

    Returns
    -------
    Metric
        Resolved Metric object.
    """
    if isinstance(dist_func, Metric):
        return dist_func

    if callable(dist_func):
        return CallableMetric(dist_func)

    if isinstance(dist_func, str):
        # Check for hamming distance
        if dist_func.lower() == "hamming":
            return HammingMetric()
        elif dist_func.lower() == "levenshtein":
            return LevenshteinMetric()
        elif dist_func.lower() == "identity":
            return IdentityMetric()
        # Otherwise treat as substitution matrix
        return SubstitutionMatrixMetric(dist_func)

    raise TypeError(
        "dist_func must be 'hamming', 'levenshtein', 'identity', substitution matrix name or a callable metric e.g. lambda function"
    )
