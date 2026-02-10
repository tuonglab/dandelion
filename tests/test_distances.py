"""Tests for dandelion.utilities._distances module."""

import inspect

import numpy as np
import pytest

from dandelion.utilities._distances import (
    CallableMetric,
    HammingMetric,
    IdentityMetric,
    LevenshteinMetric,
    Metric,
    SubstitutionMatrixMetric,
    prepare_sequences_with_separator,
    resolve_metric,
)


def _force_numpy(metric):
    """Force a metric to use the numpy backend (GPU paths are pragma: no cover)."""
    metric.backend_name = "numpy"
    return metric


# ---------------------------------------------------------------------------
# prepare_sequences_with_separator
# ---------------------------------------------------------------------------
class TestPrepareSequences:
    """Test prepare_sequences_with_separator."""

    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert prepare_sequences_with_separator([], LevenshteinMetric()) == []

    def test_empty_rows(self):
        """Test rows with no columns returns empty list."""
        assert prepare_sequences_with_separator([[]], LevenshteinMetric()) == []

    def test_single_column_no_padding(self):
        """Test single column extraction without padding."""
        seqs = [["ACGT"], ["AA"], ["CCCC"]]
        result = prepare_sequences_with_separator(seqs, LevenshteinMetric())
        assert result == ["ACGT", "AA", "CCCC"]

    def test_single_column_with_padding(self):
        """Test single column with global padding to max length."""
        seqs = [["ACGT"], ["AA"], ["CCCC"]]
        result = prepare_sequences_with_separator(
            seqs, LevenshteinMetric(), pad_to_max=True
        )
        assert all(len(s) == 4 for s in result)
        assert result[0] == "ACGT"
        assert result[1] == "AA##"

    def test_multi_column_substitution_matrix(self):
        """Test multi-column with SubstitutionMatrixMetric uses no separator."""
        seqs = [["ACG", "TGA"], ["AAA", "TTT"]]
        metric = SubstitutionMatrixMetric("BLOSUM62")
        result = prepare_sequences_with_separator(seqs, metric)
        assert result == ["ACGTGA", "AAATTT"]

    def test_multi_column_no_padding(self):
        """Test multi-column with dynamic separator."""
        seqs = [["ACGT", "CGAT"], ["AAAA", "TTTT"]]
        result = prepare_sequences_with_separator(seqs, LevenshteinMetric())
        # Dynamic separator: sep * (max_len + 1) = "#" * 5
        assert result[0] == "ACGT#####CGAT"

    def test_multi_column_with_padding(self):
        """Test multi-column with padding produces equal-length strings."""
        seqs = [["ACGT", "CG"], ["AA", "TTTT"]]
        result = prepare_sequences_with_separator(
            seqs, LevenshteinMetric(), pad_to_max=True
        )
        assert len(result) == 2
        # All results should have the same length
        assert len(result[0]) == len(result[1])


# ---------------------------------------------------------------------------
# CallableMetric
# ---------------------------------------------------------------------------
class TestCallableMetric:
    """Test CallableMetric wrapper."""

    def test_no_func_raises(self):
        """Test that providing neither func nor vectorized_func raises."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            CallableMetric()

    def test_auto_detect_pairwise(self):
        """Test auto-detection of 2-param callable as pairwise."""
        m = CallableMetric(lambda s1, s2: 1.0)
        assert m.func is not None
        assert m._vectorized_func is None

    def test_auto_detect_vectorized(self):
        """Test auto-detection of 1-param callable as vectorized."""
        m = CallableMetric(lambda seqs: np.zeros((len(seqs), len(seqs))))
        assert m.func is None
        assert m._vectorized_func is not None

    def test_wrong_param_count_raises(self):
        """Test that 3-param callable raises ValueError."""
        with pytest.raises(ValueError, match="got 3"):
            CallableMetric(lambda a, b, c: 0.0)

    def test_builtin_c_function_fallback(self, monkeypatch):
        """Test fallback for C builtins with no inspectable signature."""
        original_signature = inspect.signature

        def mock_signature(func):
            if func is len:
                raise ValueError(
                    "no signature found for builtin type <built-in function len>"
                )
            return original_signature(func)

        monkeypatch.setattr(inspect, "signature", mock_signature)
        m = CallableMetric(len)
        assert m.func is len

    def test_explicit_both_funcs(self):
        """Test providing both pairwise and vectorized functions."""
        pairwise = lambda s1, s2: 1.0
        vectorized = lambda seqs: np.ones((len(seqs), len(seqs)))
        m = CallableMetric(func=pairwise, vectorized_func=vectorized)
        assert m.func is pairwise
        assert m._vectorized_func is vectorized

    def test_compute_with_pairwise_func(self):
        """Test compute() delegates to pairwise func."""
        m = CallableMetric(lambda s1, s2: float(s1 != s2))
        assert m.compute("A", "A") == 0.0
        assert m.compute("A", "B") == 1.0

    def test_compute_fallback_to_vectorized(self):
        """Test compute() falls back to vectorized when no pairwise func."""

        def vec_fn(seqs):
            n = len(seqs)
            mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    mat[i, j] = float(seqs[i] != seqs[j])
            return mat

        m = CallableMetric(vectorized_func=vec_fn)
        assert m.compute("A", "A") == 0.0
        assert m.compute("A", "B") == 1.0

    def test_compute_vectorized_with_vectorized_func(self):
        """Test compute_vectorized() delegates to vectorized func."""

        def vec_fn(seqs):
            n = len(seqs)
            return np.eye(n)

        m = CallableMetric(vectorized_func=vec_fn)
        result = m.compute_vectorized(["A", "B", "C"])
        np.testing.assert_array_equal(result, np.eye(3))

    def test_compute_vectorized_pairwise_fallback(self):
        """Test compute_vectorized() falls back to pairwise loop."""
        m = CallableMetric(lambda s1, s2: float(s1 != s2))
        result = m.compute_vectorized(["A", "A", "B"])
        assert result.shape == (3, 3)
        assert result[0, 1] == 0.0  # "A" vs "A"
        assert result[0, 2] == 1.0  # "A" vs "B"
        np.testing.assert_array_equal(result, result.T)

    def test_compute_vectorized_empty(self):
        """Test compute_vectorized() with empty input."""
        m = CallableMetric(lambda s1, s2: 0.0)
        result = m.compute_vectorized([])
        assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# LevenshteinMetric
# ---------------------------------------------------------------------------
class TestLevenshteinMetric:
    """Test LevenshteinMetric."""

    def test_compute(self):
        """Test pairwise Levenshtein distance."""
        m = LevenshteinMetric()
        assert m.compute("ACGT", "ACGT") == 0.0
        assert m.compute("ACGT", "ACGG") == 1.0
        assert m.compute("ABC", "XYZ") == 3.0

    def test_vectorized(self):
        """Test vectorized distance matrix."""
        m = LevenshteinMetric()
        result = m.compute_vectorized(["ACGT", "ACGG", "TTTT"])
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 1.0  # 1 substitution
        assert result[0, 2] > 0

    def test_vectorized_empty(self):
        """Test vectorized with empty input."""
        m = LevenshteinMetric()
        result = m.compute_vectorized([])
        assert result.shape == (0, 0)

    def test_vectorized_multi_cpu(self):
        """Test vectorized with multiple CPUs."""
        m = LevenshteinMetric()
        result = m.compute_vectorized(["ACGT", "ACGG"], n_cpus=2)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# HammingMetric
# ---------------------------------------------------------------------------
class TestHammingMetric:
    """Test HammingMetric."""

    def test_compute(self):
        """Test pairwise Hamming distance."""
        m = HammingMetric(verbose=False)
        assert m.compute("ACGT", "ACGT") == 0.0
        assert m.compute("ACGT", "ACGG") == 1.0
        assert m.compute("AAAA", "TTTT") == 4.0

    def test_compute_unequal_length_raises(self):
        """Test that unequal-length strings raise ValueError."""
        m = HammingMetric(verbose=False)
        with pytest.raises(ValueError, match="equal length"):
            m.compute("ABC", "AB")

    def test_vectorized(self):
        """Test vectorized Hamming distance matrix."""
        m = _force_numpy(HammingMetric(verbose=False))
        result = m.compute_vectorized(["ACGT", "ACGG", "TTTT"])
        assert result.shape == (3, 3)
        assert result[0, 1] == 1.0  # 1 mismatch
        assert result[0, 0] == 0.0

    def test_vectorized_empty(self):
        """Test vectorized with empty input."""
        m = HammingMetric(verbose=False)
        result = m.compute_vectorized([])
        assert result.shape == (1, 0)


# ---------------------------------------------------------------------------
# IdentityMetric
# ---------------------------------------------------------------------------
class TestIdentityMetric:
    """Test IdentityMetric."""

    def test_compute(self):
        """Test pairwise identity distance."""
        m = IdentityMetric(verbose=False)
        assert m.compute("ACGT", "ACGT") == 0.0
        assert m.compute("ACGT", "TTTT") == 1.0

    def test_stable_hash_deterministic(self):
        """Test that hashing the same string gives the same result."""
        h1 = IdentityMetric._stable_hash("ACGT")
        h2 = IdentityMetric._stable_hash("ACGT")
        assert h1 == h2

    def test_stable_hash_different(self):
        """Test that different strings produce different hashes."""
        h1 = IdentityMetric._stable_hash("ACGT")
        h2 = IdentityMetric._stable_hash("TTTT")
        assert h1 != h2

    def test_hash_sequences(self):
        """Test batch hashing of sequences."""
        m = IdentityMetric(verbose=False)
        hashes = m._hash_sequences(["A", "B", "A"])
        assert hashes[0] == hashes[2]
        assert hashes[0] != hashes[1]

    def test_vectorized(self):
        """Test vectorized identity distance matrix."""
        m = _force_numpy(IdentityMetric(verbose=False))
        result = m.compute_vectorized(["ACGT", "ACGT", "TTTT"])
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0  # same
        assert result[0, 1] == 0.0  # same
        assert result[0, 2] == 1.0  # different

    def test_vectorized_empty(self):
        """Test vectorized with empty input."""
        m = IdentityMetric(verbose=False)
        result = m.compute_vectorized([])
        assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# SubstitutionMatrixMetric
# ---------------------------------------------------------------------------
class TestSubstitutionMatrixMetric:
    """Test SubstitutionMatrixMetric."""

    def test_init(self):
        """Test initialization with BLOSUM62."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        assert m.aligner.mode == "global"

    def test_compute(self):
        """Test pairwise BLOSUM distance."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        assert m.compute("ACDE", "ACDE") == 0.0
        assert m.compute("ACDE", "FGHK") > 0

    def test_self_score_empty(self):
        """Test self-score of empty sequence is zero."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        assert m._self_score("") == 0.0

    def test_self_score_valid(self):
        """Test self-score of a valid sequence is positive."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        score = m._self_score("ACDE")
        assert score > 0

    def test_vectorized(self):
        """Test vectorized BLOSUM distance matrix."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        result = m.compute_vectorized(["ACDE", "ACDE", "FGHK"])
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0  # identical sequences
        assert result[0, 2] > 0  # different sequences

    def test_vectorized_empty(self):
        """Test vectorized with empty input."""
        m = SubstitutionMatrixMetric("BLOSUM62")
        result = m.compute_vectorized([])
        assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# resolve_metric
# ---------------------------------------------------------------------------
class TestResolveMetric:
    """Test resolve_metric factory function."""

    def test_metric_passthrough(self):
        """Test that a Metric instance is returned as-is."""
        m = LevenshteinMetric()
        assert resolve_metric(m) is m

    def test_callable_pairwise(self):
        """Test that a callable is wrapped in CallableMetric."""
        result = resolve_metric(lambda s1, s2: 0.0)
        assert isinstance(result, CallableMetric)

    def test_string_hamming(self):
        """Test resolving 'hamming' string."""
        assert isinstance(resolve_metric("hamming"), HammingMetric)

    def test_string_levenshtein(self):
        """Test resolving 'levenshtein' string."""
        assert isinstance(resolve_metric("levenshtein"), LevenshteinMetric)

    def test_string_identity(self):
        """Test resolving 'identity' string."""
        assert isinstance(resolve_metric("identity"), IdentityMetric)

    def test_string_substitution_matrix(self):
        """Test resolving a substitution matrix name."""
        assert isinstance(resolve_metric("BLOSUM62"), SubstitutionMatrixMetric)

    def test_invalid_type_raises(self):
        """Test that an invalid type raises TypeError."""
        with pytest.raises(TypeError):
            resolve_metric(42)


# ---------------------------------------------------------------------------
# Metric Protocol
# ---------------------------------------------------------------------------
class TestMetricProtocol:
    """Test that all metrics satisfy the Metric protocol."""

    def test_builtin_metrics_satisfy_protocol(self):
        """Test built-in metrics are instances of Metric protocol."""
        assert isinstance(LevenshteinMetric(), Metric)
        assert isinstance(HammingMetric(verbose=False), Metric)
        assert isinstance(IdentityMetric(verbose=False), Metric)
        assert isinstance(SubstitutionMatrixMetric("BLOSUM62"), Metric)

    def test_callable_metric_satisfies_protocol(self):
        """Test CallableMetric satisfies the Metric protocol."""
        m = CallableMetric(lambda s1, s2: 0.0)
        assert isinstance(m, Metric)
