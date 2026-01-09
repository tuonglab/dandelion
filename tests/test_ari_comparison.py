#!/usr/bin/env python
"""
Test Adjusted Rand Index (ARI) comparison between find_clones_polars and find_clones.

This validates that the polars and pandas implementations produce identical clustering.
"""

import pytest
from sklearn.metrics import adjusted_rand_score
import polars as pl

from dandelion import Dandelion
from dandelion.utilities._polars import DandelionPolars
from dandelion.tools import find_clones
from dandelion.tools._tools_polars import concat, find_clones_polars


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_find_clones_pandas_vs_polars(airr_reannotated):
    """Test ARI between pandas and polars find_clones implementations."""
    # Create Dandelion (pandas) object
    dan_pd = Dandelion(airr_reannotated, verbose=False)

    # Create DandelionPolars object - use concat to handle dtype issues
    dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
    dan_pl = concat([dan_pl_init], verbose=False)

    # Run pandas find_clones (modifies in place)
    find_clones(dan_pd, verbose=False)
    pandas_clones = dan_pd._data["clone_id"].values

    # Run polars find_clones
    dan_pl_clones = find_clones_polars(dan_pl, verbose=False)

    # Collect polars data
    pl_data = dan_pl_clones._data
    if isinstance(pl_data, pl.LazyFrame):
        pl_data = pl_data.collect()

    # Extract clone IDs in same order as pandas
    polars_clones = pl_data["clone_id"].to_numpy()

    # Calculate ARI
    ari = adjusted_rand_score(pandas_clones, polars_clones)

    # ARI of 1.0 means perfect agreement
    assert (
        ari > 0.99
    ), f"ARI {ari:.4f} indicates clustering difference between pandas and polars versions"


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_identity_thresholds(airr_reannotated):
    """Test ARI across different identity thresholds."""
    results = []

    for identity in [0.85, 0.90, 0.95, 1.0]:
        # Create Dandelion (pandas) object
        dan_pd = Dandelion(airr_reannotated, verbose=False)

        # Create DandelionPolars object
        dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
        dan_pl = concat([dan_pl_init], verbose=False)

        # Run with identity threshold
        find_clones(dan_pd, identity=identity, verbose=False)
        pandas_clones = dan_pd._data["clone_id"].values

        dan_pl_clones = find_clones_polars(
            dan_pl, identity=identity, verbose=False
        )
        pl_data = dan_pl_clones._data
        if isinstance(pl_data, pl.LazyFrame):
            pl_data = pl_data.collect()
        polars_clones = pl_data["clone_id"].to_numpy()

        # Calculate ARI
        ari = adjusted_rand_score(pandas_clones, polars_clones)

        results.append(
            {
                "identity": identity,
                "ari": ari,
                "pandas_unique_clones": len(set(pandas_clones))
                - (1 if "" in pandas_clones else 0),
                "polars_unique_clones": len(set(polars_clones))
                - (1 if "" in polars_clones else 0),
            }
        )

        assert ari > 0.99, f"ARI mismatch at identity={identity}: {ari:.4f}"


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_by_alleles(airr_reannotated):
    """Test ARI with by_alleles parameter."""
    for by_alleles in [False, True]:
        # Create Dandelion (pandas) object
        dan_pd = Dandelion(airr_reannotated, verbose=False)

        # Create DandelionPolars object
        dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
        dan_pl = concat([dan_pl_init], verbose=False)

        # Run with by_alleles
        find_clones(dan_pd, by_alleles=by_alleles, verbose=False)
        pandas_clones = dan_pd._data["clone_id"].values

        dan_pl_clones = find_clones_polars(
            dan_pl, by_alleles=by_alleles, verbose=False
        )
        pl_data = dan_pl_clones._data
        if isinstance(pl_data, pl.LazyFrame):
            pl_data = pl_data.collect()
        polars_clones = pl_data["clone_id"].to_numpy()

        # Calculate ARI
        ari = adjusted_rand_score(pandas_clones, polars_clones)

        assert ari > 0.99, f"ARI mismatch at by_alleles={by_alleles}: {ari:.4f}"


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_junction_keys(airr_reannotated):
    """Test ARI across different junction key columns."""
    # Test columns that exist in the test data
    test_keys = ["junction", "junction_aa"]

    # Check which columns are available
    available_keys = [k for k in test_keys if k in airr_reannotated.columns]

    if not available_keys:
        pytest.skip("No junction columns available in test data")

    for junction_key in available_keys:
        # Create Dandelion (pandas) object
        dan_pd = Dandelion(airr_reannotated, verbose=False)

        # Create DandelionPolars object
        dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
        dan_pl = concat([dan_pl_init], verbose=False)

        # Run with specific junction_key
        find_clones(dan_pd, key=junction_key, verbose=False)
        pandas_clones = dan_pd._data["clone_id"].values

        dan_pl_clones = find_clones_polars(
            dan_pl, key=junction_key, verbose=False
        )
        pl_data = dan_pl_clones._data
        if isinstance(pl_data, pl.LazyFrame):
            pl_data = pl_data.collect()
        polars_clones = pl_data["clone_id"].to_numpy()

        # Calculate ARI
        ari = adjusted_rand_score(pandas_clones, polars_clones)

        assert (
            ari > 0.99
        ), f"ARI mismatch at junction_key={junction_key}: {ari:.4f}"


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_sequence_keys(airr_reannotated):
    """Test ARI using sequence-based columns for clone detection."""
    # Test sequence-based columns if available
    sequence_keys = ["sequence", "sequence_alignment"]

    # Check which columns are available
    available_keys = [k for k in sequence_keys if k in airr_reannotated.columns]

    if not available_keys:
        pytest.skip("No sequence columns available in test data")

    for seq_key in available_keys:
        # Create Dandelion (pandas) object
        dan_pd = Dandelion(airr_reannotated, verbose=False)

        # Create DandelionPolars object
        dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
        dan_pl = concat([dan_pl_init], verbose=False)

        # Run with specific sequence key
        find_clones(dan_pd, key=seq_key, verbose=False)
        pandas_clones = dan_pd._data["clone_id"].values

        dan_pl_clones = find_clones_polars(dan_pl, key=seq_key, verbose=False)
        pl_data = dan_pl_clones._data
        if isinstance(pl_data, pl.LazyFrame):
            pl_data = pl_data.collect()
        polars_clones = pl_data["clone_id"].to_numpy()

        # Calculate ARI
        ari = adjusted_rand_score(pandas_clones, polars_clones)

        assert ari > 0.99, f"ARI mismatch at sequence_key={seq_key}: {ari:.4f}"


@pytest.mark.usefixtures("airr_reannotated")
def test_ari_combined_parameters(airr_reannotated):
    """Test ARI with combined parameter variations."""
    # Test combinations of parameters
    test_cases = [
        {"identity": 0.85, "by_alleles": False, "key": "junction_aa"},
        {"identity": 0.90, "by_alleles": True, "key": "junction_aa"},
        {"identity": 0.95, "by_alleles": False, "key": "junction"},
    ]

    for params in test_cases:
        # Check if key exists
        if params["key"] not in airr_reannotated.columns:
            continue

        # Create Dandelion (pandas) object
        dan_pd = Dandelion(airr_reannotated, verbose=False)

        # Create DandelionPolars object
        dan_pl_init = DandelionPolars(airr_reannotated, verbose=False)
        dan_pl = concat([dan_pl_init], verbose=False)

        # Run pandas with params
        find_clones(
            dan_pd,
            identity=params["identity"],
            by_alleles=params["by_alleles"],
            key=params["key"],
            verbose=False,
        )
        pandas_clones = dan_pd._data["clone_id"].values

        # Run polars with params
        dan_pl_clones = find_clones_polars(
            dan_pl,
            identity=params["identity"],
            by_alleles=params["by_alleles"],
            key=params["key"],
            verbose=False,
        )
        pl_data = dan_pl_clones._data
        if isinstance(pl_data, pl.LazyFrame):
            pl_data = pl_data.collect()
        polars_clones = pl_data["clone_id"].to_numpy()

        # Calculate ARI
        ari = adjusted_rand_score(pandas_clones, polars_clones)

        assert ari > 0.99, f"ARI mismatch with params {params}: {ari:.4f}"
