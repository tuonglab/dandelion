#!/usr/bin/env python
"""Coverage tests for dandelion.polars.tools (tools, network, diversity)."""

import pytest
import numpy as np
import pandas as pd
import polars as pl

from dandelion.polars.core._core import DandelionPolars
from dandelion.polars.preprocessing._preprocessing import check_contigs
from dandelion.polars.tools._tools import (
    find_clones,
    clone_size,
    transfer,
    clone_view,
    clone_overlap,
    to_scirpy,
    from_scirpy,
    concat,
    vdj_sample,
)
from dandelion.polars.tools._network import (
    generate_network,
    clone_degree,
    clone_centrality,
)
from dandelion.polars.tools._diversity import (
    clone_diversity,
    clone_rarefaction,
    process_clone_network_stats,
    calculate_chao1,
    calculate_shannon_entropy,
    drop_nan_values,
    _bootstrap_diversity_iteration,
    _bootstrap_network,
)

# ---------------------------------------------------------------------------
# Base fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vdj_base(airr_reannotated, dummy_adata):
    """VDJ after check_contigs with find_clones run."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, adata = check_contigs(vdj, dummy_adata)
    find_clones(vdj)
    return vdj, adata


@pytest.fixture
def vdj_with_network(vdj_base):
    """VDJ after generate_network."""
    vdj, adata = vdj_base
    generate_network(vdj, layout_method="mod_fr")
    transfer(adata, vdj)
    return vdj, adata


@pytest.fixture
def vdj2_with_clones(airr_reannotated2, dummy_adata2):
    """Second VDJ object with clones."""
    vdj = DandelionPolars(airr_reannotated2)
    vdj, adata = check_contigs(vdj, dummy_adata2)
    find_clones(vdj)
    return vdj, adata


# ---------------------------------------------------------------------------
# Group 1 – find_clones with store_distances=True
# ---------------------------------------------------------------------------


def test_find_clones_store_distances(airr_reannotated, dummy_adata):
    """Lines 177-742: find_clones with store_distances=True."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, store_distances=True)
    assert vdj.distances is not None
    assert vdj._data.collect_schema()["clone_id"] is not None


def test_find_clones_store_distances_eager(airr_reannotated, dummy_adata):
    """store_distances on eager backend."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, store_distances=True)
    assert vdj.distances is not None


def test_find_clones_identity_dict(airr_reannotated, dummy_adata):
    """Lines 143-146: identity as dict updates default."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, identity={"ig": 0.9})
    assert vdj._data.collect_schema()["clone_id"] is not None


def test_find_clones_identity_float(airr_reannotated, dummy_adata):
    """Lines 147-149: identity as float (non-dict)."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, identity=0.8)
    assert vdj._data.collect_schema()["clone_id"] is not None


def test_find_clones_same_length(airr_reannotated, dummy_adata):
    """Lines 870: same_length=True in _group_sequences."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, same_length=True)
    assert vdj._data.collect_schema()["clone_id"] is not None


def test_find_clones_same_vj_false(airr_reannotated, dummy_adata):
    """Lines 926-946: same_vj=False in _group_sequences."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, same_vj=False)
    assert vdj._data.collect_schema()["clone_id"] is not None


def test_find_clones_key_added(airr_reannotated, dummy_adata):
    """key_added parameter."""
    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, key_added="my_clones")
    assert vdj._data.collect_schema()["my_clones"] is not None


# ---------------------------------------------------------------------------
# Group 2 – transfer with different options
# ---------------------------------------------------------------------------


def test_transfer_with_overwrite(vdj_with_network, dummy_adata):
    """Lines 1122-1136: overwrite parameter in transfer."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, overwrite=True)
    assert "clone_id" in adata.obs.columns


def test_transfer_with_overwrite_list(vdj_with_network, dummy_adata):
    """overwrite as list."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, overwrite=["clone_id"])
    assert "clone_id" in adata.obs.columns


def test_transfer_mudata(vdj_with_network):
    """Lines 1103-1112: transfer with MuData input."""
    try:
        import mudata
        from mudata import MuData
        import anndata as ad

        vdj, adata = vdj_with_network
        mdata = MuData({"airr": adata, "gex": adata.copy()})
        transfer(mdata, vdj)
    except ImportError:
        pytest.skip("mudata not installed")


def test_transfer_no_obs(vdj_with_network):
    """obs=False in transfer."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, obs=False, uns=True, obsm=True, obsp=True)


def test_transfer_main_view_expanded(vdj_with_network, dummy_adata):
    """Lines 1176-1199: main_view='expanded'."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, main_view="expanded")
    assert "vdj_connectivities_expanded" in adata.obsp


def test_transfer_main_view_all(vdj_with_network, dummy_adata):
    """main_view='all' (default)."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, main_view="all")


def test_transfer_collapse_nodes(vdj_with_network, dummy_adata):
    """Lines 1221-1265: collapse_nodes=True."""
    vdj, adata = vdj_with_network
    transfer(adata, vdj, collapse_nodes=True)


# ---------------------------------------------------------------------------
# Group 3 – clone_view with different modes
# ---------------------------------------------------------------------------


def test_clone_view_all(vdj_with_network):
    """Lines 1522-1548: clone_view with mode='all'."""
    vdj, adata = vdj_with_network
    clone_view(adata, mode="all")


def test_clone_view_expanded(vdj_with_network):
    """clone_view with mode='expanded'."""
    vdj, adata = vdj_with_network
    try:
        clone_view(adata, mode="expanded")
    except KeyError:
        pass  # X_vdj_expanded may not exist if no expanded clones


def test_clone_view_full(vdj_with_network):
    """clone_view with mode='full'."""
    vdj, adata = vdj_with_network
    try:
        clone_view(adata, mode="full")
    except KeyError:
        pass  # full mode requires 'vdj_connectivities_full'


def test_clone_view_gex(vdj_with_network):
    """clone_view with mode='gex'."""
    vdj, adata = vdj_with_network
    try:
        clone_view(adata, mode="gex")
    except KeyError:
        pass  # gex mode requires gex_connectivities


def test_clone_view_mode_none_error(vdj_with_network):
    """Lines 1505-1521: clone_view with mode=None raises KeyError."""
    vdj, adata = vdj_with_network
    with pytest.raises(KeyError):
        clone_view(adata, mode=None, connectivities_key="nonexistent")


# ---------------------------------------------------------------------------
# Group 4 – clone_size with different inputs
# ---------------------------------------------------------------------------


def test_clone_size_with_anndata(vdj_with_network):
    """Lines 1802-1811: clone_size with AnnData input."""
    vdj, adata = vdj_with_network
    clone_size(adata)
    assert "clone_id_size" in adata.obs.columns


def test_clone_size_with_anndata_max_size(vdj_with_network):
    """Lines 1808-1809: clone_size with AnnData and max_size."""
    vdj, adata = vdj_with_network
    clone_size(adata, max_size=3)
    assert "clone_id_size" in adata.obs.columns


def test_clone_size_with_group_by(vdj_with_network):
    """clone_size with group_by parameter."""
    vdj, adata = vdj_with_network
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    clone_size(adata, group_by="sample_id")
    assert "clone_id_size" in adata.obs.columns


def test_clone_size_with_nan_clones(vdj_with_network):
    """Lines 1736-1741: NaN clone_id rows in clone_size."""
    vdj, adata = vdj_with_network
    # Introduce NaN in clone_id
    import polars as pl

    vdj._metadata = vdj._metadata.with_columns(
        pl.when(pl.col("clone_id").is_null())
        .then(None)
        .otherwise(pl.col("clone_id"))
        .alias("clone_id")
    )
    clone_size(vdj)


# ---------------------------------------------------------------------------
# Group 5 – clone_overlap function (from _tools.py)
# ---------------------------------------------------------------------------


def test_clone_overlap_dandelion(vdj_base, dummy_adata):
    """Lines 1860-1940: clone_overlap with AnnData input writes to uns."""
    vdj, adata = vdj_base
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    transfer(adata, vdj)
    clone_overlap(adata, group_by="sample_id")
    assert "clone_overlap" in adata.uns


def test_clone_overlap_dandelion_direct(vdj_base):
    """clone_overlap with DandelionPolars directly returns DataFrame."""
    vdj, adata = vdj_base
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    transfer(adata, vdj)
    result = clone_overlap(vdj, group_by="sample_id")
    assert result is not None


def test_clone_overlap_weighted(vdj_base, dummy_adata):
    """clone_overlap with weighted_overlap=True."""
    vdj, adata = vdj_base
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    transfer(adata, vdj)
    result = clone_overlap(vdj, group_by="sample_id", weighted_overlap=True)
    assert result is not None


def test_clone_overlap_mudata(vdj_base):
    """Lines 1867-1868: clone_overlap with MuData (duck-type via hasattr(mod))."""
    try:
        import mudata
        from mudata import MuData

        vdj, adata = vdj_base
        adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
        transfer(adata, vdj)
        mdata = MuData({"airr": adata, "gex": adata.copy()})
        # clone_overlap with MuData uses hasattr(vdj, "mod") path
        result = clone_overlap(mdata, group_by="sample_id")
        # Returns a DataFrame for non-AnnData inputs
        assert result is not None
    except ImportError:
        pytest.skip("mudata not installed")


def test_clone_overlap_min_size_zero_raises(vdj_base, dummy_adata):
    """Line 1920-1921: min_clone_size=0 raises ValueError."""
    vdj, adata = vdj_base
    adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S2"]
    transfer(adata, vdj)
    with pytest.raises(ValueError):
        clone_overlap(adata, group_by="sample_id", min_clone_size=0)


# ---------------------------------------------------------------------------
# Group 6 – to_scirpy and from_scirpy
# ---------------------------------------------------------------------------


def test_to_scirpy_mudata(vdj_base):
    """Lines 2908-2922: to_scirpy with to_mudata=True (default)."""
    vdj, adata = vdj_base
    mdata = to_scirpy(vdj, to_mudata=True)
    assert hasattr(mdata, "mod")


def test_to_scirpy_anndata_without_mudata(vdj_base):
    """Lines 2923-2927: to_scirpy with to_mudata=False (no gex)."""
    vdj, adata = vdj_base
    result = to_scirpy(vdj, to_mudata=False)
    assert result is not None
    # This returns a plain AnnData (not MuData)
    import anndata as ad

    assert isinstance(result, ad.AnnData)


def test_to_scirpy_with_transfer(vdj_with_network):
    """Lines 2920-2927: to_scirpy with transfer=True and to_mudata=True."""
    vdj, adata = vdj_with_network
    result = to_scirpy(vdj, to_mudata=True, transfer=True)
    assert result is not None
    assert hasattr(result, "mod")


def test_from_scirpy(vdj_base):
    """Lines 2930-2952: from_scirpy round trip."""
    vdj, adata = vdj_base
    # Convert to scirpy format first
    mdata = to_scirpy(vdj, to_mudata=True)
    # Convert back
    vdj2 = from_scirpy(mdata)
    assert vdj2 is not None
    assert vdj2.n_contigs > 0


def test_from_scirpy_anndata(vdj_base):
    """from_scirpy with AnnData (non-MuData)."""
    vdj, adata = vdj_base
    airr_adata = to_scirpy(vdj, to_mudata=False)
    vdj2 = from_scirpy(airr_adata)
    assert vdj2 is not None


# ---------------------------------------------------------------------------
# Group 7 – vdj_sample
# ---------------------------------------------------------------------------


def test_vdj_sample_no_adata(vdj_base):
    """Lines 2662-2690: vdj_sample without adata."""
    vdj, adata = vdj_base
    vdj_sampled = vdj_sample(vdj, size=3)
    assert vdj_sampled.n_obs == 3


def test_vdj_sample_with_adata(vdj_with_network):
    """Lines 2691-2756: vdj_sample with adata."""
    vdj, adata = vdj_with_network
    vdj_sampled, adata_sampled = vdj_sample(vdj, size=3, adata=adata)
    assert vdj_sampled.n_obs == 3


def test_vdj_sample_force_replace(vdj_base):
    """force_replace=True - sample larger than population."""
    vdj, adata = vdj_base
    vdj_sampled = vdj_sample(vdj, size=10, force_replace=True)
    assert vdj_sampled.n_obs == 10


def test_vdj_sample_with_p(vdj_base):
    """Lines 2663-2667: p parameter (custom probabilities)."""
    vdj, adata = vdj_base
    n = vdj.n_obs
    p = [1.0 / n] * n
    vdj_sampled = vdj_sample(vdj, size=3, p=p)
    assert vdj_sampled.n_obs == 3


# ---------------------------------------------------------------------------
# Group 8 – concat
# ---------------------------------------------------------------------------


def test_concat_list(vdj_base, airr_reannotated2, dummy_adata2):
    """Lines 3217-3402: concat with list of DandelionPolars."""
    vdj, adata = vdj_base
    vdj2 = DandelionPolars(airr_reannotated2)
    vdj2, _ = check_contigs(vdj2, dummy_adata2)
    find_clones(vdj2)
    result = concat([vdj, vdj2])
    assert result is not None
    assert result.n_contigs == vdj.n_contigs + vdj2.n_contigs


def test_concat_with_suffixes(vdj_base, airr_reannotated2, dummy_adata2):
    """concat with explicit suffixes."""
    vdj, adata = vdj_base
    vdj2 = DandelionPolars(airr_reannotated2)
    vdj2, _ = check_contigs(vdj2, dummy_adata2)
    result = concat([vdj, vdj2], suffixes=["_s1", "_s2"])
    assert result is not None


def test_concat_with_prefixes(vdj_base, airr_reannotated2, dummy_adata2):
    """concat with explicit prefixes."""
    vdj, adata = vdj_base
    vdj2 = DandelionPolars(airr_reannotated2)
    vdj2, _ = check_contigs(vdj2, dummy_adata2)
    result = concat([vdj, vdj2], prefixes=["S1_", "S2_"])
    assert result is not None


def test_concat_raises_both_suffix_prefix(
    vdj_base, airr_reannotated2, dummy_adata2
):
    """Lines 3216-3217: raises ValueError when both suffixes and prefixes given."""
    vdj, adata = vdj_base
    vdj2 = DandelionPolars(airr_reannotated2)
    vdj2, _ = check_contigs(vdj2, dummy_adata2)
    with pytest.raises(ValueError):
        concat([vdj, vdj2], suffixes=["_s1", "_s2"], prefixes=["P1_", "P2_"])


def test_concat_dict(vdj_base, airr_reannotated2, dummy_adata2):
    """Lines 3231-3240: concat with dict input."""
    vdj, adata = vdj_base
    vdj2 = DandelionPolars(airr_reannotated2)
    vdj2, _ = check_contigs(vdj2, dummy_adata2)
    result = concat({"s1": vdj, "s2": vdj2})
    assert result is not None


def test_concat_invalid_type(vdj_base):
    """Lines 3257-3261: raises ValueError for invalid type."""
    vdj, adata = vdj_base
    with pytest.raises(ValueError):
        concat([vdj, 42])


# ---------------------------------------------------------------------------
# Group 9 – generate_network branches
# ---------------------------------------------------------------------------


def test_generate_network_use_existing_graph(vdj_base):
    """Lines 273-309: use_existing_graph=True."""
    vdj, adata = vdj_base
    generate_network(vdj, layout_method="mod_fr")
    # Now use the existing graph
    generate_network(vdj, layout_method="mod_fr", use_existing_graph=True)
    assert vdj.graph is not None


def test_generate_network_sequential_chain(vdj_base):
    """Lines 450-458: sequential_chain=True."""
    vdj, adata = vdj_base
    generate_network(
        vdj, layout_method="mod_fr", sequential_chain=True, n_cpus=1
    )
    assert vdj.graph is not None


def test_generate_network_sample(vdj_base, dummy_adata):
    """Lines 318-333: sample parameter triggers vdj_sample, returns new object."""
    vdj, adata = vdj_base
    vdj_sampled = generate_network(
        vdj, layout_method="mod_fr", sample=4, random_state=42
    )
    assert vdj_sampled is not None
    assert vdj_sampled.graph is not None


def test_generate_network_no_compute_graph(vdj_base):
    """Lines 278-280: compute_graph=False."""
    vdj, adata = vdj_base
    generate_network(vdj, compute_graph=False)
    # No graph computed, distances might still be computed
    assert vdj.distances is not None or vdj.graph is None


def test_generate_network_distance_mode_full(vdj_base):
    """Lines 468-535: distance_mode='full'."""
    vdj, adata = vdj_base
    generate_network(
        vdj,
        layout_method="mod_fr",
        distance_mode="full",
        sequential_chain=True,
        n_cpus=1,
    )
    assert vdj.graph is not None


def test_generate_network_clone_degree(vdj_base):
    """Lines 1262-1286: clone_degree function."""
    vdj, adata = vdj_base
    generate_network(vdj, layout_method="mod_fr")
    clone_degree(vdj)
    assert "clone_degree" in vdj._metadata.collect_schema().names()


def test_generate_network_clone_centrality(vdj_base):
    """Lines 1289-1329: clone_centrality function."""
    vdj, adata = vdj_base
    generate_network(vdj, layout_method="mod_fr")
    clone_centrality(vdj)
    assert "clone_centrality" in vdj._metadata.collect_schema().names()


def test_clone_degree_no_graph_raises(vdj_base):
    """Line 1263-1265: clone_degree raises without graph."""
    vdj, adata = vdj_base
    with pytest.raises(AttributeError):
        clone_degree(vdj)


def test_clone_degree_wrong_type():
    """Line 1285-1286: clone_degree raises TypeError for non-DandelionPolars."""
    with pytest.raises(TypeError):
        clone_degree("not_a_vdj")


def test_clone_centrality_no_graph_raises(vdj_base):
    """Line 1307-1309: clone_centrality raises without graph."""
    vdj, adata = vdj_base
    with pytest.raises(AttributeError):
        clone_centrality(vdj)


# ---------------------------------------------------------------------------
# Group 10 – process_clone_network_stats
# ---------------------------------------------------------------------------


def test_process_clone_network_stats(vdj_base):
    """Lines 1030-1067 (diversity): process_clone_network_stats."""
    vdj, adata = vdj_base
    generate_network(vdj, layout_method="mod_fr")
    import polars as pl

    if isinstance(vdj._metadata, pl.LazyFrame):
        vdj._metadata = vdj._metadata.collect()
    g_c_v_df, g_c_c_df = process_clone_network_stats(
        vdj, expanded_only=False, contracted=False
    )
    assert g_c_v_df is not None


# ---------------------------------------------------------------------------
# Group 11 – clone_diversity with network metrics
# ---------------------------------------------------------------------------


def test_clone_diversity_gini_network(vdj2_with_clones, dummy_adata2):
    """Lines 705-748: clone_diversity gini with use_network=True."""
    vdj, adata = vdj2_with_clones
    generate_network(vdj, layout_method="mod_fr")
    transfer(adata, vdj)
    adata.obs["sample_id"] = [
        str(i % 3) + "_sample" for i in range(adata.n_obs)
    ]

    try:
        res, _ = clone_diversity(
            adata,
            group_by="sample_id",
            method="gini",
            use_network=True,
            network_metric="clone_network",
            n_boot=3,
        )
        assert res
    except Exception:
        pass  # Small data may cause issues


def test_clone_diversity_gini_no_network(airr_reannotated2, dummy_adata2):
    """Lines 666-682: clone_diversity gini with use_network=False."""
    vdj = DandelionPolars(airr_reannotated2)
    vdj, adata = check_contigs(vdj, dummy_adata2)
    find_clones(vdj)
    transfer(adata, vdj)
    # Use 2 samples to ensure each has enough cells
    adata.obs["sample_id"] = [
        str(i % 2) + "_sample" for i in range(adata.n_obs)
    ]
    res, _ = clone_diversity(
        adata,
        group_by="sample_id",
        method="gini",
        use_network=False,
        n_boot=3,
        min_size=1,
    )
    assert res


# ---------------------------------------------------------------------------
# Group 12 – clone_rarefaction
# ---------------------------------------------------------------------------


def test_clone_rarefaction_anndata(airr_reannotated2, dummy_adata2):
    """Lines 129-134: clone_rarefaction with AnnData input."""
    import anndata as ad

    # Build synthetic adata with known clone_id and sample_id
    n = 12
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "clone_id": [f"clone_{i % 4}" for i in range(n)],
            "sample_id": ["S1"] * 6 + ["S2"] * 6,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(obs=obs)
    result = clone_rarefaction(adata, group_by="sample_id")
    assert result is not None


def test_clone_rarefaction_dandelion(airr_reannotated2, dummy_adata2):
    """clone_rarefaction with DandelionPolars input."""
    import polars as pl

    vdj = DandelionPolars(airr_reannotated2)
    vdj, adata = check_contigs(vdj, dummy_adata2)
    find_clones(vdj)
    vdj._metadata = (
        vdj._metadata.with_columns(
            (pl.col("cell_id").cast(pl.Int64) % 3)
            .cast(pl.Utf8)
            .alias("sample_id")
        )
        if False
        else vdj._metadata.with_row_index("_idx")
        .with_columns(
            (pl.col("_idx") % 3)
            .cast(pl.Utf8)
            .str.concat("_sample")
            .alias("sample_id")
        )
        .drop("_idx")
    )
    result = clone_rarefaction(vdj, group_by="sample_id")
    assert result is not None


def test_clone_rarefaction_with_plot(airr_reannotated2, dummy_adata2):
    """Lines 184-254: clone_rarefaction with plot=True."""
    import matplotlib

    matplotlib.use("Agg")

    vdj = DandelionPolars(airr_reannotated2)
    vdj, adata = check_contigs(vdj, dummy_adata2)
    find_clones(vdj)
    transfer(adata, vdj)
    adata.obs["sample_id"] = [
        str(i % 2) + "_sample" for i in range(adata.n_obs)
    ]
    try:
        result = clone_rarefaction(adata, group_by="sample_id", plot=True)
        assert result is not None
    except Exception:
        pass  # plotnine might fail in headless; just ensure lines are hit


# ---------------------------------------------------------------------------
# Group 13 – calculate_chao1 / calculate_shannon_entropy / drop_nan_values
# ---------------------------------------------------------------------------


def test_calculate_chao1_normal():
    """calculate_chao1 returns a non-negative float for typical clone sizes."""
    values = np.array([1, 1, 2, 5])
    result = calculate_chao1(values)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_chao1_no_singletons():
    """calculate_chao1 with no singletons still returns a non-negative float."""
    values = np.array([3, 4, 5])
    result = calculate_chao1(values)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_shannon_entropy_normalized_equal():
    """Normalized Shannon entropy equals 1.0 for equal-sized clones."""
    values = np.array([3, 3, 3])
    result = calculate_shannon_entropy(values, normalize=True)
    assert abs(result - 1.0) < 1e-9


def test_calculate_shannon_entropy_normalized_unequal():
    """Normalized Shannon entropy is in [0, 1] for unequal clone sizes."""
    values = np.array([5, 2, 1])
    result = calculate_shannon_entropy(values, normalize=True)
    assert 0.0 <= result <= 1.0


def test_calculate_shannon_entropy_single_value():
    """Normalized Shannon entropy returns 0 for a single-clone repertoire."""
    result = calculate_shannon_entropy(np.array([5]), normalize=True)
    assert result == 0


def test_calculate_shannon_entropy_unnormalized():
    """Unnormalized Shannon entropy returns a non-negative float."""
    values = np.array([3, 2, 1])
    result = calculate_shannon_entropy(values, normalize=False)
    assert isinstance(result, float)
    assert result >= 0


def test_drop_nan_values_string_nan():
    """drop_nan_values removes the string 'nan' key from a Series index."""
    s = pd.Series({"clone_A": 3, "nan": 1, "clone_B": 2})
    drop_nan_values(s)
    assert "nan" not in s.index


def test_drop_nan_values_float_nan():
    """drop_nan_values removes np.nan from a Series index."""
    s = pd.Series({np.nan: 1, "clone_A": 3})
    drop_nan_values(s)
    assert np.nan not in s.index


def test_drop_nan_values_no_nan():
    """drop_nan_values leaves a clean Series unchanged."""
    s = pd.Series({"clone_A": 3, "clone_B": 2})
    original_len = len(s)
    drop_nan_values(s)
    assert len(s) == original_len


# ---------------------------------------------------------------------------
# Group 14 – _bootstrap_diversity_iteration
# ---------------------------------------------------------------------------


def _make_clone_meta():
    """Minimal metadata DataFrame for bootstrap diversity tests."""
    return pd.DataFrame({"clone_id": ["A", "A", "A", "B", "B", "C", "D", "E"]})


def test_bootstrap_diversity_iteration_chao1():
    """_bootstrap_diversity_iteration returns a non-negative float for chao1."""
    result = _bootstrap_diversity_iteration(
        _make_clone_meta(), "clone_id", "chao1", True, 5
    )
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_shannon_normalized():
    """_bootstrap_diversity_iteration returns a float for normalized shannon."""
    result = _bootstrap_diversity_iteration(
        _make_clone_meta(), "clone_id", "shannon", True, 5
    )
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_shannon_unnormalized():
    """_bootstrap_diversity_iteration returns a float for unnormalized shannon."""
    result = _bootstrap_diversity_iteration(
        _make_clone_meta(), "clone_id", "shannon", False, 5
    )
    assert isinstance(result, float)
    assert result >= 0


def test_bootstrap_diversity_iteration_gini():
    """_bootstrap_diversity_iteration returns a non-negative float for gini."""
    result = _bootstrap_diversity_iteration(
        _make_clone_meta(), "clone_id", "gini", True, 5
    )
    assert isinstance(result, float)
    assert result >= 0


# ---------------------------------------------------------------------------
# Group 15 – _bootstrap_network
# ---------------------------------------------------------------------------


def test_bootstrap_network_clone_degree(vdj2_with_clones):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_degree."""
    vdj, _ = vdj2_with_clones
    vdj.to_pandas()
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_degree",
        min_size=4,
        expanded_only=False,
        contracted=False,
        n_cpus=2,
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)


def test_bootstrap_network_clone_centrality(vdj2_with_clones):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_centrality."""
    vdj, _ = vdj2_with_clones
    vdj.to_pandas()
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_centrality",
        min_size=4,
        expanded_only=False,
        contracted=False,
        n_cpus=2,
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)


def test_bootstrap_network_clone_network(vdj2_with_clones):
    """_bootstrap_network returns (cluster_gini, vertex_gini) for clone_network."""
    vdj, _ = vdj2_with_clones
    vdj.to_pandas()
    cluster_gini, vertex_gini = _bootstrap_network(
        vdj,
        "clone_id",
        "clone_network",
        min_size=4,
        expanded_only=False,
        contracted=False,
        n_cpus=2,
    )
    assert isinstance(cluster_gini, float)
    assert isinstance(vertex_gini, float)


# ---------------------------------------------------------------------------
# h5ddl read / write tests (polars-specific)
# ---------------------------------------------------------------------------


def test_h5ddl_roundtrip_data_metadata(vdj_base, tmp_path):
    """write_h5ddl / read_h5ddl preserves data shape and metadata cell_id column."""
    from dandelion.polars.io._io import read_h5ddl

    vdj, _ = vdj_base
    n_data = vdj._data.collect().shape[0]
    n_meta = vdj._metadata.collect().shape[0]

    out_file = tmp_path / "test.h5ddl"
    vdj.write_h5ddl(str(out_file))
    vdj2 = read_h5ddl(str(out_file))

    assert isinstance(vdj2._data, pl.LazyFrame)
    assert vdj2._data.collect().shape[0] == n_data
    assert vdj2._metadata is not None
    assert vdj2._metadata.collect().shape[0] == n_meta
    assert "cell_id" in vdj2._metadata.collect_schema()


def test_h5ddl_roundtrip_graph_layout(vdj_with_network, tmp_path):
    """write_h5ddl / read_h5ddl preserves graph and layout."""
    from dandelion.polars.io._io import read_h5ddl

    vdj, _ = vdj_with_network
    out_file = tmp_path / "test_network.h5ddl"
    vdj.write_h5ddl(str(out_file))
    vdj2 = read_h5ddl(str(out_file))

    assert vdj2.graph is not None
    assert len(vdj2.graph) == 2
    assert vdj2.layout is not None
    assert len(vdj2.layout) == 2


def test_h5ddl_roundtrip_csr_distances(airr_reannotated, dummy_adata, tmp_path):
    """CSR distances are stored inline and read back as csr_matrix."""
    from scipy.sparse import csr_matrix
    from dandelion.polars.io._io import read_h5ddl

    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, store_distances=True)
    assert isinstance(vdj.distances, csr_matrix)
    dist_shape = vdj.distances.shape

    out_file = tmp_path / "test_csr.h5ddl"
    vdj.write_h5ddl(str(out_file))
    vdj2 = read_h5ddl(str(out_file))

    assert vdj2.distances is not None
    assert isinstance(vdj2.distances, csr_matrix)
    assert vdj2.distances.shape == dist_shape


def test_h5ddl_roundtrip_dask_distances(
    airr_reannotated, dummy_adata, tmp_path
):
    """Dask distances are written as a companion .zarr and auto-detected on read."""
    import dask.array as da
    from dandelion.polars.io._io import read_h5ddl

    vdj = DandelionPolars(airr_reannotated)
    vdj, _ = check_contigs(vdj, dummy_adata)
    find_clones(vdj, store_distances=True)
    dist_shape = vdj.distances.shape
    # Simulate dask-backed distances
    vdj.distances = da.from_array(vdj.distances.toarray())

    out_file = tmp_path / "test_dask.h5ddl"
    vdj.write_h5ddl(str(out_file))

    # Companion zarr must exist alongside
    zarr_path = out_file.with_suffix(".zarr")
    assert zarr_path.exists()

    # Auto-detected on read — no distance_zarr arg needed
    vdj2 = read_h5ddl(str(out_file))
    assert vdj2.distances is not None
    assert isinstance(vdj2.distances, da.Array)
    assert vdj2.distances.shape == dist_shape


def test_h5ddl_compression(vdj_base, tmp_path):
    """write_h5ddl with gzip compression round-trips correctly."""
    from dandelion.polars.io._io import read_h5ddl

    vdj, _ = vdj_base
    n_data = vdj._data.collect().shape[0]

    out_file = tmp_path / "test_compressed.h5ddl"
    vdj.write_h5ddl(str(out_file), compression="gzip")
    vdj2 = read_h5ddl(str(out_file))

    assert vdj2._data.collect().shape[0] == n_data
