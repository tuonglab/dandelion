#!/usr/bin/env python
"""
Backend-agnostic tools and core tests.

Imports from the top-level ``dandelion`` namespace so that the active backend
(controlled by the ``DANDELION_BACKEND`` env-var) determines which code path
is exercised.  CI runs this file once per backend to cover both
``dandelion.base`` and ``dandelion.polars``.
"""

import pytest
import scanpy as sc

import dandelion as ddl

# -- core / preprocessing ----------------------------------------------------


@pytest.mark.usefixtures(
    "create_testfolder", "airr_reannotated", "airr_reannotated2", "dummy_adata"
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """Create Dandelion objects via check_contigs and persist for later tests."""
    vdj, adata = ddl.pp.check_contigs(airr_reannotated, dummy_adata)
    vdj2 = ddl.pp.check_contigs(airr_reannotated2)
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj.write(f)
    vdj2.write(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj2 = ddl.read(f2)
    ddl.tl.find_clones(vdj)
    ddl.tl.find_clones(vdj2)
    vdj.write(f)
    vdj2.write(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.find_clones(vdj, key_added="test_clone")


@pytest.mark.usefixtures("create_testfolder")
def test_clone_size(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_size(vdj)
    ddl.tl.clone_size(vdj, max_size=3)


@pytest.mark.usefixtures("create_testfolder")
def test_generate_network(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    f2 = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj2 = ddl.read(f2)
    ddl.tl.generate_network(vdj2, layout_method="mod_fr")
    vdj2.write(f2)
    adata = ddl.tl.to_scirpy(vdj, to_mudata=False)
    ddl.tl.generate_network(vdj, adata=adata, layout_method="mod_fr")
    ddl.tl.transfer(adata, vdj)
    vdj.write(f)
    f3 = create_testfolder / "test_backend.h5ad"
    adata.write_h5ad(f3)


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    f = create_testfolder / "test_backend2.ddl"
    vdj = ddl.read(f)
    vdj, adata = ddl.pp.check_contigs(vdj, dummy_adata2)
    ddl.tl.transfer(dummy_adata2, vdj)
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.transfer(dummy_adata2, vdj)


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.extract_edge_weights(vdj)
    ddl.tl.extract_edge_weights(vdj, expanded_only=True)


# -- diversity ----------------------------------------------------------------


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("method", ["chao1", "shannon", "gini"])
def test_diversity_anndata(create_testfolder, method):
    f = create_testfolder / "test_backend.h5ad"
    adata = sc.read_h5ad(f)
    ddl.tl.clone_diversity(adata, group_by="sample_id", method=method, n_boot=5)


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_ddl(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_diversity(vdj, group_by="sample_id", key="sequence", n_boot=5)


@pytest.mark.usefixtures("create_testfolder")
def test_clone_rarefaction(create_testfolder):
    f = create_testfolder / "test_backend.ddl"
    vdj = ddl.read(f)
    ddl.tl.clone_rarefaction(vdj, group_by="sample_id")


# -- scirpy / concat ----------------------------------------------------------


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_from_scirpy(create_testfolder, annotation_10x, fasta_10x):
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    adata = ddl.tl.to_scirpy(vdj)
    ddl.tl.from_scirpy(adata)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x")
def test_concat(create_testfolder, annotation_10x):
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    ddl.tl.concat([vdj, vdj], prefixes=["x", "y"])


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_missing_meta_cols(airr_reannotated):
    """missing_meta_cols > 0: extra metadata column from one source is preserved in result."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj1 = BaseDandelion(airr_reannotated)
    vdj2 = BaseDandelion(airr_reannotated)
    # Add an extra column only in vdj1 that vdj2 does not have
    vdj1._metadata["extra_col"] = "batch_A"
    result = base_concat([vdj1, vdj2], prefixes=["x", "y"])
    assert "extra_col" in result._metadata.columns


# -- concat comprehensive (base) ---------------------------------------------


@pytest.mark.usefixtures("airr_reannotated", "airr_reannotated2")
def test_concat_dict_base(airr_reannotated, airr_reannotated2):
    """dict input is converted to a list and concatenated."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj1 = BaseDandelion(airr_reannotated)
    vdj2 = BaseDandelion(airr_reannotated2)
    result = base_concat({"s1": vdj1, "s2": vdj2})
    assert result is not None
    assert result.n_contigs == vdj1.n_contigs + vdj2.n_contigs


@pytest.mark.usefixtures("airr_reannotated", "airr_reannotated2")
def test_concat_suffixes_base(airr_reannotated, airr_reannotated2):
    """explicit suffixes are appended to cell/contig IDs."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj1 = BaseDandelion(airr_reannotated)
    vdj2 = BaseDandelion(airr_reannotated2)
    result = base_concat([vdj1, vdj2], suffixes=["_A", "_B"])
    assert result is not None
    assert result.n_contigs == vdj1.n_contigs + vdj2.n_contigs


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_both_suffix_and_prefix_raises_base(airr_reannotated):
    """ValueError when both suffixes and prefixes are supplied."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj = BaseDandelion(airr_reannotated)
    with pytest.raises(ValueError):
        base_concat([vdj, vdj], suffixes=["_A", "_B"], prefixes=["A_", "B_"])


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_suffix_length_mismatch_raises_base(airr_reannotated):
    """ValueError when suffix list length does not match input count."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj = BaseDandelion(airr_reannotated)
    with pytest.raises(ValueError):
        base_concat([vdj, vdj], suffixes=["_only_one"])


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_check_unique_false_raises_base(airr_reannotated):
    """check_unique=False raises ValueError when indices are not unique."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj = BaseDandelion(airr_reannotated)
    with pytest.raises(ValueError):
        base_concat([vdj, vdj], check_unique=False)


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_remove_trailing_hyphen_base(airr_reannotated):
    """remove_trailing_hyphen_number strips trailing -N before adding prefix."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    # Create two independent objects to avoid in-place mutation of a shared ref
    vdj_a = BaseDandelion(airr_reannotated)
    vdj_b = BaseDandelion(airr_reannotated)
    # collapse_cells=False keeps duplicate metadata entries so that
    # metadata_index_order is None, which triggers the add_cell_prefix branch.
    result = base_concat(
        [vdj_a, vdj_b],
        prefixes=["A_", "B_"],
        remove_trailing_hyphen_number=True,
        collapse_cells=False,
    )
    assert result is not None
    assert result.n_contigs == vdj_a.n_contigs + vdj_b.n_contigs
    # Both prefixes must appear in the result cell IDs
    cell_ids = list(result._metadata.index)
    assert any(c.startswith("A_") for c in cell_ids)
    assert any(c.startswith("B_") for c in cell_ids)


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_invalid_type_raises_base(airr_reannotated):
    """ValueError when a non-Dandelion/non-DataFrame object is in the list."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj = BaseDandelion(airr_reannotated)
    with pytest.raises(ValueError):
        base_concat([vdj, 42])


@pytest.mark.usefixtures("airr_reannotated", "airr_reannotated2")
def test_concat_dataframe_input_base(airr_reannotated, airr_reannotated2):
    """pandas DataFrames are accepted alongside Dandelion objects."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj1 = BaseDandelion(airr_reannotated)
    # Pass raw pandas DataFrame as the second element
    result = base_concat([vdj1, airr_reannotated2])
    assert result is not None


@pytest.mark.usefixtures("airr_reannotated", "airr_reannotated2")
def test_concat_v_call_genotyped_partial_base(
    airr_reannotated, airr_reannotated2
):
    """v_call_genotyped present in only one object is filled from v_call in the other."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj1 = BaseDandelion(airr_reannotated)
    vdj2 = BaseDandelion(airr_reannotated2)
    vdj1._data["v_call_genotyped"] = vdj1._data["v_call"]
    result = base_concat([vdj1, vdj2])
    assert "v_call_genotyped" in result._data.columns


@pytest.mark.usefixtures("airr_reannotated")
def test_concat_auto_numbering_base(airr_reannotated):
    """Duplicate indices without explicit suffixes get auto-numbered (0, 1, ...)."""
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import concat as base_concat

    vdj = BaseDandelion(airr_reannotated)
    result = base_concat([vdj, vdj])
    assert result is not None
    assert result.n_contigs == vdj.n_contigs * 2


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_from_dataframe(airr_reannotated):
    vdj = ddl.Dandelion(airr_reannotated)
    ddl.tl.find_clones(vdj)


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_by_alleles(airr_reannotated):
    vdj = ddl.Dandelion(airr_reannotated)
    ddl.tl.find_clones(vdj, by_alleles=True)


@pytest.mark.usefixtures("airr_reannotated")
def test_find_clones_after_network(airr_reannotated):
    vdj = ddl.pp.check_contigs(airr_reannotated)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="junction_aa", layout_method="mod_fr")
    vdj2 = vdj.copy()
    ddl.tl.find_clones(vdj2)
    ddl.tl.find_clones(vdj2, key_added="cloned_idx")


# ---------------------------------------------------------------------------
# _reverse_transfer (base)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_metadata_only_base(airr_reannotated):
    """_reverse_transfer transfers obs columns into Dandelion._metadata."""
    import anndata as ad
    import pandas as pd
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    obs_names = list(vdj._metadata.index[:2])
    adata = ad.AnnData(
        obs=pd.DataFrame({"extra_rt_col": ["x", "y"]}, index=obs_names)
    )
    _reverse_transfer(adata, vdj)
    assert "extra_rt_col" in vdj._metadata.columns


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_builds_graph_base(airr_reannotated):
    """_reverse_transfer with clone_key in uns builds the clone graph (list cell_indices)."""
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    obs_names = list(vdj._metadata.index[:2])
    adata = ad.AnnData(obs=pd.DataFrame(index=obs_names))
    dist = csr_matrix([[0, 1], [1, 0]])
    adata.uns["clone_id"] = {
        "distances": dist,
        "cell_indices": {
            "0": np.array([obs_names[0]]),
            "1": np.array([obs_names[1]]),
        },
    }
    _reverse_transfer(adata, vdj)
    assert vdj.graph is not None
    assert vdj.graph[0] is not None


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_scalar_cell_indices_base(airr_reannotated):
    """_reverse_transfer with scalar cell_indices builds the clone graph."""
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    obs_names = list(vdj._metadata.index[:2])
    adata = ad.AnnData(obs=pd.DataFrame(index=obs_names))
    dist = csr_matrix([[0, 1], [1, 0]])
    adata.uns["clone_id"] = {
        "distances": dist,
        "cell_indices": {
            "0": obs_names[0],
            "1": obs_names[1],
        },
    }
    _reverse_transfer(adata, vdj)
    assert vdj.graph is not None


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_mudata_input_base(airr_reannotated):
    """_reverse_transfer with MuData input extracts the 'airr' modality."""
    import anndata as ad
    import pandas as pd
    import mudata
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    obs_names = list(vdj._metadata.index[:2])
    airr_adata = ad.AnnData(
        obs=pd.DataFrame({"mudata_rt_col": [1, 2]}, index=obs_names)
    )
    mdata = mudata.MuData({"airr": airr_adata})
    _reverse_transfer(mdata, vdj)
    assert "mudata_rt_col" in vdj._metadata.columns


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_mudata_no_airr_raises_base(airr_reannotated):
    """_reverse_transfer with MuData missing 'airr' modality raises ValueError."""
    import anndata as ad
    import pandas as pd
    import mudata
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    obs_names = list(vdj._metadata.index[:2])
    other_adata = ad.AnnData(obs=pd.DataFrame(index=obs_names))
    mdata = mudata.MuData({"other": other_adata})
    with pytest.raises(ValueError, match="airr"):
        _reverse_transfer(mdata, vdj)


@pytest.mark.usefixtures("airr_reannotated")
def test_reverse_transfer_no_duplicate_columns_base(airr_reannotated):
    """_reverse_transfer does not overwrite a column already in _metadata."""
    import anndata as ad
    import pandas as pd
    from dandelion.base.core._core import Dandelion as BaseDandelion
    from dandelion.base.tools._tools import _reverse_transfer

    vdj = BaseDandelion(airr_reannotated)
    initial_cols = list(vdj._metadata.columns)
    obs_names = list(vdj._metadata.index[:2])
    # Use an existing column name; it should not be added again
    existing_col = initial_cols[0]
    adata = ad.AnnData(
        obs=pd.DataFrame(
            {existing_col: ["a", "b"]},
            index=obs_names,
        )
    )
    _reverse_transfer(adata, vdj)
    assert list(vdj._metadata.columns).count(existing_col) == 1


# ---------------------------------------------------------------------------
# _create_anndata else branch (base)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("airr_reannotated", "dummy_adata")
def test_to_scirpy_with_gex_anndata(airr_reannotated, dummy_adata):
    """_create_anndata else branch via to_scirpy with gex_adata and to_mudata=False."""
    import anndata as ad

    vdj, adata = ddl.pp.check_contigs(airr_reannotated, dummy_adata)
    ddl.tl.find_clones(vdj)
    result = ddl.tl.to_scirpy(vdj, to_mudata=False, gex_adata=adata)
    assert result is not None
    assert isinstance(result, ad.AnnData)
    assert "airr" in result.obsm


def test_create_anndata_else_branch_base():
    """_create_anndata (base) else: branch merges AIRR array into existing AnnData."""
    import awkward as ak
    import pandas as pd
    import anndata as ad
    from dandelion.base.tools._tools import _create_anndata

    cell_ids = ["cell_A", "cell_B", "cell_C"]
    obs = pd.DataFrame(index=cell_ids)
    airr = ak.Array([{"locus": "IGH"}, {"locus": "IGL"}, {"locus": "IGK"}])
    existing = ad.AnnData(
        obs=pd.DataFrame({"gex_col": [1, 2, 3]}, index=cell_ids)
    )
    result = _create_anndata(airr, obs, existing)
    assert "airr" in result.obsm
    assert "gex_col" in result.obs.columns
    assert result.n_obs == 3


def test_create_anndata_else_branch_partial_overlap_base():
    """_create_anndata (base) else: branch filters to common cells only."""
    import awkward as ak
    import pandas as pd
    import anndata as ad
    from dandelion.base.tools._tools import _create_anndata

    airr_cells = ["cell_A", "cell_B"]
    all_cells = ["cell_A", "cell_B", "cell_C"]
    obs = pd.DataFrame(index=airr_cells)
    airr = ak.Array([{"locus": "IGH"}, {"locus": "IGL"}])
    existing = ad.AnnData(
        obs=pd.DataFrame({"gex_col": [1, 2, 3]}, index=all_cells)
    )
    result = _create_anndata(airr, obs, existing)
    assert result.n_obs == 2
    assert "airr" in result.obsm
    assert set(result.obs_names) == {"cell_A", "cell_B"}
