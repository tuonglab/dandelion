#!/usr/bin/env python
import dandelion as ddl
import pandas as pd
import pytest

from dandelion.base.io import read_10x_vdj
from dandelion.base.preprocessing import check_contigs
from dandelion.base.tools import find_clones, transfer, clone_overlap
from dandelion.base.plotting import clone_overlap as clone_overlap_plot


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x", "dummy_adata_mouse"
)
def test_clone_overlap(
    create_testfolder, annotation_10x_mouse, dummy_adata_mouse
):
    """test_clone_overlap"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(annot_file, index=False)
    vdj = read_10x_vdj(create_testfolder, filename_prefix="test_filtered")
    check_contigs(vdj)
    find_clones(vdj)
    assert vdj._data.shape[0] == 1987
    assert vdj._metadata.shape[0] == 545
    transfer(dummy_adata_mouse, vdj)
    assert dummy_adata_mouse.n_obs == 547
    # create a sample column
    label = []
    for x in range(0, dummy_adata_mouse.n_obs):
        if x < 100:
            label.append("A")
        elif x < 200:
            label.append("B")
        elif x < 300:
            label.append("C")
        elif x < 400:
            label.append("D")
        elif x < 500:
            label.append("E")
        else:
            label.append("F")
    dummy_adata_mouse.obs["sample_idx"] = label
    with pytest.raises(KeyError):
        clone_overlap_plot(
            dummy_adata_mouse,
            groupby="sample_idx",
        )
    clone_overlap(dummy_adata_mouse, groupby="sample_idx")
    assert "clone_overlap" in dummy_adata_mouse.uns
    clone_overlap_plot(
        dummy_adata_mouse,
        groupby="sample_idx",
    )
    with pytest.raises(ValueError):
        clone_overlap_plot(
            vdj,
            groupby="sample_idx",
        )
    G = clone_overlap_plot(
        dummy_adata_mouse,
        groupby="sample_idx",
        weighted_overlap=False,
        save=create_testfolder / "test.png",
        return_graph=True,
    )
    assert G is not None

    G = clone_overlap_plot(
        dummy_adata_mouse,
        groupby="sample_idx",
        weighted_overlap=True,
        scale_edge_lambda=lambda x: x * 10,
        return_graph=True,
    )
    assert G is not None

    clone_overlap_plot(
        dummy_adata_mouse,
        groupby="sample_idx",
        as_heatmap=True,
    )

    out = clone_overlap_plot(
        dummy_adata_mouse,
        groupby="sample_idx",
        as_heatmap=True,
        return_heatmap_data=True,
    )
    assert isinstance(out, pd.DataFrame)
