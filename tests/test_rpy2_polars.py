import pytest

from unittest.mock import patch

from dandelion.external.immcantation.shazam_polars import (
    quantify_mutations,
    calculate_threshold,
)
from dandelion.utilities._polars import (
    read_10x_vdj_polars,
    DandelionPolars,
    load_polars,
)


def require_r_package(pkg_name):
    """Skip test if R package is missing."""
    try:
        from rpy2.robjects.packages import importr

        return importr(pkg_name)
    except Exception:
        pytest.skip(f"R package '{pkg_name}' not installed")


def test_importrpy2_polars():
    """Ensure rpy2 import works and shazam is available."""
    sh = require_r_package("shazam")
    assert sh.__module__ == "rpy2.robjects.packages"


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_mutation_polars(create_testfolder, airr_reannotated):
    """Quantify mutations via Polars-based shazam wrapper and persist to file."""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)
    # Run mutation quantification (counts)
    # try:
    _ = quantify_mutations(str(f))
    # except Exception:
    # pytest.skip("R package 'shazam' not installed")
    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "mu_count" in vdj._data.collect_schema().names()
    # Check that mu_count column has data (collect to check non-empty)
    mu_count_col = vdj._data.select("mu_count").collect()
    assert len(mu_count_col) > 0

    # Run mutation quantification (frequency)
    # try:
    _ = quantify_mutations(str(f), frequency=True)
    # except Exception:
    # pytest.skip("R package 'shazam' not installed")
    out2 = load_polars(f)
    vdj2 = DandelionPolars(out2)
    assert "mu_freq" in vdj2._data.collect_schema().names()
    # Check that mu_freq column has data
    mu_freq_col = vdj2._data.select("mu_freq").collect()
    assert len(mu_freq_col) > 0


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
def test_calculate_threshold_polars(
    mock_show, create_testfolder, annotation_10x_mouse
):
    """Calculate threshold using Polars-based shazam wrapper on 10x annotations."""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)

    # Build DandelionPolars object from 10x annotations
    vdj_polars = read_10x_vdj_polars(out_file)

    # try:
    tr = calculate_threshold(vdj_polars)
    assert tr > 0.0
    # except Exception:
    # pytest.skip("R package 'shazam' not installed")
