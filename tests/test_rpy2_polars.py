import pytest

from unittest.mock import patch

import polars as pl

from dandelion.external.immcantation.shazam_polars import (
    quantify_mutations,
    calculate_threshold,
)
from dandelion.external.immcantation.changeo_polars import (
    define_clones,
    create_germlines,
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
    _ = quantify_mutations(str(f))
    # try:
    #     _ = quantify_mutations(str(f))
    # except Exception:
    #     pytest.skip("R package 'shazam' not installed")
    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "mu_count" in vdj._data.collect_schema().names()
    # Check that mu_count column has data (collect to check non-empty)
    mu_count_col = vdj._data.select("mu_count").collect(engine="streaming")
    assert len(mu_count_col) > 0

    _ = quantify_mutations(str(f), frequency=True)
    # Run mutation quantification (frequency)
    # try:
    #     _ = quantify_mutations(str(f), frequency=True)
    # except Exception:
    #     pytest.skip("R package 'shazam' not installed")
    out2 = load_polars(f)
    vdj2 = DandelionPolars(out2)
    assert "mu_freq" in vdj2._data.collect_schema().names()
    # Check that mu_freq column has data
    mu_freq_col = vdj2._data.select("mu_freq").collect(engine="streaming")
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

    tr = calculate_threshold(vdj_polars)
    assert tr > 0.0
    # try:
    #     tr = calculate_threshold(vdj_polars)
    #     assert tr > 0.0
    # except Exception:
    #     pytest.skip("R package 'shazam' not installed")


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_create_germlines_polars(create_testfolder, database_paths):
    """Test create germlines with Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    vdj = create_germlines(vdj, germline=database_paths["germline"])
    # Check that germline_alignment_d_mask column exists and has data
    assert "germline_alignment_d_mask" in vdj._data.collect_schema().names()
    germline_col = vdj._data.select("germline_alignment_d_mask").collect(
        engine="streaming"
    )
    assert len(germline_col) > 0


@pytest.mark.usefixtures("create_testfolder")
def test_manual_threshold_and_define_clones_polars(create_testfolder):
    """Test manual threshold and define clones with Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    define_clones(vdj, dist=0.1)
    # Check that clone_id column exists and has data
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder")
def test_define_clones_outdir_polars(create_testfolder):
    """Test define clones with output directory using Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    out_path = (
        create_testfolder / "test" / "test"
    )  # this path shouldn't exist initially
    define_clones(vdj, dist=0.1, out_dir=out_path)
    assert len(list(out_path.iterdir())) == 3
    assert len(list((out_path / "tmp").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_scoper_i_polars(create_testfolder):
    """Test identical clones from scoper with Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "clone_id" not in vdj._data.collect_schema().names()
    from dandelion.external.immcantation.scoper_polars import identical_clones

    identical_clones(vdj)
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder")
def test_scoper_h_polars(create_testfolder):
    """Test hierarchical clones from scoper with Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "clone_id" not in vdj._data.collect_schema().names()
    from dandelion.external.immcantation.scoper_polars import (
        hierarchical_clones,
    )

    hierarchical_clones(vdj, threshold=0.15)
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder")
def test_scoper_spectral_polars(create_testfolder):
    """Test spectral clones from scoper with Polars."""
    f = create_testfolder / "test.tsv"
    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "clone_id" not in vdj._data.collect_schema().names()
    from dandelion.external.immcantation.scoper_polars import spectral_clones

    spectral_clones(vdj, method="novj")
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0

    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "clone_id" not in vdj._data.collect_schema().names()
    spectral_clones(vdj, method="novj", threshold=0.15)
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0

    out = load_polars(f)
    vdj = DandelionPolars(out)
    assert "clone_id" not in vdj._data.collect_schema().names()
    spectral_clones(vdj, method="vj", threshold=0.15)
    assert "clone_id" in vdj._data.collect_schema().names()
    if isinstance(vdj._data, pl.LazyFrame):
        clone_col = vdj._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = vdj._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_define_clones_polars_dataframe(create_testfolder, airr_reannotated):
    """Test define_clones with Polars DataFrame input (not wrapped in DandelionPolars)."""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)

    # Load the data using load_polars
    out = load_polars(f)

    # Convert to Polars DataFrame
    if isinstance(out, pl.LazyFrame):
        df_polars = out.collect(engine="streaming")
    else:
        df_polars = out

    # Call define_clones with DataFrame directly
    result = define_clones(df_polars, dist=0.1)

    # Result should be a DandelionPolars object
    assert isinstance(result, DandelionPolars)
    assert "clone_id" in result._data.collect_schema().names()
    if isinstance(result._data, pl.LazyFrame):
        clone_col = result._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = result._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_define_clones_pandas_dataframe(create_testfolder, airr_reannotated):
    """Test define_clones with Pandas DataFrame input."""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)

    # Load the data using load_polars
    out = load_polars(f)

    # Convert to Pandas DataFrame
    if isinstance(out, pl.LazyFrame):
        df_pandas = out.collect(engine="streaming").to_pandas()
    else:
        df_pandas = out.to_pandas()

    # Call define_clones with pandas DataFrame
    result = define_clones(df_pandas, dist=0.1)

    # Result should be a DandelionPolars object
    assert isinstance(result, DandelionPolars)
    assert "clone_id" in result._data.collect_schema().names()
    if isinstance(result._data, pl.LazyFrame):
        clone_col = result._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = result._data.select("clone_id")
    assert len(clone_col) > 0


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_define_clones_file_path(create_testfolder, airr_reannotated):
    """Test define_clones with file path input."""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)

    # Call define_clones with file path string
    result = define_clones(str(f), dist=0.1)

    # Result should be a DandelionPolars object
    assert isinstance(result, DandelionPolars)
    assert "clone_id" in result._data.collect_schema().names()
    if isinstance(result._data, pl.LazyFrame):
        clone_col = result._data.select("clone_id").collect(engine="streaming")
    else:
        clone_col = result._data.select("clone_id")
    assert len(clone_col) > 0
