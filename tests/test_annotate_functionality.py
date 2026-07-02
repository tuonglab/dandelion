#!/usr/bin/env python
import pandas as pd
import polars as pl
import pytest

import dandelion as ddl


@pytest.fixture(autouse=True)
def _restore_backend():
    """Restore backend after each test in this module."""
    original = ddl.get_backend()
    yield
    ddl.set_backend(original)


@pytest.mark.parametrize("backend", ["base", "polars"])
@pytest.mark.parametrize("path_mode", ["vdj", "germlines_root"])
def test_annotate_functionality(backend, path_mode, database_paths):
    """Annotate v/d/j functionality from germline references."""
    ddl.set_backend(backend)

    data = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq2"],
            "cell_id": ["cell1", "cell2"],
            "v_call": ["IGHV1-18*01", "IGHV_DOES_NOT_EXIST*01"],
            "d_call": ["IGHD1-1*01", ""],
            "j_call": ["IGHJ1*01", "IGHJ1*01"],
            "locus": ["IGH", "IGH"],
            "productive": ["T", "T"],
        }
    )
    if backend == "base":
        vdj = ddl.Dandelion(data, initialize=False, verbose=False)
    else:
        # Polars backend expects standard AIRR-like columns during init.
        data_polars = data.copy()
        data_polars["c_call"] = ""
        vdj = ddl.Dandelion(data_polars, initialize=False, verbose=False)

    if path_mode == "vdj":
        germline_db = database_paths["germline"]
    else:
        germline_db = database_paths["germline"].parents[2]

    ddl.pp.annotate_functionality(vdj, germline_db=germline_db, org="human")

    dat = vdj._data
    if isinstance(dat, pl.LazyFrame):
        dat = dat.collect(engine="streaming")

    if isinstance(dat, pd.DataFrame):
        assert dat.iloc[0]["v_call_functionality"] == "F"
        assert dat.iloc[0]["d_call_functionality"] == "F"
        assert dat.iloc[0]["j_call_functionality"] == "F"
        assert pd.isna(dat.iloc[1]["v_call_functionality"])
    else:
        assert dat["v_call_functionality"][0] == "F"
        assert dat["d_call_functionality"][0] == "F"
        assert dat["j_call_functionality"][0] == "F"
        assert dat["v_call_functionality"][1] is None
