from __future__ import annotations

import gzip
import io
import json
import pytest
import zipfile

import pandas as pd
import polars as pl


from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import dandelion.polars.tools._epitope as _epitope
from dandelion.polars.tools._epitope import (
    clear_cache,
    _fetch_bytes,
    _infer_locus,
    _infer_receptor_type,
    _ensure_airr_columns,
    _normalise_na,
    _read_csv_bytes,
    _to_polars,
    _parse_iedb_receptor_zip,
    _map_iedb_columns,
    fetch_iedb,
    _parse_vdjdb_tsv,
    _get_vdjdb_url,
    fetch_vdjdb,
    fetch_all,
    fetch_db,
    get_epitope,
    _annotate_from_db,
    query,
    to_tsv,
)

# ---------------------------------------------------------------------------
# Fixtures: fake IEDB and VDJdb payloads
# ---------------------------------------------------------------------------


def _make_iedb_zip() -> bytes:
    """Build a fake receptor_full_v3.zip with two header rows + data rows.

    Row 0 of the CSV (parsed as the pandas header) is a throwaway "note" row.
    Row 1 holds the real column names IEDB actually uses. Data starts on
    row 2.

    - Includes a duplicate "cdr3_curated" column (simulating chain-2 curated
      CDR3) to exercise the header-dedup branch in `_parse_iedb_receptor_zip`
      and the chain-2 column-copy branch in `_map_iedb_columns`.
    - Includes "chain_2_type" to exercise the `_chain2_type` drop branch.
    - One row uses chain type "alphabeta" and another "gammadelta" to
      exercise the chain-splitting branch in `_map_iedb_columns`.
    """
    cols = [
        "group_iri",
        "type",
        "chain_2_type",
        "curated_v_gene",
        "curated_d_gene",
        "curated_j_gene",
        "cdr1_curated",
        "cdr2_curated",
        "cdr3_curated",
        "cdr3_curated",  # duplicate on purpose
        "protein_sequence",
        "nucleotide_sequence",
        "name",
        "source_molecule",
        "source_organism",
        "mhc_class",
        "mhc_allele_names",
        "host_organism_name",
        "iedb_ids",
    ]
    note_row = ["note"] * len(cols)
    rows = [
        # id, type, chain2type, v, d, j, cdr1, cdr2, cdr3(chain1),
        # cdr3(chain2, dup col), prot, nt, epitope, ag_prot, ag_organism,
        # mhc_class, mhc_allele, host, pubmed
        [
            "1",
            "alphabeta",
            "beta",
            "TRAV1",
            "",
            "TRAJ1",
            "CAS",
            "CAT",
            "CASSX",
            "CASSCHAIN2",
            "MEEP",
            "ATGC",
            "EP1",
            "Prot1",
            "Epstein-Barr virus",
            "I",
            "A*02:01",
            "human",
            "111",
        ],
        [
            "2",
            "gammadelta",
            "delta",
            "TRGV1",
            "",
            "TRGJ1",
            "CGS",
            "CGT",
            "CGSSX",
            "CGSSCHAIN2",
            "MEEG",
            "ATGG",
            "EP2",
            "Prot2",
            "Influenza A",
            "II",
            "DRB1*01:01",
            "human",
            "222",
        ],
        [
            "3",
            "beta",
            "",
            "TRBV1",
            "TRBD1",
            "TRBJ1",
            "CBS",
            "CBT",
            "CBSSX",
            "CBSSX",
            "MEEB",
            "ATGB",
            "EP3",
            "Prot3",
            "SARS-CoV-2",
            "I",
            "A*01:01",
            "human",
            "333",
        ],
    ]

    buf = io.StringIO()
    for r in [note_row, cols] + rows:
        buf.write(",".join(r) + "\n")
    csv_bytes = buf.getvalue().encode("utf-8")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("receptor_full_v3.csv", csv_bytes)
    return zbuf.getvalue()


def _make_iedb_zip_no_csv() -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("readme.txt", "no csv here")
    return zbuf.getvalue()


def _vdjdb_tsv_bytes() -> bytes:
    header = [
        "cdr3",
        "gene",
        "v.segm",
        "d.segm",
        "j.segm",
        "antigen.epitope",
        "antigen.gene",
        "antigen.species",
        "mhc.a",
        "mhc.class",
        "species",
        "reference.id",
        "web.method",
        "vdjdb.score",
    ]
    rows = [
        [
            "CASSX",
            "TRB",
            "TRBV1",
            "",
            "TRBJ1",
            "EP1",
            "Prot1",
            "EBV",
            "A*02:01",
            "I",
            "human",
            "111",
            "manual",
            "3",
        ],
        [
            "CASSY",
            "TRA",
            "TRAV1",
            "",
            "TRAJ1",
            "EP2",
            "Prot2",
            "CMV",
            "A*01:01",
            "I",
            "human",
            "222",
            "manual",
            "0",
        ],
        [
            "CASSZ",
            "TRB",
            "TRBV2",
            "",
            "TRBJ2",
            "EP3",
            "Prot3",
            "SARS-CoV-2",
            "A*03:01",
            "I",
            "human",
            "333",
            "manual",
            "2",
        ],
    ]
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    return ("\n".join(lines)).encode("utf-8")


def _make_vdjdb_zip() -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("vdjdb.txt", _vdjdb_tsv_bytes())
    return zbuf.getvalue()


# ---------------------------------------------------------------------------
# Reset module-level cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# clear_cache / _fetch_bytes
# ---------------------------------------------------------------------------


def test_fetch_bytes_cache_hit_and_miss(monkeypatch):
    calls = {"n": 0}

    class FakeResp:
        def __enter__(self):
            calls["n"] += 1
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"hello"

    monkeypatch.setattr(_epitope, "urlopen", lambda req, timeout: FakeResp())

    data = _fetch_bytes("http://example.com/a", timeout=5)
    assert data == b"hello"
    assert calls["n"] == 1

    # second call is a cache hit -> no new "download"
    data2 = _fetch_bytes("http://example.com/a", timeout=5)
    assert data2 == b"hello"
    assert calls["n"] == 1


def test_fetch_bytes_http_error(monkeypatch):
    def raise_http(req, timeout):
        raise HTTPError("http://x", 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(_epitope, "urlopen", raise_http)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        _fetch_bytes("http://example.com/missing")


def test_fetch_bytes_url_error(monkeypatch):
    def raise_url(req, timeout):
        raise URLError("boom")

    monkeypatch.setattr(_epitope, "urlopen", raise_url)
    with pytest.raises(RuntimeError, match="Network error"):
        _fetch_bytes("http://example.com/down")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def test_infer_locus_and_receptor_type():
    assert _infer_locus("Alpha") == "TRA"
    assert _infer_locus("heavy") == "IGH"
    assert _infer_locus(None) == ""
    assert _infer_locus("unknown") == ""
    assert _infer_receptor_type("IGH") == "BCR"
    assert _infer_receptor_type("TRA") == "TCR"
    assert _infer_receptor_type("") == ""


def test_ensure_airr_columns_and_normalise_na():
    df = pd.DataFrame({"cdr3_aa": ["N/A", "nan", "CAS"]})
    df = _ensure_airr_columns(df)
    assert (
        list(df.columns)[: len(_epitope.AIRR_CORE_FIELDS)]
        == _epitope.AIRR_CORE_FIELDS
    )
    df = _normalise_na(df)
    assert df["cdr3_aa"].tolist() == ["", "", "CAS"]


def test_read_csv_bytes():
    raw = b"a,b\n1,2\n"
    df = _read_csv_bytes(raw)
    assert df.shape == (1, 2)


def test_to_polars_roundtrip():
    df = pd.DataFrame({"a": [1, 2]})
    pldf = _to_polars(df)
    assert isinstance(pldf, pl.DataFrame)


# ---------------------------------------------------------------------------
# IEDB parsing / mapping / fetch
# ---------------------------------------------------------------------------


def test_parse_iedb_receptor_zip_no_csv_raises():
    with pytest.raises(ValueError, match="No CSV file"):
        _parse_iedb_receptor_zip(_make_iedb_zip_no_csv())


def test_map_iedb_columns_cdr3_fallback_and_chain2_drop():
    # cdr3_curated is entirely missing -> falls back to a *_cdr3_calculated_*
    # column; chain_2_type is present and must be dropped after use.
    df = pd.DataFrame(
        {
            "cdr3_curated": [None, None],
            "cdr3_calculated_x": ["FALLBACK1", "FALLBACK2"],
            "curated_v_gene": ["V1", "V2"],
            "type": ["beta", "beta"],
            "chain_2_type": ["", ""],
        }
    )
    mapped = _map_iedb_columns(df)
    assert mapped["cdr3_aa"].tolist() == ["FALLBACK1", "FALLBACK2"]
    assert "_chain2_type" not in mapped.columns


def test_fetch_iedb_full_pipeline(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_fetch_bytes", lambda url, timeout=120: _make_iedb_zip()
    )

    df = fetch_iedb()
    assert {"BCR", "TCR"}.issuperset(
        set(df["receptor_type"].unique()) - {""}
    )
    assert (df["source_db"] == "IEDB").all()
    # alphabeta row should have produced an alpha + beta pair, with the beta
    # row's cdr3_aa pulled from the duplicate "cdr3_curated" (chain-2) column
    assert "CASSCHAIN2" in df["cdr3_aa"].tolist()

    # organism filter
    df2 = fetch_iedb(organism_filter="Epstein-Barr")
    assert len(df2) >= 1
    assert df2["antigen_organism"].str.contains("Epstein-Barr").all()

    # receptor_type filter
    df3 = fetch_iedb(receptor_type="TCR")
    assert (df3["receptor_type"] == "TCR").all()

    # use_polars=True path
    pldf = fetch_iedb(use_polars=True)
    assert isinstance(pldf, pl.DataFrame)


# ---------------------------------------------------------------------------
# VDJdb parsing / mapping / fetch
# ---------------------------------------------------------------------------


def test_parse_vdjdb_tsv_all_branches():
    # zip branch
    df_zip = _parse_vdjdb_tsv(_make_vdjdb_zip())
    assert len(df_zip) == 3

    # gzip branch
    gz = gzip.compress(_vdjdb_tsv_bytes())
    df_gz = _parse_vdjdb_tsv(gz)
    assert len(df_gz) == 3

    # plain-tsv fallback branch
    df_plain = _parse_vdjdb_tsv(_vdjdb_tsv_bytes())
    assert len(df_plain) == 3


def test_get_vdjdb_url_success(monkeypatch):
    payload = {
        "assets": [
            {"name": "readme.md", "browser_download_url": "http://x/readme.md"},
            {
                "name": "vdjdb.tsv.gz",
                "browser_download_url": "http://x/vdjdb.tsv.gz",
            },
        ]
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(_epitope, "urlopen", lambda req, timeout: FakeResp())
    url = _get_vdjdb_url()
    assert url == "http://x/vdjdb.tsv.gz"


def test_get_vdjdb_url_no_suitable_asset(monkeypatch):
    payload = {
        "assets": [
            {"name": "readme.md", "browser_download_url": "http://x/readme.md"}
        ]
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(_epitope, "urlopen", lambda req, timeout: FakeResp())
    assert _get_vdjdb_url() == _epitope.VDJDB_URL


def test_get_vdjdb_url_api_failure(monkeypatch):
    def raise_err(req, timeout):
        raise URLError("down")

    monkeypatch.setattr(_epitope, "urlopen", raise_err)
    assert _get_vdjdb_url() == _epitope.VDJDB_URL


def test_fetch_vdjdb_full_pipeline(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )
    monkeypatch.setattr(
        _epitope, "_fetch_bytes", lambda url, timeout=120: _make_vdjdb_zip()
    )

    df = fetch_vdjdb()
    assert len(df) == 3

    # score filter
    df_scored = fetch_vdjdb(min_vdjdb_score=2)
    assert len(df_scored) == 2

    # antigen species / epitope / receptor_type filters
    df_species = fetch_vdjdb(antigen_species="EBV")
    assert len(df_species) == 1
    df_epi = fetch_vdjdb(antigen_epitope="EP2")
    assert len(df_epi) == 1
    df_rt = fetch_vdjdb(receptor_type="TCR")
    assert (df_rt["receptor_type"] == "TCR").all()

    # use_polars=True path
    pldf = fetch_vdjdb(use_polars=True)
    assert isinstance(pldf, pl.DataFrame)


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_both_sources(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )

    def fake_fetch_bytes(url, timeout=120):
        if "vdjdb" in url:
            return _make_vdjdb_zip()
        return _make_iedb_zip()

    monkeypatch.setattr(_epitope, "_fetch_bytes", fake_fetch_bytes)

    merged = fetch_all()
    assert len(merged) > 0

    merged_polars = fetch_all(use_polars=True)
    assert isinstance(merged_polars, pl.DataFrame)

    only_vdjdb = fetch_all(sources=["vdjdb"])
    assert (only_vdjdb["source_db"] == "VDJdb").all()


def test_fetch_all_partial_failure(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )

    def fake_fetch_bytes(url, timeout=120):
        if "vdjdb" in url:
            return _make_vdjdb_zip()
        raise RuntimeError("iedb is down")

    monkeypatch.setattr(_epitope, "_fetch_bytes", fake_fetch_bytes)
    merged = fetch_all()
    assert (merged["source_db"] == "VDJdb").all()


def test_fetch_all_total_failure(monkeypatch):
    def always_fail(url, timeout=120):
        raise RuntimeError("network down")

    monkeypatch.setattr(_epitope, "_fetch_bytes", always_fail)
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )
    with pytest.raises(RuntimeError, match="All data sources failed"):
        fetch_all()


# ---------------------------------------------------------------------------
# fetch_db / get_epitope
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_downloads(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )

    def fake_fetch_bytes(url, timeout=120):
        if "vdjdb" in url:
            return _make_vdjdb_zip()
        return _make_iedb_zip()

    monkeypatch.setattr(_epitope, "_fetch_bytes", fake_fetch_bytes)


def test_fetch_db_all_methods(patched_downloads):
    assert len(fetch_db(method="vdjdb")) > 0
    assert len(fetch_db(method="iedb", receptor_type=None)) > 0
    assert len(fetch_db(method="both", receptor_type=None)) > 0
    with pytest.raises(ValueError):
        fetch_db(method="bogus")


def _fake_vdj(cdr3_values):
    """A minimal stand-in for a Dandelion object with a plain-pandas .data."""
    df = pd.DataFrame(
        {
            "cell_id": [f"cell{i}" for i in range(len(cdr3_values))],
            "junction_aa": cdr3_values,
        }
    )
    return SimpleNamespace(data=df)


def _fake_adata(n_cells):
    return SimpleNamespace(
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)]),
        n_obs=n_cells,
    )


def test_get_epitope_with_reference():
    ref = pd.DataFrame(
        {
            "cdr3_aa": ["CASSX"],
            "antigen_epitope": ["EP1"],
            "antigen_organism": ["EBV"],
            "antigen_protein": ["Prot1"],
            "mhc_class": ["I"],
            "mhc_allele": ["A*02:01"],
            "source_db": ["VDJdb"],
        }
    )
    vdj = _fake_vdj(["CASSX", "UNKNOWN"])
    adata = _fake_adata(2)
    get_epitope(vdj, adata, reference=ref)
    assert "epitope_vdjdb_primary" in adata.obs.columns
    assert adata.obs.loc["cell0", "epitope_vdjdb_primary"] == "EP1"


def test_get_epitope_downloads_vdjdb(patched_downloads):
    vdj = _fake_vdj(["CASSX"])
    adata = _fake_adata(1)
    get_epitope(vdj, adata, method="vdjdb", receptor_type=None)


def test_get_epitope_downloads_iedb(patched_downloads):
    vdj = _fake_vdj(["CASSX"])
    adata = _fake_adata(1)
    get_epitope(vdj, adata, method="iedb", receptor_type=None)


def test_get_epitope_downloads_both(patched_downloads):
    vdj = _fake_vdj(["CASSX"])
    adata = _fake_adata(1)
    get_epitope(vdj, adata, method="both", receptor_type=None)


def test_get_epitope_invalid_method():
    vdj = _fake_vdj(["CASSX"])
    adata = _fake_adata(1)
    with pytest.raises(ValueError):
        get_epitope(vdj, adata, method="bogus")


def test_annotate_from_db_polars_and_to_pandas_paths():
    ref = pd.DataFrame(
        {
            "cdr3_aa": ["CASSX"],
            "antigen_epitope": ["EP1"],
            "antigen_organism": ["EBV"],
            "antigen_protein": ["Prot1"],
            "mhc_class": ["I"],
            "mhc_allele": ["A*02:01"],
            "source_db": ["IEDB"],
        }
    )
    adata = _fake_adata(1)

    # .collect().to_pandas() branch (e.g. a polars LazyFrame-like stand-in)
    lazy_df = pl.DataFrame(
        {"cell_id": ["cell0"], "junction_aa": ["CASSX"]}
    ).lazy()
    vdj_lazy = SimpleNamespace(data=lazy_df)
    _annotate_from_db(vdj_lazy, adata, ref)
    assert adata.obs.loc["cell0", "epitope_iedb_primary"] == "EP1"

    # .to_pandas() branch (a plain polars DataFrame)
    adata2 = _fake_adata(1)
    eager_df = pl.DataFrame({"cell_id": ["cell0"], "junction_aa": ["CASSX"]})
    vdj_eager = SimpleNamespace(data=eager_df)
    _annotate_from_db(vdj_eager, adata2, ref)
    assert adata2.obs.loc["cell0", "epitope_iedb_primary"] == "EP1"


# ---------------------------------------------------------------------------
# query / to_tsv
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "antigen_organism": ["EBV", "CMV", "SARS-CoV-2"],
            "antigen_epitope": ["EP1", "EP2", "EP3"],
            "receptor_type": ["TCR", "TCR", "BCR"],
            "locus": ["TRB", "TRA", "IGH"],
            "v_call": ["TRBV1", "TRAV1", "IGHV1"],
            "cdr3_aa": ["CASSX", "CA", "CASSLONGER"],
        }
    )


def test_query_pandas_all_filters(sample_df):
    out = query(sample_df, organism="EBV")
    assert len(out) == 1

    out = query(sample_df, epitope="EP2")
    assert len(out) == 1

    out = query(sample_df, receptor_type="TCR")
    assert len(out) == 2

    out = query(sample_df, locus="IGH")
    assert len(out) == 1

    out = query(sample_df, v_gene="TRBV1")
    assert len(out) == 1

    out = query(sample_df, min_cdr3_len=6)
    assert len(out) == 1

    # column-missing branch inside _filt
    narrow = sample_df.drop(columns=["antigen_organism"])
    out = query(narrow, organism="EBV")
    assert len(out) == len(narrow)


def test_query_polars(sample_df):
    pldf = pl.from_pandas(sample_df)
    out = query(pldf, organism="EBV", min_cdr3_len=3)
    assert isinstance(out, pl.DataFrame)
    assert out.shape[0] == 1


def test_to_tsv_pandas_and_polars(tmp_path, sample_df):
    p1 = tmp_path / "out_pandas.tsv"
    to_tsv(sample_df, str(p1))
    assert p1.exists()

    p2 = tmp_path / "out_polars.tsv"
    to_tsv(pl.from_pandas(sample_df), str(p2))
    assert p2.exists()
