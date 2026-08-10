from __future__ import annotations

import gzip
import io
import json
import types
import zipfile

import pandas as pd
import polars as pl
import pytest

from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import dandelion.polars.tools._epitope as _epitope
from dandelion.polars.tools._epitope import (
    _fetch_bytes,
    _infer_locus,
    _infer_receptor_type,
    _ensure_airr_columns,
    _normalise_na,
    _normalise_chain,
    _filter_by_chain,
    _read_csv_bytes,
    _to_polars,
    _parse_iedb_receptor_zip,
    _map_iedb_columns,
    _fetch_iedb,
    _parse_vdjdb_tsv,
    _map_vdjdb_columns,
    _get_vdjdb_url,
    _fetch_vdjdb,
    _as_pandas,
    _restore_type,
    _write_cell_level_columns,
    _annotate_from_db,
    _print_epitope_summary,
    fetch_db,
    get_epitope,
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
      exercise the chain-splitting branch in `_map_iedb_columns` (producing
      TRA+TRB and TRG+TRD rows respectively).
    - One plain "beta" row producing a lone TRB row.
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
# Reset module-level download cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    _epitope._CACHE.clear()
    yield
    _epitope._CACHE.clear()


# ---------------------------------------------------------------------------
# _fetch_bytes
# ---------------------------------------------------------------------------


class _FakeResp:
    """Minimal context-manager stand-in for urlopen()'s return value."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def test_fetch_bytes_cache_hit_and_miss(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        return _FakeResp(b"hello")

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)

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


def test_normalise_chain():
    assert _normalise_chain(None) is None
    assert _normalise_chain("tra") == ["TRA"]
    assert _normalise_chain([" tra ", "TRB"]) == ["TRA", "TRB"]
    with pytest.raises(ValueError, match="Invalid chain"):
        _normalise_chain("XYZ")


def test_filter_by_chain():
    df = pd.DataFrame({"locus": ["TRA", "TRB", "IGH"]})

    # chains=None -> unchanged passthrough
    assert _filter_by_chain(df, None) is df

    out = _filter_by_chain(df, ["TRA"])
    assert out["locus"].tolist() == ["TRA"]

    no_locus = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        _filter_by_chain(no_locus, ["TRA"])


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


def test_map_iedb_columns_no_chain_type_and_existing_junction():
    # No "type"/"chain_2_type" columns at all -> skips the whole
    # alphabeta/gammadelta split branch and the locus-derivation branch;
    # junction_aa already present -> the cdr3_aa fallback-copy is skipped
    # (must not be overwritten).
    df = pd.DataFrame(
        {
            "curated_v_gene": ["V1"],
            "cdr3_curated": ["CASX"],
            "junction_aa": ["CASXJ"],
        }
    )
    mapped = _map_iedb_columns(df)
    assert "locus" not in mapped.columns
    assert "receptor_type" not in mapped.columns
    assert mapped["junction_aa"].tolist() == ["CASXJ"]


def test_map_iedb_columns_cdr3_all_nan_no_fallback_column():
    # cdr3_curated all-NaN and no *_cdr3_calculated_* column present at all
    # -> the "if calc:" fallback branch's False path (calc is None).
    df = pd.DataFrame(
        {
            "cdr3_curated": [None, None],
            "curated_v_gene": ["V1", "V2"],
        }
    )
    mapped = _map_iedb_columns(df)
    assert mapped["cdr3_aa"].isna().all()


def test_fetch_iedb_full_pipeline(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_fetch_bytes", lambda url, timeout=120: _make_iedb_zip()
    )

    df = _fetch_iedb()
    assert {"BCR", "TCR"}.issuperset(set(df["receptor_type"].unique()) - {""})
    assert (df["source_db"] == "IEDB").all()
    # alphabeta row should have produced an alpha + beta pair, with the beta
    # row's cdr3_aa pulled from the duplicate "cdr3_curated" (chain-2) column
    assert "CASSCHAIN2" in df["cdr3_aa"].tolist()
    # loci present: TRA + TRB (alphabeta split), TRG + TRD (gammadelta split), TRB (plain)
    assert set(df["locus"].unique()) == {"TRA", "TRB", "TRG", "TRD"}

    # organism filter
    df2 = _fetch_iedb(organism_filter="Epstein-Barr")
    assert len(df2) >= 1
    assert df2["antigen_organism"].str.contains("Epstein-Barr").all()

    # receptor_type filter
    df3 = _fetch_iedb(receptor_type="TCR")
    assert (df3["receptor_type"] == "TCR").all()

    # chain filter (single string)
    df4 = _fetch_iedb(chain="TRA")
    assert (df4["locus"] == "TRA").all()
    assert len(df4) == 1

    # chain filter (list)
    df5 = _fetch_iedb(chain=["TRG", "TRD"])
    assert set(df5["locus"].unique()) == {"TRG", "TRD"}

    # use_polars=True path
    pldf = _fetch_iedb(use_polars=True)
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


def test_map_vdjdb_columns_direct():
    df = _read_csv_bytes(_vdjdb_tsv_bytes(), sep="\t")
    mapped = _map_vdjdb_columns(df)
    assert (mapped["source_db"] == "VDJdb").all()
    assert mapped["sequence_id"].str.startswith("VDJdb_").all()
    assert "locus" in mapped.columns
    assert set(mapped["locus"].unique()) == {"TRA", "TRB"}
    assert (mapped["receptor_type"] == "TCR").all()
    # web.method column must be dropped
    assert "_method" not in mapped.columns
    assert "web.method" not in mapped.columns
    # junction_aa filled from cdr3_aa
    assert mapped["junction_aa"].tolist() == mapped["cdr3_aa"].tolist()
    # db_record_id falls back to sequence_id when absent from source
    assert mapped["db_record_id"].tolist() == mapped["sequence_id"].tolist()


def test_map_vdjdb_columns_minimal_no_gene_no_method_existing_junction():
    # No "gene"/"web.method" columns at all -> skips locus/receptor_type
    # derivation and the _method drop; junction_aa already present -> the
    # cdr3_aa fallback-copy is skipped (must not be overwritten).
    df = pd.DataFrame({"cdr3": ["CASSX"], "junction_aa": ["CASSXJ"]})
    mapped = _map_vdjdb_columns(df)
    assert "locus" not in mapped.columns
    assert "receptor_type" not in mapped.columns
    assert mapped["junction_aa"].tolist() == ["CASSXJ"]
    assert (mapped["source_db"] == "VDJdb").all()


def _github_api_url() -> str:
    return "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"


def test_get_vdjdb_url_tier1_success(monkeypatch):
    payload = {
        "assets": [
            {"name": "readme.md", "browser_download_url": "http://x/readme.md"},
            {
                "name": "vdjdb.tsv.gz",
                "browser_download_url": "http://x/vdjdb.tsv.gz",
            },
        ]
    }

    def fake_urlopen(req, timeout):
        assert "api.github.com" in req.full_url
        return _FakeResp(json.dumps(payload).encode())

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)
    url = _get_vdjdb_url()
    assert url == "http://x/vdjdb.tsv.gz"


def test_get_vdjdb_url_tier1_no_asset_falls_to_tier2(monkeypatch):
    payload = {
        "assets": [
            {"name": "readme.md", "browser_download_url": "http://x/readme.md"}
        ]
    }

    def fake_urlopen(req, timeout):
        if "api.github.com" in req.full_url:
            return _FakeResp(json.dumps(payload).encode())
        # tier 2: latest-version.txt
        return _FakeResp(b"https://example.com/vdjdb-latest.zip\n")

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)
    assert _get_vdjdb_url() == "https://example.com/vdjdb-latest.zip"


def test_get_vdjdb_url_tier1_exception_falls_to_tier2(monkeypatch):
    def fake_urlopen(req, timeout):
        if "api.github.com" in req.full_url:
            raise URLError("down")
        return _FakeResp(b"https://example.com/from-changelog.zip\n")

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)
    assert _get_vdjdb_url() == "https://example.com/from-changelog.zip"


def test_get_vdjdb_url_tier2_bad_content_falls_to_tier3(monkeypatch):
    def fake_urlopen(req, timeout):
        if "api.github.com" in req.full_url:
            raise URLError("down")
        # First non-empty line isn't a URL
        return _FakeResp(b"not-a-url\nsomething-else\n")

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)
    assert _get_vdjdb_url() == _epitope.VDJDB_URL


def test_get_vdjdb_url_both_tiers_fail_falls_to_tier3(monkeypatch):
    def fake_urlopen(req, timeout):
        raise URLError("everything is down")

    monkeypatch.setattr(_epitope, "urlopen", fake_urlopen)
    assert _get_vdjdb_url() == _epitope.VDJDB_URL


def test_fetch_vdjdb_full_pipeline(monkeypatch):
    monkeypatch.setattr(
        _epitope, "_get_vdjdb_url", lambda: "http://fake/vdjdb.zip"
    )
    monkeypatch.setattr(
        _epitope, "_fetch_bytes", lambda url, timeout=120: _make_vdjdb_zip()
    )

    df = _fetch_vdjdb()
    assert len(df) == 3

    # score filter
    df_scored = _fetch_vdjdb(min_vdjdb_score=2)
    assert len(df_scored) == 2

    # antigen species / epitope / receptor_type filters
    df_species = _fetch_vdjdb(antigen_species="EBV")
    assert len(df_species) == 1
    df_epi = _fetch_vdjdb(antigen_epitope="EP2")
    assert len(df_epi) == 1
    df_rt = _fetch_vdjdb(receptor_type="TCR")
    assert (df_rt["receptor_type"] == "TCR").all()

    # chain filter
    df_chain = _fetch_vdjdb(chain="TRA")
    assert (df_chain["locus"] == "TRA").all()
    assert len(df_chain) == 1

    # use_polars=True path
    pldf = _fetch_vdjdb(use_polars=True)
    assert isinstance(pldf, pl.DataFrame)


# ---------------------------------------------------------------------------
# fetch_db
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


def test_fetch_db_all_databases(patched_downloads):
    assert len(fetch_db(database="vdjdb")) > 0
    assert len(fetch_db(database="iedb", receptor_type=None)) > 0
    assert len(fetch_db(database="both", receptor_type=None)) > 0
    with pytest.raises(ValueError):
        fetch_db(database="bogus")


# ---------------------------------------------------------------------------
# _as_pandas / _restore_type
# ---------------------------------------------------------------------------


def test_as_pandas_all_branches():
    pdf = pd.DataFrame({"a": [1]})

    # already pandas
    out, kind = _as_pandas(pdf)
    assert kind == "pandas"
    assert out is pdf

    # eager polars
    pldf = pl.DataFrame({"a": [1]})
    out, kind = _as_pandas(pldf)
    assert kind == "polars"
    assert isinstance(out, pd.DataFrame)

    # lazy polars
    lazy = pldf.lazy()
    out, kind = _as_pandas(lazy)
    assert kind == "lazy"
    assert isinstance(out, pd.DataFrame)


def test_restore_type_all_branches():
    pdf = pd.DataFrame({"a": [1]})

    assert _restore_type(pdf, "pandas") is pdf

    out = _restore_type(pdf, "polars")
    assert isinstance(out, pl.DataFrame)

    out = _restore_type(pdf, "lazy")
    assert isinstance(out, pl.LazyFrame)


def test_write_cell_level_columns():
    target = pd.DataFrame({"existing": [1, 2]}, index=["c1", "c2"])
    target.index.name = "cell_id"
    source = pd.DataFrame({"new_col": ["x", "y"]}, index=["c1", "c2"])
    source.index.name = "cell_id"

    out = _write_cell_level_columns(target, source, ["new_col"])
    assert out["new_col"].tolist() == ["x", "y"]
    assert "existing" in out.columns

    # calling again with a stale version of the column is idempotent
    source2 = pd.DataFrame({"new_col": ["z", "w"]}, index=["c1", "c2"])
    source2.index.name = "cell_id"
    out2 = _write_cell_level_columns(out, source2, ["new_col"])
    assert out2["new_col"].tolist() == ["z", "w"]
    assert list(out2.columns).count("new_col") == 1


# ---------------------------------------------------------------------------
# _annotate_from_db / get_epitope / _print_epitope_summary
# ---------------------------------------------------------------------------


def _split_update_metadata(
    vdj, retrieve, split=True, join=True, unique=True, reinitialize=False
):
    """A realistic-enough stand-in for Dandelion's own
    ``update_metadata(retrieve=..., split=True, ...)``: groups contigs into
    VDJ (IGH/TRB/TRD) vs VJ (everything else) and joins unique values per
    cell for each retrieved column, writing `{col}_VDJ` / `{col}_VJ` into
    vdj._metadata."""
    d, _ = _as_pandas(vdj._data)
    d = d.copy()
    d["_locus_group"] = d["locus"].apply(
        lambda locus: "VDJ" if locus in ("IGH", "TRB", "TRD") else "VJ"
    )
    if vdj._metadata is not None:
        meta, _ = _as_pandas(vdj._metadata)
        out = meta.set_index("cell_id")
    else:
        out = pd.DataFrame(
            index=pd.Index(d["cell_id"].unique(), name="cell_id")
        )

    for col in retrieve:
        for grp in ["VDJ", "VJ"]:
            sub = d[d["_locus_group"] == grp]
            agg = (
                sub.groupby("cell_id")[col]
                .apply(
                    lambda x: "|".join(
                        sorted(set(v for v in x.dropna() if v != ""))
                    )
                )
                .replace("", pd.NA)
            )
            out[f"{col}_{grp}"] = agg
    vdj._metadata = out.reset_index()


def _make_mock_vdj(
    data_df: pd.DataFrame,
    vdj_data_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    update_metadata="default",
):
    """Build a minimal Dandelion-like mock.

    update_metadata: "default" (working VDJ/VJ split), "missing" (no
    such attribute at all), "typeerror_then_ok" (rich-kwargs call raises
    TypeError, falls back to the reduced-kwargs call), or "noop" (does
    nothing, so no new columns appear -- exercises the "no new columns"
    warning branch).
    """
    obj = SimpleNamespace()
    obj.data = data_df
    obj._data = vdj_data_df if vdj_data_df is not None else data_df.copy()
    obj._metadata = metadata_df

    if update_metadata == "missing":
        return obj

    if update_metadata == "noop":

        def _um(self, retrieve, **kwargs):
            return None

        obj.update_metadata = types.MethodType(_um, obj)
        return obj

    if update_metadata == "typeerror_then_ok":
        calls = {"n": 0}

        def _um(self, retrieve, **kwargs):
            calls["n"] += 1
            if "split" in kwargs:
                raise TypeError("unexpected keyword argument 'split'")
            _split_update_metadata(self, retrieve)

        obj.update_metadata = types.MethodType(_um, obj)
        obj._calls = calls
        return obj

    # default
    def _um(
        self, retrieve, split=True, join=True, unique=True, reinitialize=False
    ):
        _split_update_metadata(
            self, retrieve, split, join, unique, reinitialize
        )

    obj.update_metadata = types.MethodType(_um, obj)
    return obj


def _make_reference(rows):
    """rows: list of dicts with keys cdr3_aa, antigen_epitope,
    antigen_organism, mhc_class, mhc_allele, source_db, locus."""
    base = {
        "antigen_protein": "",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _fake_adata(cell_ids):
    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    return SimpleNamespace(obs=obs, n_obs=len(cell_ids))


def _two_chain_data():
    """cellA has a matching TRA contig and a matching TRB contig (with
    *different* epitopes); cellB has a non-matching TRA contig only."""
    return pd.DataFrame(
        {
            "sequence_id": ["c1_tra", "c1_trb", "c2_tra"],
            "cell_id": ["cellA", "cellA", "cellB"],
            "locus": ["TRA", "TRB", "TRA"],
            "junction_aa": ["CAVSEQ1", "CASSEQ2", "CAVNOMATCH"],
            "productive": ["T", "T", "T"],
        }
    )


def _two_chain_reference():
    return _make_reference(
        [
            dict(
                cdr3_aa="CAVSEQ1",
                antigen_epitope="EPI_ALPHA",
                antigen_organism="EBV",
                mhc_class="I",
                mhc_allele="A*02:01",
                source_db="VDJdb",
                locus="TRA",
            ),
            dict(
                cdr3_aa="CASSEQ2",
                antigen_epitope="EPI_BETA",
                antigen_organism="CMV",
                mhc_class="I",
                mhc_allele="A*01:01",
                source_db="VDJdb",
                locus="TRB",
            ),
        ]
    )


def test_annotate_from_db_keeps_chains_separate():
    """The core bug fix: a cell's TRA match and TRB match must not be
    merged together in vdj._metadata / adata.obs."""
    data = _two_chain_data()
    vdj = _make_mock_vdj(
        data, metadata_df=pd.DataFrame({"cell_id": ["cellA", "cellB"]})
    )
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)

    # contig-level: each contig carries only its own match
    d = vdj._data.set_index("sequence_id")
    assert d.loc["c1_tra", "epitope_vdjdb"] == "EPI_ALPHA"
    assert d.loc["c1_trb", "epitope_vdjdb"] == "EPI_BETA"
    assert d.loc["c1_tra", "epitope_vdjdb_primary"] == "EPI_ALPHA"
    assert pd.isna(d.loc["c2_tra", "epitope_vdjdb"])

    # cell-level: VDJ (TRB) and VJ (TRA) groups stay distinct
    m = vdj._metadata.set_index("cell_id")
    assert m.loc["cellA", "epitope_vdjdb_VJ"] == "EPI_ALPHA"
    assert m.loc["cellA", "epitope_vdjdb_VDJ"] == "EPI_BETA"

    # adata.obs mirrors the new vdj._metadata columns
    assert adata.obs.loc["cellA", "epitope_vdjdb_VJ"] == "EPI_ALPHA"
    assert adata.obs.loc["cellA", "epitope_vdjdb_VDJ"] == "EPI_BETA"


def test_annotate_from_db_chain_filter():
    data = _two_chain_data()
    vdj = _make_mock_vdj(
        data, metadata_df=pd.DataFrame({"cell_id": ["cellA", "cellB"]})
    )
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    # restrict to TRA only -> the TRB contig must not get annotated at all
    _annotate_from_db(vdj, adata, ref, chain="TRA")

    d = vdj._data.set_index("sequence_id")
    assert d.loc["c1_tra", "epitope_vdjdb"] == "EPI_ALPHA"
    assert "epitope_vdjdb" not in d.columns or pd.isna(
        d.loc["c1_trb"].get("epitope_vdjdb", float("nan"))
    )


def test_annotate_from_db_chain_missing_locus_column_raises():
    data = pd.DataFrame({"cell_id": ["c1"], "junction_aa": ["X"]})
    vdj = _make_mock_vdj(data)
    adata = _fake_adata(["c1"])
    ref = _two_chain_reference()
    with pytest.raises(KeyError, match="locus"):
        _annotate_from_db(vdj, adata, ref, chain="TRA")


def test_annotate_from_db_missing_sequence_id_raises():
    data = pd.DataFrame(
        {"cell_id": ["c1"], "locus": ["TRA"], "junction_aa": ["CAVSEQ1"]}
    )
    vdj_data = pd.DataFrame({"cell_id": ["c1"]})  # no sequence_id column
    vdj = _make_mock_vdj(data, vdj_data_df=vdj_data)
    adata = _fake_adata(["c1"])
    ref = _two_chain_reference()
    with pytest.raises(KeyError, match="sequence_id"):
        _annotate_from_db(vdj, adata, ref, chain=None)


def test_annotate_from_db_no_matches_early_return():
    data = pd.DataFrame(
        {
            "sequence_id": ["c1"],
            "cell_id": ["c1cell"],
            "locus": ["TRA"],
            "junction_aa": ["NO_MATCH_AT_ALL"],
            "productive": ["T"],
        }
    )
    vdj = _make_mock_vdj(
        data, metadata_df=pd.DataFrame({"cell_id": ["c1cell"]})
    )
    adata = _fake_adata(["c1cell"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)
    # nothing matched -> vdj._metadata untouched, adata.obs untouched
    assert list(vdj._metadata.columns) == ["cell_id"]
    assert list(adata.obs.columns) == []


def test_annotate_from_db_no_update_metadata_method():
    data = _two_chain_data()
    vdj = _make_mock_vdj(data, update_metadata="missing")
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)
    # vdj._data still gets annotated
    d = vdj._data.set_index("sequence_id")
    assert d.loc["c1_tra", "epitope_vdjdb"] == "EPI_ALPHA"
    # but nothing propagates to adata.obs
    assert list(adata.obs.columns) == []


def test_annotate_from_db_metadata_none_initially():
    data = _two_chain_data()
    vdj = _make_mock_vdj(data, metadata_df=None)
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)
    assert vdj._metadata is not None
    assert "epitope_vdjdb_VJ" in adata.obs.columns


def test_annotate_from_db_update_metadata_typeerror_fallback():
    data = _two_chain_data()
    vdj = _make_mock_vdj(
        data,
        metadata_df=pd.DataFrame({"cell_id": ["cellA", "cellB"]}),
        update_metadata="typeerror_then_ok",
    )
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)
    assert vdj._calls["n"] == 2  # rich call failed, reduced call succeeded
    assert "epitope_vdjdb_VJ" in adata.obs.columns


def test_annotate_from_db_update_metadata_adds_nothing():
    data = _two_chain_data()
    vdj = _make_mock_vdj(
        data,
        metadata_df=pd.DataFrame({"cell_id": ["cellA", "cellB"]}),
        update_metadata="noop",
    )
    adata = _fake_adata(["cellA", "cellB"])
    ref = _two_chain_reference()

    _annotate_from_db(vdj, adata, ref, chain=None)
    # vdj._data was still annotated
    d = vdj._data.set_index("sequence_id")
    assert d.loc["c1_tra", "epitope_vdjdb"] == "EPI_ALPHA"
    # but adata.obs got nothing since update_metadata (mock) added no columns
    assert list(adata.obs.columns) == []


def test_annotate_from_db_both_source_dbs():
    data = pd.DataFrame(
        {
            "sequence_id": ["c1_tra", "c1_trb"],
            "cell_id": ["cellA", "cellA"],
            "locus": ["TRA", "TRB"],
            "junction_aa": ["CAVSEQ1", "CASSEQ2"],
            "productive": ["T", "T"],
        }
    )
    ref = _make_reference(
        [
            {
                "cdr3_aa": "CAVSEQ1",
                "antigen_epitope": "EPI_A_IEDB",
                "antigen_organism": "EBV",
                "mhc_class": "I",
                "mhc_allele": "A*02:01",
                "source_db": "IEDB",
                "locus": "TRA",
            },
            {
                "cdr3_aa": "CASSEQ2",
                "antigen_epitope": "EPI_B_VDJDB",
                "antigen_organism": "CMV",
                "mhc_class": "I",
                "mhc_allele": "A*01:01",
                "source_db": "VDJdb",
                "locus": "TRB",
            },
        ]
    )
    vdj = _make_mock_vdj(data, metadata_df=pd.DataFrame({"cell_id": ["cellA"]}))
    adata = _fake_adata(["cellA"])

    _annotate_from_db(vdj, adata, ref, chain=None)

    assert adata.obs.loc["cellA", "epitope_iedb_VJ"] == "EPI_A_IEDB"
    assert adata.obs.loc["cellA", "epitope_vdjdb_VDJ"] == "EPI_B_VDJDB"


def test_annotate_from_db_lazy_and_eager_polars_data():
    ref = _make_reference(
        [
            dict(
                cdr3_aa="CASSX",
                antigen_epitope="EP1",
                antigen_organism="EBV",
                mhc_class="I",
                mhc_allele="A*02:01",
                source_db="IEDB",
                locus="TRB",
            )
        ]
    )

    # .collect().to_pandas() branch -- vdj.data is a polars LazyFrame
    lazy_df = pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    ).lazy()
    vdj_lazy = _make_mock_vdj(
        lazy_df,
        vdj_data_df=lazy_df,
        metadata_df=pd.DataFrame({"cell_id": ["cell0"]}),
    )
    adata = _fake_adata(["cell0"])
    _annotate_from_db(vdj_lazy, adata, ref, chain=None)
    assert isinstance(vdj_lazy._data, pl.LazyFrame)
    assert adata.obs.loc["cell0", "epitope_iedb_VDJ"] == "EP1"

    # .to_pandas() branch -- vdj.data is an eager polars DataFrame
    eager_df = pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    )
    vdj_eager = _make_mock_vdj(
        eager_df,
        vdj_data_df=eager_df,
        metadata_df=pd.DataFrame({"cell_id": ["cell0"]}),
    )
    adata2 = _fake_adata(["cell0"])
    _annotate_from_db(vdj_eager, adata2, ref, chain=None)
    assert isinstance(vdj_eager._data, pl.DataFrame)
    assert adata2.obs.loc["cell0", "epitope_iedb_VDJ"] == "EP1"


def test_print_epitope_summary(capsys):
    adata = _fake_adata(["cellA", "cellB"])
    adata.obs["epitope_vdjdb_VDJ"] = ["EP1", None]
    adata.obs["epitope_vdjdb_VJ"] = [None, None]
    adata.obs["organism_vdjdb_VDJ"] = [
        "EBV",
        None,
    ]  # not epitope_-prefixed w/ VDJ/VJ match, still fine

    _print_epitope_summary(adata, "vdjdb")
    out = capsys.readouterr().out
    assert "epitope_vdjdb_VDJ" in out
    assert "epitope_vdjdb_VJ" in out
    assert "1 / 2" in out

    # no matching columns at all -> prints nothing
    adata_empty = _fake_adata(["cellA"])
    _print_epitope_summary(adata_empty, "iedb")
    out2 = capsys.readouterr().out
    assert out2 == ""


# ---------------------------------------------------------------------------
# get_epitope
# ---------------------------------------------------------------------------


def test_get_epitope_with_reference():
    ref = _make_reference(
        [
            dict(
                cdr3_aa="CASSX",
                antigen_epitope="EP1",
                antigen_organism="EBV",
                mhc_class="I",
                mhc_allele="A*02:01",
                source_db="VDJdb",
                locus="TRB",
            )
        ]
    )
    data = pd.DataFrame(
        {
            "sequence_id": ["s0", "s1"],
            "cell_id": ["cell0", "cell1"],
            "locus": ["TRB", "TRB"],
            "junction_aa": ["CASSX", "UNKNOWN"],
            "productive": ["T", "T"],
        }
    )
    vdj = _make_mock_vdj(
        data, metadata_df=pd.DataFrame({"cell_id": ["cell0", "cell1"]})
    )
    adata = _fake_adata(["cell0", "cell1"])

    get_epitope(vdj, adata, reference=ref)
    assert adata.obs.loc["cell0", "epitope_vdjdb_VDJ"] == "EP1"


def test_get_epitope_downloads_vdjdb(patched_downloads):
    data = pd.DataFrame(
        {
            "sequence_id": ["s0"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    )
    vdj = _make_mock_vdj(data, metadata_df=pd.DataFrame({"cell_id": ["cell0"]}))
    adata = _fake_adata(["cell0"])
    get_epitope(vdj, adata, database="vdjdb", receptor_type=None)


def test_get_epitope_downloads_iedb(patched_downloads):
    data = pd.DataFrame(
        {
            "sequence_id": ["s0"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    )
    vdj = _make_mock_vdj(data, metadata_df=pd.DataFrame({"cell_id": ["cell0"]}))
    adata = _fake_adata(["cell0"])
    get_epitope(vdj, adata, database="iedb", receptor_type=None)


def test_get_epitope_downloads_both(patched_downloads):
    data = pd.DataFrame(
        {
            "sequence_id": ["s0"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    )
    vdj = _make_mock_vdj(data, metadata_df=pd.DataFrame({"cell_id": ["cell0"]}))
    adata = _fake_adata(["cell0"])
    get_epitope(vdj, adata, database="both", receptor_type=None)


def test_get_epitope_invalid_database():
    data = pd.DataFrame(
        {
            "sequence_id": ["s0"],
            "cell_id": ["cell0"],
            "locus": ["TRB"],
            "junction_aa": ["CASSX"],
            "productive": ["T"],
        }
    )
    vdj = _make_mock_vdj(data)
    adata = _fake_adata(["cell0"])
    with pytest.raises(ValueError):
        get_epitope(vdj, adata, database="bogus")
