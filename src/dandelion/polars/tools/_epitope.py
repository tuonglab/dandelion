from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import polars as pl

from typing import Literal
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
IEDB_RECEPTOR_URL = (
    "https://www.iedb.org/downloader.php?file_name=doc/receptor_full_v3.zip"
)
IEDB_BCELL_URL = "https://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip"
VDJDB_URL = "https://github.com/antigenomics/vdjdb-db/releases/download/2025-12-29/vdjdb-2025-12-29.zip"
VDJDB_FALLBACK_URL = "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest-version.txt"

# ---------------------------------------------------------------------------
# Module-level download cache — raw bytes are stored here after the first
# download so that repeated calls to fetch_vdjdb / fetch_iedb (e.g. inside a
# for-loop over samples) do not trigger additional HTTP requests.
# ---------------------------------------------------------------------------
_CACHE: dict[str, bytes] = {}


def clear_cache() -> None:
    """Evict all cached raw downloads (useful if you want to force a refresh)."""
    _CACHE.clear()
    log.info("Download cache cleared.")


# ---------------------------------------------------------------------------
# AIRR Rearrangement schema — core fields we populate
# https://docs.airr-community.org/en/stable/datarep/rearrangements.html
# ---------------------------------------------------------------------------
AIRR_CORE_FIELDS: list[str] = [
    # Identifiers
    "sequence_id",
    "source_db",
    # Locus / chain
    "locus",
    "receptor_type",
    # V(D)J calls
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    # CDR / junction sequences (amino-acid)
    "cdr1_aa",
    "cdr2_aa",
    "cdr3_aa",
    "junction_aa",  # AIRR standard (CDR3 + conserved residues)
    # Full variable domain sequence
    "sequence_aa",
    "sequence_nt",
    # Antigen / epitope metadata (custom AIRR-extended fields)
    "antigen_epitope",
    "antigen_protein",
    "antigen_organism",
    "mhc_class",
    "mhc_allele",
    # Host
    "subject_species",
    # Provenance
    "pubmed_id",
    "db_record_id",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_bytes(url: str, timeout: int = 120) -> bytes:
    """Download a URL and return raw bytes with a descriptive error on failure.

    Results are cached in ``_CACHE`` so that the same URL is only downloaded
    once per Python session (or until :func:`clear_cache` is called).
    """
    if url in _CACHE:
        log.info("Cache hit for %s (%d bytes)", url, len(_CACHE[url]))
        return _CACHE[url]

    req = Request(url, headers={"User-Agent": "immune_db_retrieval/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        log.info("Downloaded %d bytes from %s", len(data), url)
        _CACHE[url] = data
        return data
    except HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} while fetching {url}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Network error fetching {url}: {exc.reason}"
        ) from exc


def _read_csv_bytes(
    raw: bytes,
    sep: str = ",",
    skip_rows: int = 0,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Parse CSV/TSV bytes into a pandas DataFrame."""
    return pd.read_csv(
        io.BytesIO(raw),
        sep=sep,
        skiprows=skip_rows,
        encoding=encoding,
        low_memory=False,
        dtype=str,  # read everything as string first — we normalise later
    )


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(df)


def _ensure_airr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing AIRR columns as empty strings so output is always schema-consistent."""
    for col in AIRR_CORE_FIELDS:
        if col not in df.columns:
            df[col] = ""
    return df[
        AIRR_CORE_FIELDS + [c for c in df.columns if c not in AIRR_CORE_FIELDS]
    ]


def _normalise_na(df: pd.DataFrame) -> pd.DataFrame:
    """Replace IEDB / VDJdb sentinel values with empty strings (AIRR null convention)."""
    return df.replace({"N/A": "", "n/a": "", "NA": "", "None": "", "nan": ""})


# ---------------------------------------------------------------------------
# Locus inference helpers
# ---------------------------------------------------------------------------

_CHAIN_TO_LOCUS = {
    # BCR / Ig
    "heavy": "IGH",
    "h": "IGH",
    "light": "IGL",
    "l": "IGL",
    "kappa": "IGK",
    "k": "IGK",
    "lambda": "IGL",
    # TCR
    "alpha": "TRA",
    "a": "TRA",
    "beta": "TRB",
    "b": "TRB",
    "gamma": "TRG",
    "g": "TRG",
    "delta": "TRD",
    "d": "TRD",
}


def _infer_locus(chain_str: str) -> str:
    if not isinstance(chain_str, str):
        return ""
    return _CHAIN_TO_LOCUS.get(chain_str.strip().lower(), "")


def _infer_receptor_type(locus: str) -> str:
    if locus.startswith("IG"):
        return "BCR"
    if locus.startswith("TR"):
        return "TCR"
    return ""


# ===========================================================================
# IEDB
# ===========================================================================

# Column mapping: IEDB receptor_full_v3 → AIRR field name
_IEDB_RECEPTOR_COL_MAP = {
    # Identifiers
    "group_iri": "db_record_id",
    "receptor_group_id": "db_record_id",  # old format fallback
    # Chain info
    "type": "_chain1_type",  # used to derive locus
    "chain_1_type": "_chain1_type",  # old format fallback
    "chain_2_type": "_chain2_type",  # old format fallback
    # V/D/J genes — chain 1 (heavy / alpha)
    "curated_v_gene": "v_call",
    "chain_1_v_gene": "v_call",  # old format fallback
    "curated_d_gene": "d_call",
    "chain_1_d_gene": "d_call",  # old format fallback
    "curated_j_gene": "j_call",
    "chain_1_j_gene": "j_call",  # old format fallback
    # CDRs — chain 1
    "cdr1_curated": "cdr1_aa",
    "chain_1_cdr1_seq": "cdr1_aa",  # old format fallback
    "cdr2_curated": "cdr2_aa",
    "chain_1_cdr2_seq": "cdr2_aa",  # old format fallback
    "cdr3_curated": "cdr3_aa",
    # "cdr3_calculated":            "cdr3_aa",        # removed, handled separately below
    "chain_1_cdr3_seq": "cdr3_aa",  # old format fallback
    "chain_1_cdr3_seq_aa": "cdr3_aa",  # old format fallback
    # Full protein sequence
    "protein_sequence": "sequence_aa",
    "chain_1_protein_seq": "sequence_aa",  # old format fallback
    "nucleotide_sequence": "sequence_nt",
    "chain_1_nucleotide_seq": "sequence_nt",  # old format fallback
    # Epitope metadata
    "name": "antigen_epitope",
    "epitope_description": "antigen_epitope",  # old format fallback
    "source_molecule": "antigen_protein",
    "antigen_name": "antigen_protein",  # old format fallback
    "source_organism": "antigen_organism",
    "antigen_species": "antigen_organism",  # old format fallback
    "mhc_class": "mhc_class",
    "mhc_allele_names": "mhc_allele",
    "mhc_allele_name": "mhc_allele",  # old format fallback
    # Host
    "host_organism_name": "subject_species",
    # Reference
    "iedb_ids": "pubmed_id",
    "pubmed_id": "pubmed_id",  # old format fallback
}

# Alternative column names present in older IEDB exports
_IEDB_ALT_COL_MAP = {
    "Receptor Group ID": "db_record_id",
    "Chain 1 Type": "_chain1_type",
    "Chain 1 CDR3 Curated": "cdr3_aa",
    "Chain 1 V Gene": "v_call",
    "Chain 1 D Gene": "d_call",
    "Chain 1 J Gene": "j_call",
    "Antigen Name": "antigen_protein",
    "Antigen Species": "antigen_organism",
    "Epitope Description": "antigen_epitope",
    "Host Species": "subject_species",
    "PubMed ID": "pubmed_id",
}


def _parse_iedb_receptor_zip(raw_zip: bytes) -> pd.DataFrame:
    """Unzip and parse the IEDB receptor_full_v3 export."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(
                "No CSV file found inside the IEDB zip. "
                f"Archive contents: {zf.namelist()}"
            )
        # Prefer the file whose name contains 'receptor'
        target = next(
            (n for n in csv_names if "receptor" in n.lower()), csv_names[0]
        )
        log.info("Parsing IEDB file: %s", target)
        raw_csv = zf.read(target)

    # IEDB exports have a header note on row 0; real column headers are on row 1
    df = _read_csv_bytes(raw_csv, sep=",", skip_rows=0)

    # Row 0 contains real column names; deduplicate by appending position index
    real_headers = df.iloc[0].tolist()
    seen = {}
    headers = []
    for i, h in enumerate(real_headers):
        key = str(h).strip()
        if key in seen:
            headers.append(f"{key}__{i}")
        else:
            seen[key] = i
            headers.append(key)

    df.columns = headers
    df = df.iloc[1:].reset_index(drop=True)
    return df


def _map_iedb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename IEDB columns to AIRR names using either known mapping."""
    # Lowercase the column names for case-insensitive matching
    col_lower = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_lower)

    # Apply primary mapping
    rename = {}
    for src, dst in _IEDB_RECEPTOR_COL_MAP.items():
        key = src.lower().replace(" ", "_")
        if key in df.columns:
            rename[key] = dst
    df = df.rename(columns=rename)

    # Deduplicate columns mapped to the same target name, keep first
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Split alphabeta/gammadelta records into two rows (one per chain)
    if "_chain1_type" in df.columns:
        for chain_type, alpha_val, beta_val in [
            ("alphabeta", "alpha", "beta"),
            ("gammadelta", "gamma", "delta"),
        ]:
            mask = df["_chain1_type"].str.lower() == chain_type
            if mask.any():
                chain1 = df.copy()
                chain1.loc[mask, "_chain1_type"] = alpha_val

                chain2 = df[mask].copy()
                chain2["_chain1_type"] = beta_val
                for col1, col2_pattern in [
                    ("cdr3_aa", "cdr3_curated__"),
                    ("v_call", "curated_v_gene__"),
                    ("j_call", "curated_j_gene__"),
                    ("cdr1_aa", "cdr1_curated__"),
                    ("cdr2_aa", "cdr2_curated__"),
                    ("sequence_aa", "protein_sequence__"),
                ]:
                    col2 = next(
                        (c for c in df.columns if c.startswith(col2_pattern)),
                        None,
                    )
                    if col2:
                        chain2[col1] = chain2[col2]

                df = pd.concat([chain1, chain2], ignore_index=True)

    # CDR3 fallback: if curated is all NaN, use calculated
    if "cdr3_aa" in df.columns and df["cdr3_aa"].isna().all():
        calc = next((c for c in df.columns if "cdr3_calculated" in c), None)
        if calc:
            df["cdr3_aa"] = df[calc]

    # Derive locus and receptor_type from chain type column
    if "_chain1_type" in df.columns:
        df["locus"] = df["_chain1_type"].apply(_infer_locus)
        df["receptor_type"] = df["locus"].apply(_infer_receptor_type)
        df = df.drop(columns=["_chain1_type"], errors="ignore")
    if "_chain2_type" in df.columns:
        df = df.drop(columns=["_chain2_type"], errors="ignore")

    # junction_aa: AIRR uses junction (includes conserved Cys / Trp)
    # CDR3 from IEDB doesn't always include the anchors — keep as cdr3_aa and
    # copy to junction_aa as best available approximation when junction is absent
    if "junction_aa" not in df.columns and "cdr3_aa" in df.columns:
        df["junction_aa"] = df["cdr3_aa"]

    df["source_db"] = "IEDB"
    df["sequence_id"] = (
        "IEDB_"
        + df.get("db_record_id", pd.Series(range(len(df)), dtype=str))
        .astype(str)
        .str.strip()
    )
    return df


def fetch_iedb(
    organism_filter: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    use_polars: bool = False,
    timeout: int = 120,
) -> DataFrame:
    """
    Download the IEDB receptor_full_v3 export and return an AIRR-compatible
    dataframe.

    Parameters
    ----------
    organism_filter : str, optional
        Case-insensitive substring to filter on ``antigen_organism``.
        e.g. ``"Epstein-Barr"``, ``"influenza"``, ``"SARS-CoV-2"``.
    receptor_type : {"BCR", "TCR"}, optional
        Restrict to B-cell or T-cell receptors.
    use_polars : bool
        Return a polars DataFrame instead of pandas. Requires polars to be
        installed.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
    """
    log.info("Fetching IEDB receptor export …")
    raw = _fetch_bytes(IEDB_RECEPTOR_URL, timeout=timeout)

    df = _parse_iedb_receptor_zip(raw)
    df = _map_iedb_columns(df)
    df = _normalise_na(df)
    df = _ensure_airr_columns(df)

    # --- filter ---
    if organism_filter:
        mask = df["antigen_organism"].str.contains(
            organism_filter, case=False, na=False
        )
        df = df[mask].reset_index(drop=True)
        log.info(
            "IEDB: %d rows after organism filter '%s'", len(df), organism_filter
        )

    if receptor_type:
        mask = df["receptor_type"].str.upper() == receptor_type.upper()
        df = df[mask].reset_index(drop=True)
        log.info(
            "IEDB: %d rows after receptor_type filter '%s'",
            len(df),
            receptor_type,
        )

    log.info("IEDB: returning %d records", len(df))

    if use_polars:
        return _to_polars(df)
    return df


# ===========================================================================
# VDJdb
# ===========================================================================

# VDJdb flat-file column → AIRR field
_VDJDB_COL_MAP = {
    # Receptor
    "cdr3": "cdr3_aa",
    "cdr3fix.aa": "cdr3_aa",  # alt col name
    "gene": "_gene",  # TRA / TRB / IGH / etc.
    "v.segm": "v_call",
    "d.segm": "d_call",
    "j.segm": "j_call",
    # Antigen
    "antigen.epitope": "antigen_epitope",
    "antigen.gene": "antigen_protein",
    "antigen.species": "antigen_organism",
    # MHC
    "mhc.a": "mhc_allele",
    "mhc.class": "mhc_class",
    # Host
    "species": "subject_species",
    # Provenance
    "reference.id": "pubmed_id",
    "web.method": "_method",  # internal, dropped after
}


def _parse_vdjdb_tsv(raw: bytes) -> pd.DataFrame:
    """Handle zip, gzipped TSV, or plain TSV from VDJdb releases."""
    import gzip as _gzip
    import zipfile

    # zip — prefer vdjdb.txt
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            target = (
                next((n for n in names if n == "vdjdb.txt"), None)
                or next((n for n in names if n == "vdjdb.slim.txt"), None)
                or next(
                    (
                        n
                        for n in names
                        if n.endswith(".tsv") or n.endswith(".txt")
                    ),
                    None,
                )
            )
            return _read_csv_bytes(zf.read(target), sep="\t")
    except zipfile.BadZipFile:
        pass
    # gzip
    try:
        return _read_csv_bytes(_gzip.decompress(raw), sep="\t")
    except OSError:
        pass
    return _read_csv_bytes(raw, sep="\t")


def _map_vdjdb_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower)

    rename = {}
    for src, dst in _VDJDB_COL_MAP.items():
        if src.lower() in df.columns:
            rename[src.lower()] = dst
    df = df.rename(columns=rename)

    # Derive locus from gene column
    if "_gene" in df.columns:
        df["locus"] = df["_gene"].str.strip()
        df["receptor_type"] = df["locus"].apply(_infer_receptor_type)
        df = df.drop(columns=["_gene"], errors="ignore")

    if "_method" in df.columns:
        df = df.drop(columns=["_method"], errors="ignore")

    # junction_aa approximation
    if "junction_aa" not in df.columns and "cdr3_aa" in df.columns:
        df["junction_aa"] = df["cdr3_aa"]

    df["source_db"] = "VDJdb"
    df["sequence_id"] = "VDJdb_" + pd.Series(range(len(df)), dtype=str)
    df["db_record_id"] = df.get("db_record_id", df["sequence_id"])

    return df


def _get_vdjdb_url() -> str:
    """Resolve the latest VDJdb release TSV URL from GitHub."""
    # Try the canonical latest-release asset name pattern
    import json as _json

    api_url = (
        "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"
    )
    req = Request(api_url, headers={"User-Agent": "immune_db_retrieval/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        assets = data.get("assets", [])
        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(".tsv.gz") or name.endswith(".tsv"):
                url = asset["browser_download_url"]
                log.info("Resolved VDJdb release asset: %s", url)
                return url
        # No suitable asset found; fall through to the static fallback
        log.warning("No .tsv/.tsv.gz asset found in latest release.")
    except Exception as exc:
        log.warning("Could not resolve latest VDJdb URL via API: %s", exc)
    # Fallback to the manually specified VDJDB_URL
    return VDJDB_URL


def fetch_vdjdb(
    antigen_species: str | None = None,
    antigen_epitope: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    min_vdjdb_score: int = 0,
    use_polars: bool = False,
    timeout: int = 120,
) -> DataFrame:
    """
    Download the latest VDJdb release and return an AIRR-compatible dataframe.

    Note: VDJdb is primarily a TCR database. BCR records exist but are sparse.

    Parameters
    ----------
    antigen_species : str, optional
        Filter on ``antigen_organism``, e.g. ``"EBV"``, ``"CMV"``,
        ``"SARS-CoV-2"``.
    antigen_epitope : str, optional
        Filter on ``antigen_epitope`` sequence substring.
    receptor_type : {"BCR", "TCR"}, optional
        Restrict to B- or T-cell receptors.
    min_vdjdb_score : int
        Minimum VDJdb confidence score (0–3). Default 0 = include all.
    use_polars : bool
        Return a polars DataFrame.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
    """
    url = _get_vdjdb_url()
    log.info("Fetching VDJdb from %s …", url)
    raw = _fetch_bytes(url, timeout=timeout)

    df = _parse_vdjdb_tsv(raw)
    df = _map_vdjdb_columns(df)
    df = _normalise_na(df)

    # VDJdb score filter
    score_col = next(
        (
            c
            for c in df.columns
            if "score" in c.lower() and "vdjdb" in c.lower()
        ),
        next((c for c in df.columns if c.lower() == "vdjdb.score"), None),
    )
    if score_col and min_vdjdb_score > 0:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        df = df[df[score_col] >= min_vdjdb_score].reset_index(drop=True)
        log.info(
            "VDJdb: %d rows after score filter (>=%d)", len(df), min_vdjdb_score
        )

    df = _ensure_airr_columns(df)

    if antigen_species:
        mask = df["antigen_organism"].str.contains(
            antigen_species, case=False, na=False
        )
        df = df[mask].reset_index(drop=True)
        log.info(
            "VDJdb: %d rows after antigen_species filter '%s'",
            len(df),
            antigen_species,
        )

    if antigen_epitope:
        mask = df["antigen_epitope"].str.contains(
            antigen_epitope, case=False, na=False
        )
        df = df[mask].reset_index(drop=True)
        log.info(
            "VDJdb: %d rows after antigen_epitope filter '%s'",
            len(df),
            antigen_epitope,
        )

    if receptor_type:
        mask = df["receptor_type"].str.upper() == receptor_type.upper()
        df = df[mask].reset_index(drop=True)
        log.info(
            "VDJdb: %d rows after receptor_type filter '%s'",
            len(df),
            receptor_type,
        )

    log.info("VDJdb: returning %d records", len(df))

    if use_polars:
        return _to_polars(df)
    return df


# ===========================================================================
# Combined fetch
# ===========================================================================


def fetch_all(
    organism_filter: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    min_vdjdb_score: int = 0,
    use_polars: bool = False,
    timeout: int = 120,
    sources: list[Literal["iedb", "vdjdb"]] | None = None,
) -> DataFrame:
    """
    Fetch from both IEDB and VDJdb and return a single merged AIRR dataframe.

    Parameters
    ----------
    organism_filter : str, optional
        Applied to both databases (``antigen_organism`` column).
    receptor_type : {"BCR", "TCR"}, optional
        Filter both databases.
    min_vdjdb_score : int
        Passed to ``fetch_vdjdb``.
    use_polars : bool
        Return a polars DataFrame.
    sources : list, optional
        Subset of ["iedb", "vdjdb"]. Defaults to both.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
    """
    if sources is None:
        sources = ["iedb", "vdjdb"]

    frames: list[pd.DataFrame] = []

    if "iedb" in sources:
        try:
            df_iedb = fetch_iedb(
                organism_filter=organism_filter,
                receptor_type=receptor_type,
                use_polars=False,
                timeout=timeout,
            )
            frames.append(df_iedb)
        except Exception as exc:
            log.warning("IEDB fetch failed — skipping: %s", exc)

    if "vdjdb" in sources:
        try:
            df_vdjdb = fetch_vdjdb(
                antigen_species=organism_filter,
                receptor_type=receptor_type,
                min_vdjdb_score=min_vdjdb_score,
                use_polars=False,
                timeout=timeout,
            )
            frames.append(df_vdjdb)
        except Exception as exc:
            log.warning("VDJdb fetch failed — skipping: %s", exc)

    if not frames:
        raise RuntimeError(
            "All data sources failed. Check network connectivity."
        )

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate on CDR3 + V + J + antigen epitope to remove exact cross-DB dups
    dedup_cols = ["cdr3_aa", "v_call", "j_call", "antigen_epitope"]
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=[c for c in dedup_cols if c in merged.columns]
    ).reset_index(drop=True)
    log.info(
        "Merged: %d total rows, %d after deduplication", before, len(merged)
    )

    if use_polars:
        return _to_polars(merged)
    return merged


# ===========================================================================
# Epitope annotation
# ===========================================================================


def _annotate_from_db(
    vdj,
    adata,
    reference: pd.DataFrame,
) -> None:
    """
    Internal helper: match CDR3 sequences from a Dandelion object against a
    pre-fetched database DataFrame and write results into adata.obs.

    Adds the following columns to adata.obs per source database:
        - epitope_{source}          : all matched epitopes, "|"-separated
        - organism_{source}         : all matched organisms, "|"-separated
        - epitope_{source}_primary  : first matched epitope (for clean visualization)
        - organism_{source}_primary : corresponding organism
    If the columns already exist they are overwritten.
    """
    # Extract sequence table from Dandelion (supports LazyFrame / DataFrame / pandas)
    if hasattr(vdj.data, "collect"):
        tmp = vdj.data.collect().to_pandas()
    elif hasattr(vdj.data, "to_pandas"):
        tmp = vdj.data.to_pandas()
    else:
        tmp = vdj.data

    # Match CDR3 against database, renaming source_db to avoid conflict with tmp
    matched = tmp.merge(
        reference[
            [
                "cdr3_aa",
                "antigen_epitope",
                "antigen_organism",
                "antigen_protein",
                "mhc_class",
                "mhc_allele",
                "source_db",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"source_db": "db_source"}),
        left_on="junction_aa",
        right_on="cdr3_aa",
        how="left",
    )

    # Aggregate per cell per source database
    for source, col_suffix in [("VDJdb", "vdjdb"), ("IEDB", "iedb")]:
        sub = matched[matched["db_source"] == source]
        if sub.empty:
            continue
        epi = (
            sub.groupby("cell_id")
            .agg(
                **{
                    f"epitope_{col_suffix}": (
                        "antigen_epitope",
                        lambda x: "|".join(x.dropna().unique()),
                    )
                },
                **{
                    f"organism_{col_suffix}": (
                        "antigen_organism",
                        lambda x: "|".join(x.dropna().unique()),
                    )
                },
            )
            .replace("", float("nan"))
        )

        # Primary columns (first value only) for clean visualization
        epi[f"epitope_{col_suffix}_primary"] = (
            epi[f"epitope_{col_suffix}"].str.split("|").str[0]
        )
        epi[f"organism_{col_suffix}_primary"] = (
            epi[f"organism_{col_suffix}"].str.split("|").str[0]
        )

        # Write into adata.obs, dropping old columns to avoid duplication on re-run
        old_cols = [
            f"epitope_{col_suffix}",
            f"organism_{col_suffix}",
            f"epitope_{col_suffix}_primary",
            f"organism_{col_suffix}_primary",
        ]
        adata.obs = adata.obs.drop(
            columns=[c for c in old_cols if c in adata.obs.columns]
        )
        adata.obs = adata.obs.join(epi, how="left")


def fetch_db(
    method: Literal["vdjdb", "iedb", "both"] = "vdjdb",
    receptor_type: Literal["BCR", "TCR"] | None = "TCR",
    min_vdjdb_score: int = 1,
    antigen_species: str | None = None,
    organism_filter: str | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Download and prepare the reference database **once** and return it as a
    DataFrame that can be reused across many :func:`get_epitope` calls.

    Calling this before a for-loop over samples avoids repeated HTTP downloads:

    Examples
    --------
    >>> db = fetch_db(method="vdjdb", receptor_type="TCR", antigen_species="EBV")
    >>> for sample_vdj, sample_adata in samples:
    ...     get_epitope(sample_vdj, sample_adata, reference=db)

    Parameters mirror :func:`get_epitope`; see its docstring for details.
    """
    method = method.lower()
    if method not in ("vdjdb", "iedb", "both"):
        raise ValueError(
            f"method must be 'vdjdb', 'iedb', or 'both', got '{method}'"
        )

    log.info("fetch_db: method=%s, receptor_type=%s", method, receptor_type)

    if method == "vdjdb":
        return fetch_vdjdb(
            antigen_species=antigen_species,
            receptor_type=receptor_type,
            min_vdjdb_score=min_vdjdb_score,
            timeout=timeout,
        )
    elif method == "iedb":
        return fetch_iedb(
            organism_filter=organism_filter,
            receptor_type=receptor_type,
            timeout=timeout,
        )
    else:  # both
        return pd.concat(
            [
                fetch_vdjdb(
                    antigen_species=antigen_species,
                    receptor_type=receptor_type,
                    min_vdjdb_score=min_vdjdb_score,
                    timeout=timeout,
                ),
                fetch_iedb(
                    organism_filter=organism_filter,
                    receptor_type=receptor_type,
                    timeout=timeout,
                ),
            ],
            ignore_index=True,
        )


def get_epitope(
    vdj,
    adata,
    method: Literal["vdjdb", "iedb", "both"] = "vdjdb",
    receptor_type: Literal["BCR", "TCR"] | None = "TCR",
    min_vdjdb_score: int = 1,
    antigen_species: str | None = None,
    organism_filter: str | None = None,
    timeout: int = 120,
    reference: pd.DataFrame | None = None,
) -> None:
    """
    Query immune receptor databases for epitope information and write results
    into AnnData.obs.

    Parameters
    ----------
    vdj : Dandelion
        Dandelion object after check_contigs / find_clones / transfer processing.
    adata : AnnData
        Gene expression AnnData object after transfer(adata, vdj).
    method : {"vdjdb", "iedb", "both"}
        Database to query.
        - "vdjdb" : VDJdb (recommended for TCR).
        - "iedb"  : IEDB  (recommended for BCR / antibody data).
        - "both"  : query both databases separately and annotate with distinct columns.
        Default is "vdjdb".
    receptor_type : {"BCR", "TCR"}, optional
        Receptor type filter. Default is "TCR".
    min_vdjdb_score : int
        Minimum VDJdb confidence score (0–3). Only used when method is
        "vdjdb" or "both". Default is 1.
    antigen_species : str, optional
        Filter VDJdb by antigen species, e.g. "EBV", "CMV", "SARS-CoV-2".
        Only used when method is "vdjdb" or "both".
    organism_filter : str, optional
        Filter IEDB by antigen organism substring, e.g. "Epstein-Barr",
        "influenza". Only used when method is "iedb" or "both".
    timeout : int
        HTTP request timeout in seconds. Default is 120.
    reference : pandas.DataFrame, optional
        **Pre-fetched** reference database DataFrame returned by
        :func:`fetch_db`.  When supplied, *all* download and filter
        parameters (``method``, ``receptor_type``, ``antigen_species``,
        ``organism_filter``, ``min_vdjdb_score``, ``timeout``) are ignored
        and no HTTP request is made.  Use this to avoid redundant downloads
        when calling get_epitope inside a loop::

            db = fetch_db(method="vdjdb", receptor_type="TCR")
            for vdj, adata in samples:
                get_epitope(vdj, adata, reference=db)

    Returns
    -------
    None
        Results are written directly to adata.obs. Columns added per database:
        - epitope_{db}          : all matched epitopes, "|"-separated
        - organism_{db}         : all matched organisms, "|"-separated
        - epitope_{db}_primary  : first matched epitope (for clean visualization)
        - organism_{db}_primary : organism of first matched epitope
        where {db} is "vdjdb" and/or "iedb" depending on method.

    Examples
    --------
    >>> get_epitope(vdj, adata)
    >>> get_epitope(vdj, adata, antigen_species="EBV")
    >>> get_epitope(vdj, adata, method="iedb", receptor_type="BCR")
    >>> get_epitope(vdj, adata, method="both", receptor_type="TCR")

    # Efficient multi-sample loop — download only once:
    >>> db = fetch_db(method="vdjdb", receptor_type="TCR", antigen_species="EBV")
    >>> for vdj_s, adata_s in sample_pairs:
    ...     get_epitope(vdj_s, adata_s, reference=db)
    """
    # --- If a pre-fetched DataFrame was supplied, skip all downloads ---
    if reference is not None:
        log.info(
            "get_epitope: using pre-fetched reference (%d rows)", len(reference)
        )
        _annotate_from_db(vdj, adata, reference)
        for col_suffix in ["vdjdb", "iedb"]:
            col = f"epitope_{col_suffix}"
            if col in adata.obs.columns:
                n = adata.obs[col].notna().sum()
                print(
                    f"Cells with matched epitope ({col_suffix}): {n} / {adata.n_obs}"
                )
        return

    method = method.lower()
    if method not in ("vdjdb", "iedb", "both"):
        raise ValueError(
            f"method must be 'vdjdb', 'iedb', or 'both', got '{method}'"
        )

    log.info("get_epitope: method=%s, receptor_type=%s", method, receptor_type)

    # Raw bytes are cached inside _fetch_bytes, so even without reference the
    # network is only hit once per session per URL.
    if method == "vdjdb":
        _reference = fetch_vdjdb(
            antigen_species=antigen_species,
            receptor_type=receptor_type,
            min_vdjdb_score=min_vdjdb_score,
            timeout=timeout,
        )

    elif method == "iedb":
        _reference = fetch_iedb(
            organism_filter=organism_filter,
            receptor_type=receptor_type,
            timeout=timeout,
        )

    else:  # both — fetch separately to preserve source_db for per-database annotation
        _reference = pd.concat(
            [
                fetch_vdjdb(
                    antigen_species=antigen_species,
                    receptor_type=receptor_type,
                    min_vdjdb_score=min_vdjdb_score,
                    timeout=timeout,
                ),
                fetch_iedb(
                    organism_filter=organism_filter,
                    receptor_type=receptor_type,
                    timeout=timeout,
                ),
            ],
            ignore_index=True,
        )

    _annotate_from_db(vdj, adata, _reference)

    # Print summary
    for col_suffix in ["vdjdb", "iedb"]:
        col = f"epitope_{col_suffix}"
        if col in adata.obs.columns:
            n = adata.obs[col].notna().sum()
            print(
                f"Cells with matched epitope ({col_suffix}): {n} / {adata.n_obs}"
            )


# ===========================================================================
# Convenience query helpers
# ===========================================================================


def query(
    df: DataFrame,
    organism: str | None = None,
    epitope: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    locus: str | None = None,
    v_gene: str | None = None,
    min_cdr3_len: int | None = None,
) -> DataFrame:
    """
    Filter an already-loaded AIRR dataframe in-memory.

    Works transparently on both pandas and polars DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
    organism : str, optional  — substring match on antigen_organism
    epitope : str, optional   — substring match on antigen_epitope
    receptor_type : str, optional — exact match ("BCR" / "TCR")
    locus : str, optional     — exact IMGT locus (IGH / IGK / IGL / TRA / TRB …)
    v_gene : str, optional    — substring match on v_call
    min_cdr3_len : int, optional — minimum CDR3 amino-acid length

    Returns
    -------
    Same type as input df
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Convert to pandas for uniform filtering, then back
        _df = df.to_pandas()
    else:
        _df = df.copy()

    def _filt(col: str, val: str, exact: bool = False) -> None:
        nonlocal _df
        if col not in _df.columns:
            return
        if exact:
            _df = _df[_df[col].str.upper() == val.upper()]
        else:
            _df = _df[_df[col].str.contains(val, case=False, na=False)]

    if organism:
        _filt("antigen_organism", organism)
    if epitope:
        _filt("antigen_epitope", epitope)
    if receptor_type:
        _filt("receptor_type", receptor_type, exact=True)
    if locus:
        _filt("locus", locus, exact=True)
    if v_gene:
        _filt("v_call", v_gene)
    if min_cdr3_len is not None and "cdr3_aa" in _df.columns:
        _df = _df[_df["cdr3_aa"].str.len() >= min_cdr3_len]

    _df = _df.reset_index(drop=True)

    if is_polars:
        return pl.from_pandas(_df)
    return _df


def to_tsv(df: DataFrame, path: str) -> None:
    """Save an AIRR-compatible dataframe to a TSV file (AIRR standard format)."""
    if isinstance(df, pl.DataFrame):
        df.write_csv(path, separator="\t")
    else:
        df.to_csv(path, sep="\t", index=False)
    log.info("Saved %d rows to %s", len(df), path)
