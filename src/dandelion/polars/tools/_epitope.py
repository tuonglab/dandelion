from __future__ import annotations

import io
import json
import logging
import zipfile

import pandas as pd
import polars as pl

from anndata import AnnData
from typing import Literal
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request

from dandelion.polars.core._core import DandelionPolars

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IEDB_RECEPTOR_URL = (
    "https://www.iedb.org/downloader.php?file_name=doc/receptor_full_v3.zip"
)
VDJDB_URL = "https://github.com/antigenomics/vdjdb-db/releases/download/2026-05-16/vdjdb-2026-05-16.zip"
VDJDB_FALLBACK_URL = "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest-version.txt"
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
_CACHE: dict[str, bytes] = {}
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
_VALID_LOCI = {"IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD"}
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


def _fetch_bytes(url: str, timeout: int = 120) -> bytes:
    """
    Download a URL and return raw bytes with a descriptive error on failure.

    Results are cached in ``_CACHE`` for the lifetime of the Python session,
    so the same URL is never downloaded twice in one process.

    Parameters
    ----------
    url : str
        The URL to download.
    timeout : int
        Request timeout in seconds. Default is 120.

    Returns
    -------
    bytes
        The raw response body.

    Raises
    ------
    RuntimeError
        If the request fails with an HTTP error status or a network-level
        error (DNS failure, connection refused, timeout, etc.).
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
    """
    Parse CSV/TSV bytes into a pandas DataFrame.

    All columns are read as strings (``dtype=str``) rather than inferring
    types, since these are external biological databases whose values
    (gene calls, sequences, IDs) should never be silently coerced to
    numbers/booleans; numeric normalisation happens later where needed
    (e.g. the VDJdb score column).

    Parameters
    ----------
    raw : bytes
        Raw CSV/TSV file content.
    sep : str, optional
        Field delimiter. Default ``","``.
    skip_rows : int, optional
        Number of leading rows to skip before the header. Default 0.
    encoding : str, optional
        Text encoding of the file. Default ``"utf-8"``.

    Returns
    -------
    pd.DataFrame
        All columns as strings.
    """
    return pd.read_csv(
        io.BytesIO(raw),
        sep=sep,
        skiprows=skip_rows,
        encoding=encoding,
        low_memory=False,
        dtype=str,  # read everything as string first — we normalise later
    )


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert a pandas DataFrame to a polars DataFrame.

    Thin wrapper around ``pl.from_pandas`` used at the end of the public
    ``fetch_*`` functions when ``use_polars=True``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to convert.

    Returns
    -------
    pl.DataFrame
        The same data as a polars DataFrame.
    """
    return pl.from_pandas(df)


def _ensure_airr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing AIRR columns as empty strings so output is always
    schema-consistent, and order columns with the AIRR core fields first.

    Every dataframe returned by ``_fetch_iedb``/``_fetch_vdjdb`` is guaranteed
    to contain every column in :data:`AIRR_CORE_FIELDS`, even if the source
    database didn't provide a value for it — callers can rely on the column
    existing without checking first.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that has already been mapped to (a subset of) AIRR column
        names.

    Returns
    -------
    pd.DataFrame
        Same data, with any missing :data:`AIRR_CORE_FIELDS` columns added
        (filled with ``""``) and columns reordered so the AIRR core fields
        come first, followed by any extra/source-specific columns.
    """
    for col in AIRR_CORE_FIELDS:
        if col not in df.columns:
            df[col] = ""
    return df[
        AIRR_CORE_FIELDS + [c for c in df.columns if c not in AIRR_CORE_FIELDS]
    ]


def _normalise_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace IEDB / VDJdb sentinel "missing value" strings with empty
    strings, matching the AIRR convention of using ``""`` (not ``NaN`` or
    the string ``"N/A"``) for absent values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of string columns straight out of a source parser.

    Returns
    -------
    pd.DataFrame
        Same shape, with ``"N/A"``, ``"n/a"``, ``"NA"``, ``"None"``, and
        ``"nan"`` values replaced by ``""``.
    """
    return df.replace({"N/A": "", "n/a": "", "NA": "", "None": "", "nan": ""})


def _infer_locus(chain_str: str) -> str:
    """
    Map a free-text chain label to its IMGT locus code.

    Parameters
    ----------
    chain_str : str
        A chain label as found in source data, e.g. ``"alpha"``, ``"beta"``,
        ``"heavy"``, ``"kappa"``, ``"a"``, ``"b"``. Matching is
        case-insensitive and whitespace-trimmed via :data:`_CHAIN_TO_LOCUS`.

    Returns
    -------
    str
        The corresponding IMGT locus code (e.g. ``"TRA"``, ``"IGH"``), or
        ``""`` if ``chain_str`` isn't a string or isn't a recognised label.
    """
    if not isinstance(chain_str, str):
        return ""
    return _CHAIN_TO_LOCUS.get(chain_str.strip().lower(), "")


def _infer_receptor_type(locus: str) -> str:
    """
    Classify an IMGT locus code as a B-cell or T-cell receptor.

    Parameters
    ----------
    locus : str
        An IMGT locus code, e.g. ``"IGH"``, ``"TRB"``.

    Returns
    -------
    str
        ``"BCR"`` if ``locus`` starts with ``"IG"``, ``"TCR"`` if it starts
        with ``"TR"``, otherwise ``""``.
    """
    if locus.startswith("IG"):
        return "BCR"
    if locus.startswith("TR"):
        return "TCR"
    return ""


def _normalise_chain(chain: str | list[str] | None) -> list[str] | None:
    """
    Validate and normalise the `chain` argument accepted by the public
    ``fetch_*``, ``fetch_db``, and ``get_epitope`` functions.

    Parameters
    ----------
    chain : str, list of str, or None
        A single IMGT locus code (e.g. ``"TRA"``) or a list of them (e.g.
        ``["TRA", "TRB"]``). Case-insensitive; leading/trailing whitespace
        is stripped. ``None`` means "no chain filter".

    Returns
    -------
    list[str] | None
        The normalised, upper-cased list of locus codes, or ``None`` if
        ``chain`` was ``None``.

    Raises
    ------
    ValueError
        If any entry in ``chain`` isn't one of :data:`_VALID_LOCI` (IGH,
        IGK, IGL, TRA, TRB, TRG, TRD).
    """
    if chain is None:
        return None
    if isinstance(chain, str):
        chain = [chain]
    normalised = [c.strip().upper() for c in chain]
    invalid = [c for c in normalised if c not in _VALID_LOCI]
    if invalid:
        raise ValueError(
            f"Invalid chain(s) {invalid}. Must be one or more of "
            f"{sorted(_VALID_LOCI)}."
        )
    return normalised


def _filter_by_chain(
    df: pd.DataFrame, chains: list[str] | None
) -> pd.DataFrame:
    """
    Filter a dataframe to rows whose `locus` column is in `chains`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``locus`` column (AIRR-mapped dataframes always do,
        via :func:`_ensure_airr_columns`).
    chains : list[str] | None
        Normalised locus codes from :func:`_normalise_chain`. If ``None``,
        ``df`` is returned unchanged.

    Returns
    -------
    pd.DataFrame
        Rows where ``locus`` (case-insensitively) is one of ``chains``,
        with the index reset. Unchanged ``df`` if ``chains`` is ``None``.

    Raises
    ------
    KeyError
        If ``chains`` is not ``None`` and ``df`` has no ``locus`` column.
    """
    if chains is None:
        return df
    if "locus" not in df.columns:
        raise KeyError(
            "Cannot apply chain filter: dataframe has no 'locus' column."
        )
    mask = df["locus"].astype(str).str.upper().isin(chains)
    return df[mask].reset_index(drop=True)


def _parse_iedb_receptor_zip(raw_zip: bytes) -> pd.DataFrame:
    """
    Unzip and parse the IEDB receptor_full_v3 export.

    IEDB's receptor CSV export ships with a duplicated/annotation header
    row: row 0 of the CSV body holds the real column names, and pandas'
    default header row (row 0 of the file) is a coarser grouping label.
    This function reads the file, promotes row 0 of the body to the real
    header (deduplicating any repeated names by appending a positional
    suffix), and drops that row from the data.

    Parameters
    ----------
    raw_zip : bytes
        Raw bytes of the downloaded IEDB receptor_full_v3.zip archive.

    Returns
    -------
    pd.DataFrame
        The parsed CSV contents with corrected column headers, still using
        IEDB's own (non-AIRR) column names — see :func:`_map_iedb_columns`
        for the AIRR rename step.

    Raises
    ------
    ValueError
        If the zip archive contains no ``.csv`` file.
    """
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
    """
    Rename IEDB columns to AIRR field names and derive AIRR-only fields.

    Handles several IEDB-specific quirks in one pass:

    - Renames columns via :data:`_IEDB_RECEPTOR_COL_MAP` (case/whitespace
      insensitive), keeping the first match if several source columns map
      to the same AIRR name.
    - Splits combined ``alphabeta`` / ``gammadelta`` receptor records into
      two separate rows (one per chain), copying over the corresponding
      chain-2 CDR3/V/J/CDR1/CDR2/protein-sequence columns.
    - Falls back to the *calculated* CDR3 if the *curated* CDR3 is entirely
      empty.
    - Derives ``locus`` and ``receptor_type`` from the chain-type column via
      :func:`_infer_locus` / :func:`_infer_receptor_type`.
    - Fills ``junction_aa`` from ``cdr3_aa`` when a true junction sequence
      isn't available (IEDB's curated CDR3 doesn't always include the
      conserved anchor residues AIRR's ``junction`` field expects).
    - Sets ``source_db`` to ``"IEDB"`` and builds a ``sequence_id`` prefixed
      with ``"IEDB_"``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`_parse_iedb_receptor_zip`, still using IEDB's
        native column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with AIRR-named columns (not yet passed through
        :func:`_ensure_airr_columns`).
    """
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


def _fetch_iedb(
    organism_filter: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    chain: str | list[str] | None = None,
    use_polars: bool = False,
    timeout: int = 120,
) -> pd.DataFrame | pl.DataFrame:
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
    chain : str or list of str, optional
        Restrict to specific IMGT locus/loci, e.g. ``"TRA"`` or
        ``["TRA", "TRB"]``. Valid values: IGH, IGK, IGL, TRA, TRB, TRG, TRD.
    use_polars : bool
        Return a polars DataFrame instead of pandas. Requires polars to be
        installed.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        AIRR-compatible dataframe of IEDB receptor records, filtered by the
        provided arguments. Column names are standardised to AIRR field
        names, with missing values as empty strings (``""``).
    """
    chains = _normalise_chain(chain)

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

    if chains:
        df = _filter_by_chain(df, chains)
        log.info("IEDB: %d rows after chain filter %s", len(df), chains)

    log.info("IEDB: returning %d records", len(df))

    if use_polars:
        return _to_polars(df)
    return df


def _parse_vdjdb_tsv(raw: bytes) -> pd.DataFrame:
    """
    Parse a VDJdb release download of unknown container format.

    VDJdb releases have shipped as a zip archive (preferring
    ``vdjdb.txt``/``vdjdb.slim.txt``, falling back to any other member
    whose filename contains ``"vdjdb"`` and ends in ``.tsv``/``.txt``), a
    gzip-compressed TSV, and a plain TSV, depending on the release. This
    tries each in turn and returns the first that parses.

    The desired member is matched by **basename**, not full path — recent
    VDJdb releases nest every file under a ``vdjdb-<date>/`` directory
    inside the zip, so a full-path equality check (e.g.
    ``name == "vdjdb.txt"``) never matches and silently falls through to
    picking the wrong file (VDJdb also ships unrelated ``.txt`` files in
    the same archive, such as ``cluster_members.txt``, which parses as a
    valid TSV but has none of the expected columns — an old bug that
    produced an all-empty ``cdr3_aa`` with no error or warning).

    Parameters
    ----------
    raw : bytes
        Raw bytes of the downloaded VDJdb release asset.

    Returns
    -------
    pd.DataFrame
        The parsed tab-separated contents, still using VDJdb's own
        (non-AIRR) column names — see :func:`_map_vdjdb_columns` for the
        AIRR rename step.

    Raises
    ------
    ValueError
        If the payload is a zip archive but no member's basename matches
        ``vdjdb.txt``, ``vdjdb.slim.txt``, or ``*vdjdb*.tsv``/``*vdjdb*.txt``.
    """
    import gzip
    import posixpath

    # zip — prefer vdjdb.txt
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            target = (
                next(
                    (n for n in names if posixpath.basename(n) == "vdjdb.txt"),
                    None,
                )
                or next(
                    (
                        n
                        for n in names
                        if posixpath.basename(n) == "vdjdb.slim.txt"
                    ),
                    None,
                )
                or next(
                    (
                        n
                        for n in names
                        if "vdjdb" in posixpath.basename(n).lower()
                        and (n.endswith(".tsv") or n.endswith(".txt"))
                    ),
                    None,
                )
            )
            if target is None:
                raise ValueError(
                    "Could not find a vdjdb.txt / vdjdb.slim.txt / "
                    "*vdjdb*.tsv member inside the VDJdb release zip. "
                    f"Archive contents: {names}"
                )
            log.info("Parsing VDJdb file: %s", target)
            return _read_csv_bytes(zf.read(target), sep="\t")
    except zipfile.BadZipFile:
        pass
    # gzip
    try:
        return _read_csv_bytes(gzip.decompress(raw), sep="\t")
    except OSError:
        pass
    return _read_csv_bytes(raw, sep="\t")


def _map_vdjdb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename VDJdb columns to AIRR field names and derive AIRR-only fields.

    Renames columns via :data:`_VDJDB_COL_MAP` (case-insensitive), derives
    ``locus``/``receptor_type`` from VDJdb's ``gene`` column (which already
    holds an IMGT-style code such as ``TRA``/``TRB``, so no lookup table is
    needed here unlike the IEDB path), drops VDJdb's internal ``web.method``
    column, fills ``junction_aa`` from ``cdr3_aa`` when absent, and sets
    ``source_db`` / ``sequence_id`` / ``db_record_id``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`_parse_vdjdb_tsv`, still using VDJdb's native
        column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with AIRR-named columns (not yet passed through
        :func:`_ensure_airr_columns`).
    """
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
    """
    Resolve the latest VDJdb release TSV/TSV.GZ (or zip) asset URL, with a
    three-tier fallback so a single point of failure doesn't leave
    :func:`_fetch_vdjdb` stuck on a stale, hardcoded release.

    Tier 1 — GitHub Releases API: queries the ``antigenomics/vdjdb-db``
    repo's ``/releases/latest`` endpoint and returns the first release
    asset ending in ``.tsv`` or ``.tsv.gz``. This is the freshest and most
    structured source, but can fail from rate limiting, network errors, or
    the release having no direct TSV asset.

    Tier 2 — ``latest-version.txt``: if tier 1 fails or finds no suitable
    asset, fetches :data:`VDJDB_FALLBACK_URL`, a changelog file in the
    vdjdb-db repo listing every release's full download URL, one per line,
    **newest first**. The first line is used. This tracks new releases
    without needing this package to be updated, unlike a hardcoded URL.

    Tier 3 — pinned :data:`VDJDB_URL`: if both of the above fail (e.g. no
    network access to GitHub at all), falls back to the release pinned in
    this module, so :func:`_fetch_vdjdb` still works, just potentially
    against an older release.

    Returns
    -------
    str
        A URL to a VDJdb release asset — from the GitHub API (tier 1), from
        ``latest-version.txt`` (tier 2), or the module-level
        :data:`VDJDB_URL` (tier 3).
    """
    # --- Tier 1: GitHub Releases API ---
    api_url = (
        "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"
    )
    req = Request(api_url, headers={"User-Agent": "immune_db_retrieval/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        assets = data.get("assets", [])
        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(".tsv.gz") or name.endswith(".tsv"):
                url = asset["browser_download_url"]
                log.info("Resolved VDJdb release asset via GitHub API: %s", url)
                return url
        # No suitable asset found; fall through to tier 2
        log.warning("No .tsv/.tsv.gz asset found in latest release via API.")
    except Exception as exc:
        log.warning("Could not resolve latest VDJdb URL via API: %s", exc)

    # --- Tier 2: latest-version.txt (newest-first changelog of URLs) ---
    try:
        req = Request(
            VDJDB_FALLBACK_URL,
            headers={"User-Agent": "immune_db_retrieval/1.0"},
        )
        with urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        if first_line.startswith("http"):
            log.info(
                "Resolved VDJdb release URL via latest-version.txt: %s",
                first_line,
            )
            return first_line
        log.warning(
            "latest-version.txt did not contain a usable URL as its first line."
        )
    except Exception as exc:
        log.warning("Could not resolve latest VDJdb URL via changelog: %s", exc)

    # --- Tier 3: pinned fallback ---
    log.warning("Falling back to pinned VDJDB_URL: %s", VDJDB_URL)
    return VDJDB_URL


def _fetch_vdjdb(
    antigen_species: str | None = None,
    antigen_epitope: str | None = None,
    receptor_type: Literal["BCR", "TCR"] | None = None,
    chain: str | list[str] | None = None,
    min_vdjdb_score: int = 0,
    use_polars: bool = False,
    timeout: int = 120,
) -> pd.DataFrame | pl.DataFrame:
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
    chain : str or list of str, optional
        Restrict to specific IMGT locus/loci, e.g. ``"TRB"`` or
        ``["TRA", "TRB"]``. Valid values: IGH, IGK, IGL, TRA, TRB, TRG, TRD.
    min_vdjdb_score : int
        Minimum VDJdb confidence score (0–3). Default 0 = include all.
    use_polars : bool
        Return a polars DataFrame.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        AIRR-compatible dataframe of VDJdb records, filtered by the provided
        arguments. Column names are standardised to AIRR field names, with
        missing values as empty strings (``""``).
    """
    chains = _normalise_chain(chain)

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

    if chains:
        df = _filter_by_chain(df, chains)
        log.info("VDJdb: %d rows after chain filter %s", len(df), chains)

    log.info("VDJdb: returning %d records", len(df))

    if use_polars:
        return _to_polars(df)
    return df


def _as_pandas(
    obj: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Coerce a pandas / polars / polars-lazy object to a pandas DataFrame,
    remembering its original type so it can be restored later.

    Dandelion's ``vdj.data`` and ``vdj._data`` may be a pandas DataFrame, a
    polars DataFrame, or a polars LazyFrame depending on version/config.
    This normalises any of those to pandas for uniform manipulation.

    Parameters
    ----------
    obj : pd.DataFrame | pl.DataFrame | pl.LazyFrame
        The object to coerce.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The data as a pandas DataFrame, and a ``kind`` tag — one of
        ``"lazy"`` (was a polars LazyFrame), ``"polars"`` (was a polars
        DataFrame), or ``"pandas"`` (was already pandas) — to be passed to
        :func:`_restore_type` to convert back.
    """
    if hasattr(obj, "collect"):
        return obj.collect().to_pandas(), "lazy"
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas(), "polars"
    return obj, "pandas"


def _restore_type(
    pdf: pd.DataFrame, kind: str
) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    """
    Convert a pandas DataFrame back to the type indicated by `kind`.

    Inverse of :func:`_as_pandas` — used after mutating a table extracted
    with ``_as_pandas`` so it can be written back onto the original object
    (e.g. ``vdj._data``) in whatever type it started as.

    Parameters
    ----------
    pdf : pd.DataFrame
        The (possibly mutated) pandas DataFrame.
    kind : str
        One of ``"lazy"``, ``"polars"``, or ``"pandas"``, as returned by
        :func:`_as_pandas`.

    Returns
    -------
    pd.DataFrame | pl.DataFrame | pl.LazyFrame
        ``pdf`` converted to a polars LazyFrame if ``kind == "lazy"``, a
        polars DataFrame if ``kind == "polars"``, or returned unchanged if
        ``kind == "pandas"``.
    """
    if kind == "lazy":
        return pl.from_pandas(pdf).lazy()
    if kind == "polars":
        return pl.from_pandas(pdf)
    return pdf


def _write_cell_level_columns(
    target_frame: pd.DataFrame, source: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """
    Drop-then-join columns onto a cell_id-indexed frame.

    Used to mirror onto ``adata.obs`` whatever new columns
    ``vdj.update_metadata`` added to ``vdj._metadata``. Dropping existing
    columns first (rather than a plain join) makes this idempotent —
    calling :func:`get_epitope` again with different filters overwrites
    rather than duplicates the annotation columns.

    Parameters
    ----------
    target_frame : pd.DataFrame
        The frame to annotate (``adata.obs``), indexed by ``cell_id``.
    source : pd.DataFrame
        Per-cell columns to write, indexed by ``cell_id``.
    cols : list[str]
        Names of the columns being written (used to drop any stale
        versions of these columns before joining).

    Returns
    -------
    pd.DataFrame
        ``target_frame`` with ``cols`` dropped (if present) and ``source``
        left-joined on the index.
    """
    target_frame = target_frame.drop(
        columns=[c for c in cols if c in target_frame.columns]
    )
    return target_frame.join(source[cols], how="left")


def _annotate_from_db(
    vdj: DandelionPolars,
    adata: AnnData,
    reference: pd.DataFrame,
    chain: str | list[str] | None = None,
) -> None:
    """
    Internal helper: match CDR3 sequences from a Dandelion object against a
    pre-fetched database DataFrame and write results into ``vdj._data``,
    ``vdj._metadata``, and ``adata.obs``.

    The two levels are populated by two different mechanisms, deliberately:

    - ``vdj._data`` (per-contig, i.e. one row per chain) is written
      directly by this function: each contig's own CDR3 match(es) are
      aggregated by ``sequence_id`` and merged in, so a cell's TRA contig
      and TRB contig each carry only their own match — no cross-chain
      bleed.
    - ``vdj._metadata`` (per-cell) is **not** built with a naive
      ``groupby("cell_id")`` over ``vdj._data``, because that would merge
      a cell's TRA and TRB (or IGH and IGL) matches into one
      undifferentiated string, discarding which chain each match came
      from. Instead, once the contig-level columns are written to
      ``vdj._data``, this delegates to Dandelion's own
      ``vdj.update_metadata(retrieve=[...])``, whose internal chain
      splitting groups contigs into VDJ (IGH/TRB/TRD) vs VJ
      (IGK/IGL/TRA/TRG) *before* joining/aggregating per cell — producing
      separate ``{column}_VDJ`` / ``{column}_VJ`` columns in
      ``vdj._metadata`` that correctly keep e.g. a TRA epitope match and a
      TRB epitope match on the same cell distinct. ``adata.obs`` then
      mirrors whatever new columns that call added.

    Adds the following columns per source database:
        Contig-level (``vdj._data``):
            - epitope_{source}, organism_{source}, mhc_class_{source},
              mhc_allele_{source}, epitope_{source}_primary,
              organism_{source}_primary — that single contig's own match(es).
        Cell-level (``vdj._metadata`` / ``adata.obs``), named by Dandelion's
        own ``update_metadata`` splitting convention:
            - epitope_{source}_VDJ / epitope_{source}_VJ
            - organism_{source}_VDJ / organism_{source}_VJ
            - mhc_class_{source}_VDJ / mhc_class_{source}_VJ
            - mhc_allele_{source}_VDJ / mhc_allele_{source}_VJ
        (The contig-level ``_primary`` convenience columns are not passed
        through to the cell level, since "primary" only has an
        unambiguous meaning for a single contig — at the cell level the
        VDJ/VJ split already narrows things to (usually) one chain, and
        further collapsing risks re-introducing the same cross-contig
        ambiguity this function exists to avoid.)
    If the columns already exist they are overwritten.

    Note that Dandelion's ``update_metadata`` defaults to
    ``productive_only=True`` — a match on a non-productive contig still
    appears in ``vdj._data``, but is excluded from the ``vdj._metadata`` /
    ``adata.obs`` cell-level aggregation, matching Dandelion's own
    convention for what counts as a cell's "real" receptor.

    Parameters
    ----------
    vdj : DandelionPolars
        The Dandelion object to annotate. Must have ``vdj.data`` and
        ``vdj._data`` populated with AIRR-compliant columns, including
        ``junction_aa`` (or ``cdr3_aa``) and ``sequence_id``.
    adata : AnnData
        The AnnData object to annotate. Must have ``adata.obs`` populated
        with a ``cell_id`` column that matches the ``cell_id`` in
        ``vdj._metadata``.
    reference : pd.DataFrame
        The reference epitope database to match against, already filtered
        to the desired source(s) (e.g. VDJdb, IEDB)
        and with AIRR-compliant column names (see :func:`_fetch_vdjdb` and
        :func:`_fetch_iedb`).
    chain : str | list[str] | None, optional
        Restrict matching to specific IMGT locus/loci (e.g. "TRA", or
        ["TRA", "TRB"]). Both the query contigs (from vdj.data) and the
        reference database are filtered to this chain before matching, so
        annotations never cross chains (e.g. a TRB CDR3 coincidentally
        matching a TRA-only reference epitope).

    Raises
    ------
    KeyError
        If ``vdj.data`` has no ``locus`` column when ``chain`` is set, or
        if ``vdj._data`` has no ``sequence_id`` column.
    """
    chains = _normalise_chain(chain)

    # Extract sequence table from Dandelion (supports LazyFrame / DataFrame / pandas)
    tmp, _ = _as_pandas(vdj.data)

    if chains is not None:
        if "locus" not in tmp.columns:
            raise KeyError(
                "vdj.data has no 'locus' column — cannot restrict annotation "
                f"to chain={chain!r}."
            )
        tmp = tmp[tmp["locus"].astype(str).str.upper().isin(chains)].copy()
        reference = _filter_by_chain(reference, chains)

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

    vdj_data_current, vdj_data_kind = _as_pandas(vdj._data)
    if "sequence_id" not in vdj_data_current.columns:
        raise KeyError(
            "vdj._data has no 'sequence_id' column — cannot align "
            "contig-level epitope annotations back onto it."
        )

    # Columns to hand to vdj.update_metadata's retrieve= — deliberately
    # excludes the "_primary" convenience columns; see docstring.
    retrieve_cols: list[str] = []

    for source, col_suffix in [("VDJdb", "vdjdb"), ("IEDB", "iedb")]:
        sub = matched[matched["db_source"] == source]
        if sub.empty:
            continue

        # --- Per-contig aggregation → vdj._data ---
        contig_epi = (
            sub.groupby("sequence_id")
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
                **{
                    f"mhc_class_{col_suffix}": (
                        "mhc_class",
                        lambda x: "|".join(x.dropna().unique()),
                    )
                },
                **{
                    f"mhc_allele_{col_suffix}": (
                        "mhc_allele",
                        lambda x: "|".join(x.dropna().unique()),
                    )
                },
            )
            .replace("", float("nan"))
        )
        contig_epi[f"epitope_{col_suffix}_primary"] = (
            contig_epi[f"epitope_{col_suffix}"].str.split("|").str[0]
        )
        contig_epi[f"organism_{col_suffix}_primary"] = (
            contig_epi[f"organism_{col_suffix}"].str.split("|").str[0]
        )

        base_cols = [
            f"epitope_{col_suffix}",
            f"organism_{col_suffix}",
            f"mhc_class_{col_suffix}",
            f"mhc_allele_{col_suffix}",
        ]
        contig_level_cols = base_cols + [
            f"epitope_{col_suffix}_primary",
            f"organism_{col_suffix}_primary",
        ]
        retrieve_cols.extend(base_cols)

        vdj_data_current = vdj_data_current.drop(
            columns=[
                c for c in contig_level_cols if c in vdj_data_current.columns
            ]
        )
        vdj_data_current = vdj_data_current.merge(
            contig_epi.reset_index(), on="sequence_id", how="left"
        )

    # Write vdj._data back once, in its original type (pandas / polars / lazy)
    vdj._data = _restore_type(vdj_data_current, vdj_data_kind)

    if not retrieve_cols:
        # No matches for any source db against this chain/reference — nothing
        # to propagate up to vdj._metadata / adata.obs.
        return

    if not hasattr(vdj, "update_metadata"):
        log.warning(
            "vdj has no update_metadata method (unexpected Dandelion "
            "object) — vdj._data was annotated, but vdj._metadata and "
            "adata.obs were not."
        )
        return

    # Snapshot vdj._metadata's columns before delegating to Dandelion's own
    # locus-aware (VDJ vs VJ) aggregation, so we can identify exactly which
    # columns it added regardless of its internal naming/suffix convention.
    if vdj._metadata is not None:
        meta_before, _ = _as_pandas(vdj._metadata)
        cols_before = set(meta_before.columns)
    else:
        cols_before = set()

    try:
        vdj.update_metadata(
            retrieve=retrieve_cols,
            split=True,
            join=True,
            unique=True,
            reinitialize=False,
        )
    except TypeError:
        # Older/newer Dandelion version with a different update_metadata
        # signature — fall back to the one argument every version accepts.
        vdj.update_metadata(retrieve=retrieve_cols, reinitialize=False)

    meta_after, _ = _as_pandas(vdj._metadata)
    new_cols = [
        c for c in meta_after.columns if c not in cols_before and c != "cell_id"
    ]
    if not new_cols:
        log.warning(
            "vdj.update_metadata did not add any new columns for %s",
            retrieve_cols,
        )
        return

    meta_after = meta_after.set_index("cell_id")
    adata.obs = _write_cell_level_columns(adata.obs, meta_after, new_cols)


def _print_epitope_summary(adata: AnnData, col_suffix: str) -> None:
    """
    Print how many cells got an epitope match for one source database.

    ``_annotate_from_db`` (via ``vdj.update_metadata``) adds columns named
    ``epitope_{col_suffix}_VDJ`` / ``epitope_{col_suffix}_VJ`` to
    ``adata.obs`` rather than a single flat ``epitope_{col_suffix}``
    column, so this reports each chain group separately.

    Parameters
    ----------
    adata : AnnData
        The annotated AnnData object.
    col_suffix : str
        ``"vdjdb"`` or ``"iedb"``.
    """
    matched_cols = [
        c
        for c in adata.obs.columns
        if c.startswith(f"epitope_{col_suffix}")
        and (c.endswith("_VDJ") or c.endswith("_VJ"))
    ]
    for col in sorted(matched_cols):
        n = adata.obs[col].notna().sum()
        print(f"Cells with matched epitope in {col}: {n} / {adata.n_obs}")


def fetch_db(
    database: Literal["vdjdb", "iedb", "both"] = "vdjdb",
    receptor_type: Literal["BCR", "TCR"] | None = "TCR",
    chain: str | list[str] | None = None,
    min_vdjdb_score: int = 1,
    antigen_species: str | None = None,
    organism_filter: str | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Download and prepare the reference database **once** and return it as a
    DataFrame that can be reused across many :func:`get_epitope` calls.

    Calling this before a for-loop over samples avoids repeated HTTP
    downloads (the underlying bytes are also cached by :func:`_fetch_bytes`,
    but ``fetch_db`` additionally avoids repeating the parsing/filtering
    work on every iteration).

    Parameters
    ----------
    database : Literal["vdjdb", "iedb", "both"], optional
        Database to query.
        - "vdjdb" : VDJdb (recommended for TCR).
        - "iedb"  : IEDB  (recommended for BCR / antibody data).
        - "both"  : concatenate both databases into a single reference.
        Default is "vdjdb".
    receptor_type : Literal["BCR", "TCR"], optional
        Restrict to B- or T-cell receptors. Default is "TCR".
    chain : str | list[str] | None, optional
        Restrict to specific IMGT locus/loci, e.g. ``"TRA"`` or
        ``["TRA", "TRB"]``. Valid values: IGH, IGK, IGL, TRA, TRB, TRG, TRD.
    min_vdjdb_score : int, optional
        Minimum VDJdb confidence score (0–3). Only used when database is
        "vdjdb" or "both". Default is 1.
    antigen_species : str, optional
        Filter VDJdb by antigen species, e.g. "EBV", "CMV", "SARS-CoV-2".
        Only used when database is "vdjdb" or "both".
    organism_filter : str, optional
        Filter IEDB by antigen organism substring, e.g. "Epstein-Barr",
        "influenza". Only used when database is "iedb" or "both".
    timeout : int
        HTTP request timeout in seconds. Default is 120.

    Returns
    -------
    pd.DataFrame
        AIRR-compatible reference dataframe, ready to pass as the
        ``reference`` argument to :func:`get_epitope`.

    Examples
    --------
    >>> db = fetch_db(database="vdjdb", receptor_type="TCR", antigen_species="EBV")
    >>> for sample_vdj, sample_adata in samples:
    ...     get_epitope(sample_vdj, sample_adata, reference=db)
    """
    database = database.lower()
    if database not in ("vdjdb", "iedb", "both"):
        raise ValueError(
            f"database must be 'vdjdb', 'iedb', or 'both', got '{database}'"
        )

    log.info(
        "fetch_db: database=%s, receptor_type=%s, chain=%s",
        database,
        receptor_type,
        chain,
    )

    if database == "vdjdb":
        return _fetch_vdjdb(
            antigen_species=antigen_species,
            receptor_type=receptor_type,
            chain=chain,
            min_vdjdb_score=min_vdjdb_score,
            timeout=timeout,
        )
    elif database == "iedb":
        return _fetch_iedb(
            organism_filter=organism_filter,
            receptor_type=receptor_type,
            chain=chain,
            timeout=timeout,
        )
    else:  # both
        return pd.concat(
            [
                _fetch_vdjdb(
                    antigen_species=antigen_species,
                    receptor_type=receptor_type,
                    chain=chain,
                    min_vdjdb_score=min_vdjdb_score,
                    timeout=timeout,
                ),
                _fetch_iedb(
                    organism_filter=organism_filter,
                    receptor_type=receptor_type,
                    chain=chain,
                    timeout=timeout,
                ),
            ],
            ignore_index=True,
        )


def get_epitope(
    vdj: DandelionPolars,
    adata: AnnData,
    database: Literal["vdjdb", "iedb", "both"] = "vdjdb",
    receptor_type: Literal["BCR", "TCR"] | None = "TCR",
    chain: str | list[str] | None = None,
    min_vdjdb_score: int = 1,
    antigen_species: str | None = None,
    organism_filter: str | None = None,
    timeout: int = 120,
    reference: pd.DataFrame | None = None,
) -> None:
    """
    Query immune receptor databases for epitope information and write results
    into adata.obs, vdj._metadata, and vdj._data.

    Parameters
    ----------
    vdj : DandelionPolars
        Dandelion object after check_contigs / find_clones / transfer processing.
    adata : AnnData
        Gene expression AnnData object after transfer(adata, vdj).
    database : Literal["vdjdb", "iedb", "both"], optional
        Database to query.
        - "vdjdb" : VDJdb (recommended for TCR).
        - "iedb"  : IEDB  (recommended for BCR / antibody data).
        - "both"  : query both databases separately and annotate with distinct columns.
        Default is "vdjdb".
    receptor_type : Literal["BCR", "TCR"] | None, optional
        Receptor type filter. Default is "TCR".
    chain : str | list[str] | None, optional
        Restrict matching to specific IMGT locus/loci, e.g. ``"TRA"`` or
        ``["TRA", "TRB"]``. Valid values: IGH, IGK, IGL, TRA, TRB, TRG, TRD.
        When set, only contigs of that chain (in vdj.data) are matched
        against reference entries of that same chain — so annotations on
        vdj._data never bleed across chains, and per-cell aggregates in
        adata.obs / vdj._metadata only reflect the requested chain(s).
    min_vdjdb_score : int, optional
        Minimum VDJdb confidence score (0–3). Only used when database is
        "vdjdb" or "both". Default is 1.
    antigen_species : str | None, optional
        Filter VDJdb by antigen species, e.g. "EBV", "CMV", "SARS-CoV-2".
        Only used when database is "vdjdb" or "both".
    organism_filter : str | None, optional
        Filter IEDB by antigen organism substring, e.g. "Epstein-Barr",
        "influenza". Only used when database is "iedb" or "both".
    timeout : int, optional
        HTTP request timeout in seconds. Default is 120.
    reference : pd.DataFrame | None, optional
        **Pre-fetched** reference database DataFrame returned by
        :func:`fetch_db`.  When supplied, *all* download and filter
        parameters (``database``, ``receptor_type``, ``antigen_species``,
        ``organism_filter``, ``min_vdjdb_score``, ``timeout``) are ignored
        and no HTTP request is made. ``chain`` still applies — it's used at
        annotation time regardless of how ``reference`` was produced. Use
        this to avoid redundant downloads when calling get_epitope inside a
        loop::

            db = fetch_db(database="vdjdb", receptor_type="TCR")
            for vdj, adata in samples:
                get_epitope(vdj, adata, reference=db)

    Returns
    -------
    None
        Results are written to three places:
        - vdj._data : per-contig (chain-specific) match columns —
          epitope_{db}, organism_{db}, mhc_class_{db}, mhc_allele_{db},
          epitope_{db}_primary, organism_{db}_primary.
        - vdj._metadata : per-cell match columns, produced by delegating to
          Dandelion's own ``vdj.update_metadata(retrieve=...)``, which
          splits contigs into VDJ (IGH/TRB/TRD) and VJ (IGK/IGL/TRA/TRG)
          groups before aggregating — giving epitope_{db}_VDJ /
          epitope_{db}_VJ (and the organism_/mhc_class_/mhc_allele_
          equivalents), rather than one flattened column that would mix a
          cell's different chains together.
        - adata.obs : mirrors whatever new columns that call added to
          vdj._metadata.
        where {db} is "vdjdb" and/or "iedb" depending on database.

    Examples
    --------
    >>> get_epitope(vdj, adata)
    >>> get_epitope(vdj, adata, antigen_species="EBV")
    >>> get_epitope(vdj, adata, database="iedb", receptor_type="BCR")
    >>> get_epitope(vdj, adata, database="both", receptor_type="TCR")
    >>> get_epitope(vdj, adata, chain="TRB")            # beta chain only
    >>> get_epitope(vdj, adata, chain=["TRA", "TRB"])   # alpha + beta

    # Efficient multi-sample loop — download only once:
    >>> db = fetch_db(database="vdjdb", receptor_type="TCR", antigen_species="EBV")
    >>> for vdj_s, adata_s in sample_pairs:
    ...     get_epitope(vdj_s, adata_s, reference=db)
    """
    # --- If a pre-fetched DataFrame was supplied, skip all downloads ---
    if reference is not None:
        log.info(
            "get_epitope: using pre-fetched reference (%d rows), chain=%s",
            len(reference),
            chain,
        )
        _annotate_from_db(vdj, adata, reference, chain=chain)
        for col_suffix in ["vdjdb", "iedb"]:
            _print_epitope_summary(adata, col_suffix)
        return

    database = database.lower()
    if database not in ("vdjdb", "iedb", "both"):
        raise ValueError(
            f"database must be 'vdjdb', 'iedb', or 'both', got '{database}'"
        )

    log.info(
        "get_epitope: database=%s, receptor_type=%s, chain=%s",
        database,
        receptor_type,
        chain,
    )

    # Raw bytes are cached inside _fetch_bytes, so even without reference the
    # network is only hit once per session per URL.
    if database == "vdjdb":
        _reference = _fetch_vdjdb(
            antigen_species=antigen_species,
            receptor_type=receptor_type,
            chain=chain,
            min_vdjdb_score=min_vdjdb_score,
            timeout=timeout,
        )

    elif database == "iedb":
        _reference = _fetch_iedb(
            organism_filter=organism_filter,
            receptor_type=receptor_type,
            chain=chain,
            timeout=timeout,
        )

    else:  # both — fetch separately to preserve source_db for per-database annotation
        _reference = pd.concat(
            [
                _fetch_vdjdb(
                    antigen_species=antigen_species,
                    receptor_type=receptor_type,
                    chain=chain,
                    min_vdjdb_score=min_vdjdb_score,
                    timeout=timeout,
                ),
                _fetch_iedb(
                    organism_filter=organism_filter,
                    receptor_type=receptor_type,
                    chain=chain,
                    timeout=timeout,
                ),
            ],
            ignore_index=True,
        )

    _annotate_from_db(vdj, adata, _reference, chain=chain)

    # Print summary
    for col_suffix in ["vdjdb", "iedb"]:
        _print_epitope_summary(adata, col_suffix)
