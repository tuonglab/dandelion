from __future__ import annotations
import h5py
import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from airr import RearrangementSchema
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from subprocess import run
from typing import TypeVar, Literal, Callable

# help silence the dtype warning?
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

F = TypeVar("F", bound=Callable)  # Define a TypeVar for any callable type

RECEPTOR_SET = {"B", "abT", "gdT"}
TRUES = ["T", "t", "True", "true", "TRUE", True, "1"]
FALSES = ["F", "f", "False", "false", "FALSE", False, "0"]
HEAVYLONG = ["IGH", "TRB", "TRD"]
LIGHTSHORT = ["IGK", "IGL", "TRA", "TRG"]
VCALL = "v_call"
JCALL = "j_call"
VCALLG = "v_call_genotyped"
JCALLG = "j_call_genotyped"
STRIPALLELENUM = "[*][0-9][0-9]"
NO_DS = [
    "129S1_SvImJ",
    "AKR_J",
    "A_J",
    "C3H_HeJ",
    "C57BL_6J",
    "BALB_c_ByJ",
    "CBA_J",
    "DBA_1J",
    "DBA_2J",
    "MRL_MpJ",
    "NOR_LtJ",
    "NZB_BlNJ",
    "SJL_J",
]
EMPTIES_STR = [
    "nan",
    "NaN",
    "",
    "None",
    "none",
]
EMPTIES = EMPTIES_STR + [
    None,
    np.nan,
    pd.NA,
]


DEFAULT_PREFIX = "all"
BOOLEAN_LIKE_COLUMNS = ["extra", "ambiguous", "full_length", "complete_vdj"]


class Tree(defaultdict):
    """Create a recursive defaultdict."""

    def __init__(self, value=None) -> None:
        super().__init__(Tree)
        self.value = value


def dict_from_table(meta: pd.DataFrame, columns: tuple[str, str]) -> dict:
    """
    Generate a dictionary from a dataframe.

    Parameters
    ----------
    meta : pd.DataFrame
        pandas data frame or file path
    columns : tuple[str, str]
        column names in data frame

    Returns
    -------
    dict
        dictionary
    """
    if (isinstance(meta, pd.DataFrame)) & (columns is not None):
        meta_ = meta
        if len(columns) == 2:
            sample_dict = dict(zip(meta_[columns[0]], meta_[columns[1]]))
    elif (os.path.isfile(str(meta))) & (columns is not None):
        meta_ = pd.read_csv(meta, sep="\t", dtype="object")
        if len(columns) == 2:
            sample_dict = dict(zip(meta_[columns[0]], meta_[columns[1]]))

    sample_dict = clean_nan_dict(sample_dict)
    return sample_dict


def clean_nan_dict(d: dict) -> dict:
    """
    Remove nan from dictionary.

    Parameters
    ----------
    d : dict
        dictionary

    Returns
    -------
    dict
        dictionary with no NAs.
    """
    return {k: v for k, v in d.items() if v is not np.nan}


def flatten(l: list) -> list:
    """
    Flatten a list-in-list-in-list.

    Parameters
    ----------
    l : list
        a list-in-list list

    Yields
    ------
    list
        a flattened list.
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def makeblastdb(ref: Path | str) -> None:
    """
    Run makeblastdb on constant region fasta file.

    Wrapper for makeblastdb.

    Parameters
    ----------
    ref : Path | str
        constant region fasta file.
    """
    cmd = ["makeblastdb", "-dbtype", "nucl", "-parse_seqids", "-in", str(ref)]
    run(cmd)


def bh(pvalues: np.array) -> np.array:
    """
    Compute the Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.array
        array of p-values to correct

    Returns
    -------
    np.array
        np.array of corrected p-values
    """
    n = int(pvalues.shape[0])
    new_pvalues = np.empty(n)
    values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = n - i
        pvalue, index = vals
        new_values.append((n / rank) * pvalue)
    for i in range(0, int(n) - 1):
        if new_values[i] < new_values[i + 1]:
            new_values[i + 1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        new_pvalues[index] = new_values[i]
    return new_pvalues


def is_categorical(array_like: pd.Series) -> bool:
    """Check if categorical."""
    return array_like.dtype.name == "category"


def type_check(dataframe: pd.DataFrame, key: str) -> bool:
    """Check dtype."""
    return (
        dataframe[key].dtype == str
        or dataframe[key].dtype == object
        or is_categorical(dataframe[key])
        or dataframe[key].dtype == bool
    )


def check_filepath(
    file_or_folder_path: Path | str,
    filename_prefix: str | None = None,
    ends_with: str | None = None,
    sub_dir: str | None = None,
    within_dandelion: bool = True,
) -> Path | None:
    """
    Checks whether file path exists.

    Parameters
    ----------
    file_or_folder_path : Path | str
        either a string or Path object pointing to a file or folder.
    filename_prefix : str | None, optional
        the prefix of the filename.
    ends_with : str | None, optional
        the suffix of the filename. Can be flexible i.e. not just the extension.
    sub_dir : str | None, optional
        the subdirectory to look for the file if specified
    within_dandelion : bool, optional
        whether to look for the file within a 'dandelion' sub folder.

    Returns
    -------
    Path | None
        Path object if file is found, else None.
    """
    filename_pre = (
        DEFAULT_PREFIX if filename_prefix is None else filename_prefix
    )

    ends_with = "" if ends_with is None else ends_with
    input_path = (
        Path(str(file_or_folder_path)).expanduser()
        if str(file_or_folder_path)[0] == "~"
        else Path(str(file_or_folder_path))
    )
    if input_path.is_file() and str(input_path).endswith(ends_with):
        return input_path
    elif input_path.is_dir():
        if within_dandelion:
            for child in input_path.iterdir():
                if child.name[0] != ".":
                    if child.is_dir() and child.name == "dandelion":
                        out_dir = child
                        if sub_dir is not None:
                            out_dir = out_dir / sub_dir
                        for file in out_dir.iterdir():
                            if file.name[0] != ".":
                                if file.is_file() and str(file).endswith(
                                    ends_with
                                ):
                                    if file.name.startswith(
                                        filename_pre + "_contig"
                                    ):
                                        return file
        else:
            if sub_dir is not None:
                input_path = input_path / sub_dir
            for file in input_path.iterdir():
                if file.name[0] != ".":
                    if file.is_file() and str(file).endswith(ends_with):
                        if file.name.startswith(filename_pre + "_contig"):
                            return file
    else:
        return None


def cmp(a, b):
    """Python2.x cmp function."""
    return (a > b) - (a < b)


def cmp_str_emptylast(s1, s2):
    """Help sort empty string to last."""
    if not s1 or not s2:
        return bool(s2) - bool(s1)

    return cmp(s1, s2)


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function."""

    class K:
        """Key class"""

        def __init__(self, obj, *args) -> None:
            self.obj = obj

        def __lt__(self, other) -> bool:
            """Less than."""
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other) -> bool:
            """Greater than."""
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other) -> bool:
            """Equal."""
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other) -> bool:
            """Less than or equal."""
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other) -> bool:
            """Greater than or equal."""
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other) -> bool:
            """Not equal."""
            return mycmp(self.obj, other.obj) != 0

    return K


def not_same_call(a: str, b: str, pattern: str) -> bool:
    """Utility function to check if a == b in terms of pattern."""
    return (re.search(pattern, a) and not re.search(pattern, b)) or (
        re.search(pattern, b) and not re.search(pattern, a)
    )


def same_call(a: str, b: str, c: str, pattern: str) -> bool:
    """Utility function to check if a == b == c in terms of pattern."""
    queries = [a, b, c]
    queries = [q for q in queries if pd.notnull(q)]
    return all([re.search(pattern, x) for x in queries])


def present(x: str | None) -> bool:
    """Utility function to check if x is not null or blank."""
    return pd.notnull(x) and x not in [
        "",
        "None",
        "none",
        "NA",
        "na",
        "NaN",
        "nan",
    ]


def check_missing(x: str | None) -> bool:
    """Utility function to check if x is null or blank."""
    return pd.isnull(x) or x == ""


def all_missing(x: str | None) -> bool:
    """Utility function to check if all x is not null or blank."""
    return all(pd.isnull(x)) or all(x == "")


def all_missing2(x: str | None) -> bool:
    """Utility function to check if all x is not null or blank or the word None."""
    return all(pd.isnull(x)) or all(x == "") or all(x == "None")


def return_mix_dtype(data: pd.DataFrame) -> list:
    """Utility function to return mixed dtypes columns."""
    check = [
        c
        for c in data.columns
        if pd.api.types.infer_dtype(data[c]).startswith("mixed")
    ]
    return check


def get_numpy_dtype(series: pd.Series) -> str:
    """
    Map a Pandas dtype to an appropriate NumPy dtype.

    Parameters
    ----------
    series : pd.Series
        The Pandas Series.

    Returns
    -------
    str
        A string representing the NumPy dtype corresponding to the Pandas dtype.

    Raises
    ------
    TypeError
        If the Pandas dtype is unsupported.
    """
    if pd.api.types.is_integer_dtype(series):
        return "i4"  # 32-bit integer
    elif pd.api.types.is_float_dtype(series):
        return "f8"  # 64-bit float
    elif pd.api.types.is_bool_dtype(series):
        return "i1"  # 8-bit integer for booleans (True/False)
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
        series
    ):
        # Handle object or string columns; dynamically calculate the max string length
        max_length = series.astype(str).map(len).max()
        return "S{}".format(max(1, max_length))  # String with max length
    else:
        raise TypeError(f"Unsupported data type: {series.name}")


def sanitize_data_for_saving(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Quick sanitize dtypes for saving.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        DataFrame and corresponding NumPy structured dtype.
    """
    tmp = data.copy()
    dtype_dict = {}

    for col in tmp:
        if col in RearrangementSchema.properties:
            dtype = RearrangementSchema.properties[col]["type"]
            tmp[col] = sanitize_column(tmp[col], dtype)
        elif col in BOOLEAN_LIKE_COLUMNS:
            dtype = "boolean"
            tmp[col] = sanitize_column(tmp[col], dtype)
        else:
            tmp[col] = try_numeric_conversion(tmp[col])

        dtype_dict[col] = get_numpy_dtype(tmp[col])

        # 🔧 Fix: ensure string columns use Unicode dtype, not ASCII bytes
        if tmp[col].dtype == object or tmp[col].dtype == "string":
            dtype_dict[col] = h5py.string_dtype(encoding="utf-8")

    dtypes = [(key, record) for key, record in dtype_dict.items()]
    return tmp, dtypes


def sanitize_boolean(value: str | bool) -> str:
    """
    Sanitize a boolean-like value to 'T' or 'F'.
    Parameters
    ----------
    value : str | bool
        The value to sanitize.
    Returns
    -------
    str
        'T' for True, 'F' for False, or the original value if not boolean-like.
    """
    if isinstance(value, bool):
        return "T" if value else "F"
    elif isinstance(value, str):
        stripped_value = value.strip().lower()
        if stripped_value in ["true", "t"]:
            return "T"
        elif stripped_value in ["false", "f"]:
            return "F"
    elif isinstance(value, (int, float)):
        if value == 1:
            return "T"
        elif value == 0:
            return "F"
    return value


def sanitize_column(series: pd.Series, dtype: str) -> pd.Series:
    """
    Sanitize a column based on the specified dtype.

    Parameters
    ----------
    series : pd.Series
        The column to be sanitized.
    dtype : str
        The expected data type of the column (`string`, `boolean`, `integer`, or `number`).

    Returns
    -------
    pd.Series
        The sanitized column with replaced values and appropriate data type.
    """
    pd.set_option("future.no_silent_downcasting", True)
    if dtype == "boolean":
        series = series.apply(lambda x: "" if check_missing(x) else x)
        series = series.replace([None, np.nan, "nan", "na", "NaN", ""], "")
        return series.apply(sanitize_boolean)
    elif dtype == "string":
        series = series.apply(lambda x: "" if check_missing(x) else x)
        return (
            series.replace([None, np.nan, "nan", "na", "NaN", ""], "")
            .astype(str)
            .apply(clean_unicode)
        )
    elif dtype in ["number"]:
        series = series.apply(lambda x: np.nan if check_missing(x) else x)
        series = series.replace([None, np.nan, "nan", "na", "NaN", ""], np.nan)
        # for dtype to be float
        return series.astype("float64").fillna(np.nan)
    elif dtype in ["integer"]:
        series = series.apply(lambda x: "" if check_missing(x) else int(x))
        series = series.replace([None, np.nan, "nan", "na", "NaN", ""], "")
        return series.astype(str).apply(clean_unicode)
    return series


def clean_unicode(x: str) -> str:
    """Normalize and ensure valid UTF-8 text."""
    if not isinstance(x, str):
        return ""
    # Normalize to NFKC form (handles Greek/Unicode nicely)
    x = unicodedata.normalize("NFKC", x)
    # Remove invalid or unencodable characters safely
    return x.encode("utf-8", "ignore").decode("utf-8")


def try_numeric_conversion(series: pd.Series) -> pd.Series:
    """
    Attempt to convert a column to numeric, or fallback to treating it as a string.

    Parameters
    ----------
    series : pd.Series
        The column to be converted.

    Returns
    -------
    pd.Series
        The column converted to numeric if possible, or sanitized as a string if not.
    """
    if series.dtype.name == "category":
        series = sanitize_column(series, "string")

    if series.apply(lambda x: isinstance(x, str) and "|" in x).any():
        return sanitize_column(series, "string")
    try:
        return pd.to_numeric(series)
    except:
        return sanitize_column(series, "string")


def sanitize_data(data: pd.DataFrame, ignore: str = "clone_id") -> None:
    """Quick sanitize dtypes."""
    data = data.astype("object")
    data = data.infer_objects()
    for d in data:
        if d in BOOLEAN_LIKE_COLUMNS:
            data[d] = data[d].apply(sanitize_boolean)
        if d in RearrangementSchema.properties:
            if RearrangementSchema.properties[d]["type"] in [
                "string",
                "boolean",
                "integer",
            ]:
                data[d] = data[d].replace(
                    EMPTIES,
                    "",
                )
                if RearrangementSchema.properties[d]["type"] == "integer":
                    data[d] = [
                        int(x) if present(x) else ""
                        for x in pd.to_numeric(data[d])
                    ]
                if RearrangementSchema.properties[d]["type"] == "boolean":
                    data[d] = data[d].apply(sanitize_boolean)
            else:
                data[d] = data[d].replace(
                    EMPTIES,
                    np.nan,
                )
        else:
            if d != ignore:
                try:
                    data[d] = pd.to_numeric(data[d])
                except:
                    data[d] = data[d].replace(
                        to_replace=EMPTIES,
                        value="",
                    )
        if re.search("mu_freq", d):
            data[d] = [
                float(x) if present(x) else np.nan
                for x in pd.to_numeric(data[d])
            ]
        if re.search("mu_count", d):
            data[d] = [
                int(x) if present(x) else "" for x in pd.to_numeric(data[d])
            ]
    if (
        pd.Series(["cell_id", "umi_count", "productive"])
        .isin(data.columns)
        .all()
    ):  # sort so that the productive contig with the largest umi is first
        data.sort_values(
            by=["cell_id", "productive", "umi_count"],
            inplace=True,
            ascending=[True, False, False],
        )

    # check if airr-standards is happy
    validate_airr(data)
    return data


def sanitize_blastn(data: pd.DataFrame) -> None:
    """Quick sanitize dtypes."""
    data = data.astype("object")
    data = data.infer_objects()
    for d in data:
        if d in RearrangementSchema.properties:
            if RearrangementSchema.properties[d]["type"] in [
                "string",
                "boolean",
                "integer",
            ]:
                data[d] = data[d].replace(
                    EMPTIES,
                    "",
                )
                if RearrangementSchema.properties[d]["type"] == "integer":
                    data[d] = [
                        int(x) if present(x) else ""
                        for x in pd.to_numeric(data[d])
                    ]
            else:
                data[d] = data[d].replace(
                    EMPTIES,
                    np.nan,
                )
        else:
            try:
                data[d] = pd.to_numeric(data[d])
            except:
                data[d] = data[d].replace(
                    to_replace=EMPTIES,
                    value="",
                )
    return data


def validate_airr(data: pd.DataFrame) -> None:
    """Validate dtypes in airr table."""
    tmp = data.copy()
    int_columns = []
    for d in tmp:
        try:
            tmp[d].replace(np.nan, pd.NA).astype("Int64")
            int_columns.append(d)
        except:
            pass
    for _, row in tmp.iterrows():
        contig = Contig(row).contig
        for required in [
            "sequence",
            "rev_comp",
            "sequence_alignment",
            "germline_alignment",
            "v_cigar",
            "d_cigar",
            "j_cigar",
        ]:
            if required not in contig:
                contig.update({required: ""})
    RearrangementSchema.validate_header(contig.keys())
    RearrangementSchema.validate_row(contig)


class ContigDict(dict):
    """Class Object to extract the contigs as a dictionary."""

    def __setitem__(self, key: str, value: str) -> None:
        """Standard __setitem__."""
        super().__setitem__(key, value)

    def __hash__(self) -> int:
        """Make it hashable."""
        return hash(tuple(self))


class Contig:
    """Class Object to hold contig."""

    def __init__(self, contig: dict, mapper: dict | None = None) -> None:
        if mapper is not None:
            mapper.update({k: k for k in contig.keys() if k not in mapper})
            self._contig = ContigDict(
                {mapper[key]: vals for (key, vals) in contig.items()}
            )
        else:
            self._contig = ContigDict(contig)
        for key, value in self._contig.items():
            if isinstance(value, float) and np.isnan(value):
                self._contig[key] = ""

    @property
    def contig(self) -> ContigDict:
        """Contig slot."""
        return self._contig


def deprecated(
    details: str, deprecated_in: str, removed_in: str
) -> Callable[[F], F]:
    """Decorator to mark a function as deprecated"""

    def deprecated_decorator(func: F) -> F:
        """Deprecate dectorator"""

        def deprecated_func(*args, **kwargs):
            """Deprecate function"""
            warnings.warn(
                "{} is a deprecated in {} and will be removed in {}."
                " {}".format(func.__name__, deprecated_in, removed_in, details),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def format_call(
    metadata: pd.DataFrame,
    call: str,
    suffix_vdj: str = "_VDJ",
    suffix_vj: str = "_VJ",
) -> list:
    """Extract v/d/j/c call values from data."""
    call_dict = {
        "Multi": "Multi",
        "None": "None",
        "": "None",
        "unassigned": "None",
    }
    if suffix_vj is not None:
        call_1 = {
            x[0]: x[1] if present(x[1]) else "None"
            for x, y in zip(
                metadata[call + suffix_vdj].items(),
                list(metadata[call + suffix_vj]),
            )
        }
        call_2 = {
            x[0]: x[1] if present(x[1]) else "None"
            for x, y in zip(
                metadata[call + suffix_vj].items(),
                list(metadata[call + suffix_vdj]),
            )
        }
        call_2 = {x: y if "|" not in y else "Multi" for x, y in call_2.items()}
        call_4 = {
            x: "Single" if y not in call_dict else call_dict[y]
            for x, y in call_2.items()
        }
    else:
        call_1 = {
            x: y if present(y) else "None"
            for x, y in metadata[call + suffix_vdj].items()
        }
        call_2 = {x: "None" for x in call_1.keys()}
        call_4 = call_3 = call_2
    call_1 = {x: y if "|" not in y else "Multi" for x, y in call_1.items()}
    call_3 = {
        x: "Single" if y not in call_dict else call_dict[y]
        for x, y in call_1.items()
    }
    return (
        list(call_1.values()),
        list(call_2.values()),
        list(call_3.values()),
        list(call_4.values()),
    )


def format_isotype1(metadata: pd.DataFrame) -> list[str]:
    """Quick format isotype."""
    isotype_status = [
        (
            "IgM/IgD"
            if (i == "IgM|IgD") or (i == "IgD|IgM")
            else "Multi" if "|" in i else i
        )
        for i in metadata["isotype"]
    ]
    return isotype_status


def format_isotype2(metadata: pd.DataFrame) -> list[str]:
    """Format isotype status so that if the chain is called "exception" it is allowed to have the special isotype "IgM/IgD"."""
    isotype_status = [
        (x if "exception" in y else ("Multi" if y == "Extra pair" else x))
        for x, y in zip(metadata["isotype_status"], metadata["chain_status"])
    ]
    return isotype_status


def format_locus(
    metadata: pd.DataFrame,
    vcall: str,
    suffix_vdj: str = "_VDJ",
    suffix_vj: str = "_VJ",
    productive_only: bool = True,
) -> pd.Series:
    """Extract locus call value from data."""
    locus_1 = dict(metadata["locus" + suffix_vdj])
    locus_2 = dict(metadata["locus" + suffix_vj])
    constant_1 = dict(metadata["isotype_status"])
    prod_1 = dict(metadata["productive" + suffix_vdj])
    prod_2 = dict(metadata["productive" + suffix_vj])

    # also extract the v/d/j calls
    v_call_1 = dict(metadata[vcall + suffix_vdj])
    j_call_1 = dict(metadata["j_call" + suffix_vdj])
    d_call_1 = dict(metadata["d_call" + suffix_vdj])

    locus_dict = {}
    for i in metadata.index:
        if productive_only:
            loc1 = {
                e: l
                for e, l in enumerate(
                    [
                        ll
                        for ll, p in zip(
                            locus_1[i].split("|"), prod_1[i].split("|")
                        )
                        if p in TRUES
                    ]
                )
            }
            loc2 = {
                e: l
                for e, l in enumerate(
                    [
                        ll
                        for ll, p in zip(
                            locus_2[i].split("|"), prod_2[i].split("|")
                        )
                        if p in TRUES
                    ]
                )
            }
        else:
            loc1 = {
                e: l for e, l in enumerate([ll for ll in locus_1[i].split("|")])
            }
            loc2 = {
                e: l for e, l in enumerate([ll for ll in locus_2[i].split("|")])
            }
        loc1x, loc2x = [], []
        if not all([px == "None" for px in loc1.values()]):
            loc1xx = list(loc1.values())
            loc1x = [ij[:2] for ij in loc1.values()]

        if not all([px == "None" for px in loc2.values()]):
            loc2xx = list(loc2.values())
            loc2x = [ij[:2] for ij in loc2.values()]

        if len(loc1x) > 0:
            if len(list(set(loc1x))) > 1:
                tmp1 = "ambiguous"
                if len(loc2x) > 0:
                    if len(list(set(loc2x))) > 1:
                        tmp2 = "ambiguous"
                    else:
                        if len(loc2x) > 1:
                            if (all(x in ["TRA", "TRG"] for x in loc2xx)) and (
                                len(list(set(loc2xx))) == 2
                            ):
                                tmp2 = "Extra VJ-exception"
                            else:
                                tmp2 = "Extra VJ"
                        else:
                            tmp2 = loc2xx[0]
                else:
                    tmp2 = "None"
            else:
                if len(loc1x) > 1:
                    if constant_1[i] == "IgM/IgD":
                        # for BCR e.g. IgM/IgD, also check that the v/d/j calls are the same
                        v1 = v_call_1[i].split("|")
                        d1 = d_call_1[i].split("|")
                        j1 = j_call_1[i].split("|")
                        if productive_only:
                            v1 = [
                                vv
                                for vv, pp in zip(v1, prod_1[i].split("|"))
                                if pp in TRUES
                            ]
                            d1 = [
                                dd
                                for dd, pp in zip(d1, prod_1[i].split("|"))
                                if pp in TRUES
                            ]
                            j1 = [
                                jj
                                for jj, pp in zip(j1, prod_1[i].split("|"))
                                if pp in TRUES
                            ]
                        same_vdj = True
                        if len(v1) == 2 and len(d1) == 2 and len(j1) == 2:
                            if not (
                                v1[0] == v1[1]
                                and d1[0] == d1[1]
                                and j1[0] == j1[1]
                            ):
                                same_vdj = False
                        else:
                            same_vdj = False
                        if same_vdj:
                            tmp1 = "Extra VDJ-exception"
                        else:
                            tmp1 = "Extra VDJ"
                    elif (all(x in ["TRB", "TRD"] for x in loc1xx)) and (
                        len(list(set(loc1xx))) == 2
                    ):
                        tmp1 = "Extra VDJ-exception"
                    else:
                        tmp1 = "Extra VDJ"
                else:
                    tmp1 = loc1xx[0]

                if len(loc2x) > 0:
                    if len(list(set(loc2x))) > 1:
                        tmp2 = "ambiguous"
                    else:
                        if len(loc2x) > 1:
                            if (all(x in ["TRA", "TRG"] for x in loc2xx)) and (
                                len(list(set(loc2xx))) == 2
                            ):
                                tmp2 = "Extra VJ-exception"
                            else:
                                tmp2 = "Extra VJ"
                        else:
                            tmp2 = loc2xx[0]
                else:
                    tmp2 = "None"

                if (
                    tmp1 not in ["None", "Extra VDJ", "Extra VDJ-exception"]
                ) and (tmp2 not in ["None", "Extra VJ", "Extra VJ-exception"]):
                    if list(set(loc1x)) != list(set(loc2x)):
                        tmp1 = "ambiguous"
                        tmp2 = "ambiguous"
        else:
            tmp1 = "None"
            if len(loc2x) > 0:
                if len(list(set(loc2x))) > 1:
                    tmp2 = "ambiguous"
                else:
                    if len(loc2x) > 1:
                        if (all(x in ["TRA", "TRG"] for x in loc2xx)) and (
                            len(list(set(loc2xx))) == 2
                        ):
                            tmp2 = "Extra VJ-exception"
                        else:
                            tmp2 = "Extra VJ"
                    else:
                        tmp2 = loc2xx[0]
            else:
                tmp2 = "None"
        if any(tmp == "ambiguous" for tmp in [tmp1, tmp2]):
            locus_dict.update({i: "ambiguous"})
        else:
            locus_dict.update({i: tmp1 + " + " + tmp2})

        if any(tmp == "None" for tmp in [tmp1, tmp2]):
            if tmp1 == "None":
                locus_dict.update({i: "Orphan " + tmp2})
            elif tmp2 == "None":
                locus_dict.update({i: "Orphan " + tmp1})
        if any(re.search("No_contig", tmp) for tmp in [tmp1, tmp2]):
            locus_dict.update({i: "No_contig"})
    result = pd.Series(locus_dict)
    return result


def lib_type(lib: str):
    """Dictionary of acceptable loci for library type."""
    librarydict = {
        "tr-ab": ["TRA", "TRB"],
        "tr-gd": ["TRG", "TRD"],
        "ig": ["IGH", "IGK", "IGL"],
    }
    return librarydict[lib]


def movecol(
    df: pd.DataFrame,
    cols_to_move: list = [],
    ref_col: str = "",
) -> pd.DataFrame:
    """A way to order columns."""
    # https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    cols = df.columns.tolist()
    seg1 = cols[: list(cols).index(ref_col) + 1]
    seg2 = cols_to_move

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]


def format_chain_status(locus_status):
    """Format chain status from locus status."""
    chain_status = []
    for ls in locus_status:
        if ("Orphan" in ls) and (re.search("TRB|IGH|TRD|VDJ", ls)):
            if not re.search("exception", ls):
                if re.search("Extra", ls):
                    chain_status.append("Orphan Extra VDJ")
                else:
                    chain_status.append("Orphan VDJ")
            else:
                chain_status.append("Orphan VDJ-exception")
        elif ("Orphan" in ls) and (re.search("TRA|TRG|IGK|IGL|VJ", ls)):
            if not re.search("exception", ls):
                if re.search("Extra", ls):
                    chain_status.append("Orphan Extra VJ")
                else:
                    chain_status.append("Orphan VJ")
            else:
                chain_status.append("Orphan VJ-exception")
        elif re.search("exception|IgM/IgD", ls):
            chain_status.append("Extra pair-exception")
        elif re.search("Extra", ls):
            chain_status.append("Extra pair")
        elif re.search("ambiguous|None", ls):
            chain_status.append("ambiguous")
        else:
            chain_status.append("Single pair")
    return chain_status


def set_germline_env(
    germline: str | None = None,
    org: Literal["human", "mouse"] = "human",
    input_file: Path | str | None = None,
    db: Literal["imgt", "ogrdb"] = "imgt",
) -> tuple[dict[str, str], Path, Path]:
    """
    Set the paths to germline database and environment variables and relevant input files.

    Parameters
    ----------
    germline : str | None, optional
        path to germline database. None defaults to environmental variable $GERMLINE.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    input_file : Path | str | None, optional
        path to input file.
    db : Literal["imgt", "ogrdb"], optional
        database to use. Defaults to imgt.
    Returns
    -------
    tuple[dict[str, str], Path, Path]
        environment dictionary and path to germline database.

    Raises
    ------
    KeyError
        if $GERMLINE environmental variable is not set.
    """
    env = os.environ.copy()
    if germline is None:
        try:
            gml = Path(env["GERMLINE"])
        except KeyError:
            raise KeyError(
                "Environmental variable $GERMLINE is missing. "
                "Please 'export GERMLINE=/path/to/database/germlines/'"
            )
        gml = gml / db / org / "vdj"
    else:
        gml = env["GERMLINE"] = Path(germline)
    if input_file is not None:
        input_file = Path(input_file)
    return env, gml, input_file


def set_igblast_env(
    igblast_db: Path | str | None = None,
    input_file: Path | str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    """
    Set the igblast database and environment variables and relevant input files.

    Parameters
    ----------
    igblast_db : str | None, optional
        path to igblast database. None defaults to environmental variable $IGDATA.
    input_file : Path | str | None, optional
        path to input file.

    Returns
    -------
    tuple[dict[str, str], Path, Path]
        environment dictionary and path to igblast database.

    Raises
    ------
    KeyError
        if $IGDATA environmental variable is not set.
    """
    env = os.environ.copy()
    if igblast_db is None:
        try:
            igdb = Path(env["IGDATA"])
        except KeyError:
            raise KeyError(
                "Environmental variable $IGDATA is missing. "
                "Please 'export IGDATA=/path/to/database/igblast/'"
            )
    else:
        igdb = env["IGDATA"] = Path(igblast_db)
    if input_file is not None:
        input_file = Path(input_file)
    return env, igdb, input_file


def set_blast_env(
    blast_db: str | None = None,
    input_file: Path | str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    """
    Set the blast database and environment variables and relevant input files.

    Parameters
    ----------
    blast_db : str | None, optional
        path to blast database. None defaults to environmental variable $BLASTDB.
    input_file : Path | str | None, optional
        path to input file.
    Returns
    -------
    tuple[dict[str, str], Path, Path]
        environment dictionary and path to igblast database.

    Raises
    ------
    KeyError
        if $BLASTDB environmental variable is not set.
    """
    env = os.environ.copy()
    if blast_db is None:
        try:
            bdb = Path(env["BLASTDB"])
        except KeyError:
            raise KeyError(
                "Environmental variable $BLASTDB is missing. "
                "Please 'export BLASTDB=/path/to/database/blast/'"
            )
    else:
        bdb = env["BLASTDB"] = Path(blast_db)
    if input_file is not None:
        input_file = Path(input_file)
    return env, bdb, input_file


def sum_col(vals: list) -> float | int:
    """Sum columns if not NaN."""
    if all(pd.isnull(vals)):
        return np.nan
    else:
        return sum(vals)


def check_data(
    data: list[Path | str] | Path | str, filename_prefix: list[str] | str | None
) -> tuple[list[str], list[str]]:
    """Quick check for data and filename prefixes"""
    if type(data) is not list:
        data = [data]
    if not isinstance(filename_prefix, list):
        filename_prefix = [filename_prefix]
        if len(filename_prefix) == 1:
            if len(data) > 1:
                filename_prefix = filename_prefix * len(data)
    if all(t is None for t in filename_prefix):
        filename_prefix = [None for d in data]
    return data, filename_prefix


def check_same_celltype(clone_def1: str, clone_def2: str) -> bool:
    """Check if the first key is the same."""
    return clone_def1.split("_", 1)[0] == clone_def2.split("_", 1)[0]


def clear_h5file(filename: Path | str) -> None:
    """Little hack to overwrite an existing h5 file."""
    with h5py.File(filename, "w") as hf:
        for datasetname in hf.keys():
            del hf[datasetname]


def get_vcall_key(data: dict, v_call_key: str) -> str:
    """
    Determine which V-call key to use based on the provided data and key.

    Parameters
    ----------
    data : dict
        The data dictionary containing possible keys.
    v_call_key : str
        The requested key to check (e.g. "v_call" or "v_call_genotyped").

    Returns
    -------
    str
        The best matching V-call key, following this priority:
        1. "v_call_genotyped" if it exists in data and matches v_call_key
        2. "v_call" if it exists in data and matches v_call_key
        3. v_call_key if it exists in data
        4. "v_call" as a default fallback
    """
    if "v_call_genotyped" in data and v_call_key == "v_call_genotyped":
        return "v_call_genotyped"
    elif "v_call" in data and v_call_key == "v_call":
        return "v_call"
    elif v_call_key in data:
        return v_call_key
    else:
        return "v_call"
