"""
loader.py — Shared data-loading layer for the Entity Resolution pipeline.

Provides standardized loading of both *raw* and *preprocessed* TSV files,
with correct separator handling (tab-delimited), entity_id preservation,
safe missing-value treatment, and outputs compatible with the blocking
and matching stages.

Usage examples
--------------
    from src.preprocessing.loader import (
        load_raw_source,
        load_preprocessed_source,
        load_ground_truth,
        preprocess_raw_source,
    )

    # Load a raw TSV → dict keyed by entity_id (raw fields only)
    raw = load_raw_source("dataset/train/train_source1.tsv")

    # Load a preprocessed TSV (15-column clean schema)
    clean = load_preprocessed_source(
        "dataset/preprocessed/train/train_source1_clean.tsv"
    )

    # Load + normalize a raw TSV on-the-fly (no pre-saved clean file)
    normed = preprocess_raw_source("dataset/train/train_source1.tsv")

    # Load ground truth
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import polars as pl

try:
    from .normalizer import preprocess_row
    from .interface import load_entities, DEFAULT_COLUMNS
except ImportError:
    from src.preprocessing.normalizer import preprocess_row
    from src.preprocessing.interface import load_entities, DEFAULT_COLUMNS

log = logging.getLogger("preprocessing.loader")

# Raw source columns — the canonical 4-column schema of unprocessed TSVs.
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


# =====================================================================
#  1. Safe value helpers
# =====================================================================

def _safe_str(value: Any) -> str:
    """Convert a value to string, treating None / NaN / 'null' etc. as ''."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("", "null", "nan", "none", "n/a", "na"):
        return ""
    return s


# =====================================================================
#  2. Raw TSV loading
# =====================================================================

def load_raw_source(
    path: Union[str, Path],
    return_type: str = "dict",
) -> Union[Dict[str, Dict[str, str]], pl.DataFrame]:
    """Load a raw 4-column source TSV (entity_id, business_name,
    business_address, country).

    Parameters
    ----------
    path : str or Path
        Path to the raw TSV file.
    return_type : {"dict", "polars"}, default "dict"
        - "dict": ``{entity_id: {"business_name": ..., "business_address": ..., "country": ...}}``
        - "polars": raw ``pl.DataFrame``

    Returns
    -------
    dict or pl.DataFrame

    Notes
    -----
    * All columns are read as ``Utf8`` to preserve entity_id exactly.
    * Common null literals are normalised to empty strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found: {path}")

    schema_overrides = {c: pl.Utf8 for c in RAW_COLUMNS}

    df = pl.read_csv(
        str(path),
        separator="\t",
        schema_overrides=schema_overrides,
        null_values=["", "null", "NULL", "nan", "NaN", "None", "N/A", "NA"],
        ignore_errors=True,
    )

    # Fill any remaining nulls with empty string
    df = df.with_columns([
        pl.col(c).fill_null("").alias(c) for c in df.columns
    ])

    if return_type == "polars":
        return df

    # Build dict keyed by entity_id
    records: Dict[str, Dict[str, str]] = {}
    id_list = df["entity_id"].to_list()
    name_list = df["business_name"].to_list()
    addr_list = df["business_address"].to_list()
    country_list = df["country"].to_list()

    for i in range(df.height):
        eid = id_list[i]
        records[eid] = {
            "business_name": name_list[i] or "",
            "business_address": addr_list[i] or "",
            "country": country_list[i] or "",
        }

    return records


def load_raw_source_matching(
    path: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    """Load a raw TSV into the format expected by the matching pipeline.

    Returns ``{entity_id: {"name": ..., "addr": ..., "country": ...}}`` —
    matching the key names used in ``matching/pipeline.py::load_source()``.

    This is a **drop-in replacement** for the matching model's
    ``load_source()`` function, but using robust Polars-based loading
    instead of bare ``csv.DictReader``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found: {path}")

    schema_overrides = {c: pl.Utf8 for c in RAW_COLUMNS}

    df = pl.read_csv(
        str(path),
        separator="\t",
        schema_overrides=schema_overrides,
        null_values=["", "null", "NULL", "nan", "NaN", "None", "N/A", "NA"],
        ignore_errors=True,
    )

    df = df.with_columns([
        pl.col(c).fill_null("").alias(c) for c in df.columns
    ])

    records: Dict[str, Dict[str, str]] = {}
    id_list = df["entity_id"].to_list()
    name_list = df["business_name"].to_list()
    addr_list = df["business_address"].to_list()
    country_list = df["country"].to_list()

    for i in range(df.height):
        eid = id_list[i]
        records[eid] = {
            "name": name_list[i] or "",
            "addr": addr_list[i] or "",
            "country": country_list[i] or "",
        }

    return records


# =====================================================================
#  3. Preprocessed TSV loading  (delegates to interface.load_entities)
# =====================================================================

def load_preprocessed_source(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    return_type: str = "dict",
) -> Union[Dict[str, Dict[str, str]], pl.DataFrame]:
    """Load a preprocessed 15-column clean TSV.

    This is a thin wrapper around ``interface.load_entities()`` that adds
    consistent logging and explicit null safety.

    Parameters
    ----------
    path : str or Path
        Path to the preprocessed TSV (e.g. ``*_clean.tsv``).
    columns : list[str] | None
        Columns to load.  None = all 15 standard columns.
    return_type : {"dict", "polars"}
        Output format.

    Returns
    -------
    dict or pl.DataFrame
    """
    path = Path(path)
    log.info(f"Loading preprocessed source: {path.name}")
    result = load_entities(path, columns=columns, return_type=return_type)
    if return_type == "polars":
        # Fill nulls for safety
        result = result.with_columns([
            pl.col(c).fill_null("").alias(c) for c in result.columns
        ])
    log.info(
        f"  Loaded {'DataFrame' if return_type == 'polars' else 'dict'} "
        f"from {path.name}"
    )
    return result


# =====================================================================
#  4. On-the-fly preprocessing  (raw TSV → normalized dict)
# =====================================================================

def preprocess_raw_source(
    path: Union[str, Path],
    return_type: str = "dict",
) -> Union[Dict[str, Dict[str, str]], pl.DataFrame]:
    """Load a raw TSV, normalize every row, and return the enriched result.

    This is the *online* equivalent of ``run_preprocess.py``, useful when
    a preprocessed file does not exist yet.  Normalization reuses
    ``normalizer.preprocess_row()`` — no duplication.

    Parameters
    ----------
    path : str or Path
        Path to the raw source TSV.
    return_type : {"dict", "polars"}
        Output format.

    Returns
    -------
    dict[str, dict] or pl.DataFrame
        Full 15-column normalized records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found: {path}")

    log.info(f"Loading + normalizing raw source: {path.name}")

    df = pl.read_csv(
        str(path),
        separator="\t",
        schema_overrides={c: pl.Utf8 for c in RAW_COLUMNS},
        null_values=["", "null", "NULL", "nan", "NaN", "None", "N/A", "NA"],
        ignore_errors=True,
    )

    df = df.with_columns([
        pl.col(c).fill_null("").alias(c) for c in df.columns
    ])

    ids = df["entity_id"].to_list()
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()

    rows: list[dict[str, str]] = []
    for i in range(df.height):
        row = preprocess_row(
            entity_id=ids[i] or "",
            business_name=names[i] or "",
            business_address=addrs[i] or "",
            country=countries[i] or "",
        )
        rows.append(row)

    if return_type == "polars":
        out_cols = DEFAULT_COLUMNS
        return pl.DataFrame(rows, schema={c: pl.Utf8 for c in out_cols})

    # dict keyed by entity_id
    records: Dict[str, Dict[str, str]] = {}
    for row in rows:
        eid = row.pop("entity_id")
        records[eid] = row
    return records


# =====================================================================
#  5. Ground truth loading
# =====================================================================

def load_ground_truth(
    path: Union[str, Path],
) -> Dict[str, Set[str]]:
    """Load ground truth TSV → ``{s1_entity_id: set(matched_ids)}``.

    Handles both column name variants used across the codebase:
    ``source1_entity_id`` / ``entity_id`` and
    ``matched_entity_ids`` / ``matches``.

    Parameters
    ----------
    path : str or Path
        Path to the ground truth TSV.

    Returns
    -------
    dict[str, set[str]]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    log.info(f"Loading ground truth: {path.name}")

    gt: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = (
                row.get("source1_entity_id")
                or row.get("entity_id")
                or ""
            ).strip()
            if not s1_id:
                continue

            matches_str = (
                row.get("matched_entity_ids")
                or row.get("matches")
                or ""
            ).strip()

            if matches_str:
                matched = {
                    m.strip()
                    for m in matches_str.split(",")
                    if m.strip()
                }
            else:
                matched = set()

            gt[s1_id] = matched

    log.info(f"  Loaded {len(gt):,} S1 entities from ground truth")
    return gt
