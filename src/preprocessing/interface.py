"""
interface.py — Shared interface for downstream agents (Blocking & Matching).

Provides fast, standardized loading of preprocessed entity records and
extracts blocking keys / matching attributes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

log = logging.getLogger("preprocessing.interface")

DEFAULT_COLUMNS = [
    "entity_id",
    "country",
    "business_name",
    "business_address",
    "name_original",
    "name_clean",
    "name_translit",
    "name_core",
    "legal_suffix",
    "addr_original",
    "addr_clean",
    "addr_translit",
    "addr_city",
    "addr_state",
    "addr_postal_code",
]


def load_entities(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    return_type: str = "dict",
) -> Union[Dict[str, Dict[str, str]], pl.DataFrame]:
    """Load preprocessed entities from a clean TSV file.

    Parameters
    ----------
    path : str or Path
        Path to the preprocessed TSV file (e.g. `train_source1_clean.tsv`).
    columns : list of str, optional
        Subset of columns to load. If None, loads all standard columns.
    return_type : {"dict", "polars"}, default "dict"
        - "dict": Returns `{entity_id: {col: val, ...}}` (optimized for matching model lookup).
        - "polars": Returns the raw `pl.DataFrame` (optimized for vectorised blocking).

    Returns
    -------
    dict or pl.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {path}")

    cols = columns or DEFAULT_COLUMNS
    schema_overrides = {c: pl.Utf8 for c in cols}

    df = pl.read_csv(
        str(path),
        separator="\t",
        columns=cols,
        schema_overrides=schema_overrides,
        null_values=["", "null", "NULL", "nan", "NaN", "None"],
        ignore_errors=True,
    )

    if return_type == "polars":
        return df

    # Convert to Python dict keyed by entity_id
    records: Dict[str, Dict[str, str]] = {}
    id_list = df["entity_id"].to_list()
    col_data = {col: df[col].to_list() for col in cols if col != "entity_id"}

    n_rows = df.height
    for i in range(n_rows):
        eid = id_list[i]
        rec = {col: (col_data[col][i] or "") for col in col_data}
        records[eid] = rec

    return records


def get_blocking_keys(entity: Dict[str, str]) -> Dict[str, str]:
    """Generate high-utility candidate blocking keys for an entity.

    Intended for Agent 2 (Blocking/Candidate Generation):
    - `postal_code`: Extracted 5-digit ZIP / 6-digit PIN / French postal code
    - `city_state`: Combined canonical city and state (e.g. 'mumbai_maharashtra')
    - `name_first_token`: First significant word from `name_core`
    - `country`: Lowercased country identifier

    Parameters
    ----------
    entity : dict
        A record dictionary from `load_entities()` or `preprocess_row()`.

    Returns
    -------
    dict of str -> str
    """
    country = (entity.get("country") or "").strip().lower()
    postal = (entity.get("addr_postal_code") or "").strip()
    city = (entity.get("addr_city") or "").strip().lower()
    state = (entity.get("addr_state") or "").strip().lower()

    # City-state block
    city_state = f"{city}_{state}" if (city and state) else ""

    # Name first token from core name
    name_core = (entity.get("name_core") or entity.get("name_clean") or "").strip().lower()
    first_token = name_core.split()[0] if name_core else ""

    return {
        "country": country,
        "postal_code": postal,
        "city_state": city_state,
        "name_first_token": first_token,
    }
