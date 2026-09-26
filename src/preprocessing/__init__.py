"""Data preprocessing and normalization module for business entities."""

from .normalizer import (
    clean_business_name,
    clean_address,
    preprocess_row,
    transliterate,
    normalize_conjunctions,
    strip_accents,
    collapse_whitespace,
)
from .interface import load_entities, get_blocking_keys
from .loader import (
    load_raw_source,
    load_raw_source_matching,
    load_preprocessed_source,
    preprocess_raw_source,
    load_ground_truth,
)

__all__ = [
    # normalizer
    "clean_business_name",
    "clean_address",
    "preprocess_row",
    "transliterate",
    "normalize_conjunctions",
    "strip_accents",
    "collapse_whitespace",
    # interface
    "load_entities",
    "get_blocking_keys",
    # loader
    "load_raw_source",
    "load_raw_source_matching",
    "load_preprocessed_source",
    "preprocess_raw_source",
    "load_ground_truth",
]
