"""
Feature engineering for entity-pair matching.
Computes similarity features between S1 and candidate (S2/S3) records.
Optimized for parallel computation using joblib.
"""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import jellyfish
from typing import Optional

from src.matching.normalizer import (
    normalize_name,
    normalize_name_aggressive,
    normalize_address,
    expand_address_abbreviations,
    extract_numbers,
    extract_postal_code,
    tokenize,
    strip_accents,
)


def _jaccard_tokens(tokens_a: list, tokens_b: list) -> float:
    """Jaccard similarity on token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _containment_ratio(tokens_a: list, tokens_b: list) -> float:
    """Containment: |A ∩ B| / min(|A|, |B|). Catches subset relationships."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def _length_ratio(a: str, b: str) -> float:
    """min(len_a, len_b) / max(len_a, len_b)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _number_overlap(nums_a: set, nums_b: set) -> float:
    """Jaccard on numeric tokens."""
    if not nums_a and not nums_b:
        return 1.0
    if not nums_a or not nums_b:
        return 0.0
    intersection = len(nums_a & nums_b)
    union = len(nums_a | nums_b)
    return intersection / union if union > 0 else 0.0


def _soundex_overlap(tokens_a: list, tokens_b: list) -> float:
    """Overlap of Soundex codes for tokens (ASCII tokens only)."""
    def _get_soundex_set(tokens):
        codes = set()
        for t in tokens:
            # Soundex only works on ASCII alphabetic
            cleaned = ''.join(c for c in t if c.isascii() and c.isalpha())
            if len(cleaned) >= 2:
                try:
                    codes.add(jellyfish.soundex(cleaned))
                except Exception:
                    pass
        return codes

    sa = _get_soundex_set(tokens_a)
    sb = _get_soundex_set(tokens_b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def compute_pair_features(
    name1: str, addr1: str, country1: str,
    name2: str, addr2: str, country2: str,
    source_type: int = 0,  # 0 = S2, 1 = S3
) -> np.ndarray:
    """
    Compute all similarity features for a single candidate pair.

    Returns a numpy array of ~20 features.
    """
    # ── Normalize ────────────────────────────────────────────────────────
    n_name1 = normalize_name(name1 or "")
    n_name2 = normalize_name(name2 or "")
    agg_name1 = normalize_name_aggressive(name1 or "")
    agg_name2 = normalize_name_aggressive(name2 or "")

    n_addr1 = normalize_address(addr1 or "")
    n_addr2 = normalize_address(addr2 or "")
    exp_addr1 = normalize_address(expand_address_abbreviations(addr1 or "", country1))
    exp_addr2 = normalize_address(expand_address_abbreviations(addr2 or "", country2))

    # ── Name tokens ──────────────────────────────────────────────────────
    name_tok1 = tokenize(n_name1)
    name_tok2 = tokenize(n_name2)
    agg_tok1 = tokenize(agg_name1)
    agg_tok2 = tokenize(agg_name2)

    # ── Address tokens ───────────────────────────────────────────────────
    addr_tok1 = tokenize(exp_addr1)
    addr_tok2 = tokenize(exp_addr2)

    # ── Name features ────────────────────────────────────────────────────

    # 0: Jaro-Winkler on normalized names
    f_name_jw = JaroWinkler.similarity(n_name1, n_name2) if (n_name1 and n_name2) else 0.0

    # 1: Levenshtein ratio on normalized names
    f_name_lev = fuzz.ratio(n_name1, n_name2) / 100.0 if (n_name1 and n_name2) else 0.0

    # 2: Token Jaccard on name
    f_name_jaccard = _jaccard_tokens(name_tok1, name_tok2)

    # 3: Token-sort ratio (sort tokens, then Levenshtein)
    f_name_token_sort = fuzz.token_sort_ratio(n_name1, n_name2) / 100.0 if (n_name1 and n_name2) else 0.0

    # 4: Token-set ratio (handles subsets)
    f_name_token_set = fuzz.token_set_ratio(n_name1, n_name2) / 100.0 if (n_name1 and n_name2) else 0.0

    # 5: Containment ratio on name tokens
    f_name_containment = _containment_ratio(name_tok1, name_tok2)

    # 6: Exact match after normalization
    f_name_exact = 1.0 if (n_name1 and n_name2 and n_name1 == n_name2) else 0.0

    # 7: Name length ratio
    f_name_len_ratio = _length_ratio(n_name1, n_name2)

    # 8: Aggressive (ASCII-only) Jaro-Winkler
    f_name_agg_jw = JaroWinkler.similarity(agg_name1, agg_name2) if (agg_name1 and agg_name2) else 0.0

    # 9: Soundex token overlap
    f_name_soundex = _soundex_overlap(agg_tok1, agg_tok2)

    # ── Address features ─────────────────────────────────────────────────

    addr1_empty = 1.0 if not n_addr1 else 0.0
    addr2_empty = 1.0 if not n_addr2 else 0.0

    # 10: Address Jaro-Winkler
    f_addr_jw = JaroWinkler.similarity(exp_addr1, exp_addr2) if (exp_addr1 and exp_addr2) else 0.0

    # 11: Address token Jaccard
    f_addr_jaccard = _jaccard_tokens(addr_tok1, addr_tok2)

    # 12: Address token-sort ratio
    f_addr_token_sort = fuzz.token_sort_ratio(exp_addr1, exp_addr2) / 100.0 if (exp_addr1 and exp_addr2) else 0.0

    # 13: Number overlap in addresses
    nums1 = extract_numbers(addr1 or "")
    nums2 = extract_numbers(addr2 or "")
    f_addr_num_overlap = _number_overlap(nums1, nums2)

    # 14: Postal code match
    pc1 = extract_postal_code(addr1 or "", country1)
    pc2 = extract_postal_code(addr2 or "", country2)
    f_postal_match = 1.0 if (pc1 and pc2 and pc1 == pc2) else (0.0 if (pc1 or pc2) else 0.5)

    # 15: Address length ratio
    f_addr_len_ratio = _length_ratio(exp_addr1, exp_addr2)

    # 16: Address containment
    f_addr_containment = _containment_ratio(addr_tok1, addr_tok2)

    # 17: Either address empty flag
    f_addr_any_empty = 1.0 if (addr1_empty or addr2_empty) else 0.0

    # ── Country features ─────────────────────────────────────────────────

    # 18: Country exact match
    c1 = (country1 or "").strip().lower()
    c2 = (country2 or "").strip().lower()
    f_country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    # ── Cross-field features ─────────────────────────────────────────────

    # 19: Source type (S2=0, S3=1)
    f_source = float(source_type)

    # 20: Geometric mean of name_jw * addr_jw
    f_geo_mean = np.sqrt(f_name_jw * f_addr_jw) if (f_name_jw > 0 and f_addr_jw > 0) else 0.0

    # 21: Min of name and address best scores
    f_min_sim = min(f_name_token_sort, f_addr_token_sort)

    # 22: Max of name and address best scores
    f_max_sim = max(f_name_token_sort, f_addr_token_sort)

    return np.array([
        f_name_jw,           # 0
        f_name_lev,          # 1
        f_name_jaccard,      # 2
        f_name_token_sort,   # 3
        f_name_token_set,    # 4
        f_name_containment,  # 5
        f_name_exact,        # 6
        f_name_len_ratio,    # 7
        f_name_agg_jw,       # 8
        f_name_soundex,      # 9
        f_addr_jw,           # 10
        f_addr_jaccard,      # 11
        f_addr_token_sort,   # 12
        f_addr_num_overlap,  # 13
        f_postal_match,      # 14
        f_addr_len_ratio,    # 15
        f_addr_containment,  # 16
        f_addr_any_empty,    # 17
        f_country_match,     # 18
        f_source,            # 19
        f_geo_mean,          # 20
        f_min_sim,           # 21
        f_max_sim,           # 22
    ], dtype=np.float32)


# Feature names for interpretability
FEATURE_NAMES = [
    "name_jaro_winkler",
    "name_levenshtein",
    "name_jaccard",
    "name_token_sort",
    "name_token_set",
    "name_containment",
    "name_exact",
    "name_length_ratio",
    "name_aggressive_jw",
    "name_soundex_overlap",
    "addr_jaro_winkler",
    "addr_jaccard",
    "addr_token_sort",
    "addr_number_overlap",
    "addr_postal_match",
    "addr_length_ratio",
    "addr_containment",
    "addr_any_empty",
    "country_match",
    "source_type",
    "geo_mean_name_addr",
    "min_name_addr_sim",
    "max_name_addr_sim",
]
