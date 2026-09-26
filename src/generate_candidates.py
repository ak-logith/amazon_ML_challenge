#!/usr/bin/env python3
"""
Entity Resolution — Candidate Generation / Blocking Pipeline (V4)
==================================================================
High-Recall Multi-Channel Inverted-Index Blocking Engine with:
- Global Candidate Pooling (Zero Early Quota Choking)
- Priority-Weighted Global Scoring (Rare Name & Composite Dominate Noise)
- True Character 3-Gram Fallback for Low-Candidate Entities (<40)
- Script / Transliteration Resilience (Open-Set ASCII + Original Script)
- Proportional Source 2 / Source 3 Balancing Without Artificial Inflation
- Open-Set Country Discovery (US, India, France, Germany, Arbitrary)
- Dynamic Elastic Final Budgeting [BUDGET_MIN=50, BUDGET_MAX=600]

Designed for memory-constrained environments (<500 MB peak RAM)
and maximum pair completeness (Recall >= 95%).

Usage:
------
    python src/generate_candidates.py validate --data-dir "path/to/dataset"   # Evaluate on full train GT
    python src/generate_candidates.py validate --sample-s1 1000              # Diagnostic small-sample run
    python src/generate_candidates.py generate --data-dir "path/to/dataset"   # Generate candidates for test
"""

import argparse
import csv
import gc
import heapq
import os
import re
import sys
import time
import unicodedata
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s):
        return s

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ───────────────────────────── Paths & Configuration ─────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "output"


def _find_dataset_root(p: Path) -> Optional[Path]:
    """Check if `p` or any of its standard child directories contains train/test dataset splits."""
    if not p.exists():
        return None
    # 1. Direct folder containing 'train' (and optionally 'test')
    if (p / "train").is_dir():
        return p
    # 2. Direct folder containing 'dataset/train'
    if (p / "dataset" / "train").is_dir():
        return p / "dataset"
    # 3. Known challenge nested directory structures
    nested_patterns = [
        p / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
        p / "student_resource" / "dataset",
    ]
    for n in nested_patterns:
        if (n / "train").is_dir():
            return n
    return None


def resolve_dataset_dir(user_path: Optional[str] = None) -> Path:
    """
    Resolve the root dataset directory containing 'train' and 'test' subdirectories.

    Precedence:
    1. Explicit user_path passed via --data-dir CLI argument
    2. Environment variables: AMAZON_ML_DATA_DIR, DATASET_DIR
    3. Auto-discovery relative to repository root
    """
    if user_path:
        p = Path(user_path).resolve()
        found = _find_dataset_root(p)
        if found:
            return found
        return p

    # Check environment variables
    for env_var in ("AMAZON_ML_DATA_DIR", "DATASET_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            found = _find_dataset_root(Path(env_val).resolve())
            if found:
                return found

    # Auto-discovery relative to repo root
    candidates = [
        REPO_ROOT / "dataset",
        REPO_ROOT / "data",
        REPO_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
        REPO_ROOT.parent / "dataset",
        REPO_ROOT.parent / "student_resource" / "dataset",
    ]
    for cand in candidates:
        found = _find_dataset_root(cand)
        if found:
            return found

    return REPO_ROOT / "dataset"


def get_dataset_paths(data_dir: Path, mode: str) -> dict[str, Path]:
    """
    Validate that required files for `mode` ('validate' or 'generate') exist.
    Raises FileNotFoundError with a clear informative message if any are missing.
    """
    split = "train" if mode == "validate" else "test"
    split_dir = data_dir / split
    if not split_dir.is_dir():
        if (data_dir / f"{split}_source1.tsv").exists():
            split_dir = data_dir

    if mode == "validate":
        required = {
            "s1": split_dir / "train_source1.tsv",
            "s2": split_dir / "train_source2.tsv",
            "s3": split_dir / "train_source3.tsv",
            "gt": split_dir / "train_ground_truth.tsv",
        }
    else:
        required = {
            "s1": split_dir / "test_source1.tsv",
            "s2": split_dir / "test_source2.tsv",
            "s3": split_dir / "test_source3.tsv",
        }

    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        missing_list = "\n".join(f"    - {name}: {required[name]}" for name in missing)
        raise FileNotFoundError(
            f"\n[ERROR] Missing required dataset files for mode '{mode}':\n"
            f"{missing_list}\n"
            f"  Resolved dataset directory: {data_dir}\n"
            f"  Expected location of '{split}' split: {split_dir}\n"
            f"Please specify the correct dataset path using:\n"
            f"  python src/generate_candidates.py {mode} --data-dir <path_to_dataset_root>\n"
            f"Or set the AMAZON_ML_DATA_DIR environment variable."
        )
    return required


# ── Configurable Blocking & Indexing Hyper-parameters ──
NAME_FREQ_CAP       = 6_000   # Drop name tokens appearing > N times per country
ADDR_FREQ_CAP       = 3_000   # Drop generic addr tokens appearing > N times per country
PREFIX_FREQ_CAP     = 8_000   # Drop prefixes appearing > N times per country
POSTAL_FREQ_CAP     = 4_000   # Drop postal/numeric tokens appearing > N times per country
C3_FREQ_CAP         = 12_000  # Drop character 3-grams appearing > N times per country

RARE_TOKEN_THRESH   = 250     # Tokens with document frequency <= N are marked "rare"
MIN_TOKEN_LEN       = 3       # Minimum token length for blocking keys
PREFIX_LEN          = 5       # Chars used for prefix blocking keys

# ── Dynamic Candidate Budget & Fallback Settings (V4) ──
# No early channel quota choking! These constants control global budgeting.
BUDGET_MIN          = 50      # Target minimum candidate floor
BUDGET_MAX          = 600     # Dynamic candidate cap for dense metropolitan entities
FALLBACK_TRIGGER    = 40      # If total primary candidates < N, activate true 3-gram fallback
FALLBACK_MAX_ADD    = 50      # Maximum candidate expansion from 3-gram fallback

# ── Scoring Weights for Global Ranking (Prioritize High-Specificity Evidence) ──
W_RARE_NAME         = 16      # High priority: exact rare name-token agreement
W_NAME              = 6       # Standard name token agreement
W_COMPOSITE_BONUS   = 14      # High priority: name + address intersection
W_POSTAL_NUM        = 6       # Postal code / PIN / building number match
W_ADDR              = 2       # Generic address token agreement (cannot overpower name)
W_PREFIX            = 3       # 5-char prefix agreement
W_3GRAM             = 4       # Score per shared character 3-gram in fallback

# ── Legal Suffixes (Multi-Lingual: US, India, France, EU, Global) ──
SUFFIXES = frozenset({
    # English / Global
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "pvt", "private", "co", "company", "llp", "plc", "gmbh", "ag",
    "the", "of", "and", "for", "group", "holdings", "enterprise", "enterprises",
    "com", "www", "http", "https", "org", "net",
    # French / European
    "sa", "sas", "sarl", "sci", "eurl", "bv", "nv", "pty", "pte",
    "groupe", "societe", "association", "ets", "cie", "sasu", "snc", "scs",
    "ste", "direction", "service", "services",
})

# ── Address Stop-Words (Multi-Lingual: US, India, France, Global) ──
ADDR_STOPS = frozenset({
    # Common English / US / Global
    "near", "behind", "opp", "opposite", "beside", "above", "below",
    "floor", "fl", "block", "shop", "flat", "unit", "apt", "apartment",
    "no", "number", "plot", "house", "building", "bldg", "tower", "room",
    "road", "rd", "street", "st", "avenue", "ave", "drive", "dr",
    "lane", "ln", "boulevard", "blvd", "highway", "hwy", "way", "path",
    "place", "plaza", "square", "sq", "court", "ct", "circle", "cir",
    "po", "box", "null", "none", "na", "nil", "unknown",
    "north", "south", "east", "west", "new", "old", "central",
    # India specific
    "nagar", "colony", "marg", "sector", "phase", "zone", "area",
    "district", "dist", "tehsil", "taluk", "mandal", "ward", "chowk",
    "bazaar", "complex", "enclave", "vihar", "puram", "gram",
    # France / French specific
    "ville", "rue", "place", "chemin", "allee", "impasse", "boulevard",
    "avenue", "route", "rte", "quai", "cours", "passage", "square",
    "cedex", "bp", "cs", "immeuble", "batiment", "bat", "porte",
    "residence", "res", "lotissement", "zone", "zi", "za", "zac",
})

# ───────────────────────── Text Pre-processing ───────────────────────────

_PUNCT = re.compile(r"[^\w\s]")
_WS    = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def _norm(text: str) -> str:
    """Lower-case, normalize unicode (NFKD), strip diacritics, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return _WS.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def _translit_clean(text: str) -> str:
    """Transliterate text to clean ASCII (handles Indic Devanagari/Tamil, Greek, etc.)."""
    if not text:
        return ""
    return _norm(unidecode(text))


def _name_tokens(name: str) -> list[str]:
    """Substantive name tokens for blocking (suffixes & short/numeric words removed)."""
    tokens = [
        t for t in _norm(name).split()
        if t not in SUFFIXES and len(t) >= MIN_TOKEN_LEN and not t.isdigit()
    ]
    # Add transliterated tokens if original contains non-ASCII characters
    t_name = _translit_clean(name)
    if t_name and t_name != _norm(name):
        t_tokens = [
            t for t in t_name.split()
            if t not in SUFFIXES and len(t) >= MIN_TOKEN_LEN and not t.isdigit()
        ]
        tokens = list(dict.fromkeys(tokens + t_tokens))
    return tokens


def _name_prefixes(name: str, length: int = PREFIX_LEN) -> list[str]:
    """First `length` characters of each substantive name token."""
    seen = set()
    out = []
    for t in _name_tokens(name):
        if len(t) >= length:
            p = t[:length]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _addr_tokens(addr: str) -> list[str]:
    """Address tokens for blocking (stop-words removed)."""
    return [
        t for t in _norm(addr).split()
        if t not in ADDR_STOPS and len(t) >= MIN_TOKEN_LEN
    ]


def _postal_and_numeric_tokens(addr: str) -> list[str]:
    """Postal codes, PIN/ZIP codes, plot/building numbers (containing digits)."""
    tokens = _norm(addr).split()
    return [t for t in tokens if _DIGIT.search(t) and len(t) >= 2]


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams (without whitespace) for typo/variation fallback."""
    cleaned = _norm(text).replace(" ", "")
    if not cleaned:
        return []
    if len(cleaned) < n:
        return [cleaned]
    return [cleaned[i:i+n] for i in range(len(cleaned)-n+1)]


# ─────────────────── Entity-ID Integer Encoding ─────────────────────────
# S2-xxx  →  +xxx   |   S3-xxx  →  −xxx
# 4 bytes per entity in array('i'), saving >15x memory over Python str/set.

def _enc(eid: str) -> int:
    if eid[1] == "2":          # S2-...
        return int(eid[3:])
    return -int(eid[3:])       # S3-...


def _dec(code: int) -> str:
    if code > 0:
        return f"S2-{code}"
    return f"S3-{-code}"


# ───────────────────── Multi-Channel Index Building ───────────────────────

def _build_indexes(s2_path: str, s3_path: str, country: str):
    """
    Build multi-channel inverted indexes for S2/S3 entities in `country`.

    Returns:
        ni: dict[str, array('i')]      -- Name tokens
        pi: dict[str, array('i')]      -- Name prefixes (5-char)
        ai: dict[str, array('i')]      -- Address tokens
        pos_i: dict[str, array('i')]   -- Postal / Numeric tokens
        c3_i: dict[str, array('i')]    -- Character 3-grams for true fallback
        rare_tokens: set[str]          -- Set of name tokens with df <= RARE_TOKEN_THRESH
    """
    print(f"  [{country}] Building multi-channel inverted indexes …", flush=True)
    t0 = time.time()

    ni = defaultdict(lambda: array("i"))
    pi = defaultdict(lambda: array("i"))
    ai = defaultdict(lambda: array("i"))
    pos_i = defaultdict(lambda: array("i"))
    c3_i = defaultdict(lambda: array("i"))
    n = 0

    for fp in (s2_path, s3_path):
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            fh.readline()  # skip header
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3].strip() != country:
                    continue
                code = _enc(p[0].strip())
                name = p[1].strip() if len(p) > 1 else ""
                addr = p[2].strip() if len(p) > 2 else ""

                # Channel A: Name tokens & 5-char prefixes (with transliteration)
                for t in _name_tokens(name):
                    ni[t].append(code)
                for pr in _name_prefixes(name, PREFIX_LEN):
                    pi[pr].append(code)

                # Channel B: Address tokens & Postal/Numeric tokens
                for t in _addr_tokens(addr):
                    ai[t].append(code)
                for pt in _postal_and_numeric_tokens(addr):
                    pos_i[pt].append(code)

                # Channel D: True Character 3-grams for names (and transliterations)
                ngrams = set(_char_ngrams(name, 3))
                t_name = _translit_clean(name)
                if t_name and t_name != _norm(name):
                    ngrams.update(_char_ngrams(t_name, 3))
                for ng in ngrams:
                    c3_i[ng].append(code)

                n += 1
                if n % 1_000_000 == 0:
                    print(f"    … {n:,} S2/S3 entities indexed ({time.time()-t0:.1f}s)", flush=True)

    # ── Identify Rare Name Tokens ──
    rare_tokens = {k for k, v in ni.items() if len(v) <= RARE_TOKEN_THRESH}

    # ── Apply Frequency Caps ──
    ni_before, ai_before, pi_before = len(ni), len(ai), len(pi)
    pos_before, c3_before = len(pos_i), len(c3_i)

    ni = {k: v for k, v in ni.items() if len(v) <= NAME_FREQ_CAP}
    pi = {k: v for k, v in pi.items() if len(v) <= PREFIX_FREQ_CAP}
    ai = {k: v for k, v in ai.items() if len(v) <= ADDR_FREQ_CAP}
    pos_i = {k: v for k, v in pos_i.items() if len(v) <= POSTAL_FREQ_CAP}
    c3_i = {k: v for k, v in c3_i.items() if len(v) <= C3_FREQ_CAP}

    dt = time.time() - t0
    print(f"  [{country}] {n:,} S2/S3 entities indexed in {dt:.1f}s", flush=True)
    print(f"    Channel A Name tokens   : {len(ni):,} kept / {ni_before:,} total ({len(rare_tokens):,} rare)", flush=True)
    print(f"    Channel A Name prefixes : {len(pi):,} kept / {pi_before:,} total", flush=True)
    print(f"    Channel B Addr tokens   : {len(ai):,} kept / {ai_before:,} total", flush=True)
    print(f"    Channel B Postal tokens : {len(pos_i):,} kept / {pos_before:,} total", flush=True)
    print(f"    Channel D Char 3-grams  : {len(c3_i):,} kept / {c3_before:,} total", flush=True)

    return ni, pi, ai, pos_i, c3_i, rare_tokens


# ──────────────── Ground-Truth Loader (Compact numpy storage) ────────────

def _load_gt(gt_path: str):
    """
    Load ground truth as a sorted numpy int64 array of shape (N, 2).
    Column 0 = S1 entity number (positive).
    Column 1 = matched entity code (+/- as per _enc).
    """
    print("  Loading ground truth into numpy array …", flush=True)
    t0 = time.time()
    pairs: list[tuple[int, int]] = []
    with open(gt_path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip():
                s1_num = int(p[0].strip()[3:])
                for mid in p[1].split(","):
                    mid = mid.strip()
                    if mid:
                        pairs.append((s1_num, _enc(mid)))

    arr = np.array(pairs, dtype=np.int64)
    arr = arr[arr[:, 0].argsort()]
    dt = time.time() - t0
    print(f"  Loaded {len(arr):,} match pairs in {dt:.1f}s ({arr.nbytes / 1e6:.1f} MB)", flush=True)
    return arr


def _gt_lookup(gt_arr, s1_num: int) -> set[int]:
    """Return the set of encoded match codes for `s1_num`."""
    if gt_arr is None:
        return set()
    col = gt_arr[:, 0]
    lo = np.searchsorted(col, s1_num, side="left")
    hi = np.searchsorted(col, s1_num, side="right")
    if lo == hi:
        return set()
    return set(gt_arr[lo:hi, 1].tolist())


# ────────────── Candidate Generation per Country (V4 Engine) ───────────────

def _generate_country_v4(
    s1_path: str,
    country: str,
    ni: dict,
    pi: dict,
    ai: dict,
    pos_i: dict,
    c3_i: dict,
    rare_tokens: set,
    gt_arr,
    out_fh,
    budget_min: int = BUDGET_MIN,
    budget_max: int = BUDGET_MAX,
    fallback_trigger: int = FALLBACK_TRIGGER,
    sample_s1: Optional[int] = None,
):
    """
    Stream S1 entities for `country`, compute raw channel candidate sets,
    perform priority-weighted global scoring without early channel choking,
    apply true 3-gram fallback for low-candidate entities,
    and enforce dynamic elastic budgeting with Source 2/3 balancing.
    """
    stats = dict(
        total=0, with_cands=0, total_cands=0,
        total_cands_before_trunc=0,
        zero_candidates=0,
        capped_budget_max=0,
        under_budget_min=0,
        budget_between=0,
        fallback_triggered=0,
        cands_from_fallback=0,
        cands_s2=0, cands_s3=0,
        # Evidence tracking
        cands_with_rare_name=0,
        cands_with_composite=0,
        cands_with_postal=0,
        cands_with_addr=0,
        # Ground Truth metrics
        gt_total=0, gt_found=0, gt_missed=0,
        gt_found_s2=0, gt_found_s3=0,
        gt_total_s2=0, gt_total_s3=0,
        gt_found_by_a=0, gt_found_by_b=0, gt_found_by_c=0, gt_found_by_d=0,
        perfect=0, partial=0, zero_recall=0,
        cand_counts=[],
    )

    t0 = time.time()

    with open(s1_path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4 or p[3].strip() != country:
                continue

            if sample_s1 is not None and stats["total"] >= sample_s1:
                break

            eid  = p[0].strip()
            name = p[1].strip() if len(p) > 1 else ""
            addr = p[2].strip() if len(p) > 2 else ""
            stats["total"] += 1

            ntoks = _name_tokens(name)
            prefs = _name_prefixes(name, PREFIX_LEN)
            atoks = _addr_tokens(addr)
            post_toks = _postal_and_numeric_tokens(addr)

            # ──────────────────────────────────────────────────────────
            # CHANNEL A: Name Tokens & Prefixes (Raw Scores)
            # ──────────────────────────────────────────────────────────
            scores_a: dict[int, int] = {}
            for t in ntoks:
                arr_ = ni.get(t)
                if arr_ is not None:
                    w = W_RARE_NAME if t in rare_tokens else W_NAME
                    for code in arr_:
                        scores_a[code] = scores_a.get(code, 0) + w

            for pr in prefs:
                arr_ = pi.get(pr)
                if arr_ is not None:
                    for code in arr_:
                        scores_a[code] = scores_a.get(code, 0) + W_PREFIX

            # ──────────────────────────────────────────────────────────
            # CHANNEL B: Address Tokens & Postal/Numeric Tokens (Raw Scores)
            # ──────────────────────────────────────────────────────────
            scores_b: dict[int, int] = {}
            for t in atoks:
                arr_ = ai.get(t)
                if arr_ is not None:
                    for code in arr_:
                        scores_b[code] = scores_b.get(code, 0) + W_ADDR

            for pt in post_toks:
                arr_ = pos_i.get(pt)
                if arr_ is not None:
                    for code in arr_:
                        scores_b[code] = scores_b.get(code, 0) + W_POSTAL_NUM

            # ──────────────────────────────────────────────────────────
            # CHANNEL C: Composite Name ∩ Address (Raw Scores)
            # ──────────────────────────────────────────────────────────
            intersection_codes = set(scores_a.keys()) & set(scores_b.keys())
            scores_c: dict[int, int] = {}
            for code in intersection_codes:
                scores_c[code] = scores_a[code] + scores_b[code] + W_COMPOSITE_BONUS

            # ──────────────────────────────────────────────────────────
            # UNIFIED CANDIDATE POOL (No Early Quota Choking)
            # ──────────────────────────────────────────────────────────
            all_raw_codes = set(scores_a.keys()) | set(scores_b.keys())
            composite_scores: dict[int, int] = {}
            for code in all_raw_codes:
                sa = scores_a.get(code, 0)
                sb = scores_b.get(code, 0)
                sc = scores_c.get(code, 0)
                composite_scores[code] = sc if sc > 0 else (sa + sb)

            # ──────────────────────────────────────────────────────────
            # CHANNEL D: True Character 3-Gram Fallback for Low-Candidate Entities
            # ──────────────────────────────────────────────────────────
            scores_d: dict[int, int] = {}
            if len(composite_scores) < fallback_trigger:
                stats["fallback_triggered"] += 1
                c3_counts = Counter()
                s1_ngrams = set(_char_ngrams(name, 3))
                t_name = _translit_clean(name)
                if t_name and t_name != _norm(name):
                    s1_ngrams.update(_char_ngrams(t_name, 3))

                for ng in s1_ngrams:
                    arr_ = c3_i.get(ng)
                    if arr_ is not None:
                        for code in arr_:
                            c3_counts[code] += 1

                # Take candidates with >= 2 shared 3-grams that aren't already included
                fallback_candidates = [
                    (code, count * W_3GRAM)
                    for code, count in c3_counts.items()
                    if count >= 2 and code not in composite_scores
                ]

                if fallback_candidates:
                    top_fallback = heapq.nlargest(FALLBACK_MAX_ADD, fallback_candidates, key=lambda x: x[1])
                    for code, sc in top_fallback:
                        composite_scores[code] = sc
                        scores_d[code] = sc
                        stats["cands_from_fallback"] += 1

            # ──────────────────────────────────────────────────────────
            # Dynamic Budget Clamping & Source 2 / Source 3 Balancing
            # ──────────────────────────────────────────────────────────
            n_merged = len(composite_scores)
            stats["total_cands_before_trunc"] += n_merged

            if n_merged > budget_max:
                stats["capped_budget_max"] += 1
                # Separate by Source (code > 0 is S2, code < 0 is S3)
                s2_cands = [(c, score) for c, score in composite_scores.items() if c > 0]
                s3_cands = [(c, score) for c, score in composite_scores.items() if c < 0]

                target_per_source = budget_max // 2

                # Select top candidates per source without artificial inflation
                top_s2 = heapq.nlargest(min(target_per_source, len(s2_cands)), s2_cands, key=lambda x: x[1])
                top_s3 = heapq.nlargest(min(target_per_source, len(s3_cands)), s3_cands, key=lambda x: x[1])

                selected_codes = set(c for c, _ in top_s2) | set(c for c, _ in top_s3)

                # If remaining budget exists, fill with highest remaining scores regardless of source
                remaining_budget = budget_max - len(selected_codes)
                if remaining_budget > 0:
                    remaining_pool = [
                        item for item in composite_scores.items()
                        if item[0] not in selected_codes
                    ]
                    top_remaining = heapq.nlargest(remaining_budget, remaining_pool, key=lambda x: x[1])
                    selected_codes.update(c for c, _ in top_remaining)

                final_cands = selected_codes
            else:
                final_cands = set(composite_scores.keys())
                if n_merged <= budget_min:
                    stats["under_budget_min"] += 1
                else:
                    stats["budget_between"] += 1

            num_cands = len(final_cands)
            stats["total_cands"] += num_cands
            if num_cands > 0:
                stats["with_cands"] += 1
                for c in final_cands:
                    if c > 0:
                        stats["cands_s2"] += 1
                    else:
                        stats["cands_s3"] += 1
                    # Evidence breakdown
                    if c in scores_a:
                        stats["cands_with_rare_name"] += 1
                    if c in scores_c:
                        stats["cands_with_composite"] += 1
                    if c in scores_b:
                        stats["cands_with_addr"] += 1
            else:
                stats["zero_candidates"] += 1

            if len(stats["cand_counts"]) < 100_000:
                stats["cand_counts"].append(num_cands)

            # ── Write Output TSV ──
            if final_cands:
                out_fh.write(eid)
                out_fh.write("\t")
                out_fh.write(",".join(_dec(c) for c in sorted(final_cands)))
                out_fh.write("\n")
            else:
                out_fh.write(eid)
                out_fh.write("\t\n")

            # ── Validation Against Ground Truth ──
            if gt_arr is not None:
                s1_num = int(eid[3:])
                true = _gt_lookup(gt_arr, s1_num)
                if true:
                    found = len(true & final_cands)
                    missed = len(true) - found
                    stats["gt_total"]  += len(true)
                    stats["gt_found"]  += found
                    stats["gt_missed"] += missed

                    # Source split (S2 vs S3)
                    for code in true:
                        if code > 0:
                            stats["gt_total_s2"] += 1
                            if code in final_cands:
                                stats["gt_found_s2"] += 1
                        else:
                            stats["gt_total_s3"] += 1
                            if code in final_cands:
                                stats["gt_found_s3"] += 1

                        # Channel attribution
                        if code in final_cands:
                            if code in scores_c:
                                stats["gt_found_by_c"] += 1
                            elif code in scores_a:
                                stats["gt_found_by_a"] += 1
                            elif code in scores_b:
                                stats["gt_found_by_b"] += 1
                            elif code in scores_d:
                                stats["gt_found_by_d"] += 1

                    if missed == 0:
                        stats["perfect"] += 1
                    elif found > 0:
                        stats["partial"] += 1
                    else:
                        stats["zero_recall"] += 1

            if stats["total"] % 200_000 == 0:
                avg = stats["total_cands"] / stats["total"]
                print(f"    … {stats['total']:,} S1 processed ({time.time()-t0:.1f}s, avg {avg:.1f} cands/entity)", flush=True)

    return stats


# Backward-compatible alias for unit tests
_generate_country_v3 = _generate_country_v4


# ──────────────────────── Main Pipeline Runner ───────────────────────────

def run(
    mode: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    budget_min: int = BUDGET_MIN,
    budget_max: int = BUDGET_MAX,
    fallback_trigger: int = FALLBACK_TRIGGER,
    sample_s1: Optional[int] = None,
):
    assert mode in ("validate", "generate"), f"Unknown mode: {mode}"
    base_dir = resolve_dataset_dir(data_dir)
    paths = get_dataset_paths(base_dir, mode)
    out_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUT

    tag = "VALIDATION (train ground truth)" if mode == "validate" else "PRODUCTION TEST GENERATION"

    print(f"\n{'=' * 70}", flush=True)
    print(f"  BLOCKING ENGINE V4 — {tag}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Resolved Dataset Dir : {base_dir}", flush=True)
    print(f"  Output Directory     : {out_dir}", flush=True)
    print(f"  Dynamic Budget       : [{budget_min}, {budget_max}] | Fallback Trigger < {fallback_trigger}", flush=True)
    if sample_s1:
        print(f"  Diagnostic Sample Cap: {sample_s1:,} S1 entities per country", flush=True)

    # ── Open-Set Country Discovery from S1 ──
    countries: set[str] = set()
    with open(paths["s1"], "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                c = p[3].strip()
                if c:
                    countries.add(c)
    sorted_countries = sorted(countries)
    print(f"Discovered Countries in S1 ({len(sorted_countries)}): {sorted_countries}", flush=True)

    # ── Ground Truth (Validation Mode Only) ──
    gt_arr = None
    if mode == "validate":
        gt_arr = _load_gt(str(paths["gt"]))

    # ── Output TSV Setup ──
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / ("train_candidate_pairs.tsv" if mode == "validate" else "candidate_pairs.tsv")
    out_fh = open(out_path, "w", encoding="utf-8", newline="")
    out_fh.write("source1_entity_id\tcandidate_entity_ids\n")

    cumulative = defaultdict(int)
    all_sample_cand_counts = []
    country_reports = {}
    t_start = time.time()

    for country in sorted_countries:
        print(f"\n{'-' * 60}", flush=True)
        print(f"  Processing Country Partition: {country}", flush=True)
        print(f"{'-' * 60}", flush=True)
        tc = time.time()

        # Build Multi-Channel Indexes
        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(str(paths["s2"]), str(paths["s3"]), country)

        # Generate Candidates & Evaluate
        stats = _generate_country_v4(
            str(paths["s1"]), country, ni, pi, ai, pos_i, c3_i, rare_tokens, gt_arr, out_fh,
            budget_min=budget_min,
            budget_max=budget_max,
            fallback_trigger=fallback_trigger,
            sample_s1=sample_s1,
        )

        dt = time.time() - tc
        tot = stats["total"]
        wc  = stats["with_cands"]
        tc_ = stats["total_cands"]
        avg = tc_ / max(tot, 1)

        print(f"\n  [{country}] Completed in {dt:.1f}s", flush=True)
        print(f"    S1 entities          : {tot:,}", flush=True)
        print(f"    With candidates      : {wc:,} ({100 * wc / max(tot,1):.2f}%)", flush=True)
        print(f"    Zero candidates      : {stats['zero_candidates']:,} ({100 * stats['zero_candidates'] / max(tot,1):.2f}%)", flush=True)
        print(f"    Total candidate pairs: {tc_:,}", flush=True)
        print(f"    Avg cands/entity     : {avg:.1f}", flush=True)
        print(f"    Capped at BUDGET_MAX : {stats['capped_budget_max']:,} ({100 * stats['capped_budget_max'] / max(tot,1):.2f}%)", flush=True)
        print(f"    Fallback Triggered   : {stats['fallback_triggered']:,} ({100 * stats['fallback_triggered'] / max(tot,1):.2f}%)", flush=True)
        print(f"    Cands from Fallback  : {stats['cands_from_fallback']:,}", flush=True)

        if gt_arr is not None and stats["gt_total"] > 0:
            rec = stats["gt_found"] / stats["gt_total"]
            rec_s2 = stats["gt_found_s2"] / max(stats["gt_total_s2"], 1)
            rec_s3 = stats["gt_found_s3"] / max(stats["gt_total_s3"], 1)
            print(f"    -- RECALL CEILING    : {rec * 100:.2f}% ({stats['gt_found']:,} / {stats['gt_total']:,})", flush=True)
            print(f"       S2 Recall         : {rec_s2 * 100:.2f}% ({stats['gt_found_s2']:,} / {stats['gt_total_s2']:,})", flush=True)
            print(f"       S3 Recall         : {rec_s3 * 100:.2f}% ({stats['gt_found_s3']:,} / {stats['gt_total_s3']:,})", flush=True)

        country_reports[country] = stats
        for k, v in stats.items():
            if k != "cand_counts":
                cumulative[k] += v
        all_sample_cand_counts.extend(stats["cand_counts"])

        del ni, pi, ai, pos_i, c3_i, rare_tokens
        gc.collect()

    out_fh.close()

    # ──────────────────────── Final Report ──────────────────────────────
    total_time = time.time() - t_start
    T = cumulative["total"]
    C = cumulative["total_cands"]
    fsize_mb = os.path.getsize(out_path) / (1024 * 1024)

    counts_arr = np.array(all_sample_cand_counts) if all_sample_cand_counts else np.array([0])
    p50 = float(np.median(counts_arr))
    p90 = float(np.percentile(counts_arr, 90))
    p95 = float(np.percentile(counts_arr, 95))
    p99 = float(np.percentile(counts_arr, 99))
    max_cands = int(np.max(counts_arr))

    peak_ram_mb = 0.0
    if psutil is not None:
        try:
            peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            peak_ram_mb = 0.0

    print(f"\n{'=' * 70}", flush=True)
    print(f"  V4 BLOCKING EVALUATION REPORT ({total_time / 60:.1f} min)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Total S1 entities processed    : {T:,}", flush=True)
    print(f"  Candidate count BEFORE trunc   : {cumulative['total_cands_before_trunc']:,}", flush=True)
    print(f"  Candidate count AFTER trunc    : {C:,}", flush=True)
    print(f"  Candidate reduction by budget  : {cumulative['total_cands_before_trunc'] - C:,} ({(cumulative['total_cands_before_trunc'] - C)/max(cumulative['total_cands_before_trunc'],1)*100:.2f}%)", flush=True)
    print(f"  Output TSV File Size           : {fsize_mb:.1f} MB ({fsize_mb/1024:.2f} GB)", flush=True)
    print(f"  Peak RAM Usage                 : {peak_ram_mb:.1f} MB", flush=True)
    print(f"  Zero-candidate S1 entities     : {cumulative['zero_candidates']:,} ({cumulative['zero_candidates']/max(T,1)*100:.3f}%)", flush=True)
    print(f"  Entities capped at BUDGET_MAX  : {cumulative['capped_budget_max']:,} ({cumulative['capped_budget_max']/max(T,1)*100:.2f}%)", flush=True)
    print(f"  Entities <= BUDGET_MIN         : {cumulative['under_budget_min']:,} ({cumulative['under_budget_min']/max(T,1)*100:.2f}%)", flush=True)
    print(f"  Entities between MIN & MAX     : {cumulative['budget_between']:,} ({cumulative['budget_between']/max(T,1)*100:.2f}%)", flush=True)
    print(f"  Fallback (Channel D) triggered : {cumulative['fallback_triggered']:,} ({cumulative['fallback_triggered']/max(T,1)*100:.2f}%)", flush=True)
    print(f"  Candidates from 3-Gram Fallback: {cumulative['cands_from_fallback']:,}", flush=True)
    print(f"  Candidates by Source           : S2 = {cumulative['cands_s2']:,} ({cumulative['cands_s2']/max(C,1)*100:.1f}%), S3 = {cumulative['cands_s3']:,} ({cumulative['cands_s3']/max(C,1)*100:.1f}%)", flush=True)
    print(f"\n  Candidate Distribution per S1:", flush=True)
    print(f"    Mean   : {C / max(T,1):.1f}", flush=True)
    print(f"    Median : {p50:.0f}", flush=True)
    print(f"    p90    : {p90:.0f}", flush=True)
    print(f"    p95    : {p95:.0f}", flush=True)
    print(f"    p99    : {p99:.0f}", flush=True)
    print(f"    Max    : {max_cands}", flush=True)

    if gt_arr is not None and cumulative["gt_total"] > 0:
        overall_rec = cumulative["gt_found"] / cumulative["gt_total"]
        s2_rec = cumulative["gt_found_s2"] / max(cumulative["gt_total_s2"], 1)
        s3_rec = cumulative["gt_found_s3"] / max(cumulative["gt_total_s3"], 1)

        print(f"\n  * FINAL MEASURED BLOCKING RECALL (Pair Completeness):", flush=True)
        print(f"    OVERALL RECALL : {overall_rec * 100:.2f}% ({cumulative['gt_found']:,} / {cumulative['gt_total']:,})", flush=True)
        print(f"    S2 RECALL      : {s2_rec * 100:.2f}% ({cumulative['gt_found_s2']:,} / {cumulative['gt_total_s2']:,})", flush=True)
        print(f"    S3 RECALL      : {s3_rec * 100:.2f}% ({cumulative['gt_found_s3']:,} / {cumulative['gt_total_s3']:,})", flush=True)
        print(f"    Total Missed   : {cumulative['gt_missed']:,} ({cumulative['gt_missed']/cumulative['gt_total']*100:.2f}%)", flush=True)

        print(f"\n  Channel Recovery Attribution:", flush=True)
        print(f"    Channel C (Name + Addr Composite) : {cumulative['gt_found_by_c']:,} matches", flush=True)
        print(f"    Channel A (Protected Name)        : {cumulative['gt_found_by_a']:,} matches", flush=True)
        print(f"    Channel B (Protected Address)     : {cumulative['gt_found_by_b']:,} matches", flush=True)
        print(f"    Channel D (3-Gram Fallback)       : {cumulative['gt_found_by_d']:,} matches", flush=True)

    print(f"\n  Output File: {out_path}", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    return cumulative, country_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Recall Multi-Channel Inverted-Index Blocking Engine (V4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/generate_candidates.py validate
  python src/generate_candidates.py validate --sample-s1 1000
  python src/generate_candidates.py validate --data-dir "path/to/dataset"
  python src/generate_candidates.py generate --data-dir "path/to/dataset" --output-dir "path/to/output"
"""
    )
    parser.add_argument(
        "mode",
        choices=["validate", "generate"],
        help="Pipeline execution mode: 'validate' (train ground truth) or 'generate' (test candidates)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the dataset directory containing 'train' and 'test' subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated candidate TSV files (defaults to '<repo>/output')"
    )
    parser.add_argument(
        "--budget-min",
        type=int,
        default=BUDGET_MIN,
        help=f"Minimum candidate count target (default: {BUDGET_MIN})"
    )
    parser.add_argument(
        "--budget-max",
        type=int,
        default=BUDGET_MAX,
        help=f"Maximum candidate budget cap (default: {BUDGET_MAX})"
    )
    parser.add_argument(
        "--fallback-trigger",
        type=int,
        default=FALLBACK_TRIGGER,
        help=f"Threshold to trigger character 3-gram fallback (default: {FALLBACK_TRIGGER})"
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Limit number of S1 entities per country for fast diagnostic evaluation"
    )
    args = parser.parse_args()
    try:
        run(
            mode=args.mode,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            budget_min=args.budget_min,
            budget_max=args.budget_max,
            fallback_trigger=args.fallback_trigger,
            sample_s1=args.sample_s1,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
