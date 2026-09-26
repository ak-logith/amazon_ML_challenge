#!/usr/bin/env python3
"""
Entity Resolution — Candidate Generation / Blocking Pipeline (V3)
==================================================================
High-Recall Multi-Channel Inverted-Index Blocking Engine with
Protected Channel Quotas, Open-Set Country Partitioning,
and Dynamic Elastic Candidate Budgeting.

Designed for memory-constrained environments (<500 MB peak RAM)
and maximum pair completeness (Recall).

Key V3 Architecture Features:
1. Multi-Channel Protected Quotas:
   - Channel A: Rare and exact name tokens + prefixes
   - Channel B: High-specificity address, postal, and numeric tokens
   - Channel C: Composite high-priority intersection (Name ∩ Address)
   - Channel D: Character prefix & relaxed fallback for low-candidate entities
2. Protection against Address Noise Displacement:
   - Dedicated channel quotas ensure address density never displaces
     rare name matches.
3. Open-Set Country Partitioning:
   - Dynamically discovers all country partitions from Source 1 (e.g. US, India, France, etc.).
4. Dynamic Candidate Budgeting:
   - Distinctive entities use 20–100 slots, dense metropolitan entities
     dynamically expand up to BUDGET_MAX slots.
5. Strict Zero Data Leakage:
   - Ground truth is strictly used for offline evaluation/recall metrics
     and NEVER used during candidate generation or ranking.

Usage:
------
    python src/generate_candidates.py validate --data-dir "path/to/dataset"   # Evaluate on full train GT
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
    3. Auto-discovery relative to repository root:
       - <repo>/dataset
       - <repo>/data
       - <repo>/../6ab10eb3b23ba_student_resource/student_resource/dataset
       - <repo>/../dataset
       - <repo>/../student_resource/dataset
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

    # Sensible default fallback relative to repository
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

RARE_TOKEN_THRESH   = 250     # Tokens with document frequency <= N are marked "rare"
MIN_TOKEN_LEN       = 3       # Minimum token length for blocking keys
PREFIX_LEN          = 5       # Chars used for prefix blocking keys
FALLBACK_PREFIX_LEN = 4       # Chars used for relaxed fallback prefix keys

# ── Multi-Channel Protected Quotas & Dynamic Budgets ──
CHANNEL_A_QUOTA     = 180     # Guaranteed slots for Name Token & Prefix candidates
CHANNEL_B_QUOTA     = 180     # Guaranteed slots for Address / Postal / Numeric candidates
CHANNEL_C_QUOTA     = 180     # Guaranteed slots for Composite Name ∩ Addr candidates
CHANNEL_D_MAX       = 100     # Max fallback candidates for low-candidate entities

BUDGET_MIN          = 50      # Min candidate limit for low-density entities
BUDGET_MAX          = 650     # Dynamic candidate cap for high-density metropolitan entities
FALLBACK_TRIGGER    = 40      # If total candidates < N, activate Channel D fallback

# ── Scoring Weights for Ranking ──
W_RARE_NAME         = 12      # Score per shared rare name token (df <= RARE_TOKEN_THRESH)
W_NAME              = 5       # Score per shared standard name token
W_COMPOSITE_BONUS   = 8       # Bonus for candidate present in both Name & Addr channels
W_POSTAL_NUM        = 6       # Score per shared postal code / numeric plot/building token
W_ADDR              = 3       # Score per shared standard address token
W_PREFIX            = 2       # Score per shared 5-char name prefix
W_FALLBACK_PREF     = 2       # Score per shared 4-char prefix (in fallback mode)

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
    """Lower-case, normalize unicode (NFKD), strip accents/diacritics, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return _WS.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def _name_tokens(name: str) -> list[str]:
    """Substantive name tokens for blocking (suffixes & short/numeric words removed)."""
    return [
        t for t in _norm(name).split()
        if t not in SUFFIXES and len(t) >= MIN_TOKEN_LEN and not t.isdigit()
    ]


def _name_prefixes(name: str, length: int = PREFIX_LEN) -> list[str]:
    """First `length` characters of each long-enough name token."""
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
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
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
        fpi: dict[str, array('i')]     -- Fallback 4-char prefixes
        rare_tokens: set[str]          -- Set of name tokens with df <= RARE_TOKEN_THRESH
    """
    print(f"  [{country}] Building multi-channel inverted indexes …", flush=True)
    t0 = time.time()

    ni = defaultdict(lambda: array("i"))
    pi = defaultdict(lambda: array("i"))
    ai = defaultdict(lambda: array("i"))
    pos_i = defaultdict(lambda: array("i"))
    fpi = defaultdict(lambda: array("i"))
    n = 0

    for fp in (s2_path, s3_path):
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            fh.readline() # skip header
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3].strip() != country:
                    continue
                code = _enc(p[0].strip())
                name = p[1].strip() if len(p) > 1 else ""
                addr = p[2].strip() if len(p) > 2 else ""

                # Channel A: Name tokens & 5-char prefixes
                for t in _name_tokens(name):
                    ni[t].append(code)
                for pr in _name_prefixes(name, PREFIX_LEN):
                    pi[pr].append(code)

                # Channel B: Address tokens & Postal/Numeric tokens
                for t in _addr_tokens(addr):
                    ai[t].append(code)
                for pt in _postal_and_numeric_tokens(addr):
                    pos_i[pt].append(code)

                # Channel D: 4-char fallback prefixes
                for fp_pref in _name_prefixes(name, FALLBACK_PREFIX_LEN):
                    fpi[fp_pref].append(code)

                n += 1
                if n % 1_000_000 == 0:
                    print(f"    … {n:,} S2/S3 entities indexed ({time.time()-t0:.1f}s)", flush=True)

    # ── Identify Rare Name Tokens ──
    rare_tokens = {k for k, v in ni.items() if len(v) <= RARE_TOKEN_THRESH}

    # ── Apply Frequency Caps ──
    ni_before, ai_before, pi_before = len(ni), len(ai), len(pi)
    pos_before, fpi_before = len(pos_i), len(fpi)

    ni = {k: v for k, v in ni.items() if len(v) <= NAME_FREQ_CAP}
    pi = {k: v for k, v in pi.items() if len(v) <= PREFIX_FREQ_CAP}
    ai = {k: v for k, v in ai.items() if len(v) <= ADDR_FREQ_CAP}
    pos_i = {k: v for k, v in pos_i.items() if len(v) <= POSTAL_FREQ_CAP}
    fpi = {k: v for k, v in fpi.items() if len(v) <= PREFIX_FREQ_CAP}

    dt = time.time() - t0
    print(f"  [{country}] {n:,} S2/S3 entities indexed in {dt:.1f}s", flush=True)
    print(f"    Channel A Name tokens   : {len(ni):,} kept / {ni_before:,} total ({len(rare_tokens):,} rare)", flush=True)
    print(f"    Channel A Name prefixes : {len(pi):,} kept / {pi_before:,} total", flush=True)
    print(f"    Channel B Addr tokens   : {len(ai):,} kept / {ai_before:,} total", flush=True)
    print(f"    Channel B Postal tokens : {len(pos_i):,} kept / {pos_before:,} total", flush=True)
    print(f"    Channel D Fallback pref : {len(fpi):,} kept / {fpi_before:,} total", flush=True)

    return ni, pi, ai, pos_i, fpi, rare_tokens


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
    col = gt_arr[:, 0]
    lo = np.searchsorted(col, s1_num, side="left")
    hi = np.searchsorted(col, s1_num, side="right")
    if lo == hi:
        return set()
    return set(gt_arr[lo:hi, 1].tolist())


# ────────────── Candidate Generation per Country (Multi-Channel V3) ───────

def _generate_country_v3(
    s1_path: str,
    country: str,
    ni: dict,
    pi: dict,
    ai: dict,
    pos_i: dict,
    fpi: dict,
    rare_tokens: set,
    gt_arr,            # numpy array or None
    out_fh,            # open file handle
):
    """
    Stream through S1 entities for `country`, query multi-channel indexes,
    allocate protected quotas, apply dynamic budgets, and write candidates.

    Multi-Channel Quota Logic:
      1. Query Channel A (Name tokens + prefixes) -> score candidates by name overlap
      2. Query Channel B (Address tokens + postal/numeric) -> score candidates by address overlap
      3. Identify Channel C (Composite Name ∩ Address) -> guaranteed top priority
      4. If total candidates < FALLBACK_TRIGGER -> query Channel D (4-char prefixes)
      5. Form final candidate set by taking UNION of protected channel quotas:
         - All Channel C candidates (up to CHANNEL_C_QUOTA)
         - Top Channel A candidates (up to CHANNEL_A_QUOTA)
         - Top Channel B candidates (up to CHANNEL_B_QUOTA)
         - Top Channel D candidates (up to CHANNEL_D_MAX, if triggered)
      6. Dynamically clamp total candidates to [BUDGET_MIN, BUDGET_MAX].

    Returns a comprehensive stats dictionary.
    """
    stats = dict(
        total=0, with_cands=0, total_cands=0,
        total_cands_before_trunc=0,
        zero_candidates=0,
        capped_budget_max=0,
        under_budget_min=0,
        budget_between=0,
        fallback_triggered=0,
        # GT metrics
        gt_total=0, gt_found=0, gt_missed=0,
        gt_found_s2=0, gt_found_s3=0,
        gt_total_s2=0, gt_total_s3=0,
        gt_found_by_a=0, gt_found_by_b=0, gt_found_by_c=0, gt_found_by_d=0,
        perfect=0, partial=0, zero_recall=0,
        cand_counts=[],  # Sample of candidate counts for percentile calculations
    )

    t0 = time.time()

    with open(s1_path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4 or p[3].strip() != country:
                continue

            eid  = p[0].strip()
            name = p[1].strip() if len(p) > 1 else ""
            addr = p[2].strip() if len(p) > 2 else ""
            stats["total"] += 1

            ntoks = _name_tokens(name)
            prefs = _name_prefixes(name, PREFIX_LEN)
            atoks = _addr_tokens(addr)
            post_toks = _postal_and_numeric_tokens(addr)

            # ──────────────────────────────────────────────────────────
            # CHANNEL A: Name Tokens & Prefixes
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
            # CHANNEL B: Address Tokens & Postal/Numeric Tokens
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
            # CHANNEL C: Composite Name ∩ Address (Priority Tier 1)
            # ──────────────────────────────────────────────────────────
            intersection_codes = set(scores_a.keys()) & set(scores_b.keys())
            scores_c: dict[int, int] = {}
            for code in intersection_codes:
                scores_c[code] = scores_a[code] + scores_b[code] + W_COMPOSITE_BONUS

            # ──────────────────────────────────────────────────────────
            # CHANNEL D: Fallback for Low-Candidate Entities (< FALLBACK_TRIGGER)
            # ──────────────────────────────────────────────────────────
            scores_d: dict[int, int] = {}
            raw_unique_count = len(set(scores_a.keys()) | set(scores_b.keys()))

            if raw_unique_count < FALLBACK_TRIGGER:
                stats["fallback_triggered"] += 1
                for fp_pref in _name_prefixes(name, FALLBACK_PREFIX_LEN):
                    arr_ = fpi.get(fp_pref)
                    if arr_ is not None:
                        for code in arr_:
                            scores_d[code] = scores_d.get(code, 0) + W_FALLBACK_PREF

            # ──────────────────────────────────────────────────────────
            # Multi-Channel Quota Allocation & Dynamic Budget Assembly
            # ──────────────────────────────────────────────────────────
            final_cands: set[int] = set()

            # 1. Take top Channel C candidates (guaranteed priority)
            if scores_c:
                if len(scores_c) > CHANNEL_C_QUOTA:
                    top_c = heapq.nlargest(CHANNEL_C_QUOTA, scores_c.items(), key=lambda x: x[1])
                    final_cands.update(code for code, _ in top_c)
                else:
                    final_cands.update(scores_c.keys())

            # 2. Take top Channel A candidates (protected name quota)
            if scores_a:
                if len(scores_a) > CHANNEL_A_QUOTA:
                    top_a = heapq.nlargest(CHANNEL_A_QUOTA, scores_a.items(), key=lambda x: x[1])
                    final_cands.update(code for code, _ in top_a)
                else:
                    final_cands.update(scores_a.keys())

            # 3. Take top Channel B candidates (protected address quota)
            if scores_b:
                if len(scores_b) > CHANNEL_B_QUOTA:
                    top_b = heapq.nlargest(CHANNEL_B_QUOTA, scores_b.items(), key=lambda x: x[1])
                    final_cands.update(code for code, _ in top_b)
                else:
                    final_cands.update(scores_b.keys())

            # 4. Take top Channel D fallback candidates (if triggered)
            if scores_d:
                if len(scores_d) > CHANNEL_D_MAX:
                    top_d = heapq.nlargest(CHANNEL_D_MAX, scores_d.items(), key=lambda x: x[1])
                    final_cands.update(code for code, _ in top_d)
                else:
                    final_cands.update(scores_d.keys())

            # 5. Dynamic Budget Clamping
            n_merged = len(final_cands)
            stats["total_cands_before_trunc"] += n_merged
            if n_merged >= BUDGET_MAX:
                stats["capped_budget_max"] += 1
                composite_scores: dict[int, int] = {}
                for code in final_cands:
                    sa = scores_a.get(code, 0)
                    sb = scores_b.get(code, 0)
                    sc = scores_c.get(code, 0)
                    sd = scores_d.get(code, 0)
                    composite_scores[code] = sa + sb + sc + sd
                top_final = heapq.nlargest(BUDGET_MAX, composite_scores.items(), key=lambda x: x[1])
                final_cands = set(code for code, _ in top_final)
            elif n_merged <= BUDGET_MIN:
                stats["under_budget_min"] += 1
            else:
                stats["budget_between"] += 1

            num_cands = len(final_cands)
            stats["total_cands"] += num_cands
            if num_cands > 0:
                stats["with_cands"] += 1
            else:
                stats["zero_candidates"] += 1

            if len(stats["cand_counts"]) < 100_000:
                stats["cand_counts"].append(num_cands)

            # ── Write Output ──
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
                            if code in scores_c: stats["gt_found_by_c"] += 1
                            elif code in scores_a: stats["gt_found_by_a"] += 1
                            elif code in scores_b: stats["gt_found_by_b"] += 1
                            elif code in scores_d: stats["gt_found_by_d"] += 1

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


# ──────────────────────── Main Pipeline Runner ───────────────────────────

def run(mode: str, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
    assert mode in ("validate", "generate"), f"Unknown mode: {mode}"
    base_dir = resolve_dataset_dir(data_dir)
    paths = get_dataset_paths(base_dir, mode)
    out_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUT

    tag = "VALIDATION (train ground truth)" if mode == "validate" else "PRODUCTION TEST GENERATION"

    print(f"\n{'=' * 70}", flush=True)
    print(f"  BLOCKING ENGINE V3 — {tag}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Resolved Dataset Dir : {base_dir}", flush=True)
    print(f"  Output Directory     : {out_dir}", flush=True)

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
        ni, pi, ai, pos_i, fpi, rare_tokens = _build_indexes(str(paths["s2"]), str(paths["s3"]), country)

        # Generate Candidates & Evaluate
        stats = _generate_country_v3(
            str(paths["s1"]), country, ni, pi, ai, pos_i, fpi, rare_tokens, gt_arr, out_fh,
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

        del ni, pi, ai, pos_i, fpi, rare_tokens
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
    try:
        import psutil
        peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        peak_ram_mb = 0.0

    print(f"\n{'=' * 70}", flush=True)
    print(f"  V3 BLOCKING EVALUATION REPORT ({total_time / 60:.1f} min)", flush=True)
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
        print(f"    Channel D (Fallback Prefix)       : {cumulative['gt_found_by_d']:,} matches", flush=True)

    print(f"\n  Output File: {out_path}", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    return cumulative, country_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Recall Multi-Channel Inverted-Index Blocking Engine (V3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/generate_candidates.py validate
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
        help="Path to the dataset directory containing 'train' and 'test' subdirectories (or parent folder)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated candidate TSV files (defaults to '<repo>/output')"
    )
    args = parser.parse_args()
    try:
        run(mode=args.mode, data_dir=args.data_dir, output_dir=args.output_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
