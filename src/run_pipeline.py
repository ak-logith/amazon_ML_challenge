#!/usr/bin/env python3
"""
End-to-End Entity Resolution Inference Pipeline
================================================
Blocking (V5.1) → Feature Extraction → XGBoost Scoring → Threshold → matching_results.tsv

This script orchestrates the complete pipeline from raw test data to submission files:
  1. Blocking: Generate candidate_pairs.tsv using the V5.1 blocker
  2. Matching: Score each (S1, candidate) pair with the trained XGBoost model
  3. Threshold: Apply the optimized threshold to produce matching_results.tsv
  4. Validation: Run the submission validator

Usage:
    python -m src.run_pipeline                          # Full pipeline
    python -m src.run_pipeline --skip-blocking          # Reuse existing candidate_pairs.tsv
    python -m src.run_pipeline --mode validate          # Validate on training data
    python -m src.run_pipeline --mode validate --sample-s1 5000  # Quick validation
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import logging
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.matching.features import compute_pair_features, FEATURE_NAMES
from src.generate_candidates import run as run_blocker, resolve_dataset_dir

# ── Configuration ────────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
NUM_WORKERS = 8
CHUNK_SIZE = 5_000       # Pairs per parallel chunk for feature computation
SCORE_BATCH = 200_000    # S1 entities per scoring batch (controls peak RAM)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_source_records(path: Path) -> dict:
    """Load a source TSV → {entity_id: {name, addr, country}}."""
    log.info(f"Loading {path.name}...")
    records = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records[row["entity_id"]] = {
                "name": row.get("business_name", "") or "",
                "addr": row.get("business_address", "") or "",
                "country": row.get("country", "") or "",
            }
    log.info(f"  Loaded {len(records):,} records")
    return records


def load_candidates(path: Path) -> dict:
    """Load candidate_pairs.tsv → {s1_id: [candidate_ids]}."""
    log.info(f"Loading candidates from {path.name}...")
    cands = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                cand_ids = parts[1].split(",")
            else:
                cand_ids = []
            cands[s1_id] = cand_ids
    log.info(f"  Loaded candidates for {len(cands):,} S1 entities")
    total_pairs = sum(len(v) for v in cands.values())
    log.info(f"  Total candidate pairs: {total_pairs:,}")
    return cands


def load_ground_truth(path: Path) -> dict:
    """Load ground truth → {s1_id: set(matched_ids)}."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id") or row.get("entity_id")
            matches_str = (row.get("matched_entity_ids") or row.get("matches") or "").strip()
            if matches_str:
                gt[s1_id] = set(m.strip() for m in matches_str.split(",") if m.strip())
            else:
                gt[s1_id] = set()
    return gt


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE COMPUTATION (Parallel)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_chunk(chunk_data: list) -> np.ndarray:
    """Worker: compute features for a chunk of pairs."""
    rows = []
    for s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, source_type in chunk_data:
        feats = compute_pair_features(
            s1_name, s1_addr, s1_country,
            cand_name, cand_addr, cand_country,
            source_type=source_type,
        )
        rows.append(feats)
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)


def compute_features_for_batch(
    pairs: list,  # [(s1_id, cand_id), ...]
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
) -> np.ndarray:
    """Compute features for a batch of pairs using parallel workers."""
    # Build payloads
    payloads = []
    for s1_id, cand_id in pairs:
        s1_rec = s1_records.get(s1_id, {"name": "", "addr": "", "country": ""})
        if cand_id.startswith("S2-"):
            cand_rec = s2_records.get(cand_id, {"name": "", "addr": "", "country": ""})
            st = 0
        else:
            cand_rec = s3_records.get(cand_id, {"name": "", "addr": "", "country": ""})
            st = 1
        payloads.append((
            s1_rec["name"], s1_rec["addr"], s1_rec["country"],
            cand_rec["name"], cand_rec["addr"], cand_rec["country"],
            st
        ))

    n = len(payloads)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    chunks = [payloads[i:i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        matrices = list(executor.map(_compute_chunk, chunks))

    return np.vstack(matrices)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL SCORING
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Load trained XGBoost model and config."""
    config_path = MODEL_DIR / "matching_config.pkl"
    model_path = MODEL_DIR / "matching_model.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    threshold = config["best_threshold"]
    log.info(f"  Model loaded: {config['model_type']}, {config['num_features']} features")
    log.info(f"  Optimal threshold: {threshold:.3f} (validation F0.5: {config['best_f05']:.4f})")
    return model, threshold, config


# ═══════════════════════════════════════════════════════════════════════════
# 4. STREAMING SCORING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def score_candidates_streaming(
    candidates: dict,       # {s1_id: [cand_ids]}
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    model,
    threshold: float,
    output_path: Path,
    gt: Optional[dict] = None,
):
    """
    Score all candidate pairs in streaming batches to keep RAM bounded.
    Writes matching_results.tsv as it goes.

    Returns: (total_s1, total_pairs_scored, total_matches, eval_metrics)
    """
    log.info("=" * 70)
    log.info("SCORING CANDIDATES → matching_results.tsv")
    log.info(f"  Threshold: {threshold:.3f}")
    log.info(f"  Batch size: {SCORE_BATCH:,} S1 entities")
    log.info("=" * 70)

    os.makedirs(output_path.parent, exist_ok=True)
    out_fh = open(output_path, "w", encoding="utf-8", newline="")
    out_fh.write("source1_entity_id\tmatched_entity_ids\n")

    s1_ids = list(candidates.keys())
    total_s1 = len(s1_ids)
    total_pairs = 0
    total_matches = 0
    total_scored = 0

    # For evaluation
    all_f05_scores = [] if gt is not None else None

    t_start = time.time()
    batch_idx = 0

    for batch_start in range(0, total_s1, SCORE_BATCH):
        batch_end = min(batch_start + SCORE_BATCH, total_s1)
        batch_s1 = s1_ids[batch_start:batch_end]
        batch_idx += 1

        # Build pairs for this batch
        batch_pairs = []
        batch_s1_offsets = {}  # s1_id -> (start_idx, count) in batch_pairs
        for s1_id in batch_s1:
            cand_ids = candidates.get(s1_id, [])
            start = len(batch_pairs)
            for cid in cand_ids:
                batch_pairs.append((s1_id, cid))
            batch_s1_offsets[s1_id] = (start, len(cand_ids))

        n_pairs = len(batch_pairs)
        total_pairs += n_pairs

        if n_pairs > 0:
            # Compute features
            t_feat = time.time()
            X = compute_features_for_batch(batch_pairs, s1_records, s2_records, s3_records)
            feat_time = time.time() - t_feat

            # Predict
            t_pred = time.time()
            probs = model.predict_proba(X)[:, 1]
            pred_time = time.time() - t_pred

            total_scored += n_pairs
            rate = n_pairs / max(feat_time, 0.001)
            log.info(
                f"  Batch {batch_idx}: {len(batch_s1):,} S1, {n_pairs:,} pairs | "
                f"features {feat_time:.1f}s ({rate:,.0f}/s) | predict {pred_time:.1f}s"
            )
        else:
            probs = np.array([])

        # Apply threshold and write results
        batch_matches = 0
        for s1_id in batch_s1:
            start, count = batch_s1_offsets.get(s1_id, (0, 0))
            matched_ids = []
            if count > 0:
                s1_probs = probs[start:start + count]
                for i, prob in enumerate(s1_probs):
                    if prob >= threshold:
                        matched_ids.append(batch_pairs[start + i][1])

            total_matches += len(matched_ids)
            batch_matches += len(matched_ids)

            # Write row (every S1 must have a row, even if no matches)
            if matched_ids:
                out_fh.write(f"{s1_id}\t{','.join(matched_ids)}\n")
            else:
                out_fh.write(f"{s1_id}\t\n")

            # Evaluate against ground truth
            if gt is not None:
                true_ids = gt.get(s1_id, set())
                pred_set = set(matched_ids)
                if len(true_ids) == 0 and len(pred_set) == 0:
                    f05 = 1.0
                elif len(true_ids) == 0:
                    f05 = 0.0
                elif len(pred_set) == 0:
                    f05 = 0.0
                else:
                    tp = len(pred_set & true_ids)
                    fp = len(pred_set - true_ids)
                    fn = len(true_ids - pred_set)
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f05 = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
                all_f05_scores.append(f05)

        # Free batch memory
        del X, probs if n_pairs > 0 else None
        gc.collect()

        elapsed = time.time() - t_start
        log.info(
            f"    → {batch_end:,}/{total_s1:,} S1 done | "
            f"{total_matches:,} matches | {elapsed:.0f}s elapsed"
        )

    out_fh.close()

    # Final report
    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("MATCHING COMPLETE")
    log.info(f"  S1 entities:     {total_s1:,}")
    log.info(f"  Pairs scored:    {total_scored:,}")
    log.info(f"  Matches found:   {total_matches:,}")
    log.info(f"  Match rate:      {total_matches / max(total_s1, 1):.2f} per S1")
    log.info(f"  Time:            {elapsed / 60:.1f} min")
    log.info(f"  Output:          {output_path}")

    eval_metrics = None
    if gt is not None and all_f05_scores:
        macro_f05 = float(np.mean(all_f05_scores))
        perfect = sum(1 for s in all_f05_scores if s == 1.0)
        zero = sum(1 for s in all_f05_scores if s == 0.0)
        log.info(f"\n  ★ MACRO F0.5:    {macro_f05:.4f}")
        log.info(f"    Perfect (1.0): {perfect:,} ({100 * perfect / len(all_f05_scores):.1f}%)")
        log.info(f"    Zero (0.0):    {zero:,} ({100 * zero / len(all_f05_scores):.1f}%)")
        eval_metrics = {"macro_f05": macro_f05, "perfect": perfect, "zero": zero}

    log.info("=" * 70)
    return total_s1, total_scored, total_matches, eval_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    mode: str = "generate",
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_blocking: bool = False,
    sample_s1: Optional[int] = None,
    budget_max: int = 1500,
):
    """
    Complete end-to-end pipeline:
      blocking → feature computation → XGBoost scoring → matching_results.tsv
    """
    t0 = time.time()

    # Resolve paths
    base_dir = resolve_dataset_dir(data_dir)
    out_dir = Path(output_dir).resolve() if output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    is_validate = mode == "validate"
    split = "train" if is_validate else "test"
    split_dir = base_dir / split

    log.info("=" * 70)
    log.info(f"ENTITY RESOLUTION — FULL PIPELINE ({'VALIDATION' if is_validate else 'TEST'})")
    log.info(f"  Dataset: {base_dir}")
    log.info(f"  Output:  {out_dir}")
    log.info(f"  Budget:  {budget_max}")
    if sample_s1:
        log.info(f"  Sample:  {sample_s1:,} S1 per country")
    log.info("=" * 70)

    # ── Step 1: Blocking ─────────────────────────────────────────────────
    cand_filename = "train_candidate_pairs.tsv" if is_validate else "candidate_pairs.tsv"
    cand_path = out_dir / cand_filename

    if skip_blocking and cand_path.exists():
        log.info(f"[SKIP BLOCKING] Using existing {cand_path}")
    else:
        log.info("[STEP 1] Running blocker...")
        run_blocker(
            mode=mode,
            data_dir=data_dir,
            output_dir=str(out_dir),
            budget_max=budget_max,
            sample_s1=sample_s1,
        )
        log.info(f"[STEP 1] Blocking complete → {cand_path}")

    gc.collect()

    # ── Step 2: Load data ────────────────────────────────────────────────
    log.info("[STEP 2] Loading source records...")
    s1 = load_source_records(split_dir / f"{split}_source1.tsv")
    s2 = load_source_records(split_dir / f"{split}_source2.tsv")
    s3 = load_source_records(split_dir / f"{split}_source3.tsv")

    candidates = load_candidates(cand_path)

    gt = None
    if is_validate:
        gt = load_ground_truth(split_dir / f"{split}_ground_truth.tsv")
        log.info(f"  Ground truth: {len(gt):,} S1 entities")

    # ── Step 3: Load model ───────────────────────────────────────────────
    log.info("[STEP 3] Loading trained model...")
    model, threshold, config = load_model()

    # ── Step 4: Score and produce matching_results.tsv ────────────────────
    log.info("[STEP 4] Scoring candidates...")
    match_filename = "train_matching_results.tsv" if is_validate else "matching_results.tsv"
    match_path = out_dir / match_filename

    total_s1, total_scored, total_matches, eval_metrics = score_candidates_streaming(
        candidates=candidates,
        s1_records=s1,
        s2_records=s2,
        s3_records=s3,
        model=model,
        threshold=threshold,
        output_path=match_path,
        gt=gt,
    )

    # ── Step 5: Validate submission format ───────────────────────────────
    if not is_validate:
        log.info("[STEP 5] Running submission validator...")
        test_dir = str(split_dir)
        try:
            from utils.validate_submission import validate
            errors, warnings = validate(
                str(match_path),
                str(cand_path),
                test_dir,
                check_ids=False,
            )
            for w in warnings:
                log.warning(f"  {w}")
            if errors:
                for e in errors:
                    log.error(f"  {e}")
                log.error("SUBMISSION VALIDATION FAILED")
            else:
                log.info("  ✓ Submission validation PASSED")
        except Exception as e:
            log.warning(f"  Validator error: {e}")

    elapsed = time.time() - t0
    log.info(f"\nTotal pipeline time: {elapsed / 60:.1f} minutes")
    log.info(f"Output files:")
    log.info(f"  Candidates: {cand_path}")
    log.info(f"  Matching:   {match_path}")

    return match_path, cand_path, eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-End Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["validate", "generate"], default="generate",
        help="'validate' (train data with GT) or 'generate' (test data for submission)"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-blocking", action="store_true", help="Reuse existing candidate_pairs.tsv")
    parser.add_argument("--sample-s1", type=int, default=None, help="S1 per country (diagnostic)")
    parser.add_argument("--budget-max", type=int, default=1500, help="Blocker budget cap (default: 1500)")

    args = parser.parse_args()
    run_full_pipeline(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_blocking=args.skip_blocking,
        sample_s1=args.sample_s1,
        budget_max=args.budget_max,
    )
