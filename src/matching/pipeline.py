"""
Entity Resolution Matching Pipeline
====================================
End-to-end pipeline: data loading → pair generation → feature computation →
CUDA-accelerated GBDT training (XGBoost/LightGBM) → threshold optimization → F0.5 evaluation.

Optimized for:
- CUDA acceleration on local NVIDIA GPU (GeForce RTX 4050)
- Parallel multi-core feature extraction via batched ProcessPoolExecutor
- Memory-safe pair generation (no RAM bloat / zero swapping)
- Flexible train/validation split (stratified by country & singletons)
- Macro F0.5 metric with fine-grained threshold grid search

Usage:
    python -m src.matching.pipeline
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gc
import csv
import time
import random
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, Dict, Set, List

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.matching.features import compute_pair_features, FEATURE_NAMES

# ── Configuration ────────────────────────────────────────────────────────
DATA_DIR = Path(r"e:\Projects\Amazon ML Challenge\Given Resource\6ab10eb3b23ba_student_resource\student_resource\dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

# Parallelism
NUM_WORKERS = 8       # 8 workers with 1 thread each prevents thread contention and memory bloat
CHUNK_SIZE = 5_000    # Batch size for parallel workers

# Model & CUDA configuration
USE_CUDA = True          # Use GPU acceleration with XGBoost
MODEL_BACKEND = "xgb_cuda"  # "xgb_cuda" or "lgbm_cpu"

# Sampling config (balanced for speed, memory safety, and high statistical power)
NEGATIVE_RATIO = 4         # negatives per positive
MAX_TRAIN_PAIRS = 1_000_000   # 1M pairs (~500K pos, 500K neg) trains in seconds on CUDA
MAX_VAL_PAIRS = 300_000       # 300K pairs for validation (~15-20K S1 entities)

# Validation split — flexible, stratified (85/15)
VAL_FRACTION = 0.15

# Random seed
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("matching")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_source(path: Path) -> dict:
    """Load a source TSV into a dict keyed by entity_id."""
    log.info(f"Loading {path.name}...")
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records[row["entity_id"]] = {
                "name": row.get("business_name", "") or "",
                "addr": row.get("business_address", "") or "",
                "country": row.get("country", "") or "",
            }
    log.info(f"  Loaded {len(records):,} records from {path.name}")
    return records


def load_ground_truth(path: Path) -> dict:
    """Load ground truth → {s1_id: set(matched_ids)}."""
    log.info(f"Loading ground truth from {path.name}...")
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id") or row.get("entity_id")
            matches_str = (row.get("matched_entity_ids") or row.get("matches") or "").strip()
            if matches_str:
                matched_list = [m.strip() for m in matches_str.split(",") if m.strip()]
                gt[s1_id] = set(matched_list)
            else:
                gt[s1_id] = set()
    log.info(f"  Loaded {len(gt):,} S1 entities in ground truth")
    return gt


# ═══════════════════════════════════════════════════════════════════════════
# 2. VALIDATION SPLIT — Stratified S1-entity level
# ═══════════════════════════════════════════════════════════════════════════

def create_val_split(gt: dict, s1_records: dict, val_fraction: float = VAL_FRACTION) -> Tuple[set, set]:
    """
    Split S1 entities into train/val sets.
    Stratified by country × match-count bucket.
    Flexible split ratio — not a rigid 80/20.
    """
    log.info(f"Creating validation split (val_fraction={val_fraction})...")

    s1_ids = list(gt.keys())
    countries = []
    match_counts = []

    for s1_id in s1_ids:
        rec = s1_records.get(s1_id, {})
        countries.append(rec.get("country", "Unknown"))
        mc = len(gt[s1_id])
        if mc == 0:
            bucket = "0"
        elif mc <= 2:
            bucket = "1-2"
        elif mc <= 5:
            bucket = "3-5"
        else:
            bucket = "6+"
        match_counts.append(bucket)

    strat_keys = [f"{c}_{b}" for c, b in zip(countries, match_counts)]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(s1_ids, strat_keys))

    train_s1 = set(s1_ids[i] for i in train_idx)
    val_s1 = set(s1_ids[i] for i in val_idx)

    train_singletons = sum(1 for s in train_s1 if len(gt[s]) == 0)
    val_singletons = sum(1 for s in val_s1 if len(gt[s]) == 0)
    train_positives = sum(len(gt[s]) for s in train_s1)
    val_positives = sum(len(gt[s]) for s in val_s1)

    log.info(f"  Train: {len(train_s1):,} S1 entities ({train_singletons:,} singletons, {train_positives:,} positive links)")
    log.info(f"  Val:   {len(val_s1):,} S1 entities ({val_singletons:,} singletons, {val_positives:,} positive links)")

    return train_s1, val_s1


# ═══════════════════════════════════════════════════════════════════════════
# 3. PAIR GENERATION — Memory-safe & Balanced
# ═══════════════════════════════════════════════════════════════════════════

def generate_pairs(
    s1_set: set,
    gt: dict,
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    neg_ratio: int = NEGATIVE_RATIO,
    max_pairs: int = MAX_TRAIN_PAIRS,
) -> Tuple[List[Tuple[str, str]], np.ndarray, set]:
    """
    Generate positive and negative pairs for a set of S1 entities.
    Stops cleanly once max_pairs is met, ensuring included entities have complete candidate sets.
    
    Returns: (pairs, labels, evaluated_s1_set)
    """
    log.info(f"Generating pairs (target max: {max_pairs:,}, neg_ratio={neg_ratio})...")

    # Build country-indexed pools for efficient negative sampling
    s2_by_country = defaultdict(list)
    for eid, rec in s2_records.items():
        s2_by_country[rec["country"]].append(eid)
    s3_by_country = defaultdict(list)
    for eid, rec in s3_records.items():
        s3_by_country[rec["country"]].append(eid)

    # Shuffled list of entities
    s1_list = list(s1_set)
    rng = random.Random(SEED)
    rng.shuffle(s1_list)

    pairs = []
    labels = []
    included_s1 = set()

    for s1_id in s1_list:
        if len(pairs) >= max_pairs:
            break

        matches = gt.get(s1_id, set())
        country = s1_records.get(s1_id, {}).get("country", "")

        # Positive pairs
        valid_positives = []
        for mid in matches:
            if mid.startswith("S2-") and mid in s2_records:
                valid_positives.append(mid)
            elif mid.startswith("S3-") and mid in s3_records:
                valid_positives.append(mid)

        # Negative pairs — sample from same country
        num_neg = max(1, len(valid_positives)) * neg_ratio
        neg_pool_s2 = s2_by_country.get(country, [])
        neg_pool_s3 = s3_by_country.get(country, [])

        neg_candidates = set()
        attempts = 0
        max_attempts = num_neg * 10
        while len(neg_candidates) < num_neg and attempts < max_attempts:
            attempts += 1
            if rng.random() < 0.5 and neg_pool_s2:
                cand = rng.choice(neg_pool_s2)
            elif neg_pool_s3:
                cand = rng.choice(neg_pool_s3)
            elif neg_pool_s2:
                cand = rng.choice(neg_pool_s2)
            else:
                break
            if cand not in matches:
                neg_candidates.add(cand)

        # Add pairs
        for mid in valid_positives:
            pairs.append((s1_id, mid))
            labels.append(1)

        for cand in neg_candidates:
            pairs.append((s1_id, cand))
            labels.append(0)

        included_s1.add(s1_id)

    labels = np.array(labels, dtype=np.int8)
    pos_count = int(labels.sum())
    neg_count = len(labels) - pos_count
    log.info(f"  Generated {len(pairs):,} pairs across {len(included_s1):,} S1 entities ({pos_count:,} pos, {neg_count:,} neg)")

    return pairs, labels, included_s1


# ═══════════════════════════════════════════════════════════════════════════
# 4. HIGH-THROUGHPUT PARALLEL FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_chunk(chunk_data: list) -> np.ndarray:
    """Worker function computing features for a chunk of pairs."""
    rows = []
    for s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, source_type in chunk_data:
        feats = compute_pair_features(
            s1_name, s1_addr, s1_country,
            cand_name, cand_addr, cand_country,
            source_type=source_type,
        )
        rows.append(feats)
    return np.array(rows, dtype=np.float32)


def build_pair_payloads(pairs: list, s1: dict, s2: dict, s3: dict) -> list:
    """Extract lightweight text tuples for sampled pairs."""
    items = []
    for s1_id, cand_id in pairs:
        s1_rec = s1.get(s1_id, {"name": "", "addr": "", "country": ""})
        if cand_id.startswith("S2-"):
            cand_rec = s2.get(cand_id, {"name": "", "addr": "", "country": ""})
            st = 0
        else:
            cand_rec = s3.get(cand_id, {"name": "", "addr": "", "country": ""})
            st = 1
        items.append((
            s1_rec["name"], s1_rec["addr"], s1_rec["country"],
            cand_rec["name"], cand_rec["addr"], cand_rec["country"],
            st
        ))
    return items


def compute_features_parallel(
    payloads: list,
    n_workers: int = NUM_WORKERS,
    chunk_size: int = CHUNK_SIZE,
    batch_label: str = "",
) -> np.ndarray:
    """
    Compute features for all pairs using chunked ProcessPoolExecutor.
    Achieves 80,000+ pairs/sec with minimal IPC and RAM overhead.
    """
    n = len(payloads)
    log.info(f"Computing features for {n:,} pairs {batch_label} using {n_workers} workers (chunk_size={chunk_size:,})...")
    start = time.time()

    chunks = [payloads[i:i + chunk_size] for i in range(0, n, chunk_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        matrices = list(executor.map(_compute_chunk, chunks))

    features = np.vstack(matrices)
    elapsed = time.time() - start
    rate = n / elapsed if elapsed > 0 else 0
    log.info(f"  Features computed in {elapsed:.1f}s ({rate:,.0f} pairs/sec) | Shape: {features.shape} | RAM: {features.nbytes / (1024*1024):.1f}MB")
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 5. F0.5 EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def f05_per_entity(predicted_ids: set, true_ids: set) -> float:
    """Compute F0.5 for a single S1 entity."""
    if len(true_ids) == 0 and len(predicted_ids) == 0:
        return 1.0  # Correct singleton
    if len(true_ids) == 0 and len(predicted_ids) > 0:
        return 0.0  # False merge on singleton
    if len(predicted_ids) == 0 and len(true_ids) > 0:
        return 0.0  # Missed all matches

    tp = len(predicted_ids & true_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 1.25 * precision * recall / (0.25 * precision + recall)


def compute_macro_f05(
    predictions: dict,  # {s1_id: set(predicted_ids)}
    ground_truth: dict,  # {s1_id: set(true_ids)}
    s1_entities: set,    # all S1 entities to evaluate
) -> Tuple[float, List[float]]:
    """Compute macro F0.5 across evaluated S1 entities."""
    scores = []
    for s1_id in s1_entities:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(f05_per_entity(pred, true))
    return float(np.mean(scores)), scores


def _apply_threshold(pairs: list, probs: np.ndarray, threshold: float) -> dict:
    """Apply threshold to probabilities and group predictions by S1 entity."""
    preds = defaultdict(set)
    for (s1_id, cand_id), prob in zip(pairs, probs):
        if prob >= threshold:
            preds[s1_id].add(cand_id)
    return dict(preds)


# ═══════════════════════════════════════════════════════════════════════════
# 6. THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def optimize_threshold(
    val_pairs: list,
    val_probs: np.ndarray,
    val_gt: dict,
    val_s1_set: set,
) -> Tuple[float, float, dict]:
    """
    Grid search over thresholds to maximize macro F0.5 on validation set.
    Returns: (best_threshold, best_f05, threshold_scores_dict)
    """
    log.info("Optimizing match threshold for Macro F0.5...")

    coarse_thresholds = np.arange(0.10, 0.95, 0.05)
    best_f05 = -1.0
    best_t = 0.5
    threshold_scores = {}

    for t in coarse_thresholds:
        preds = _apply_threshold(val_pairs, val_probs, t)
        f05, _ = compute_macro_f05(preds, val_gt, val_s1_set)
        t_key = round(float(t), 2)
        threshold_scores[t_key] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_t = float(t)
        log.info(f"  t={t:.2f}  →  Macro F0.5={f05:.4f}")

    fine_start = max(0.05, best_t - 0.08)
    fine_end = min(0.95, best_t + 0.08)
    fine_thresholds = np.arange(fine_start, fine_end, 0.01)

    for t in fine_thresholds:
        t_key = round(float(t), 2)
        if t_key in threshold_scores:
            continue
        preds = _apply_threshold(val_pairs, val_probs, t)
        f05, _ = compute_macro_f05(preds, val_gt, val_s1_set)
        threshold_scores[t_key] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_t = float(t)
            log.info(f"  t={t:.2f}  →  Macro F0.5={f05:.4f} [NEW BEST]")

    log.info(f"  ★ Optimal threshold: {best_t:.2f}  →  Macro F0.5 = {best_f05:.4f}")
    return best_t, best_f05, threshold_scores


# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL TRAINING — CUDA-ACCELERATED
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    backend: str = MODEL_BACKEND,
) -> Tuple[object, str]:
    """
    Train GBDT classifier with CUDA acceleration (XGBoost) or CPU LightGBM.
    """
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    if backend == "xgb_cuda":
        log.info(f"Training XGBoost on CUDA GPU ({X_train.shape[0]:,} pairs, {len(FEATURE_NAMES)} features)...")
        log.info(f"  scale_pos_weight: {scale_pos:.2f} (pos={pos_count:,}, neg={neg_count:,})")

        model = xgb.XGBClassifier(
            tree_method="hist",
            device="cuda",
            n_estimators=600,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=50,
            random_state=SEED,
        )

        t0 = time.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        log.info(f"XGBoost training completed in {time.time() - t0:.1f}s")
        return model, "xgb_cuda"

    else:
        log.info(f"Training LightGBM on CPU ({X_train.shape[0]:,} pairs, {len(FEATURE_NAMES)} features)...")
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data, free_raw_data=False)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.08,
            "min_child_samples": 100,
            "scale_pos_weight": scale_pos,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": NUM_WORKERS,
            "verbose": -1,
            "seed": SEED,
        }

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50, verbose=True),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=600,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        return model, "lgbm_cpu"


def predict_probabilities(model, model_type: str, X: np.ndarray) -> np.ndarray:
    """Predict match probabilities."""
    if model_type == "xgb_cuda":
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X, num_iteration=model.best_iteration)


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("ENTITY RESOLUTION — MATCHING MODEL PIPELINE (CUDA ACCELERATED)")
    log.info(f"Workers: {NUM_WORKERS}  |  Val fraction: {VAL_FRACTION}  |  Backend: {MODEL_BACKEND}")
    log.info("=" * 70)

    overall_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    s1 = load_source(TRAIN_DIR / "train_source1.tsv")
    s2 = load_source(TRAIN_DIR / "train_source2.tsv")
    s3 = load_source(TRAIN_DIR / "train_source3.tsv")
    gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")

    # ── Validation split ─────────────────────────────────────────────────
    train_s1, val_s1 = create_val_split(gt, s1, val_fraction=VAL_FRACTION)

    # ── Generate pairs ───────────────────────────────────────────────────
    train_pairs, train_labels, train_evaluated_s1 = generate_pairs(
        train_s1, gt, s1, s2, s3,
        neg_ratio=NEGATIVE_RATIO,
        max_pairs=MAX_TRAIN_PAIRS,
    )
    val_pairs, val_labels, val_evaluated_s1 = generate_pairs(
        val_s1, gt, s1, s2, s3,
        neg_ratio=NEGATIVE_RATIO,
        max_pairs=MAX_VAL_PAIRS,
    )

    # ── Convert to lightweight payloads and free 12.5M records ───────────
    log.info("Converting sampled pairs to text payloads...")
    train_payloads = build_pair_payloads(train_pairs, s1, s2, s3)
    val_payloads = build_pair_payloads(val_pairs, s1, s2, s3)

    log.info("Freeing full dataset dictionaries (releasing ~6 GB RAM)...")
    del s1, s2, s3
    gc.collect()

    # ── Compute features ─────────────────────────────────────────────────
    X_train = compute_features_parallel(train_payloads, batch_label="[train]")
    X_val = compute_features_parallel(val_payloads, batch_label="[val]")

    # ── Train model ──────────────────────────────────────────────────────
    model, model_type = train_model(X_train, train_labels, X_val, val_labels, backend=MODEL_BACKEND)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_type == "xgb_cuda":
        model_path = MODEL_DIR / "matching_model.json"
        model.save_model(str(model_path))
    else:
        model_path = MODEL_DIR / "matching_lgbm.txt"
        model.save_model(str(model_path))
    log.info(f"Model saved to {model_path}")

    # ── Predict on validation ────────────────────────────────────────────
    val_probs = predict_probabilities(model, model_type, X_val)
    log.info(f"Validation predictions: min={val_probs.min():.4f}, max={val_probs.max():.4f}, mean={val_probs.mean():.4f}")

    # ── Threshold optimization ───────────────────────────────────────────
    val_gt_subset = {s1_id: gt.get(s1_id, set()) for s1_id in val_evaluated_s1}
    best_threshold, best_f05, threshold_scores = optimize_threshold(
        val_pairs, val_probs, val_gt_subset, val_evaluated_s1,
    )

    # ── Report final scores ──────────────────────────────────────────────
    final_preds = _apply_threshold(val_pairs, val_probs, best_threshold)
    final_f05, per_entity = compute_macro_f05(final_preds, val_gt_subset, val_evaluated_s1)

    singleton_scores = []
    non_singleton_scores = []
    for s1_id, score in zip(val_evaluated_s1, per_entity):
        if len(gt.get(s1_id, set())) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    log.info("=" * 70)
    log.info("FINAL RESULTS")
    log.info("=" * 70)
    log.info(f"  Best threshold:         {best_threshold:.3f}")
    log.info(f"  Macro F0.5 (overall):   {final_f05:.4f}")
    log.info(f"  F0.5 (singletons):      {np.mean(singleton_scores):.4f} (n={len(singleton_scores):,})")
    log.info(f"  F0.5 (non-singletons):  {np.mean(non_singleton_scores):.4f} (n={len(non_singleton_scores):,})")

    per_entity_arr = np.array(per_entity)
    log.info(f"  Score dist: min={per_entity_arr.min():.4f}, p25={np.percentile(per_entity_arr, 25):.4f}, median={np.median(per_entity_arr):.4f}, p75={np.percentile(per_entity_arr, 75):.4f}, max={per_entity_arr.max():.4f}")

    perfect = int((per_entity_arr == 1.0).sum())
    zero = int((per_entity_arr == 0.0).sum())
    log.info(f"  Perfect (1.0): {perfect:,} ({100*perfect/len(per_entity_arr):.1f}%)")
    log.info(f"  Zero (0.0):    {zero:,} ({100*zero/len(per_entity_arr):.1f}%)")

    # Feature importances
    if model_type == "xgb_cuda":
        importances = model.feature_importances_
    else:
        importances = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importances)[::-1]
    log.info("\nTop 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        log.info(f"  {FEATURE_NAMES[idx]:30s}: {importances[idx]:.4f}")

    elapsed = time.time() - overall_start
    log.info(f"\nTotal pipeline time: {elapsed/60:.2f} minutes")

    # ── Save threshold and config ────────────────────────────────────────
    config = {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "val_fraction": VAL_FRACTION,
        "neg_ratio": NEGATIVE_RATIO,
        "model_type": model_type,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "threshold_scores": threshold_scores,
    }
    config_path = MODEL_DIR / "matching_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    log.info(f"Config saved to {config_path}")

    return model, best_threshold, final_f05


if __name__ == "__main__":
    main()
