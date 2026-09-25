"""
Entity Resolution Matching Pipeline
====================================
End-to-end pipeline: data loading → pair generation → feature computation →
LightGBM training → threshold optimization → F0.5 evaluation.

Optimized for maximum parallelism via joblib + multiprocessing.

Usage:
    python -m src.matching.pipeline
"""

import os
import sys
import csv
import time
import random
import hashlib
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

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

# Parallelism — max workers for feature computation
NUM_WORKERS = max(1, os.cpu_count() - 1)  # Leave 1 core for OS

# Sampling config
NEGATIVE_RATIO = 4       # negatives per positive
MAX_TRAIN_PAIRS = 5_000_000   # cap training pairs to manage memory/time
MAX_VAL_PAIRS = 1_500_000     # cap validation pairs

# Validation split — flexible, not rigid 80/20
VAL_FRACTION = 0.15  # 85/15 split — more training data while still validating

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
                "name": row.get("business_name", ""),
                "addr": row.get("business_address", ""),
                "country": row.get("country", ""),
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
            s1_id = row["source1_entity_id"]
            ids_str = row["matched_entity_ids"].strip()
            if ids_str:
                gt[s1_id] = set(ids_str.split(","))
            else:
                gt[s1_id] = set()
    log.info(f"  Loaded {len(gt):,} S1 entities in ground truth")
    return gt


# ═══════════════════════════════════════════════════════════════════════════
# 2. VALIDATION SPLIT — Stratified S1-entity level
# ═══════════════════════════════════════════════════════════════════════════

def create_val_split(gt: dict, s1_records: dict, val_fraction: float = VAL_FRACTION):
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
        # Bucket: 0, 1-2, 3-5, 6+
        if mc == 0:
            bucket = "0"
        elif mc <= 2:
            bucket = "1-2"
        elif mc <= 5:
            bucket = "3-5"
        else:
            bucket = "6+"
        match_counts.append(bucket)

    # Create stratification key
    strat_keys = [f"{c}_{b}" for c, b in zip(countries, match_counts)]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(s1_ids, strat_keys))

    train_s1 = set(s1_ids[i] for i in train_idx)
    val_s1 = set(s1_ids[i] for i in val_idx)

    # Log statistics
    train_singletons = sum(1 for s in train_s1 if len(gt[s]) == 0)
    val_singletons = sum(1 for s in val_s1 if len(gt[s]) == 0)
    train_positives = sum(len(gt[s]) for s in train_s1)
    val_positives = sum(len(gt[s]) for s in val_s1)

    log.info(f"  Train: {len(train_s1):,} S1 entities ({train_singletons:,} singletons, {train_positives:,} positive links)")
    log.info(f"  Val:   {len(val_s1):,} S1 entities ({val_singletons:,} singletons, {val_positives:,} positive links)")

    return train_s1, val_s1


# ═══════════════════════════════════════════════════════════════════════════
# 3. PAIR GENERATION — Positive + Negative sampling
# ═══════════════════════════════════════════════════════════════════════════

def generate_pairs(
    s1_set: set,
    gt: dict,
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    neg_ratio: int = NEGATIVE_RATIO,
    max_pairs: int = MAX_TRAIN_PAIRS,
) -> tuple:
    """
    Generate positive and negative pairs for a set of S1 entities.

    Returns: (pair_list, labels)
        pair_list: list of (s1_id, candidate_id)
        labels: np.array of 0/1
    """
    log.info(f"Generating pairs for {len(s1_set):,} S1 entities (neg_ratio={neg_ratio})...")

    # Build country-indexed pools for efficient negative sampling
    s2_by_country = defaultdict(list)
    for eid, rec in s2_records.items():
        s2_by_country[rec["country"]].append(eid)
    s3_by_country = defaultdict(list)
    for eid, rec in s3_records.items():
        s3_by_country[rec["country"]].append(eid)

    pairs = []
    labels = []
    rng = random.Random(SEED)

    for s1_id in s1_set:
        matches = gt.get(s1_id, set())
        country = s1_records.get(s1_id, {}).get("country", "")

        # Positive pairs
        valid_positives = []
        for mid in matches:
            if mid.startswith("S2-") and mid in s2_records:
                valid_positives.append(mid)
            elif mid.startswith("S3-") and mid in s3_records:
                valid_positives.append(mid)

        for mid in valid_positives:
            pairs.append((s1_id, mid))
            labels.append(1)

        # Negative pairs — sample from same country
        num_neg = max(1, len(valid_positives)) * neg_ratio
        # Mix S2 and S3 negatives
        neg_pool_s2 = s2_by_country.get(country, [])
        neg_pool_s3 = s3_by_country.get(country, [])

        neg_candidates = set()
        attempts = 0
        max_attempts = num_neg * 10
        while len(neg_candidates) < num_neg and attempts < max_attempts:
            attempts += 1
            # Alternate S2 and S3
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

        for cand in neg_candidates:
            pairs.append((s1_id, cand))
            labels.append(0)

    labels = np.array(labels, dtype=np.int8)
    log.info(f"  Generated {len(pairs):,} pairs ({labels.sum():,} positive, {len(pairs) - labels.sum():,} negative)")

    # Subsample if too many
    if len(pairs) > max_pairs:
        log.info(f"  Subsampling to {max_pairs:,} pairs...")
        indices = list(range(len(pairs)))
        rng.shuffle(indices)

        # Keep all positives, subsample negatives
        pos_idx = [i for i in indices if labels[i] == 1]
        neg_idx = [i for i in indices if labels[i] == 0]

        # Keep all positives up to half the budget
        max_pos = min(len(pos_idx), max_pairs // 2)
        max_neg = max_pairs - max_pos

        selected = pos_idx[:max_pos] + neg_idx[:max_neg]
        rng.shuffle(selected)

        pairs = [pairs[i] for i in selected]
        labels = np.array([labels[i] for i in selected], dtype=np.int8)
        log.info(f"  After subsampling: {len(pairs):,} pairs ({labels.sum():,} pos, {len(pairs) - labels.sum():,} neg)")

    return pairs, labels


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARALLEL FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_features_for_pair(args):
    """Worker function for computing features for a single pair."""
    s1_id, cand_id, s1_rec, cand_rec = args
    source_type = 0 if cand_id.startswith("S2-") else 1
    return compute_pair_features(
        s1_rec["name"], s1_rec["addr"], s1_rec["country"],
        cand_rec["name"], cand_rec["addr"], cand_rec["country"],
        source_type=source_type,
    )


def compute_features_parallel(
    pairs: list,
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    n_workers: int = NUM_WORKERS,
    batch_label: str = "",
) -> np.ndarray:
    """
    Compute features for all pairs using parallel workers.
    Returns: (N, num_features) numpy array.
    """
    n = len(pairs)
    log.info(f"Computing features for {n:,} pairs {batch_label} using {n_workers} workers...")
    start = time.time()

    # Prepare args
    def _gen_args():
        for s1_id, cand_id in pairs:
            s1_rec = s1_records.get(s1_id, {"name": "", "addr": "", "country": ""})
            if cand_id.startswith("S2-"):
                cand_rec = s2_records.get(cand_id, {"name": "", "addr": "", "country": ""})
            else:
                cand_rec = s3_records.get(cand_id, {"name": "", "addr": "", "country": ""})
            yield (s1_id, cand_id, s1_rec, cand_rec)

    # Use joblib for parallel computation
    results = Parallel(n_jobs=n_workers, batch_size=1000, prefer="processes")(
        delayed(_compute_features_for_pair)(args) for args in _gen_args()
    )

    features = np.stack(results)
    elapsed = time.time() - start
    rate = n / elapsed if elapsed > 0 else 0
    log.info(f"  Features computed in {elapsed:.1f}s ({rate:,.0f} pairs/sec)")
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
) -> tuple:
    """Compute macro F0.5 across all S1 entities. Returns (score, per_entity_scores)."""
    scores = []
    for s1_id in s1_entities:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(f05_per_entity(pred, true))
    return np.mean(scores), scores


# ═══════════════════════════════════════════════════════════════════════════
# 6. THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def optimize_threshold(
    val_pairs: list,
    val_probs: np.ndarray,
    val_gt: dict,
    val_s1_set: set,
) -> tuple:
    """
    Grid search over thresholds to maximize macro F0.5 on validation.
    Returns: (best_threshold, best_f05, threshold_scores_dict)
    """
    log.info("Optimizing match threshold...")

    # Coarse sweep
    coarse_thresholds = np.arange(0.10, 0.95, 0.05)
    best_f05 = 0.0
    best_t = 0.5
    threshold_scores = {}

    for t in coarse_thresholds:
        preds = _apply_threshold(val_pairs, val_probs, t)
        f05, _ = compute_macro_f05(preds, val_gt, val_s1_set)
        threshold_scores[round(t, 2)] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
        log.info(f"  t={t:.2f}  →  F0.5={f05:.4f}")

    # Fine sweep around the best coarse threshold
    fine_start = max(0.05, best_t - 0.08)
    fine_end = min(0.95, best_t + 0.08)
    fine_thresholds = np.arange(fine_start, fine_end, 0.01)

    for t in fine_thresholds:
        t_round = round(t, 2)
        if t_round in threshold_scores:
            continue
        preds = _apply_threshold(val_pairs, val_probs, t)
        f05, _ = compute_macro_f05(preds, val_gt, val_s1_set)
        threshold_scores[t_round] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
        log.info(f"  t={t:.2f}  →  F0.5={f05:.4f} {'← NEW BEST' if f05 >= best_f05 else ''}")

    log.info(f"  ★ Best threshold: {best_t:.2f}  →  F0.5={best_f05:.4f}")
    return best_t, best_f05, threshold_scores


def _apply_threshold(pairs, probs, threshold):
    """Apply threshold to probabilities and group by S1 entity."""
    preds = defaultdict(set)
    for (s1_id, cand_id), prob in zip(pairs, probs):
        if prob >= threshold:
            preds[s1_id].add(cand_id)
    return dict(preds)


# ═══════════════════════════════════════════════════════════════════════════
# 7. LIGHTGBM TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.Booster:
    """Train a LightGBM binary classifier."""
    log.info(f"Training LightGBM on {X_train.shape[0]:,} pairs ({FEATURE_NAMES[:3]}... {len(FEATURE_NAMES)} features)")

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
    log.info(f"  Positive: {pos_count:,}, Negative: {neg_count:,}, scale_pos_weight: {scale_pos:.2f}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.1,
        "min_child_samples": 100,
        "scale_pos_weight": scale_pos,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
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
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    log.info(f"  Best iteration: {model.best_iteration}")
    log.info(f"  Best val AUC: {model.best_score.get('val', {}).get('auc', 'N/A')}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    log.info("  Feature importances (gain):")
    for i in sorted_idx[:10]:
        log.info(f"    {FEATURE_NAMES[i]:30s}: {importance[i]:,.0f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("ENTITY RESOLUTION — MATCHING MODEL PIPELINE")
    log.info(f"Workers: {NUM_WORKERS}  |  Val fraction: {VAL_FRACTION}")
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
    train_pairs, train_labels = generate_pairs(
        train_s1, gt, s1, s2, s3,
        neg_ratio=NEGATIVE_RATIO,
        max_pairs=MAX_TRAIN_PAIRS,
    )
    val_pairs, val_labels = generate_pairs(
        val_s1, gt, s1, s2, s3,
        neg_ratio=NEGATIVE_RATIO,
        max_pairs=MAX_VAL_PAIRS,
    )

    # ── Compute features ─────────────────────────────────────────────────
    X_train = compute_features_parallel(train_pairs, s1, s2, s3, batch_label="[train]")
    X_val = compute_features_parallel(val_pairs, s1, s2, s3, batch_label="[val]")

    log.info(f"Feature matrices: train={X_train.shape}, val={X_val.shape}")

    # ── Train model ──────────────────────────────────────────────────────
    model = train_model(X_train, train_labels, X_val, val_labels)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "matching_lgbm.txt"
    model.save_model(str(model_path))
    log.info(f"Model saved to {model_path}")

    # ── Predict on validation ────────────────────────────────────────────
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    log.info(f"Validation predictions: min={val_probs.min():.4f}, max={val_probs.max():.4f}, mean={val_probs.mean():.4f}")

    # ── Threshold optimization ───────────────────────────────────────────
    val_gt_subset = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1}
    best_threshold, best_f05, threshold_scores = optimize_threshold(
        val_pairs, val_probs, val_gt_subset, val_s1,
    )

    # ── Report final scores ──────────────────────────────────────────────
    final_preds = _apply_threshold(val_pairs, val_probs, best_threshold)
    final_f05, per_entity = compute_macro_f05(final_preds, val_gt_subset, val_s1)

    # Breakdown by category
    singleton_scores = []
    non_singleton_scores = []
    for s1_id, score in zip(val_s1, per_entity):
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

    # Score distribution
    per_entity_arr = np.array(per_entity)
    log.info(f"  Per-entity score dist:  min={per_entity_arr.min():.4f}, "
             f"p25={np.percentile(per_entity_arr, 25):.4f}, "
             f"median={np.median(per_entity_arr):.4f}, "
             f"p75={np.percentile(per_entity_arr, 75):.4f}, "
             f"max={per_entity_arr.max():.4f}")

    perfect = (per_entity_arr == 1.0).sum()
    zero = (per_entity_arr == 0.0).sum()
    log.info(f"  Perfect (1.0): {perfect:,} ({100*perfect/len(per_entity_arr):.1f}%)")
    log.info(f"  Zero (0.0):    {zero:,} ({100*zero/len(per_entity_arr):.1f}%)")

    elapsed = time.time() - overall_start
    log.info(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")

    # ── Save threshold and config ────────────────────────────────────────
    config = {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "val_fraction": VAL_FRACTION,
        "neg_ratio": NEGATIVE_RATIO,
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
