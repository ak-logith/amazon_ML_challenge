#!/usr/bin/env python3
"""
Train Production Matching Model with Blocker Hard Negatives
===========================================================
Replaces the easy-negative baseline model with a high-precision
model trained on candidate-generation hard negatives + random negatives.

Evaluated on held-out validation S1 entities against ALL their blocker candidates.
Saves model to models/matching_model.json and config to models/matching_config.pkl.
"""

import sys
import gc
import csv
import time
import random
import shutil
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_pipeline import (
    load_source_records,
    load_candidates,
    load_ground_truth,
    compute_features_for_batch,
    FEATURE_NAMES,
)
from src.generate_candidates import resolve_dataset_dir

def train_and_save():
    print("=" * 75)
    print("PRODUCTION MATCHING MODEL TRAINING (HARD NEGATIVES + CALIBRATED THRESHOLD)")
    print("=" * 75)

    base_dir = resolve_dataset_dir() / "train"
    cand_path = PROJECT_ROOT / "output" / "train_candidate_pairs.tsv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backup baseline model if not already backed up
    baseline_model = models_dir / "matching_model.json"
    backup_model = models_dir / "matching_model_baseline_random.json"
    if baseline_model.exists() and not backup_model.exists():
        shutil.copy2(baseline_model, backup_model)
        print(f"Backed up baseline model to {backup_model.name}")

    # 2. Load candidates and ground truth
    cands = load_candidates(cand_path)
    all_s1 = list(cands.keys())
    print(f"Loaded {len(all_s1):,} S1 entities from {cand_path.name}")

    gt = load_ground_truth(base_dir / "train_ground_truth.tsv")

    # 3. Train / Validation split (80/20 on the 2,000 S1 entities)
    rng = random.Random(42)
    shuffled_s1 = list(all_s1)
    rng.shuffle(shuffled_s1)
    n_train = int(len(shuffled_s1) * 0.80)
    train_s1 = set(shuffled_s1[:n_train])
    val_s1 = set(shuffled_s1[n_train:])
    print(f"Dataset split: {len(train_s1):,} train S1 | {len(val_s1):,} val S1")

    # 4. Construct training set with:
    # - All true positives
    # - Up to 60 hard negatives per S1 from blocker candidate pool
    # - 5 random negatives per S1 to preserve global negative calibration
    train_pairs = []
    train_labels = []
    needed_cand_ids = set()

    for sid in train_s1:
        sid_cands = cands[sid]
        true_ids = gt.get(sid, set())
        
        pos = [c for c in sid_cands if c in true_ids]
        neg = [c for c in sid_cands if c not in true_ids]

        # Positives
        for c in pos:
            train_pairs.append((sid, c))
            train_labels.append(1)
            needed_cand_ids.add(c)

        # Hard negatives from candidate pool (60 per S1)
        sample_k = min(len(neg), 60)
        sampled_neg = rng.sample(neg, sample_k) if sample_k > 0 else []
        for c in sampled_neg:
            train_pairs.append((sid, c))
            train_labels.append(0)
            needed_cand_ids.add(c)

    # Validation pairs: ALL candidate pairs for val_s1
    val_pairs = []
    for sid in val_s1:
        for cid in cands[sid]:
            val_pairs.append((sid, cid))
            needed_cand_ids.add(cid)

    print(f"Train pairs: {len(train_pairs):,} ({sum(train_labels):,} pos, {len(train_labels)-sum(train_labels):,} neg)")
    print(f"Validation pairs (ALL blocker candidates): {len(val_pairs):,}")

    # 5. Load needed source records
    needed_s1_ids = set(all_s1)
    print(f"\nLoading source records for {len(needed_s1_ids):,} S1 and {len(needed_cand_ids):,} candidates...")
    def load_needed(path, needed):
        rec = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                eid = r["entity_id"]
                if eid in needed:
                    rec[eid] = {
                        "name": r.get("business_name", "") or "",
                        "addr": r.get("business_address", "") or "",
                        "country": r.get("country", "") or "",
                    }
        return rec

    t0 = time.time()
    s1_rec = load_needed(base_dir / "train_source1.tsv", needed_s1_ids)
    s2_rec = load_needed(base_dir / "train_source2.tsv", needed_cand_ids)
    s3_rec = load_needed(base_dir / "train_source3.tsv", needed_cand_ids)
    print(f"Loaded all required source records in {time.time()-t0:.1f}s.")

    # 6. Feature computation
    print("\nComputing features for train pairs...")
    t0 = time.time()
    X_train = compute_features_for_batch(train_pairs, s1_rec, s2_rec, s3_rec)
    y_train = np.array(train_labels, dtype=np.int8)
    print(f"X_train computed: {X_train.shape} in {time.time()-t0:.1f}s.")

    print("Computing features for val pairs...")
    t0 = time.time()
    X_val = compute_features_for_batch(val_pairs, s1_rec, s2_rec, s3_rec)
    print(f"X_val computed: {X_val.shape} in {time.time()-t0:.1f}s.")

    # 7. Model training (XGBoost GPU)
    print("\nTraining XGBoost model on GPU...")
    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time()-t0:.2f}s.")

    # 8. Evaluation & Threshold Optimization
    print("\nEvaluating and optimizing threshold on validation set (ALL blocker candidates)...")
    val_probs = model.predict_proba(X_val)[:, 1]

    val_pairs_by_s1 = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        val_pairs_by_s1[sid].append((cid, prob))

    threshold_scores = {}
    best_thresh = 0.90
    best_f05 = -1.0
    best_avg_matches = 0.0

    test_thresholds = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.980, 0.985, 0.990, 0.995]
    for thresh in test_thresholds:
        s1_f05_list = []
        tot_matches = 0
        for sid in val_s1:
            true_set = gt.get(sid, set())
            pred_set = set(cid for cid, p in val_pairs_by_s1[sid] if p >= thresh)
            tot_matches += len(pred_set)

            if len(true_set) == 0 and len(pred_set) == 0:
                score = 1.0
            elif len(true_set) == 0 or len(pred_set) == 0:
                score = 0.0
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
            s1_f05_list.append(score)

        macro_f05 = float(np.mean(s1_f05_list))
        avg_m = tot_matches / len(val_s1)
        threshold_scores[round(thresh, 3)] = macro_f05
        print(f"  Threshold {thresh:.3f}: Macro F0.5 = {macro_f05:.4f} (Avg matches/S1: {avg_m:.2f})")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh
            best_avg_matches = avg_m

    print("-" * 75)
    print(f"★ OPTIMAL THRESHOLD: {best_thresh:.3f} | Macro F0.5: {best_f05:.4f} | Avg matches/S1: {best_avg_matches:.2f}")
    print("-" * 75)

    # 9. Save production model and config
    out_model_path = models_dir / "matching_model.json"
    out_config_path = models_dir / "matching_config.pkl"

    model.save_model(str(out_model_path))
    print(f"Saved production model to {out_model_path}")

    config = {
        "best_threshold": float(best_thresh),
        "best_f05": float(best_f05),
        "val_fraction": 0.20,
        "neg_strategy": "blocker_hard_negatives",
        "model_type": "xgb_cuda",
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "threshold_scores": threshold_scores,
    }
    with open(out_config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Saved model config to {out_config_path}")
    print("=" * 75)
    print("MATCHING MODEL TRAINING COMPLETE & READY FOR INFERENCE")
    print("=" * 75)

if __name__ == "__main__":
    train_and_save()
