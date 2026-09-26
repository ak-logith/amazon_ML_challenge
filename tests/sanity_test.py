"""
Quick sanity test of the matching pipeline on a small subset.
Tests that all components work end-to-end before the full run.
"""

__test__ = False  # Standalone benchmark script, not a pytest test case

import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.matching.pipeline import (
    load_source, load_ground_truth, create_val_split,
    generate_pairs, compute_features_parallel, train_model,
    optimize_threshold, compute_macro_f05, _apply_threshold,
    TRAIN_DIR, NUM_WORKERS, FEATURE_NAMES,
)
import logging
log = logging.getLogger("matching")

def main():
    log.info("=== SANITY TEST (small subset) ===")
    start = time.time()

    # Load FULL data (we need it all for pair generation)
    s1 = load_source(TRAIN_DIR / "train_source1.tsv")
    s2 = load_source(TRAIN_DIR / "train_source2.tsv")
    s3 = load_source(TRAIN_DIR / "train_source3.tsv")
    gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")

    # Take a small subset of S1 for testing
    import random
    rng = random.Random(42)
    all_s1 = list(gt.keys())
    # Take 5000 S1 entities
    subset_s1 = set(rng.sample(all_s1, min(5000, len(all_s1))))

    # Split into mini train/val
    subset_list = list(subset_s1)
    split_idx = int(len(subset_list) * 0.85)
    rng.shuffle(subset_list)
    train_s1 = set(subset_list[:split_idx])
    val_s1 = set(subset_list[split_idx:])

    log.info(f"Subset: {len(train_s1)} train, {len(val_s1)} val")

    # Generate pairs
    train_pairs, train_labels = generate_pairs(
        train_s1, gt, s1, s2, s3, neg_ratio=3, max_pairs=200_000
    )
    val_pairs, val_labels = generate_pairs(
        val_s1, gt, s1, s2, s3, neg_ratio=3, max_pairs=50_000
    )

    # Compute features
    X_train = compute_features_parallel(train_pairs, s1, s2, s3, n_workers=NUM_WORKERS, batch_label="[sanity-train]")
    X_val = compute_features_parallel(val_pairs, s1, s2, s3, n_workers=NUM_WORKERS, batch_label="[sanity-val]")

    log.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    log.info(f"Train pos rate: {train_labels.mean():.3f}, Val pos rate: {val_labels.mean():.3f}")

    # Train model
    model = train_model(X_train, train_labels, X_val, val_labels)

    # Predict + threshold
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1}
    best_t, best_f05, scores = optimize_threshold(val_pairs, val_probs, val_gt, val_s1)

    log.info(f"\n=== SANITY TEST RESULTS ===")
    log.info(f"Best threshold: {best_t:.3f}")
    log.info(f"Best F0.5: {best_f05:.4f}")
    log.info(f"Time: {time.time() - start:.1f}s")
    log.info("=== Sanity test PASSED ===")


if __name__ == "__main__":
    main()
