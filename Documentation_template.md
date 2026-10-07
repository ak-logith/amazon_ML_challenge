# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** The Entity Resolvers  
**Team Members:** Team Antigravity  
**Submission Date:** October 6, 2026  

---

## 1. Executive Summary

We developed an end-to-end, memory-bounded, GPU-accelerated business entity resolution pipeline capable of linking millions of multi-source records (Source 1 to Source 2 and Source 3) across multilingual entity names (English, French, Hindi/Indic) and noisy address representations. Our solution pairs a High-Recall Multi-Channel Inverted-Index Blocker (achieving **91.41% bounded recall** under dynamic budgeting) with a 23-feature CUDA-accelerated XGBoost classifier trained on blocker hard negatives. Operating with precision-preserving decision thresholding (τ = 0.950), our pipeline achieves an unbiased held-out **Macro F0.5 score of 0.8036** (Precision: 87.44%, Recall: 80.44%) while guaranteeing strict memory bounds (< 3.5 GB peak RAM) and sub-second GPU inference throughput (187,000 pairs/sec).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the 24.2M records (Train: 2.2M S1, 5.0M S2, 5.3M S3; Test: 1.7M S1, 4.9M S2, 5.1M S3), we identified five key challenges:
1. **Severe False-Merge Penalty (F0.5 Metric)**: The evaluation metric weights precision twice as heavily as recall (β = 0.5). False positive merges across disparate business branches or sibling companies severely degrade the macro score.
2. **Cross-Script and Multilingual Divergence**: Names span Latin and Indic scripts (Devanagari, Tamil, Bengali, Telugu), while the test set introduces France (259K entities) with European legal corporate suffixes (SARL, SAS, SASU, EURL).
3. **Address Noise & Incomplete Geocodes**: Street addresses suffer from phonetic transliterations, abbreviations (St, Rd, Ave, Ste, Fl), and divergent postal code formats (5-digit US ZIPs, 6-digit Indian PIN codes, 5-digit French postal codes).
4. **Negative Distribution Shift (Easy vs Hard Negatives)**: Models trained on uniform random negatives collapse on dense candidate pools (producing ~37 false positives per entity and F0.5 = 0.2186) because candidate-generation pools contain high-similarity negatives (same city, same state, shared common name words).
5. **Scale and Memory Limits**: Pairwise comparison of 1.7M S1 against 10M S2/S3 yields >10^13 pairs, requiring strict blocking and streaming memory management.

### 2.2 Solution Strategy

**Approach Type:** Multi-Channel Inverted Index Blocking + Pairwise 23-Feature CUDA GBDT Classifier + Streaming Output Generation  
**Core Innovation:** 
1. **Decoupled Multi-Channel Global Pooling**: Independent channel quotas with priority-weighted global scoring and true character 3-gram fallback, preventing dense address noise from crowding out rare business name tokens.
2. **Blocker-Derived Hard-Negative Retraining**: Training XGBoost directly on hard negative candidates retrieved by the blocker, aligning the model's decision boundary with candidate pool distributions.
3. **Line-by-Line Streaming Architecture**: Scoring candidate pairs in streaming batches of 10,000 entities directly from disk, preventing memory bloat on large-scale candidate sets.

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from 1.7x10^13 potential pairs to a bounded candidate set without sacrificing true matches, we implemented the V5.1 Multi-Channel Blocking Engine:

- **Partitioning & Discovery**: Strict country-partitioned blocking with open-set discovery (India, US, France). S1 entities are linked only against S2/S3 entities within the same country.
- **Channel A (Protected Name Tokens & 5-Char Prefixes)**: Rare name tokens (DF <= 250) receive weight W=16, standard tokens (250 < DF <= 6,000) receive W=6, common tokens (6,000 < DF <= 25,000) receive W=1, and 5-character prefixes receive W=3.
- **Channel B (Protected Address & Postal/Numeric Tokens)**: Standard address tokens receive W=2; extracted postal codes and street/plot numbers receive W=6.
- **Channel C (Composite Name ∩ Address)**: Candidates appearing in both Channel A and Channel B receive guaranteed inclusion and a composite bonus (W=14).
- **Channel D (Character 3-Gram Fallback)**: Automatically triggers when primary name candidates < 10, indexing character 3-grams from original and transliterated names (W=6 per shared trigram).
- **V5.1 Specificity Boosting**: Multi-evidence candidates receive additive bonuses: +20 for >= 2 distinct name token hits, +18 for strong name + postal match, +16 for strong composite agreement.
- **Evidence-Based Dynamic Budgeting**: Base budget 800, scaling up to BUDGET_MAX=1,500 (via pipeline default) based on high-quality candidate count. Minimum floor 50.
- **Source Balancing**: Selected candidates maintain a balanced proportion between Source 2 and Source 3 via per-source quota allocation (budget/2 per source, remainder filled globally).

### Measured Blocking Performance (on Training Ground Truth):
- **Final Bounded Blocking Recall**: **91.41%** (6,394 / 6,995 ground-truth pairs recovered)
- **Source 2 Recall**: **91.74%**
- **Source 3 Recall**: **91.10%**
- **Zero-Candidate S1 Entities**: **0.05%** (1 in 2,000)
- **Peak RAM Usage**: **< 300 MB**

---

## 4. Matching Model

### 4.1 Feature Engineering (23 Pairwise Similarity Features)
Features are computed in parallel across 8 CPU worker processes at 51,000+ pairs/sec:
1. `name_jaro_winkler`: Jaro-Winkler similarity on normalized business names.
2. `name_levenshtein`: Normalized Levenshtein ratio on cleaned names.
3. `name_jaccard`: Word-level Jaccard similarity on name token sets.
4. `name_token_sort`: Token-sort fuzzy matching ratio.
5. `name_token_set`: Token-set fuzzy matching ratio (handles subsets and permutations).
6. `name_containment`: Token containment ratio |A ∩ B| / min(|A|, |B|).
7. `name_exact`: Binary flag for exact normalized name equality.
8. `name_length_ratio`: min(len_a, len_b) / max(len_a, len_b).
9. `name_aggressive_jw`: Jaro-Winkler similarity after stripping legal suffixes and filler words.
10. `name_soundex_overlap`: Jaccard overlap of phonetic Soundex codes for name tokens.
11. `addr_jaro_winkler`: Jaro-Winkler similarity on cleaned address strings.
12. `addr_jaccard`: Word-level Jaccard similarity on address tokens.
13. `addr_token_sort`: Token-sort fuzzy ratio on addresses.
14. `addr_number_overlap`: Jaccard overlap on numeric/building/plot tokens.
15. `addr_postal_match`: Exact agreement flag for 5/6-digit postal/PIN codes.
16. `addr_length_ratio`: Address character length ratio.
17. `addr_containment`: Token containment ratio on addresses.
18. `addr_any_empty`: Binary indicator if either address field is null/empty.
19. `country_match`: Exact country agreement flag.
20. `source_type`: Binary flag (0 for Source 2, 1 for Source 3).
21. `geo_mean_name_addr`: Geometric mean of name and address Jaro-Winkler similarities.
22. `min_name_addr_sim`: Minimum of name and address Jaro-Winkler similarities.
23. `max_name_addr_sim`: Maximum of name and address Jaro-Winkler similarities.

### 4.2 Model Type & Hard-Negative Training
- **Model**: Gradient Boosted Decision Trees via XGBoost with CUDA GPU acceleration (`tree_method="hist"`, `device="cuda"`).
- **Hyperparameters**: 350 estimators, max depth 6, learning rate 0.07, subsample 0.85, colsample_bytree 0.85, min_child_weight 5, scale_pos_weight 1.0.
- **Hard-Negative Training Pool**: 75,498 pairs (3,873 positives + 71,625 hard negatives at 60 hard negatives per S1 entity) directly sampled from blocker candidate pools alongside true ground-truth matches.

### 4.3 Threshold Selection Method
Decision threshold τ was optimized using a proper **3-way data split** to ensure validation integrity:
- **Training set** (1,200 S1 entities): Used exclusively for XGBoost model training.
- **Calibration set** (400 S1 entities, 432,036 candidate pairs): Used exclusively for threshold grid search over 13 candidate thresholds from 0.80 to 0.995.
- **Held-out set** (400 S1 entities, 422,553 candidate pairs): Used exclusively for final unbiased evaluation at the frozen threshold. **Never used for threshold tuning or model training.**

Optimal Decision Threshold: **τ = 0.950** (selected by maximizing Macro F0.5 on the calibration set).

---

## 5. Results & Error Analysis

### 5.1 Validation Metrics Summary
Results were evaluated across three independent, non-overlapping splits to guarantee validation integrity:

| Split | Description | Entities (S1) | Candidate Pairs | Macro F0.5 | Precision | Recall | Avg Matches / S1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | Used for XGBoost feature learning | 1,200 | 75,498 | 0.8965 | 0.9893 | 0.8071 | 2.81 |
| **Calibration Set** | Used strictly for threshold grid search (τ = 0.950) | 400 | 432,036 | 0.8006 | — | — | 3.02 |
| **Unbiased Held-Out** | Never seen during training OR threshold tuning | 400 | 422,553 | **0.8036** | **0.8744** | **0.8044** | **3.14** |

*(Note: The above scores represent local validation against training ground truth. The official leaderboard score is determined exclusively by the test set evaluation.)*

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**:
  - *Franchise and Chain Stores*: Multiple retail branches (e.g. "Subway", "State Bank of India") situated in the same commercial district sharing city, state, and partial road names.
  - *Shared Corporate Shells*: Companies sharing generic industrial complex addresses (e.g., "MIDC Industrial Area", "Suite 100") with partially overlapping descriptive words.
- **False Negatives (Missed Matches)**:
  - *Drastic Name Abbreviations*: Acronyms without common character n-grams (e.g. "TCS" vs "Tata Consultancy Services Ltd").
  - *Missing Address Context*: Entities where address fields were entirely blank or unpopulated in Source 2/Source 3.

---

## 6. Conclusion

We delivered an end-to-end, reproducible entity resolution system achieving a verified **0.8036 unbiased held-out Macro F0.5** (Precision: 87.44%, Recall: 80.44%) and **91.41% blocking recall**. By diagnosing the baseline negative distribution shift and retraining with blocker hard negatives, we eliminated severe false-positive inflation, normalizing predictions to **3.14 matches per entity**. The streaming architecture guarantees full scalability, processing millions of candidate pairs in bounded memory (< 3.5 GB) at 51,000+ pairs/sec.

### Limitations
1. **Blocking ceiling**: 91.41% blocking recall means ~8.6% of true matches are permanently lost before the matching stage. Entities with zero overlapping name tokens or extreme abbreviations are the primary failure mode.
2. **Small validation pool**: The 3-way validation used 2,000 S1 entities from the training set (a 0.09% sample of 2.2M). While the held-out score is unbiased, it has limited statistical power.
3. **GPU dependency**: XGBoost training uses CUDA. CPU fallback is available but not tested for production-scale scoring.
4. **France domain shift**: The test set introduces France entities not present in training. The model relies on cross-lingual features (transliteration, Jaro-Winkler) rather than French-specific learned patterns.

---

## Appendix

### A. Code Artefacts
The complete source code is structured as follows:
- **`src/generate_candidates.py`**: V5.1 Multi-channel blocking engine with dynamic budgeting and open-set country discovery.
- **`src/matching/features.py`**: 23 pairwise similarity features computed via parallel multi-processing.
- **`src/matching/normalizer.py`**: Multilingual name and address normalization (legal suffixes, abbreviation expansion, postal code extraction).
- **`src/matching/train_with_hard_negatives.py`**: Hard-negative training script for CUDA XGBoost.
- **`src/run_pipeline.py`**: Production entry point orchestrating blocking, parallel feature computation, and streaming scoring.
- **`utils/validate_submission.py`**: Official submission format and consistency validator.
- **`models/matching_model.json`**: Frozen production XGBoost model weights.
- **`models/matching_config.pkl`**: Frozen model configuration and decision threshold (τ = 0.950).

**Reproducing Test Outputs:**
```bash
# Full pipeline (blocking + matching):
python -m src.run_pipeline --mode generate

# Reuse existing candidates (skip blocking):
python -m src.run_pipeline --mode generate --skip-blocking
```

### B. Additional Results: Decision Threshold Sensitivity
Below is the empirical trade-off between decision threshold τ, Macro F0.5, and average predicted matches per entity (measured on the calibration set of 400 S1 entities):

| Threshold (τ) | Macro F0.5 | Avg Matches / S1 | Status |
| :---: | :---: | :---: | :--- |
| 0.800 | 0.7564 | 3.83 | High recall, precision penalty |
| 0.850 | 0.7757 | 3.60 | Improving |
| 0.900 | 0.7898 | 3.35 | High Precision |
| 0.940 | 0.7973 | 3.10 | Near-optimal |
| **0.950** | **0.8006** | **3.02** | **Optimal Operating Point** |
| 0.960 | 0.7990 | 2.96 | Slightly conservative |
| 0.970 | 0.7958 | 2.83 | Overly conservative |
| 0.990 | 0.7822 | 2.41 | Recall drop |

### C. Frozen Production Configuration

| Parameter | Value |
|-----------|-------|
| Blocker Version | V5.1 |
| Budget Range | [50, 1500] |
| Budget Base | 800 |
| Matching Model | XGBoost CUDA, 350 trees, depth 6, lr 0.07 |
| Decision Threshold | τ = 0.950 |
| Feature Count | 23 pairwise features |
| Training Strategy | Blocker hard negatives (60/S1) |
| Validation Split | 3-way: 1200 train / 400 calibration / 400 held-out |
