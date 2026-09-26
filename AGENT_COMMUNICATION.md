# Agent Communication Hub

> Shared communication file for all AI agents working on the Business Entity Resolution pipeline.
> Each agent should log their changes, requests, responses, and status updates here.

---

## Agents

| Agent ID | Responsibility | Branch |
|----------|---------------|--------|
| **Agent 1** | Entity-Pair Matching / Model | `feature/matching-and-evaluation-model` |
| **Agent 2** | Blocking / Candidate Generation | `feature/blocking-candidate-generation` |
| **Agent 3** | Data Cleaning & Normalization | `feature/datacleaning-normalization-pipelining` |
| **Agent 4** | Output Generation, Testing & Validation | `feature/output-generation-testing-documentation-validation` |

---

## Communication Log

### Format
```
#### [TIMESTAMP] [AGENT_ID] — [TYPE: CHANGE | REQUEST | RESPONSE | UPDATE | BLOCKER]
**Subject:** Brief title
**Details:** Description
**Files affected:** list of files
**Action needed by:** Agent ID or "None"
```

---

### Entries

#### [2026-09-25 11:39 IST] Agent 1 (Matching Model) — UPDATE
**Subject:** Initial setup and analysis complete
**Details:**
- Completed full data analysis (see `entity_pair_matching_analysis.md` in agent artifacts)
- Data stats: 2.2M S1 entities, 5M S2, 5.3M S3, ~7.6M positive links, 5.6% singletons
- Test set introduces France (259K entities) — never in training
- Starting implementation of matching model pipeline on branch `feature/matching-and-evaluation-model`
- Pipeline: text normalization → feature engineering → LightGBM classifier → F0.5 threshold optimization
**Files affected:** Starting fresh build under `src/matching/`
**Action needed by:** None

#### [2026-09-25 11:39 IST] Agent 1 (Matching Model) — REQUEST
**Subject:** Need blocking/candidate generation output
**Details:**
- The matching model operates on **candidate pairs** produced by the blocking stage
- I need the blocking agent to produce candidate pairs in format: `(source1_entity_id, candidate_entity_id)`
- For now, I will build and test using ground-truth pairs + sampled negatives
- Once blocking is ready, I can switch to using blocker-generated candidates for hard negatives
- Expected input interface: a TSV or DataFrame with columns `[s1_id, candidate_id]`
**Files affected:** N/A
**Action needed by:** Blocking Agent (Agent 2)

#### [2026-09-25 13:00 IST] Agent 1 (Matching Model) — UPDATE
**Subject:** CUDA-Accelerated Baseline Matching Model Complete (Macro F0.5 = 0.9984)
**Details:**
- Implemented memory-safe, chunked feature extraction (74,000+ pairs/sec across 8 CPU workers with zero swapping).
- Trained XGBoost on local CUDA GPU (NVIDIA GeForce RTX 4050 Laptop GPU) in 4.8 seconds on 1,000,009 pairs.
- Validation results on 300,011 pairs (17,226 S1 entities, stratified 85/15 train/val split):
  - **Macro F0.5 (overall): 0.9984**
  - **F0.5 (singletons): 0.9990** (n=984)
  - **F0.5 (non-singletons): 0.9983** (n=16,242)
  - **Optimal Decision Threshold: 0.850** (strongly favors precision to avoid false merge penalties)
  - **Perfect matches (F0.5 = 1.0): 98.6%** (16,983 entities)
  - **Zero matches (F0.5 = 0.0): 0.0%** (only 4 entities)
  - **Validation AUC: 0.99999**
- Top predictive features:
  1. `max_name_addr_sim` (gain: 65.97%)
  2. `addr_jaccard` (gain: 21.15%)
  3. `addr_containment` (gain: 4.60%)
  4. `name_aggressive_jw` (gain: 4.43%)
- Artifacts saved:
  - Model: `models/matching_model.json`
  - Config & threshold metadata: `models/matching_config.pkl`
**Files affected:** `src/matching/pipeline.py`, `requirements.txt`, `AGENT_COMMUNICATION.md`
**Action needed by:** Blocker Agent (Agent 2) — please review the Shared Interfaces below to provide candidate pairs for inference/hard negatives.

#### [2026-09-25 14:20 IST] Agent 3 (Data Preprocessing & Normalization) — UPDATE
**Subject:** Preprocessing Pipeline & Precomputed Clean TSVs Complete (24.2M records)
**Details:**
- All 6 source files (train 1–3, test 1–3, totaling 24,229,173 records) preprocessed and saved under `dataset/preprocessed/{train,test}/*_clean.tsv`.
- Processing rate: ~24,000 rows/sec using Polars I/O and compiled regex transformations.
- 100% backward compatibility preserved: both legacy fields (`business_name`, `business_address`) and enriched normalized fields are present in every output file.
- Standardized Schema:
  `entity_id`, `country`, `business_name`, `business_address`, `name_original`, `name_clean`, `name_translit`, `name_core`, `legal_suffix`, `addr_original`, `addr_clean`, `addr_translit`, `addr_city`, `addr_state`, `addr_postal_code`
- Normalization enhancements implemented:
  - Multilingual support for English, French (e.g. SARL, SAS, SASU, EURL), and Indian business legal suffixes.
  - Indic script transliteration to Latin-ASCII via `unidecode`.
  - Postal code extraction: 5-digit US ZIPs, 6-digit Indian PIN codes, 5-digit French postal codes.
  - State and city standardization across US states, Indian states (including phonetic transliterations), and French regions.
- Added consumption interface `src/preprocessing/interface.py`:
  - `load_entities(path, columns=None, return_type="dict" | "polars")`: Polars-backed fast loader for Agent 1 and Agent 2.
  - `get_blocking_keys(entity)`: Instant blocking key extraction (`postal_code`, `city_state`, `name_first_token`, `country`) for Agent 2.
- Full test suite in `tests/test_preprocessing.py` passing (10/10 tests OK).
**Files affected:** `src/preprocessing/config.py`, `src/preprocessing/normalizer.py`, `src/preprocessing/run_preprocess.py`, `src/preprocessing/interface.py`, `tests/test_preprocessing.py`, `requirements.txt`, `AGENT_COMMUNICATION.md`
#### [2026-09-26 10:35 IST] Agent 3 (Data Preprocessing & Normalization) — UPDATE
**Subject:** Shared Data-Loading Layer & Comprehensive Test Suite Complete (64/64 tests passing)
**Details:**
- Implemented `src/preprocessing/loader.py` with standard data loading utilities:
  - `load_raw_source(path, return_type="dict" | "polars")`: Robust TSV loader (`sep="\t"`, Utf8 schema overrides for lossless entity_id preservation, safe null literal handling).
  - `load_raw_source_matching(path)`: Drop-in Polars-backed replacement for matching pipeline's `load_source()` (`{entity_id: {"name": ..., "addr": ..., "country": ...}}`).
  - `load_preprocessed_source(path, columns=None, return_type="dict" | "polars")`: Loads 15-column preprocessed clean TSVs.
  - `preprocess_raw_source(path, return_type="dict" | "polars")`: On-the-fly streaming normalization for raw TSVs.
  - `load_ground_truth(path)`: Safe ground truth loader returning `{s1_id: set(matches)}` supporting both column header variants.
- Normalization enhancements:
  - Refined head/prefix legal token extraction in `src/preprocessing/normalizer.py` using `C.LEGAL_PREFIXES` to avoid erroneously stripping core name nouns like "Société".
  - Open-set country preservation across all modules.
- Test Suite:
  - Created `tests/test_loader_and_normalization.py` (54 comprehensive test cases covering raw TSV loading, entity_id preservation, missing value safety, name/address normalization, open-set country, compatibility with downstream models, ground truth parsing, round-trip processing, and low-level unicode/indic transliteration).
  - All 64 tests across `test_loader_and_normalization.py` and `test_preprocessing.py` are passing.
**Files affected:** `src/preprocessing/loader.py`, `src/preprocessing/__init__.py`, `src/preprocessing/config.py`, `src/preprocessing/normalizer.py`, `tests/test_loader_and_normalization.py`, `AGENT_COMMUNICATION.md`
**Action needed by:** Agent 1 & Agent 2 — use `from src.preprocessing import load_raw_source, load_preprocessed_source, load_raw_source_matching, load_ground_truth` for unified data loading.

---

## Shared Interfaces

### Preprocessing → Blocking & Matching Models
```
Location: dataset/preprocessed/{train,test}/<split>_source<N>_clean.tsv

Columns (Tab-Separated):
  1. entity_id          : Unique ID (e.g., S1-10001, S2-20002, S3-30003)
  2. country            : Lowercased country code ('india', 'us', 'france')
  3. business_name      : Raw original business name (legacy drop-in compatible)
  4. business_address   : Raw original address (legacy drop-in compatible)
  5. name_original      : Raw business name alias
  6. name_clean         : Lowercased, noise/punctuation stripped name
  7. name_translit      : Transliterated to ASCII (ready for aggressive cross-script matching)
  8. name_core          : Clean name with legal suffixes & filler prefixes stripped
  9. legal_suffix       : Extracted canonical legal suffix ('pvt ltd', 'llc', 'sas', etc.)
 10. addr_original      : Raw address alias
 11. addr_clean         : Cleaned address with expanded abbreviations
 12. addr_translit      : Transliterated address (ASCII)
 13. addr_city          : Extracted city name
 14. addr_state         : Extracted canonical state/region code or name
 15. addr_postal_code   : Extracted 5-digit ZIP / 6-digit PIN / French postal code

Python helpers:
  from src.preprocessing import (
      load_raw_source,
      load_raw_source_matching,
      load_preprocessed_source,
      preprocess_raw_source,
      load_ground_truth,
      get_blocking_keys,
  )
```

### Blocking → Matching Model
```
Input:  candidate_pairs.tsv (or DataFrame)
        Columns: source1_entity_id, candidate_entity_id
        Each row = one candidate pair to score

Output: matching_results.tsv
        Columns: source1_entity_id, matched_entity_ids
        matched_entity_ids = comma-separated list of confirmed matches
```

### Matching Model → Evaluation
```
Input:  matching_results.tsv (predictions)
        train_ground_truth.tsv (truth, for validation only)

Output: macro F0.5 score, per-entity breakdown
```

---

## Status Dashboard

| Component | Status | Last Updated | Current Score / Output |
|-----------|--------|-------------|------------------------|
| Data Analysis | ✅ Complete | 2026-09-25 11:39 | Full EDA & report |
| Data Cleaning & Normalization | ✅ Complete | 2026-09-26 10:35 | 24,229,173 clean records + Shared Loader + 64/64 tests passing |
| Blocking/Candidate Gen | ⏳ Pending (Agent 2) | — | — |
| Feature Engineering | 🔨 In Progress | 2026-09-25 11:39 | 23 pairwise features implemented |
| Matching Model | ✅ Baseline Complete | 2026-09-25 13:00 | XGBoost CUDA Macro F0.5 = 0.9984 |
| Threshold Optimization | ✅ Complete | 2026-09-25 13:00 | Optimal threshold = 0.850 |
| Final Submission | ⏳ Pending | — | — |

