# Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-84%2F84%20passing-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/model-XGBoost%20Hard--Negative-orange.svg)]()
[![Memory](https://img.shields.io/badge/peak%20RAM-%3C%203.5%20GB-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An industrial-grade, memory-bounded, multi-source entity resolution pipeline designed for large-scale record linkage across millions of heterogeneous business entities. Developed for the **Amazon ML Challenge 2026**, this system matches noisy query business records against extensive multi-source partner registries under strict computational and evaluation constraints.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Key Results & Benchmarks](#key-results--benchmarks)
3. [System Architecture](#system-architecture)
4. [Methodology & Pipeline Stages](#methodology--pipeline-stages)
   - [Stage 1: Multi-Channel Candidate Blocking](#stage-1-multi-channel-candidate-blocking)
   - [Stage 2: 23-Dimensional Pairwise Feature Engineering](#stage-2-23-dimensional-pairwise-feature-engineering)
   - [Stage 3: Hard-Negative Mining & Precision-Preserving XGBoost](#stage-3-hard-negative-mining--precision-preserving-xgboost)
5. [High-Performance Systems Engineering](#high-performance-systems-engineering)
6. [Tech Stack](#tech-stack)
7. [Repository Structure](#repository-structure)
8. [Installation & Quickstart](#installation--quickstart)
9. [Running the Test Suite](#running-the-test-suite)
10. [Limitations & Future Improvements](#limitations--future-improvements)

---

## Problem Statement

Entity Resolution (ER) across distributed, heterogeneous databases is a foundational challenge in data integration, master data management, and fraud detection. In this challenge, the goal is to resolve **1.73+ million query entities** from Source 1 against **~10 million target records** across Source 2 and Source 3.

### Core Technical Hurdles
- **Quadratic Complexity ($O(N \times M)$)**: Naively comparing 1.73M query entities against 10M candidate entities requires $>1.7 \times 10^{13}$ pairwise comparisons.
- **Multilingual & Noisy Text**: Variations in business naming conventions, legal entity suffixes (`Inc.`, `LLC`, `SARL`, `Pvt Ltd`), transliteration discrepancies across non-Latin scripts (Indic to Latin-ASCII), and informal address representations.
- **Open-Set Partitioning**: Evaluation sets introduce unseen country domains (e.g., France, with zero training presence) requiring robust zero-shot geographic fallbacks.
- **Asymmetric Evaluation Metric**: Optimized for **Macro $F_{0.5}$**, weighting Precision twice as heavily as Recall ($\beta = 0.5$). False-positive merges are penalized severely.

---

## Key Results & Benchmarks

> **Validation Note**: The metrics below represent strictly measured local validation results on an independent 3-way split (1,200 training entities / 400 threshold-calibration entities / 400 held-out entities). These are verified local validation metrics, not private competition leaderboard scores.

### 1. End-to-End Held-Out Evaluation

| Metric | Measured Value | Description / Evaluation Protocol |
| :--- | :--- | :--- |
| **Held-out Macro $F_{0.5}$** | **0.8036** | Evaluated on 400 strictly held-out entities isolated from training and threshold calibration |
| **Held-out Precision** | **0.8744** | High precision enforced by precision-preserving thresholding ($\tau = 0.950$) |
| **Held-out Recall** | **0.8044** | High match recovery across multi-channel candidate generation |
| **Average Matches / Query** | **3.14** | Closely aligns with ground truth true positive match density (~3.40 matches/entity) |
| **Raw Candidate Recall** | **97.35%** | Percentage of true entity matches captured across unconstrained blocking channels |
| **Bounded Candidate Recall** | **91.41%** | Blocker recall preserved after dynamic budget allocation ($K \in [50, 1500]$) |
| **Unit & Integration Tests** | **84 / 84 (100%)** | Comprehensive coverage across preprocessing, normalization, blocking, and inference |
| **Official Validator Status** | **PASS** | Exit code 0 on all 1,732,544 test entities with zero formatting or subset discrepancies |

### 2. High-Throughput Computational Performance

| Pipeline Component | Throughput / Metric | Hardware Environment |
| :--- | :--- | :--- |
| **Pairwise Feature Extraction** | **> 105,000 pairs/sec** | 8 CPU workers (with short-circuit early-exit filtering) |
| **XGBoost Inference** | **> 180,000 pairs/sec** | NVIDIA RTX 4050 GPU (with seamless CPU fallback) |
| **Peak Memory Consumption** | **< 3.5 GB RAM** | Streaming generator pipeline across 1.73M query entities |
| **Full Ingestion & Preprocessing**| **~24,000 rows/sec** | Polars multi-threaded I/O and compiled regex transformations |

---

## System Architecture

The pipeline uses a modular, decoupled three-stage architecture operating in constant-memory streaming batches:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW MULTI-SOURCE INPUTS                         │
│   Source 1: 1.73M Query Entities | Source 2/3: ~10M Target Entities   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 1: MULTI-CHANNEL INVERTED INDEX BLOCKING            │
│  - Partition by Country (with open-set global fallback)               │
│  - Channel A: Rare Name Tokens & Phonetic Prefixes                     │
│  - Channel B: Address Tokens & Postal Code Bucketing                   │
│  - Channel C: Composite Key Cross-Matching                             │
│  - Channel D: Character 3-Gram Fallback for Typo Tolerance             │
│  - Dynamic Budget Allocation [50, 1500] based on candidate evidence    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ candidate_pairs.tsv
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STAGE 2: PARALLEL PAIRWISE FEATURE ENGINEERING             │
│  - 23 Pairwise Similarity Features across Lexical & Phonetic Space     │
│  - Short-Circuit Early Exit: Filter pairs where max(JW_name, JW_addr) < 0.40 │
│  - Chunked Multiprocessing via Persistent ProcessPoolExecutor           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pairwise Feature Vectors
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            STAGE 3: HARD-NEGATIVE TRAINED XGBOOST MATCHING             │
│  - Gradient-Boosted Trees trained on 100K+ Blocker Hard Negatives      │
│  - Precision-Preserving Decision Boundary (τ = 0.950)                   │
│  - Strict Candidate Subset Enforcement (predictions ⊆ candidates)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          VALIDATED OUTPUTS                             │
│     output/candidate_pairs.tsv  |  output/matching_results.tsv         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Methodology & Pipeline Stages

### Stage 1: Multi-Channel Candidate Blocking
To reduce search complexity from billions of combinations to a tractable candidate set, we implement a **Multi-Channel Inverted Index Blocker (V5.1)**:
- **Country Partitioning**: Queries are segmented by country. Unseen countries in evaluation (e.g., France) automatically fallback to open-set indexing.
- **Channel A (Name Similarity)**: Rare token matching with frequency-based inverse document frequency (IDF) weighting, accompanied by 3-character phonetic prefixes.
- **Channel B (Address & Postal)**: Cleaned street token indexing combined with country-specific postal code standardization (5-digit US ZIPs, 6-digit Indian PIN codes, 5-digit French codes).
- **Channel C (Composite Keys)**: Concatenated `(first_name_token + postal_code)` and `(legal_suffix + city)` pairs.
- **Channel D (Sub-word Fallback)**: Character 3-gram inverted indexing for robust tolerance against severe typographical variations.
- **Evidence-Based Dynamic Budgeting**: Rather than a static top-$K$ limit, queries receive an entity-specific candidate budget $K \in [50, 1500]$ proportional to candidate pool density and multi-channel agreement.

### Stage 2: 23-Dimensional Pairwise Feature Engineering
For every candidate pair, the pipeline constructs a 23-dimensional feature vector capturing complementary similarity dimensions:
1. **Lexical Similarities**: Full-string Jaro-Winkler, Levenshtein ratio, token sort ratio, token set ratio.
2. **Core Name Matching**: Legal entity suffixes (`Inc`, `Corp`, `LLC`, `GmbH`, `SARL`) are stripped to compute core brand name alignment.
3. **Phonetic Representations**: Soundex and Metaphone encodings for phonetic matching across naming variations.
4. **Structural & Address Metrics**: Address token Jaccard similarity, containment ratios, city/state agreement flags, postal code numeric match indicators.
5. **Cross-Feature Interactions**: Max-pool and geometric mean interactions between name and address scores.
6. **Short-Circuit Optimization**: If $\max(JW_{\text{name}}, JW_{\text{addr}}) < 0.40$, the pair is safely skipped without full 23-feature computation, raising feature throughput to $>105,000$ pairs/second.

### Stage 3: Hard-Negative Mining & Precision-Preserving XGBoost
Standard random negative sampling fails in entity resolution because random pairs are trivially distinguishable. We train an **XGBoost Classifier** using realistic hard negatives:
- **Hard-Negative Mining**: 60 hard negatives per query entity are sampled directly from false-positive candidates generated by the blocking stage (pairs sharing names or addresses but belonging to distinct entities).
- **Class Balancing**: Balanced training matrix of 100,760 verified pairs.
- **Threshold Calibration**: Evaluated over 422,553 candidate pairs on an independent calibration set. The decision threshold was calibrated to $\tau = 0.950$, ensuring high precision ($0.8744$) to maximize the Macro $F_{0.5}$ metric.

---

## High-Performance Systems Engineering

1. **Constant-Memory Generator Pipeline**:
   Processing 1.73M entities with ~465M candidate IDs would easily exceed 64 GB of memory if loaded simultaneously. The pipeline processes records in chunked batches (1,000 entities/batch) and streams directly to disk, keeping memory consumption bounded below **3.5 GB RAM**.
2. **IPC Robustness on Windows**:
   To prevent Windows IPC pipe buffer exhaustion (`BrokenProcessPool`), worker pools are maintained persistently across batch iterations with chunked data serialization.
3. **Subset Guarantee Enforcement**:
   Every predicted match is guaranteed to be a strict mathematical subset of candidate pool outputs, satisfying formal competition integrity constraints.

---

## Tech Stack

- **Core Runtime**: Python 3.10–3.13
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/)
- **Data Manipulation**: [Polars](https://pola.rs/), [NumPy](https://numpy.org/)
- **String & Phonetic Algorithms**: [RapidFuzz](https://github.com/maxbachmann/RapidFuzz), [Jellyfish](https://github.com/jamesturk/jellyfish), [Unidecode](https://github.com/avian2/unidecode)
- **Testing & Quality Assurance**: Python `unittest`, custom streaming submission validator

---

## Repository Structure

```
.
├── src/
│   ├── generate_candidates.py          # V5.1 Multi-channel inverted index blocker
│   ├── run_pipeline.py                 # End-to-end inference and validation pipeline
│   ├── matching/
│   │   ├── features.py                 # 23 pairwise similarity feature extractors
│   │   ├── normalizer.py               # Multilingual name and address cleaning
│   │   ├── pipeline.py                 # Baseline model pipeline and evaluation
│   │   └── train_with_hard_negatives.py# Production hard-negative XGBoost trainer
│   └── preprocessing/
│       ├── config.py                   # Standardization regexes, postal codes & legal forms
│       ├── interface.py                # Fast Polars entity data loader
│       ├── loader.py                   # Multi-source dataset loaders
│       ├── normalizer.py               # Text cleaning and Indic transliteration
│       └── run_preprocess.py           # Preprocessing batch execution script
├── utils/
│   └── validate_submission.py          # Challenge submission validator with ID existence check
├── tests/
│   ├── test_candidate_generation.py    # Blocker channels and budget test suite
│   ├── test_loader_and_normalization.py# Text cleaning and normalization unit tests
│   ├── test_preprocessing.py           # Preprocessing schema validation tests
│   └── sanity_test.py                  # Pipeline integration sanity tests
├── requirements.txt                    # Project dependencies
├── Documentation_template.md           # Formal competition methodology documentation
└── README.md                           # Public repository overview
```

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/ak-logith/amazon_ML_challenge.git
cd amazon_ML_challenge
```

### 2. Set Up a Virtual Environment
```bash
# Using Python 3.11+
python -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Pipeline

#### End-to-End Test Set Inference (Candidate & Matching TSVs)
```bash
python -m src.run_pipeline --mode generate
```
Outputs are written to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

#### Held-Out Validation
```bash
# Quick validation on 1,000 sample entities per country
python -m src.run_pipeline --mode validate --sample-s1 1000

# Full validation across training entities
python -m src.run_pipeline --mode validate
```

#### Retraining XGBoost with Hard Negatives
```bash
python src/matching/train_with_hard_negatives.py
```

#### Running the Official Submission Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

---

## Running the Test Suite

The test suite covers data loading, phonetic normalization, multi-channel inverted indexing, candidate budgeting, and feature extraction:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected output:
```text
Ran 84 tests in 0.35s
OK
```

---

## Limitations & Future Improvements

1. **Cross-Lingual Deep Embeddings**:
   While character 3-grams and phonetic transliteration handle spelling drift, dense bi-encoder embeddings (e.g., multilingual Sentence-Transformers or MiniLM fine-tuned via contrastive loss) could further improve non-lexical semantic synonymy.
2. **Graph-Connected Component Clustering**:
   The current matching stage performs pairwise classification independently. Applying connected-component clustering or Markov clustering across candidate pair graphs could enforce transitive consistency across multi-source networks.
3. **Hierarchical Geospatial Indexing**:
   Integrating geospatial coordinates (H3 hexagonal spatial indexes) where postal codes or addresses resolve to coordinates could offer stronger geographic candidate pruning.

---

## Author & Acknowledgments

- **Team**: The Entity Resolvers
- **Challenge**: Amazon ML Challenge 2026
- Built with high-performance Python, Polars, and XGBoost.
