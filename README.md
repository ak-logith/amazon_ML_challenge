# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, memory-bounded, CUDA-accelerated multi-source entity resolution pipeline designed for large-scale business record linkage across heterogeneous schemas, multilingual entity names, and cross-source variations (Source 1 vs Source 2 & Source 3).

---

## 1. System Architecture

The pipeline consists of three decoupled, memory-bounded stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           1. BLOCKING STAGE                             │
│ Multi-Channel Inverted Indexes (Name, Address, Composite, 3-Gram Fallback)│
│ Evidence-based Dynamic Budgeting [50, 1500] | Open-set Country Partition│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ candidate_pairs.tsv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    2. PARALLEL FEATURE EXTRACTION                       │
│ 23 Similarity Features: Jaro-Winkler, Levenshtein, Token Overlap,       │
│ Numeric/Postal Match, Soundex, Legal Suffix, Country Agreement          │
│ Chunked ProcessPoolExecutor (50,000+ pairs/sec across 8 CPU workers)    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Pairwise Feature Matrix
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    3. MATCHING & THRESHOLD DECISION                     │
│ CUDA XGBoost Classifier trained with Blocker Hard Negatives             │
│ Calibrated Precision-Preserving Decision Threshold (τ = 0.950)          │
│ Streaming Output Generation (Peak RAM < 4 GB)                           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                        matching_results.tsv
```

---

## 2. Directory Structure

```
.
├── src/
│   ├── generate_candidates.py          # V5.1 Multi-channel blocking engine
│   ├── run_pipeline.py                 # End-to-end inference and scoring pipeline
│   ├── matching/
│   │   ├── features.py                 # 23 pairwise similarity features
│   │   ├── normalizer.py               # Multilingual name and address cleaning
│   │   ├── pipeline.py                 # Baseline model pipeline and evaluation
│   │   └── train_with_hard_negatives.py# Production hard-negative trainer
│   └── preprocessing/                  # Data ingestion and normalization
├── models/
│   ├── matching_model.json             # Serialized production XGBoost model
│   └── matching_config.pkl             # Model metadata and calibrated threshold
├── utils/
│   └── validate_submission.py          # Official challenge submission validator
├── tests/                              # Unit and integration test suite (84 tests)
├── output/                             # Generated candidate and matching files
├── requirements.txt                    # Python package dependencies
└── README.md                           # System documentation
```

---

## 3. Installation & Setup

### Environment Requirements
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **RAM**: Minimum 8 GB (Pipeline operates under 4 GB peak RAM)
- **GPU**: NVIDIA GPU with CUDA support recommended (e.g. RTX 4050/3060 or better; automatic CPU fallback available)
- **OS**: Windows / Linux / macOS

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Execution Commands

### A. Run Full Test Set Generation (Production Submission)
Generates `output/candidate_pairs.tsv` and `output/matching_results.tsv` from test split:
```bash
python -m src.run_pipeline --mode generate
```

### B. Run End-to-End Validation
Validates blocking and matching against the training ground truth:
```bash
# Fast diagnostic validation on 1,000 S1 sample entities per country
python -m src.run_pipeline --mode validate --sample-s1 1000

# Full validation across all training entities
python -m src.run_pipeline --mode validate
```

### C. Retrain Matching Model with Hard Negatives
To retrain the model with fresh hard negatives directly from candidate pairs:
```bash
python src/matching/train_with_hard_negatives.py
```

### D. Validate Submission Format
Runs the official challenge validator to verify TSV format, headers, entity IDs, and candidate subset rules:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### E. Run Test Suite
Runs all 84 unit and integration tests across data ingestion, normalization, and candidate generation:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 5. Key Verified Metrics

- **End-to-End Held-Out Macro F0.5**: **0.8036** (evaluated on 400 strictly held-out entities never seen during model training or threshold tuning)
- **Validation Precision**: **0.8744**
- **Validation Recall**: **0.8044**
- **Decision Threshold**: **τ = 0.950** (calibrated on independent 400-entity split)
- **Average Matches per S1**: **3.14** (closely matching true positive ground truth density ~3.4)
- **Blocking Recall**: **91.41%** (measured on training ground truth)
- **Feature Extraction Throughput**: **51,000+ pairs/sec** (8 CPU workers)
- **CUDA Prediction Throughput**: **187,000+ pairs/sec** (NVIDIA RTX 4050 GPU)
- **Peak RAM Usage**: **< 3.5 GB** (bounded via line-by-line candidate streaming)
