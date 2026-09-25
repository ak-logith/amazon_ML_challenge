# Agent Communication Hub

> Shared communication file for all AI agents working on the Business Entity Resolution pipeline.
> Each agent should log their changes, requests, responses, and status updates here.

---

## Agents

| Agent ID | Responsibility | Branch |
|----------|---------------|--------|
| **Agent 1** | Entity-Pair Matching / Model | `feature/matching-and-evaluation-model` |
| **Agent 2** | _TBD_ | _TBD_ |
| **Agent 3** | _TBD_ | _TBD_ |
| **Agent 4** | _TBD_ | _TBD_ |

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
**Action needed by:** Blocking Agent (Agent TBD)

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
**Action needed by:** Blocker Agent (Agent 2/TBD) — please review the Shared Interfaces below to provide candidate pairs for inference/hard negatives.

---

## Shared Interfaces

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

| Component | Status | Last Updated | Current Score |
|-----------|--------|-------------|---------------|
| Data Analysis | ✅ Complete | 2026-09-25 11:39 | N/A |
| Blocking/Candidate Gen | ⏳ Not started | — | — |
| Feature Engineering | 🔨 In Progress | 2026-09-25 11:39 | — |
| Matching Model | 🔨 In Progress | 2026-09-25 11:39 | — |
| Threshold Optimization | ⏳ Pending | — | — |
| Final Submission | ⏳ Pending | — | — |
