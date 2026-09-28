# ML Challenge 2026: Business Entity Resolution

**Repository:** https://github.com/Mrudula-itsjuzme/Amazon-ML-challenge  
**Current stage:** Final Model Submission

---

## 1. Executive Summary

This repository contains our work for the Amazon ML Challenge 2026 Business Entity Resolution task. 
Our final model uses a robust, high-speed Sparse TF-IDF candidate generation strategy followed by a LightGBM classifier with carefully tuned string similarity features.

---

## 2. Methodology

### 2.1 Candidate Generation / Blocking Strategy
Exhaustive pairwise comparison is impossible for millions of records. To generate candidates efficiently:
- We construct an inverted index using normalized tokens of `len >= 3`.
- We apply a **max_df** frequency threshold to filter out the top 1% most frequent tokens (like "Ltd", "Private"). This makes the sparse matrix dot product extremely fast without losing meaningful entity components.
- For each S1 query, we compute the TF-IDF weighted overlap score against all targets in S2 and S3 using a highly optimized sparse matrix multiplication (`scipy.sparse`).
- We retain the top 5 nearest neighbors as candidates. 

### 2.2 Feature Engineering
For the generated candidates, we extract pairwise features focusing on Name and Address similarity:
- Normalized Sequence Ratio (difflib)
- Character 3-gram Jaccard Similarity
- Token Jaccard Similarity
- TF-IDF Weighted Token Overlap (using a custom IDF dictionary)
- Numeric Address Overlap (Shared PIN/ZIP extraction)
- Missingness flags (e.g., `addr_missing`)
- Country Match flags

### 2.3 Model Training
We use a **LightGBM Classifier** trained on a large subset of the generated training candidates. 
The training pairs are dynamically generated with our candidate strategy so the feature distribution closely matches inference. Ground-truth matches are appended to ensure sufficient positive representation. 
Hyperparameters: `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`.
The output probability is thresholded at 0.5 to form the final matches.

---

## 3. Results and Outputs
- `output/candidate_pairs.tsv` contains the candidates generated.
- `output/matching_results.tsv` contains the final predictions.
The complete pipeline runs well within the memory and time limits, processing 1.7M test queries in under 30 minutes on standard hardware using parallel multiprocessing.

