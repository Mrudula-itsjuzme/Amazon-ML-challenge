# Amazon ML Challenge: Business Entity Resolution

This repository contains the exploratory data analysis and feature engineering strategy for the Business Entity Resolution challenge. 

## Problem Statement

The goal is to match business records across three independent data sources that lack common identifiers. 
- **Source 1** is a clean, deduplicated, 100% Latin-script reference table.
- **Sources 2 and 3** are noisy, multi-lingual datasets containing duplicates, missing addresses, and transliterated names.

The objective is to find all matching records from Sources 2 and 3 for each Source 1 entity. 

## Dataset Insights & EDA

A comprehensive pairwise feature analysis was conducted to evaluate how well string similarity metrics separate true matches from deceptive false friends. 

### Key Findings
1. **Severe Distribution Shift**: The training data contains only the `US` and `India`. The test data introduces `France` (15% of the test set). Hard-coded vocabulary rules (e.g., matching "Pvt Ltd") will fail on French data. 
2. **Cross-Script Matching**: Source 1 contains exactly 0% non-Latin scripts, while the candidate sources contain ~15% non-Latin scripts (e.g., Hindi, Odia). Standard character-based similarity metrics fail completely (score 0.0) on these true matches.
3. **Deceptive False Friends**: Many businesses share long boilerplate legal suffixes (e.g., "Constructions Private Limited"). This artificially inflates basic string overlap metrics (like standard Jaccard or SequenceMatcher) between completely different businesses. 
4. **Target Skew**: 94.4% of Source 1 entities have at least one match, with a median of 3 matches.

## Feature Engineering Strategy

Based on the analysis, a single string-distance threshold is insufficient. The matching pipeline must rely on a combination of specific pairwise features to handle missing data and hard negatives.

### Recommended Features
* **Name & Address 3-Gram Jaccard**: Highly robust to word reordering.
* **Name & Address SequenceMatcher (Levenshtein proxy)**: Strongest baseline string metric.
* **TF-IDF Weighted Word Jaccard**: Punishes generic tokens like "LLC", "Road", and "Private", preventing false friends from scoring highly.
* **Normalized Exact Match**: Raw exact match is useless due to punctuation/casing. Normalized exact match guarantees a subset of high-confidence predictions.
* **Missingness Flags**: `addr_missing_B` acts as an indicator to the model to rely solely on Name features without penalizing the overall score.
* **Country (Categorical)**: Baseline similarities in India are much lower (mean 0.68) than in the US (mean 0.86). The model needs the country column to dynamically shift its own decision boundaries.

## Next Steps (Modeling Plan)

1. **Candidate Generation (Blocking)**: Due to dataset size (billions of possible pairs), implement a fast TF-IDF + nearest-neighbor index (e.g., MinHash LSH or Faiss) to retrieve the top 20 candidates per Source 1 entity.
2. **Pairwise Feature Extraction**: Compute the recommended features above for all generated candidate pairs.
3. **Classification**: Train a Gradient Boosted Tree (XGBoost/LightGBM) using `GroupKFold` (grouped by country) to predict the binary match probability. 
