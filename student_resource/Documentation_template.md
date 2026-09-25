# ML Challenge 2026: Business Entity Resolution

**Repository:** https://github.com/Mrudula-itsjuzme/Amazon-ML-challenge  
**Current stage:** Dataset analysis and feature analysis  
**Training status:** Not started yet by design

---

## 1. Executive Summary

This repository contains our work for the Amazon ML Challenge 2026 Business Entity Resolution task. The current phase focuses on understanding the dataset, quantifying the noise patterns, and identifying a compact set of pairwise name/address features before any model training begins.

The strongest findings so far are that normalized string similarity, token overlap, and character n-gram features provide strong match signal, but naive similarity breaks on hard negatives with generic legal/business terms and on multilingual/transliterated Indian records. Address similarity is especially useful for resolving cases where business names look deceptively similar.

---

## 2. Methodology

### 2.1 Problem Analysis

The task is to match each Source 1 business record against zero, one, or many records from Source 2 and Source 3.

Current dataset findings:

- Source 1 contains **2,206,821** records and behaves as the deduplicated reference source.
- Source 2 contains **5,034,616** records.
- Source 3 contains **5,285,603** records.
- The full training corpus contains more than **12 million** records across the three sources.
- Source 1 has no missing names or addresses in the inspected data.
- Source 2 has about **168,967** missing addresses.
- Source 3 has about **175,916** missing addresses.
- Source 2 contains at least **25,891** exact duplicate name+address rows.
- **5.58%** of Source 1 entities have no ground-truth match.
- Median matches per Source 1 entity: **3**.
- Maximum observed matches for a Source 1 entity: **11**.
- Training countries are **US** and **India**.
- Test additionally contains **France**, creating an explicit unseen-country distribution shift.

The practical implication is that exhaustive pairwise comparison is impossible. Candidate generation will eventually be necessary, but it has intentionally not been implemented yet because the present phase is limited to dataset and feature analysis.

### 2.2 Current Analysis Strategy

The current work is deliberately split into two stages:

1. **Dataset analysis** to understand scale, missingness, duplicates, country shift, scripts, and likely leakage risks.
2. **Pairwise feature analysis** using true matches, random negatives, hard-name negatives, and hard-address negatives.

No final model has been trained yet.

---

## 3. Feature Analysis

### 3.1 Pair Construction

Feature behaviour is evaluated on several pair types:

- **True matches:** ground-truth Source 1 ↔ Source 2/3 pairs.
- **Random negatives:** unrelated records.
- **Hard-name negatives:** different businesses sharing meaningful name tokens.
- **Hard-address negatives:** different businesses sharing meaningful address tokens.

Hard negatives are essential because random negatives make most string features appear unrealistically strong.

### 3.2 Observed Feature Separation

Representative mean feature values:

| Pair type | Name sequence ratio | Name Jaccard | Name 3-gram | Address sequence ratio | Address Jaccard |
|---|---:|---:|---:|---:|---:|
| Random negative | 0.251 | 0.014 | 0.013 | 0.240 | 0.007 |
| Hard address negative | 0.283 | 0.042 | 0.052 | 0.435 | 0.134 |
| Hard name negative | 0.522 | 0.227 | 0.264 | 0.268 | 0.019 |
| True match | 0.789 | 0.609 | 0.569 | 0.820 | 0.716 |

The important observation is not that similarity works against random negatives. That case is easy. The useful result is that hard-name negatives significantly close the gap, while address similarity often remains low and can help disambiguate them.

### 3.3 Normalization

Normalization materially improves exact-match features.

Observed correlation with the match label:

- Raw name exact match: **0.064**
- Normalized name exact match: **0.368**

Current normalization includes lowercasing, Unicode normalization/folding, punctuation cleanup, and whitespace normalization.

This confirms that raw exact equality is too brittle for the dataset.

### 3.4 Threshold Behaviour

A single name-similarity threshold is not sufficient.

Observed sequence-ratio behaviour:

| Threshold | Precision | Recall | TP | FP |
|---|---:|---:|---:|---:|
| > 0.6 | 0.120 | 0.851 | 63 | 461 |
| > 0.7 | 0.295 | 0.797 | 59 | 141 |
| > 0.8 | 0.557 | 0.595 | 44 | 35 |
| > 0.9 | 0.769 | 0.405 | 30 | 9 |

Lower thresholds recover more true matches but admit many hard false positives. High thresholds improve precision but lose a large fraction of true matches.

This is a strong indication that the eventual matcher will need multiple complementary features rather than one hand-written similarity cutoff.

### 3.5 Feature Interactions

The most useful interaction observed so far is:

> **High name similarity + low address similarity → often a deceptive non-match.**

Conversely, high similarity in both fields is much more convincing.

Address information therefore acts as an important second signal when names share generic words or legal suffixes.

### 3.6 Feature Redundancy

Observed correlations among selected name features:

| Feature pair | Correlation |
|---|---:|
| Jaccard vs Dice | 0.977 |
| Jaccard vs 3-gram | 0.895 |
| Sequence ratio vs 3-gram | 0.835 |
| Sequence ratio vs Dice | 0.826 |

Jaccard and Dice are effectively redundant for this task. Keeping both is unlikely to add meaningful information.

Sequence-based similarity and character 3-gram similarity are related but still different enough to retain for further analysis.

### 3.7 Country Differences

Name similarity behaves differently across countries.

Observed mean sequence similarity:

| Country | Non-match mean | Match mean |
|---|---:|---:|
| India | 0.385 | 0.685 |
| US | 0.351 | 0.864 |

Indian true matches have substantially lower raw string similarity than US true matches.

A major cause observed in the data is multilingual and cross-script matching.

Example true match:

```text
"Shakti Agro Limited"
"ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍"
```

Character overlap and basic token similarity can collapse to zero despite the records representing the same business.

This makes transliteration/script handling an important issue to quantify before modelling.

### 3.8 Hard Negative Example

A representative deceptive negative:

```text
"One Constructions Private Limited"
"Sai Energy Constructions Private Limited"
```

The strings share most of their tokens, but the differentiating business names are different.

This demonstrates why common terms such as:

- Private
- Limited
- Pvt
- Ltd
- Corporation
- Enterprises
- Constructions

can inflate token-overlap features and create false positives.

---

## 4. Current Compact Feature Set

The feature set currently considered strongest for future modelling is:

### Name
- `norm_name_exact`
- `name_seq_ratio`
- `name_jaccard`
- `name_char_3gram_jaccard`

### Address
- `norm_addr_exact`
- `addr_seq_ratio`
- `addr_jaccard`

### Structural / missingness
- `addr_missing_A`
- `addr_missing_B`
- `same_country`

Features currently considered redundant or weak:

- Dice similarity when Jaccard is already present
- raw unnormalized exact-match features
- simple length-difference features as primary signals

These are not final model features yet. They are the current shortlist produced by analysis.

---

## 5. What Is Done

- [x] Read and documented the challenge format and submission requirements
- [x] Inspected all training sources and ground truth structure
- [x] Measured dataset size and memory footprint
- [x] Analysed missing values
- [x] Analysed source-level duplicates
- [x] Analysed number of matches per Source 1 entity
- [x] Identified singleton/reference entities with zero matches
- [x] Compared training and test country distributions
- [x] Identified France as an unseen test country
- [x] Built sampled positive and negative pair analysis
- [x] Added hard-name negatives
- [x] Added hard-address negatives
- [x] Analysed normalized exact-match behaviour
- [x] Analysed sequence similarity
- [x] Analysed word-level Jaccard similarity
- [x] Analysed character 3-gram similarity
- [x] Analysed feature redundancy
- [x] Analysed feature interactions
- [x] Inspected difficult true matches
- [x] Inspected deceptive false matches
- [x] Compared India and US feature behaviour
- [x] Identified multilingual/transliteration failure cases
- [x] Produced an initial compact feature shortlist

---

## 6. Not Done Yet

These are intentionally not marked complete:

- [ ] Full script/language distribution analysis
- [ ] More systematic generic-token/legal-suffix frequency analysis
- [ ] Address component analysis such as PIN/ZIP/house number matching
- [ ] Raw vs normalized feature ablations across all pair types
- [ ] Full train/test text-distribution comparison beyond country counts
- [ ] Final feature selection
- [ ] Candidate generation / blocking implementation
- [ ] Model training
- [ ] Validation strategy implementation
- [ ] Threshold selection
- [ ] Final matching pipeline
- [ ] Submission generation

The project is currently at the **dataset + feature analysis stage**, not the modelling stage.

---

## 7. Next Analysis Steps

Before training, the next useful analysis tasks are:

1. Quantify non-Latin scripts and transliteration frequency.
2. Measure common-token/legal-suffix frequencies and how much they inflate similarity.
3. Analyse address-specific components such as numeric tokens, PIN/ZIP codes, and reordered address parts.
4. Compare raw vs normalized similarities consistently across random and hard negatives.
5. Finalize the smallest non-redundant feature set.

Only after these are understood should the project move into model training.

---

## 8. Challenge Constraints to Preserve

Future work must remain within the challenge rules:

- No external business lookup APIs or databases.
- No external geocoding.
- Every Source 1 test record must be represented in the final output.
- Final matching output is evaluated using macro **F0.5**, so false positive merges are especially costly.
- France is unseen during training and must still be handled in test inference.

---

## Appendix

### Repository

https://github.com/Mrudula-itsjuzme/Amazon-ML-challenge

### Current repository structure

```text
Amazon-ML-challenge/
├── .gitignore
└── student_resource/
    ├── README.md
    ├── Documentation_template.md
    ├── eda_and_feature_analysis.py
```

The analysis scripts used locally should be added to the repository once their structure is cleaned and finalized.
