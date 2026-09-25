# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains our work for the Amazon ML Challenge 2026 Business Entity Resolution task.

**Repository:** https://github.com/Mrudula-itsjuzme/Amazon-ML-challenge  
**Current stage:** Dataset analysis + pairwise feature analysis  
**Training status:** Not started yet by design

## Problem

The task is to match each Source 1 business against zero, one, or many records from Source 2 and Source 3 despite noisy names, addresses, missing fields, spelling variation, legal suffixes, word reordering, and multilingual/transliterated text.

Source 1 is the deduplicated reference source. Sources 2 and 3 are the noisy candidate sources.

## What is done

- [x] Dataset structure and scale analysis
- [x] Missing-value analysis
- [x] Duplicate analysis
- [x] Ground-truth match-count analysis
- [x] Train/test country comparison
- [x] France identified as an unseen test-country shift
- [x] Positive-pair construction from ground truth
- [x] Random-negative analysis
- [x] Hard-name-negative analysis
- [x] Hard-address-negative analysis
- [x] Normalized exact-match analysis
- [x] Sequence similarity analysis
- [x] Word-level Jaccard analysis
- [x] Character 3-gram similarity analysis
- [x] Feature redundancy analysis
- [x] Feature interaction analysis
- [x] India vs US feature-behaviour comparison
- [x] Difficult true-match inspection
- [x] Deceptive hard-negative inspection
- [x] Multilingual/transliteration failure-case identification
- [x] Initial compact feature shortlist

## Dataset findings

- Source 1: **2,206,821** records
- Source 2: **5,034,616** records
- Source 3: **5,285,603** records
- More than **12 million** training records overall
- Source 2 has about **168,967** missing addresses
- Source 3 has about **175,916** missing addresses
- Source 2 contains at least **25,891** exact duplicate name+address rows
- **5.58%** of Source 1 entities have no match
- Median matches per Source 1 entity: **3**
- Maximum observed matches per Source 1 entity: **11**
- Training countries: **US, India**
- Test adds **France**, creating an unseen-country distribution shift

## Feature-analysis findings

### Random negatives are too easy

Random mismatches make string-similarity features look stronger than they really are. Hard negatives are much more informative.

Representative mean values:

| Pair type | Name seq. | Name Jaccard | Name 3-gram | Address seq. | Address Jaccard |
|---|---:|---:|---:|---:|---:|
| Random negative | 0.251 | 0.014 | 0.013 | 0.240 | 0.007 |
| Hard address negative | 0.283 | 0.042 | 0.052 | 0.435 | 0.134 |
| Hard name negative | 0.522 | 0.227 | 0.264 | 0.268 | 0.019 |
| True match | 0.789 | 0.609 | 0.569 | 0.820 | 0.716 |

### Normalization matters

Observed correlation with the match label:

- raw name exact match: **0.064**
- normalized name exact match: **0.368**

Lowercasing, Unicode normalization, punctuation cleanup, and whitespace normalization make exact matching substantially more useful.

### One threshold is not enough

For name sequence similarity:

| Threshold | Precision | Recall |
|---|---:|---:|
| > 0.6 | 0.120 | 0.851 |
| > 0.7 | 0.295 | 0.797 |
| > 0.8 | 0.557 | 0.595 |
| > 0.9 | 0.769 | 0.405 |

A single similarity cutoff cannot give both good precision and recall.

### Address similarity complements name similarity

A recurring pattern is:

> **high name similarity + low address similarity = often a deceptive non-match**

This is especially important when businesses share generic legal or industry terms.

### Redundant features

Name Jaccard and Dice similarity are almost identical in behaviour:

- Jaccard vs Dice correlation: **0.977**

Keeping both is unlikely to add much value.

### Country differences

Observed mean name sequence similarity:

| Country | Non-match | Match |
|---|---:|---:|
| India | 0.385 | 0.685 |
| US | 0.351 | 0.864 |

Indian true matches are harder for raw character-based similarity.

One observed cause is cross-script matching, for example:

```text
Shakti Agro Limited
ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍
```

Standard character-overlap features can completely miss such pairs.

### Hard-negative example

```text
One Constructions Private Limited
Sai Energy Constructions Private Limited
```

Generic overlapping terms make the strings look much more similar than the underlying entities really are.

## Current feature shortlist

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

This is a **feature-analysis shortlist**, not a final trained-model feature set.

## Still to analyse before training

- [ ] Full script / non-Latin frequency analysis
- [ ] Systematic generic-token and legal-suffix frequency analysis
- [ ] Address components such as PIN/ZIP, house numbers, and numeric-token overlap
- [ ] Raw vs normalized feature comparison across every pair type
- [ ] More train/test text-distribution checks
- [ ] Final non-redundant feature selection

## Not started yet

- [ ] Candidate generation / blocking
- [ ] Model training
- [ ] Validation pipeline
- [ ] Threshold tuning
- [ ] Final matching pipeline
- [ ] Submission generation

That separation is intentional. The project is still in the **dataset + feature understanding** phase.

## Documentation

The detailed running methodology and progress log is in:

- [student_resource/Documentation_template.md](student_resource/Documentation_template.md)

The official challenge statement and submission requirements are preserved in:

- [student_resource/README.md](student_resource/README.md)
- [student_resource/utils/validate_submission.py](student_resource/utils/validate_submission.py)

## Repository

https://github.com/Mrudula-itsjuzme/Amazon-ML-challenge
