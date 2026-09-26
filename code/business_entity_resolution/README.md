# Business entity resolution

`01_pipeline.py` preprocesses Unicode names and addresses and retrieves bounded candidates without reading labels. `02_match.py` builds pair features and runs LightGBM inference. `03_validate.py` performs grouped model comparison and locked confirmation. The system permits zero, one, or many matches per S1 and never filters candidates by country. Dependencies are pinned in `requirements.txt` (Python 3.10).

## Reproduce the full-pool matcher study

The study used a label-independent, disjoint 1,000-S1 sample beginning after the previous 300-S1 exploratory cohort in deterministic source hash order. Both retrieval steps scanned **all 10,320,219 S2/S3 targets**. Settings were frozen before inspecting this cohort's labels: top 50 per route and cap 60. Neither stage inserts or protects ground-truth targets.

```bash
python code/business_entity_resolution/01_pipeline.py --split train \
  --stream-target --sample-s1 1000 --sample-offset 300 --audit-retrieval \
  --target-chunk 50000 --workers 4 --route-top 50 --cap 60 \
  --output-dir research_runs/ber_fullpool_1000_offset300
python code/business_entity_resolution/01_pipeline.py --split train \
  --augment-from research_runs/ber_fullpool_1000_offset300 \
  --target-chunk 50000 --workers 4 --route-top 50 --cap 60 \
  --output-dir research_runs/ber_fullpool_1000_offset300_aug1
python code/business_entity_resolution/03_validate.py \
  --input-dir research_runs/ber_fullpool_1000_offset300_aug1 \
  --fullpool-model-design
python code/business_entity_resolution/03_validate.py \
  --input-dir research_runs/ber_fullpool_1000_offset300_aug1 \
  --fullpool-model-confirm
```

The last two commands have **already completed** for this cohort. Their outputs are `fullpool_model_selection.json` and `fullpool_model_confirmation.json`. The validator refuses to repeat design or confirmation after those outputs exist. The confirmation cohort must not be reopened to tune this model. Only retrieved candidates entered the feature table; missed true targets were hydrated separately for diagnostic slices and never inserted into candidates or model features.

The 1,000 S1 entities contained 3,533 true links and 47 no-match singletons. Retrieval retained 60,000 candidate pairs, including 3,378 positive and 56,622 negative pairs. Development used 800 grouped S1 entities (2,828 true links; 48,000 candidate pairs; 2,703 positives). The locked 200 S1 entities contained 705 true links, 9 singletons, and 12,000 candidate pairs (675 positive, 11,325 negative). Negatives are retrieved hard negatives, not random target pairs.

| K | Full-cohort retrieved true links | Recall |
|---:|---:|---:|
| 40 | 3,150 / 3,533 | 0.8916 |
| 60 | 3,378 / 3,533 | 0.9561 |
| 80 | 3,416 / 3,533 | 0.9669 |
| 100 | 3,430 / 3,533 | 0.9708 |
| 150 | 3,444 / 3,533 | 0.9748 |

These are **blocker recall** figures, not model accuracy. K above 60 comes from the raw route audit; the matcher was trained and scored on K=60.

## Learned matcher results

Both models used the same 48,000 development pairs, fixed features, grouped three-fold outer evaluation, and grouped inner out-of-fold threshold selection from a small fixed grid. The model with higher mean macro entity F0.5 was selected before locked confirmation. XGBoost was unavailable and omitted.

| Model / evaluation | Blocker recall | Macro F0.5 | Pair precision | Pair recall | Singleton accuracy | Mean fold F0.5 | Fold std |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM, development folds | 0.9558 | 0.8984 | 0.9544 | 0.8499 | 0.7074 | 0.8984 | 0.0134 |
| CatBoost, development folds | 0.9558 | 0.8925 | 0.9705 | 0.8010 | 0.7778 | 0.8925 | 0.0246 |
| Selected LightGBM, locked confirmation | 0.9574 | 0.9309 | 0.9791 | 0.8638 | 1.0000 (9/9) | 0.8984 | 0.0134 |

LightGBM currently generalizes best by the prespecified development-fold criterion. Its locked threshold was 0.7, chosen using only development data. The locked cross-script F0.5 was 0.9293 on 21 entities; missing-address F0.5 was 0.8404 on 24; short-name F0.5 was 0.9409 on 36. Locked country slices were India 0.9230 (83 entities) and US 0.9365 (117). A development-only US country holdout scored 0.8599 macro F0.5 and 0.9101 pair precision; it is a diagnostic, not evidence of unseen-country generalization.

The locked failure counts were 30 true links missed by retrieval, 66 retrieved true links rejected by the matcher, and 13 incorrect accepted pairs. Among the blocker misses, 8 were cross-script and 6 had a missing address. Among matcher false negatives, 14 had a numeric conflict, 12 a missing address, 11 a short source name, and 3 were cross-script. Of 13 false positives, 2 had a numeric conflict and 1 a missing address. Categories overlap. LightGBM's largest split-count importances were address 3-gram similarity (416), rare-token route score (405), numeric score relative to the entity's best candidate (390), name length ratio (369), and native-name token overlap (358). These are importance diagnostics, not causal feature effects.

The retrieval gate of at least 99% remains unmet. The locked matcher score is legitimate for the full target universe and fixed K=60 retrieval system; it does not imply that increasing K or changing routes would improve F0.5.

## Wider-union reranking study

The earlier 200-S1 confirmation remains closed. A new study on the original
800 development S1 entities compared a precision-constrained K=60 baseline
against K=150 with unchanged features, and K=100/K=150 with explicit
missing-address name features. Thresholds were selected by inner grouped OOF
predictions with a 0.97 pair-precision target. K=100 with missing-address
features was promising (mean F0.5 0.8978 vs 0.8904 for K=60; all three folds
above 0.97 precision), but one fold was lower by 0.0001, so the strict
all-fold improvement criterion did not select it. K=150 did not meet the
precision-and-stability criterion.

Before reading labels of the next cohort, the frozen external comparison is:
train both fixed LightGBM policies on only those 800 development S1 groups;
select each threshold from grouped OOF predictions on those groups; evaluate
K=60 unchanged features and K=100 missing-address features once on a disjoint
500-S1 cohort at hash offset 1300, retrieved against the full target pool.
Confirm K=100 only if its external macro F0.5 exceeds K=60 and its pair
precision is at least 0.97. The validator writes a single
`external_rerank_confirmation.json` and refuses to reopen it. No route or
model setting will be changed based on that confirmation.

## Submission path and limits

The earlier reduced-target pilot excluded most true links and its model scores must not be used as estimates of full-pool quality. The previous mru pipeline also seeded its candidate pool with ground-truth targets and ASCII-stripped native name text; these issues are avoided in this study.

`01_pipeline.py --split test` and `02_match.py --mode test` write the official `candidate_pairs.tsv` and `matching_results.tsv` schemas. A 100-S1 smoke verified both files, but full-scale inference for all 2.2 million S1 entities has not been validated on this 15 GiB host. The streaming full-pool research assay handles bounded S1 cohorts; the default full test path still materializes the target pool in memory. Keep existing research files until a full-scale replacement passes validation.
