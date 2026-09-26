# Business entity resolution

The latest frozen full-pool IDF matcher audit, grouped ablations, blocker miss
taxonomy, and one-time 500-S1 confirmation are in
[`IDF_MATCHER_AUDIT.md`](IDF_MATCHER_AUDIT.md). The selected K100 LightGBM
scored **0.9393 macro F0.5** and **0.9762 pair precision** on that untouched
cohort. The 0.95 target was not reached.

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

The external check completed on 500 disjoint S1 entities and 1,725 true links,
retrieved from all 10,320,219 targets. It **confirmed** the fixed K=100
candidate. Both models trained on the same earlier 800 development S1 groups;
their thresholds were 0.9 for K=60 and 0.8 for K=100, each selected from
development-only grouped OOF predictions.

| External metric | K=60 baseline | K=100 missing-address model |
|---|---:|---:|
| Blocker recall | 0.9467 | 0.9670 |
| Macro entity F0.5 | 0.8875 | 0.8932 |
| Pair precision | 0.9818 | 0.9704 |
| Pair recall | 0.7797 | 0.8186 |
| Missing-address F0.5 (67 entities) | 0.8513 | 0.8725 |
| Cross-script F0.5 (69 entities) | 0.7800 | 0.8112 |
| Singleton accuracy (28 entities) | 0.8929 | 0.7857 |
| Blocker misses / matcher false negatives / false positives | 92 / 288 / 25 | 57 / 256 / 43 |

The K=100 gain came with more false positives and lower singleton accuracy.
Precision is only slightly above the frozen 0.97 floor, so this cohort alone
does not establish a comfortable precision margin for deployment. The result
is saved in `research_runs/ber_fullpool_500_offset1300_aug1/external_rerank_confirmation.json`.

## Reverse retrieval research status

`01_pipeline.py --reverse-from` contains an experimental target-to-S1 route.
It builds native-name, compact-name and address sparse views over the complete
2,206,821-record training S1 population and streams target records, retaining
only targets for which a study S1 appears in the global top ten. The bounded
4,000-target / 2,000-S1 smoke preserved all earlier candidate pairs and added
reverse pairs. This is correctness plumbing, not a challenge recall result.

The full target run was stopped at the runtime gate. On this 15 GiB host,
scoring **5,000 targets against all 2,206,821 S1** took **88 seconds** after
index construction, with 4,060 targets requiring full-index scoring. At that
measured throughput, 10,320,219 targets would take roughly **50 hours** of
scoring alone. The bounded run is explicitly marked
`reverse_profile_targets=5000` in its manifest. No full-pool reverse recall,
99% blocker result, or matcher gain has been measured. The reverse route has
not been added to the learned matcher or promoted to the submission path;
target competition, residual features, and exclusivity remain untested.

### Colab ANN experiment

`reverse_ann_fullpool_colab.ipynb` runs an approximate reverse route with a
cached 128-dimensional sparse random projection and Faiss IVF index over all
training S1 entities. It first times 100,000 targets, refuses a projected scan
above the notebook's session guard, and only reports blocker recall after its
manifest confirms all 10,320,219 targets were scanned. The notebook compares
raw and K=40/60/80/100/150 recall on the 800 development S1 entities; it does
not open the previously used 200-entity confirmation partition. Its output is
an experiment for route selection, not a matcher metric or submission change.

Upload `research_runs/colab_reverse_bundle.zip` and the four original training
TSVs to `MyDrive/ml_challenge/` as shown in the notebook. The bundle contains
the code and existing forward retrieval artifacts, but no dataset or labels.
The notebook saves full retrieval outputs to `MyDrive/ml_challenge/reverse_ann_result`.

The equivalent local ANN run completed against all 10,320,219 targets on the
`mru` checkout. It used 64 projection components, 1,024 IVF lists, `nprobe=8`,
and top ten reverse S1 neighbors. Target-scan checkpoints made the 76-minute
run restartable; peak resident memory was about 2.4 GiB. Its manifest and
development-only report are in `research_runs/reverse_ann_local_full/`.
For a local restart, repeat the same command; the scan manifest skips finished
target chunks:

```bash
python code/business_entity_resolution/01_pipeline.py --split train \
  --data-dir student_resource/dataset \
  --reverse-ann-from research_runs/ber_fullpool_1000_offset300_aug1 \
  --output-dir research_runs/reverse_ann_local_full \
  --ann-cache-dir research_runs/reverse_ann_local_cache \
  --ann-checkpoint-dir research_runs/reverse_ann_local_checkpoints \
  --ann-checkpoint-interval 100000 --reverse-source-chunk 20000 \
  --target-chunk 10000 --reverse-hash-features 131072 \
  --ann-components 64 --ann-nlist 1024 --ann-nprobe 8 \
  --ann-threads 4 --cap 150
```

| Development blocker recall, 2,828 true links | K40 | K60 | K80 | K100 | K150 | Raw union |
|---|---:|---:|---:|---:|---:|---:|
| Forward routes | 0.8893 | 0.9558 | 0.9678 | 0.9714 | 0.9760 | 0.9802 |
| Forward plus reverse ANN | 0.8907 | 0.9562 | 0.9653 | 0.9699 | 0.9752 | 0.9802 |

The ANN route added **zero** of the 56 true links absent from the raw forward
union on the 800 development S1 entities. It displaced four true links from
the K100 selection. It is therefore not promoted to the learned matcher.
The previously opened 200-S1 confirmation partition was not used for this
route decision.

On the full-pool ANN K150 candidates, a grouped three-fold development
comparison gave LightGBM macro F0.5 0.9080 and pair precision 0.9521, and
CatBoost 0.8939 and 0.9637. Both missed the precision target without a
constraint. With the established development-only 0.97 precision-constrained
threshold rule, LightGBM reached mean macro F0.5 0.9026 (folds 0.9012,
0.9178, 0.8887) and mean pair precision 0.9783 (minimum fold 0.9761).
The prior K100 missing-address model scored mean F0.5 0.8978 (folds 0.8973,
0.9202, 0.8759); the ANN K150 model lost on the second fold and lowered mean
missing-address F0.5 from 0.8457 to 0.8347. The wider ANN model is not
selected under the all-fold stability criterion. These are development
comparisons, not a new locked-confirmation result.

A paired K100 ablation kept the original forward candidate IDs fixed and
added only reverse ANN rank/score features. LightGBM mean F0.5 rose from
0.8978 to 0.9018 with precision 0.9787, but the first grouped fold fell
from 0.8973 to 0.8936 and mean missing-address F0.5 fell from 0.8457 to
0.8266. This feature-only variant also fails the stability criterion. The
learned matcher therefore retains the seven established routes; reverse ANN
stays as a research-only artifact.

The retained K100 LightGBM's development-fit importance file is
`research_runs/ber_fullpool_1000_offset300_aug1/best_k100_development_feature_importance.json`.
Its leading features are name length ratio, numeric score relative to the
source's best candidate, rare-token route score, address character similarity,
address token overlap, and number overlap. Importance is descriptive of that
development fit, not an independent feature-selection result.

## Submission path and limits

The earlier reduced-target pilot excluded most true links and its model scores must not be used as estimates of full-pool quality. The previous mru pipeline also seeded its candidate pool with ground-truth targets and ASCII-stripped native name text; these issues are avoided in this study.

`01_pipeline.py --split test` and `02_match.py --mode test` write the official `candidate_pairs.tsv` and `matching_results.tsv` schemas. A 100-S1 smoke verified both files, but full-scale inference for all 2.2 million S1 entities has not been validated on this 15 GiB host. The streaming full-pool research assay handles bounded S1 cohorts; the default full test path still materializes the target pool in memory. Keep existing research files until a full-scale replacement passes validation.
