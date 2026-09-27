# Structural audit and decision experiments (development only)

This audit uses the complete provided train/test TSVs for label-free structure, all
7,638,365 training links for labeled structure, and the existing 800-S1 K100
full-target-pool development cohort for error and decision experiments. The
previous confirmation cohorts were not reopened. Public repository metrics are
author claims, not reproduced results.

## 1. TSV structure and supported invariants

| Observation | Measured result | Use and limit |
|---|---:|---|
| S1 rows / true links | 2,206,821 / 7,638,365 | Full training GT scan |
| Distinct true target IDs / conflicting owners | 7,638,365 / **0** | Target exclusivity is exact in train; only use globally if conflicts arise at prediction time |
| Country mismatches in true links | **0** | Partition by the original, open-set country string; do not hard-code US/India |
| S1 no-match / one-match / two-or-more | 123,247 / 119,157 / 1,964,417 | A match-exists decision matters, but most S1s have multiple links |
| S1 with both S2 and S3 links | 1,776,047 | Cross-source collective evidence is plausible |
| Linked S2 / S3 targets | 3,693,619 / 3,944,746 | Source-specific corruption and decisions deserve testing |
| Missing linked target address, S2 / S3 | 165,184 (4.47%) / 171,834 (4.36%) | The source-specific missing-address rates are similar |
| Exact raw linked address, S2 / S3 | 382,899 (10.37%) / 170,383 (4.32%) | S3 changes the address more often |
| Exact raw linked name, S2 / S3 | 407,332 (11.03%) / 413,693 (10.49%) | Name copies alone are weak retrieval evidence |

The per-S1 S2 count distribution is 0:287,745, 1:789,108, 2:652,779,
3:333,957, 4:119,078, 5:24,154. The S3 distribution is 0:266,276,
1:716,417, 2:668,375, 3:372,443, 4:145,116, 5:35,378, 6:2,816.
No S1 has more than five S2 or six S3 links in the provided GT.

In a deterministic sample of linked targets (numeric target ID divisible by 100),
S2/S3 cross-script proxies are 15.4%/12.0%, numeric conflict 7.7%/6.8%,
token reordering 5.1%/4.9%, and alias-like weak-name/strong-address 1.3%/2.5%.
These are operational proxies, not an exhaustive corruption ontology. Target
address truncation by the sample's strict prefix test is 0.06%/0.86%.

## 2. Train/test shift and copies

Train S1 countries are US 1,323,633 and India 883,188. Test S1 has India
809,986, US 663,106, and **France 259,452 (15.0%)**. There is no labeled
France analogue. Test France has 703,378 S2 and 731,615 S3 targets; target
non-ASCII names occur in 24.5%/23.9% respectively. A new France-specific
learned rule cannot be validated with these labels. Open-set country equality
and label-free corpus statistics transfer naturally.

Median name length is 24 for S1 and 25 for targets in both train and test.
Median name token count is four throughout. Test target missing-address rates
and short-name rates are broadly similar to train after conditioning on country,
but the country mixture changes. Test target counts are 4,887,273 S2 and
5,082,316 S3 versus train's 5,034,616 and 5,285,603.

Using exact NFKC, casefold, whitespace-normalized keys, train has 528,922
name keys shared across S2/S3, but only 2,094 shared name-and-address keys.
Test has 4,838 shared name-and-address keys. Within the 800-S1 development
K100 candidate universe, 1,013 exact name/address duplicate groups cover
2,073 targets; 47 groups have at least two labeled links and **zero** of those
split across different S1 owners. Only 40 groups cross S2/S3. These counts
motivate an OOF sibling feature, but exact copies have limited reach. Near-copy
ownership and train/test candidate-density shift are not yet measured; no
claim depends on them.

## 3. Current error topology

The saved grouped OOF 50/50 blend on 800 development S1s has 58 false-positive
candidate pairs and 264 false-negative *retrieved* pairs. Additional blocker
losses are separate. Of FNs, 149 are S3 and 115 S2; 59 lack target address,
63 have numeric conflict, 35 have short S1 names, 32 meet the cross-script
proxy, and 19 have strong address but weak transliterated name. Categories
overlap. Among FPs, 37 are S2 and 21 S3; 12 lack target address.

Crucially, 247/264 FNs and 52/58 FPs occur on S1s with another correctly
predicted link. Only four true candidates at rank 2 or 3 sit behind a false
rank-1 distractor. There are zero predicted target ownership collisions in
this cohort, so greedy exclusivity would change no current predictions.
Automatic six-cluster error grouping finds recurring weak-address/strong-name,
weak-name/strong-address, number-conflict, and moderate-both-field cases;
cluster counts and centroids are in the ignored development artifact
`research_runs/forensic_20260927/development_blend_error_topology.json`.

## 4. Public approach audit

| Public project | Structural idea worth testing | Validation caveat |
|---|---|---|
| [Sid-techweb/AmazonML-New](https://github.com/Sid-techweb/AmazonML-New) | Target-to-S1 retrieval; learned Indic dictionary; one-sided differences and candidate margins | Author reports ~0.978 local and ~0.970 leaderboard. Local threshold sweep uses its validation fold; scores are not our grouped full-pool benchmark. Its earlier sampled-corpus score was explicitly invalid for full-pool inference. |
| [Akash-bardia/amazon-ml-challenge-2026](https://github.com/Akash-bardia/amazon-ml-challenge-2026) | Source thresholds, first-token and number conflicts, hard negatives | Its training script builds a reduced pool from all validation true targets plus 300,000 distractors, injects training positives into training pairs, and selects thresholds on the reported validation set. Its reported ~0.976 is therefore not comparable to our full-pool locked score. |
| [AyanAhmedKhan/amazon-ml-challenge](https://github.com/AyanAhmedKhan/amazon-ml-challenge) | Sparse multi-view TF-IDF, exact sparse target-to-S1 reverse, OOF collective features, expected-F sets | The reverse representation differs from our failed 64-D FAISS ANN. Its Modal stage requests roughly 128 GB RAM; local feasibility and exact full-pool recall must be checked. Its code correctly guards target competition when the graph is incomplete. |

No LICENSE file was found in the checked public clones; concepts should be
reimplemented independently rather than copying source. Public README results
do not establish generalization under our protocol.

## 5. Ranked hypotheses and experiment plan

1. **Per-S1 set decision / cross-source support.** Most matcher errors are
   partial matches on S1s with existing TPs. Test expected-F first, then a
   genuinely nested OOF stage-2 model using source-level support and an explicit
   no-match probability. Stage-1 predictions for stage-2 training must be made
   by models that never saw the relevant S1 labels.
2. **Source-specific calibration.** S3 contributes more FNs and has fewer exact
   address copies. Tune a small S2/S3 offset grid only on inner grouped OOF.
3. **Fold-safe corruption mappings.** Learn Indic/native-Latin and abbreviation
   mappings from GT of training folds only; quantify cross-script rescues and
   wrong-alias FPs separately.
4. **Sparse reverse retrieval.** First audit the 56 development raw misses and
   25 K100 losses with label-free sparse retrieval. Add a route only if unique
   rescues justify its candidate growth and full-pool runtime.
5. **Collective copy/competition evidence.** OOF sibling support has a plausible
   but limited exact-copy base. Target exclusivity is a true GT invariant, but
   zero observed predicted collisions make its immediate gain small.

All candidates must use the complete 10,320,219-target training pool, K100
unless a measured blocker rescue justifies a change, grouped S1 folds, and
untouched confirmation. A new confirmation cohort is warranted only after
stable, material development gains; 0.9516 remains a prior predeclared control,
not a threshold-tuning set. Test data can contribute label-free statistics only.

## 6. Completed decision ablations

These use identical saved K100 grouped OOF pair probabilities. They change only
the decision rule. Expected-F assumes independent, calibrated pair probabilities
and includes an empty-set option. The source-offset grid is selected on the
other two folds with pair precision at least 0.97, then applied to the held-out
fold. Means and population std are across three grouped folds.

| Decision rule | Folds | Mean | Std | Finding |
|---|---|---:|---:|---|
| IDF control | .9272 / .9356 / .9312 | .9313 | .0035 | Reference |
| 50/50 blend | .9339 / .9403 / .9316 | .9353 | .0037 | Current development baseline |
| Expected-F, raw probabilities | .9348 / .9393 / .9303 | .9348 | .0037 | Worse mean; fold 3 and singleton slice regress |
| Expected-F, 0.03 missing-link allowance | .9348 / .9376 / .9303 | .9342 | .0031 | Worse |
| Expected-F, 5% probability shrink | .9345 / .9392 / .9303 | .9347 | .0036 | Worse |
| Nested S2/S3 threshold offsets | .9378 / .9430 / .9308 | .9372 | .0050 | Higher mean, but fold 3 regresses; not promotable |
| Earlier cross-source stage-2 experiment | invalid | invalid | invalid | Saved score and feature rows were misaligned; see the corrected nested study |

The nested source-offset model had per-fold precision .9790/.9805/.9734,
recall .8776/.8712/.8871, singleton accuracy .9000/.8889/.8000,
missing-address F0.5 .8664/.9166/.8947, cross-script F0.5
.9190/.9311/.8808, and short-name F0.5 .9659/.9346/.9333. Blocker recall
is fixed at .9713 overall; neither decision changes candidates. Detailed
fold metrics, FP/FN counts, and exact S1 IDs/pair IDs changed are in the ignored development artifact
`research_runs/forensic_20260927/expected_f_ablation.json`.
Against the blend, raw expected-F improves 8 S1s and worsens 23; it adds 8
pairs and removes 25. Nested source offsets improve 18 S1s and worsen 15;
they add 18 pairs and remove 17. These are entity-score changes rather than
independent candidate recalls.

These ablations do **not** support a new locked confirmation or a changed
selected pipeline. The next highest-value experiment is a nested OOF
cross-source stage-2 model **with matched stage-1 training density**, grouped
calibration, exact train/test feature parity, and a separate no-match head.
The first cross-source stage-2 implementation had an indexing error: its saved
OOF score order differed from the saved feature order (only 100/80,000 rows
aligned). Its .5527 result and its apparent calibration failure are invalid,
not evidence about the method. The corrected nested study explicitly joins on
S1 and target IDs and separates stage-1 fitting, stage-2 fitting, threshold
selection, and outer evaluation by disjoint S1 cohorts. See
[COLLECTIVE_SIBLING_AUDIT.md](COLLECTIVE_SIBLING_AUDIT.md). The current saved
OOF scores cannot be reused naively to train a new head because their models
overlap a prospective outer holdout.

## 7. Ideas already rejected on `mru`

The SVD+FAISS reverse ANN scanned all 10.32M targets and rescued zero raw
true links; its K100 recall fell. Prior relative competition, lexical-shape,
numeric-context direct model, address canonicalization, source-floor gates,
and sup exact-posting routes lacked stable grouped or locked improvement.
These are recorded in [BRANCH_AUDIT_BLEND.md](BRANCH_AUDIT_BLEND.md) and
[README.md](README.md). The sparse reverse hypothesis is a different
representation, not a repeat of FAISS ANN.
