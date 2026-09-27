# Cross-branch audit and frozen K100 blend

This is a full-target-pool research result on `mru`. The **frozen selected policy** scored **0.9496 macro entity F0.5** on one untouched 500-S1 confirmation cohort. A **predeclared 50/50 blend control** scored **0.9516** on that same cohort. The control crossed 0.95 as a legitimate one-time measurement, but it was not the frozen selected policy; the confirmation result was not used to change the policy. Both were trained on the original 800 development S1 groups and used thresholds chosen from those groups only. The result is stored in `research_runs/ber_fullpool_500_offset2800_aug1/frozen_blend_confirmation.json`.

## Branch audit

The remote had `main`, `mru`, and `sup` when checked. A stale local `origin/meg` ref contains EDA and no matcher; the local backup branch has no separate entity-resolution pipeline. No branch was merged.

| Branch code | Different idea | Development evidence | Decision |
|---|---|---|---|
| `sup/01_preprocess.py` | Unidecode then ASCII-only normalization | Removes native-script information already preserved on `mru`; exact normalized name finds 0/56 raw misses. | Reject replacement preprocessing. |
| `sup/02_generate_candidates.py` | Exact name, transliterated name, exact address, address token, address number postings with 500-posting cap | Exact name/address paths find 0/56 raw misses. The exact full-pool `sup` address-DF scan found one potential raw miss (`villano`) and one cap miss (`houlton`). The raw source's eligible token union exceeded 500 targets, so `sup` itself drops it. **Actual raw rescues: 0/56**. The cap-loss posting contains 486 targets and is not a measured K100 rescue. | No new route. |
| `sup/03_filter_candidates.py` | Fixed similarity/country thresholds | Hand rules would discard candidates before the learned matcher; no grouped evidence of better macro F0.5. | Do not import thresholds. |
| `sup/04_evaluate.py` | Chunked pair and macro metrics | Useful reporting pattern, but `mru` already measures grouped full-pool macro F0.5 and blocker recall separately. | No model change. |
| `sup/05_make_submission.py` | Group filtered candidates into output | No learned scoring or useful retrieval idea. | No model change. |
| `main/src/preprocess.py` | Legal suffix mapping, address abbreviations, postal/numeric tokens, Unicode-safe basic normalization | Legal/core and numeric representations largely already exist on `mru`. Three new canonical-address features lowered grouped mean F0.5: IDF .9336→.9306; IDF+numeric .9384→.9338 on the same folds. | Reject address feature addition. |

All blocker checks used the 800-S1 development partition and the complete 10,320,219-target pool. The `sup` token DF scan was label-free; labels were used afterward only to count development misses. The previously opened confirmation cohorts were not reopened.

## Development comparisons

The original 800-S1 development set contains 2,828 true links. K100 generated 80,000 pairs: 2,747 positives and 77,253 retrieved hard negatives. The following grouped three-fold experiments used nested grouped out-of-fold threshold selection with a 0.97 pair-precision floor. Standard deviations are across folds.

| Model | Mean macro F0.5 | Std | Folds | Precision | Recall | Singleton accuracy | Missing-address F0.5 | Cross-script F0.5 | Short-name F0.5 | Blocker recall |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| IDF baseline, original seed | .9313 | .0035 | .9272 / .9356 / .9312 | .9766 | .8664 | .8778 | .8783 | .9036 | .9235 | .9713 |
| IDF + numeric, 50/50 score blend | .9353 | .0037 | .9339 / .9403 / .9316 | .9772 | .8780 | .8630 | .8915 | .9086 | .9386 | .9713 |
| **Frozen blend + IDF-top floor 0.5** | **.9365** | **.0036** | .9377 / .9403 / .9316 | .9776 | .8780 | .8963 | .8915 | .9086 | .9386 | .9713 |
| IDF + numeric direct model, alternate seed | .9384 | .0079 | .9343 / .9494 / .9314 | .9752 | .8883 | .8630 | .8951 | .9165 | .9360 | .9713 |
| IDF + numeric + relative competition | .9337 | .0094 | .9365 / .9435 / .9210 | .9815 | .8718 | .8593 | .8883 | .9014 | .9267 | .9713 |
| IDF + numeric + lexical shape | .9375 | .0079 | .9321 / .9487 / .9316 | .9766 | .8842 | .8630 | .8949 | .9163 | .9369 | .9713 |
| IDF + numeric + both groups | .9356 | .0075 | .9332 / .9458 / .9278 | .9751 | .8859 | .8444 | .8925 | .9051 | .9378 | .9713 |

The direct IDF+numeric model had the highest **experimental grouped mean** (.9384), but fold 3 fell below its same-run IDF control and singleton accuracy stayed low. Relative competition, lexical shape, and `main` address abbreviations did not provide stable improvements. The frozen source rule suppresses all blend predictions for an S1 only when the IDF model's highest K100 probability is below 0.5. This rule was suggested by a development singleton error, so its old-fold gain is exploratory; it was then checked unchanged on a disjoint development cohort.

The label-free hash-offset-2300 **new development** cohort had 500 S1s and 1,807 true links. With thresholds trained on the prior 800 only, IDF scored .9376; the blend and source-floor rule both scored **.9518** macro F0.5, .9854 precision, .8993 recall, .8889 singleton accuracy, .9285 missing-address F0.5, .9124 cross-script F0.5, and .9579 short-name F0.5. The source floor changed no prediction there. This was development evidence, not a confirmation result.

## One-time untouched confirmation

The frozen policy, code hashes, global target-DF hashes, model seed, K100 cap, thresholds, and source rule are in `frozen_blend_policy.json`. The hash-offset-2800 confirmation cohort was disjoint from the original 1,000 S1s and all earlier 500-S1 cohorts. Preflight verified full-pool label-free retrieval, candidate counts, cohort disjointness, and exact feature parity before opening labels. The evaluator wrote a once-only marker and refuses a repeat.

There were **500 S1 entities, 1,696 true links, and 50,000 K100 retrieved candidate pairs**: 1,642 positives and 48,358 retrieved negatives. Candidate recall at K40/60/80/100/150 was **.9051/.9552/.9646/.9682/.9746**. These are blocker recall measurements, not matcher accuracy.

| Predeclared model | Blocker recall | Macro F0.5 | Pair precision | Pair recall | Singleton accuracy | Missing-address F0.5 | Cross-script F0.5 | Short-name F0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IDF control | .9682 | .9462 | .9867 | .8744 | 1.0000 | .9263 | .9084 | .9591 |
| 50/50 blend control | .9682 | **.9516** | .9819 | .8980 | .9000 | .9390 | .9131 | .9635 |
| **Frozen selected blend + source floor** | .9682 | **.9496** | .9819 | .8974 | .9000 | .9390 | .9131 | .9635 |

The selected policy improved over the same-cohort IDF control by .0033 macro F0.5 and over the previous best 0.9393 confirmation by .0103. It did **not** exceed 0.95. The predeclared blend control did exceed 0.95 by .0016, but selecting it after seeing confirmation would use the locked cohort for model selection. The source floor removed one true link and did not improve singleton accuracy on this cohort. No threshold, model, or rule was changed after confirmation. The 0.95 margin for the control is small, and this is one cohort rather than a guarantee across future S1 populations.

Confirmation failure counts were 54 true links missed by K100 retrieval for every arm. The IDF control had 159 matcher false negatives and 20 false positives; the blend control had 119 and 28; the selected source-floor policy had 120 and 28. The numeric model's leading split-count features included target-only name max IDF (683), name length ratio (356), source name length (334), relative number difference (317), and source-only name max IDF (306). Split importance is descriptive; grouped ablations are the evidence for feature effect.
