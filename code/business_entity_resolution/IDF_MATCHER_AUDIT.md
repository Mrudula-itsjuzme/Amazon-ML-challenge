# Frozen full-pool IDF matcher audit

The selected K100 LightGBM improved macro entity F0.5 from **0.9225 to 0.9393** on one previously untouched 500-S1 confirmation cohort. Pair precision was **0.9762**. The 0.95 goal was **not reached**. The confirmation was opened once, after the candidate set, feature groups, model, and threshold grid were frozen in `frozen_idf_policy.json`. All candidates came from the complete 10,320,219-record S2/S3 target pool without labels or positive injection. The confirmation report is `research_runs/ber_fullpool_500_offset1800_aug1/frozen_idf_confirmation.json`.

## Why the earlier 0.9309 was higher than the later 0.8932

| Factor | Earlier 200-S1 locked result | Later 500-S1 external result | Finding |
|---|---|---|---|
| Cohort | Hash offset 300, sealed 200 of 1,000; 705 true links | Disjoint hash offset 1300; 1,725 true links | Different cohorts. Earlier development-fold mean was only 0.8984, so its 0.9309 confirmation was optimistic relative to those folds. |
| Candidate generation and pool | Seven forward routes; 10,320,219 targets | Same routes and full target pool | No reduced-pool explanation. Both retrieval runs were label-free. |
| Cap | K60, blocker recall 0.9574 | K100, blocker recall 0.9670 | Wider cap raised recall; it also changed the hard-negative distribution. |
| Features | Base features, including missing-address features | Base plus explicit missing-address features | The earlier external K60 control removed missing-address columns and was not an exact replay of the 0.9309 arm. The new same-cohort replay includes them. |
| Model | LightGBM, 350 trees, learning rate 0.05, 31 leaves | Same LightGBM parameters | No model-parameter explanation. |
| Threshold | 0.7, development macro-only grid | 0.8, development grouped OOF with 0.97 precision floor | Different decision policies affect recall and precision. |
| Leakage | Grouped S1 splits; no labels in retrieval | Same safeguard | Code and manifests show no positive injection or target-pool leakage. The old locked labels were not reopened for this audit. |
| Sampling and difficulty | 117 US / 83 India; 21 cross-script entities | 285 US / 215 India; 69 cross-script entities | Similar country mix but different entity difficulty. Cohort variation and threshold policy are confounded in the historical comparison. |

An exact earlier-policy K60 arm trained on the 800 original development S1 groups and evaluated on the **new same 500-S1 cohort** scored **0.9215**; the retained K100 control scored **0.9225**. This reproduces the old setup's code/configuration and its development-selected 0.7 threshold on new data, but deliberately does not rerun the old sealed 200 labels. The historical 0.9309 is a valid one-time full-pool result, not evidence that the older setup generalizes better. This new same-cohort comparison largely removes model, pool, and cohort confounding.

## Development-only blocker mining

The 800-S1 development partition has 2,828 true links. The raw route union found 2,772 (0.9802); K100 found 2,747 (0.9714). All **56 raw misses** and **25 additional cap misses** are listed in `research_runs/ber_fullpool_1000_offset300_aug1/development_all_k100_misses.tsv`. Counts below overlap because one link can have several failure modes.

| Failure mode | Raw misses / 56 | Additional K100 cap misses / 25 |
|---|---:|---:|
| Cross-script or transliteration | 23 | 9 |
| Short or generic name | 41 | 19 |
| Missing address | 15 | 4 |
| Heavy corruption | 6 | 1 |
| Numeric conflict | 11 | 3 |
| Truncated address | 4 | 2 |
| Legal suffix or tokenization | 0 | 1 |
| Exact-name distractor saturation | 47 | 21 |
| Country conflict | 0 | 0 |

Raw misses were 38 India and 18 US; cap misses were 16 India and 9 US. The exact-name predicates rescued no raw miss. A label-free full-pool shadow scan of source address number plus token keys found at most 10/56 raw and 5/25 cap misses with target document frequency at most 100, but could add up to 65,408 duplicate candidate pairs across 800 sources. The top three rare keys per source reduced that upper bound to six raw and three cap rescues for up to 22,571 pairs. These are *upper bounds*, not measured route gains. No new route was selected. Reverse ANN had previously rescued zero raw misses and remains excluded.

## Grouped development ablations

All rows used the same full-pool, label-free K100 candidates, 800 S1 groups, 80,000 retrieved pairs, 2,747 positives and 77,253 retrieved hard negatives. Three outer grouped folds and inner grouped out-of-fold threshold selection were used. The mean/std are across outer folds. The base retained model scored 0.8978 (folds 0.8973/0.9202/0.8759), precision 0.9793 and recall 0.8028. These scores are **development only**.

| Change from base unless specified | Mean macro F0.5 | Std | Fold scores | Mean pair precision | Decision |
|---|---:|---:|---|---:|---|
| Global full-target token IDF | 0.9299 | 0.0045 | .9239 / .9349 / .9310 | .9780 | All-fold gain |
| Lexical shape | 0.9008 | 0.0169 | .8941 / .9241 / .8843 | .9809 | Rejected: unstable |
| Numeric and source interactions | 0.9171 | 0.0204 | .9099 / .9448 / .8965 | .9802 | Gain over base, but weaker than IDF |
| Relative ranks, margins and competition | 0.8976 | 0.0153 | .8926 / .9184 / .8820 | .9806 | Rejected |
| IDF plus numeric interactions | 0.9356 | 0.0073 | .9337 / .9453 / .9278 | .9783 | One fold below IDF alone |
| Regularized LightGBM on IDF plus numeric | 0.9338 | 0.0125 | .9337 / .9492 / .9186 | .9797 | Rejected |
| CatBoost on IDF | 0.9104 | 0.0104 | .9084 / .9241 / .8988 | .9760 | Rejected |
| Two-stage LightGBM on IDF | 0.9241 | 0.0113 | .9214 / .9391 / .9119 | .9773 | Rejected |
| IDF plus finer development-only threshold grid | **0.9313** | **0.0035** | .9272 / .9356 / .9312 | .9766 | Selected before confirmation |

The selected model's grouped development blocker recall was 0.9713, mean pair recall 0.8664, mean singleton accuracy 0.8778, missing-address F0.5 0.8783, cross-script F0.5 0.9036, and short-name F0.5 0.9235. Source-specific missing-address/cross-script thresholds provided no change in grouped development. Greedy target exclusivity was skipped: no baseline predicted target collided across sources in development OOF predictions.

The global token DF tables were computed once from the unlabeled complete target pool with `04_global_df.py`. `02_match.py` adds eight candidate-specific IDF features: weighted name/address overlap, maximum shared-token IDF, and maximum source-only/target-only token IDF for names and addresses. The selected model kept the existing LightGBM parameters and seven retrieval routes. Top confirmation split-count importance was target-only name max IDF (796), target-only address max IDF (389), name length ratio (366), source-only name max IDF (318), and numeric score relative to the best candidate (305). Importance does not establish causal effect; the grouped ablation does.

## One-time locked confirmation

This disjoint hash-offset-1800 cohort has **500 S1 entities, 1,730 true links, and 50,000 K100 candidate pairs**. K100 contains 1,669 positives and 48,331 retrieved negatives. K40/60/80/100/150 blocker recall was **0.8867 / 0.9474 / 0.9578 / 0.9647 / 0.9688**. Confirmation threshold 0.7 was selected on development grouped OOF data. This table is the final untouched-cohort result; fold mean/std refer to development, not repeated confirmation samples.

| Model | Blocker Recall | Macro F0.5 | Precision | Recall | Singleton Accuracy | Mean Fold Score | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier K60 policy | .9474 | .9215 | .9691 | .8509 | .9167 | .8984 | .0134 |
| Retained K100 LightGBM | .9647 | .9225 | .9751 | .8358 | .9167 | .8978 | .0181 |
| **Selected K100 LightGBM + global IDF** | **.9647** | **.9393** | **.9762** | **.8786** | .8333 | **.9313** | **.0035** |

The selected model has the highest full-pool locked macro F0.5 and precision remains above 0.97. It gained on cross-script entities (.8822 versus .8566 for retained K100, 61 entities) and short names (.9236 versus .9078, 67 entities). Missing-address F0.5 **fell** slightly (.8536 versus .8586, 69 entities), and singleton accuracy fell to 10/12 from 11/12. These slices are small and were not used to revise the frozen model. Failure counts for the selected model: **61 blocker misses, 149 matcher false negatives, 37 matcher false positives**. The retained K100 control had 61, 223 and 37 respectively. The model improved recall mostly by accepting retrieved true links; blocker recall remains below 0.99.

The selected architecture **did not reach 0.95** on grouped development or locked confirmation. Its improvement over the same-cohort retained K100 control is 0.0168 macro F0.5, with precision improving by 0.0012. The historical 0.9309 result and the earlier 0.8932 external result remain separately documented as single-cohort outcomes; neither was used to retune this confirmation result.
