# K100 sibling evidence, corrected nested models, and structural constraints

Development-only work on `mru`. The existing K100 candidate set came from the
complete 10,320,219-target training pool without labels, positive injection,
or positive protection. All decisions below use S1-grouped development folds.
No earlier confirmation cohort was reopened, no new one was created, and the
selected inference pipeline was not changed.

## Critical correction to the preceding audit

The prior cross-source stage-2 result of **0.5527** was **invalid**. Its saved
OOF score rows and saved pair-feature rows were combined by position, although
only 100/80,000 pairs occupied the same row in both files. The apparent
calibration failure was an indexing error. This study aligns pairs by the
unique `(source1_entity_id, candidate_entity_id)` key and checks one-to-one
membership before fitting. The preceding [forensic audit](FORENSIC_STRUCTURAL_AUDIT.md)
has been corrected. Expected-F, sibling diagnostics, and graph diagnostics use
the OOF score/pair IDs directly and are not affected by this mistake.

## 1. The 247 collective false negatives

The current saved OOF 50/50 blend has 264 retrieved false-negative pairs. Of
these, 247 have at least one correctly predicted target for the same S1. For
each of the 247, the complete K100 neighborhood was inspected. All 247 have a
true predicted sibling as their highest-probability selected sibling; 17 also
have a predicted false-positive sibling. There are 57 missing-address FNs.
The strongest selected sibling is from the opposite source for 168 FNs; 100
missed S2 targets have a selected S3 sibling, and 129 missed S3 targets have a
selected S2 sibling. The latter counts overlap the broader 168 measure because
several selected siblings may exist.

The table evaluates **nonselected development candidates that already have a
selected sibling**: 247 collective FNs versus 72,810 retrieved hard negatives.
"FP exposure" is the number of currently rejected negatives that a blanket
boost would add, rather than the current model's existing FP count. Precision
is the measured incremental `FN/(FN+exposure)` if every flagged pair were
promoted. Thresholds are exploratory diagnostics, not selected rules.

| Label-free sibling signal | FN coverage / 247 | FP exposure | Incremental precision |
|---|---:|---:|---:|
| Sibling probability ≥ .90 | 245 | 72,218 | .003 |
| Candidate p ≥ .20 and sibling p ≥ .90 | 128 | 90 | .587 |
| Candidate p ≥ .40 and sibling p ≥ .90 | 94 | 35 | .729 |
| Native name similarity ≥ .85 | 94 | 8,971 | .010 |
| Transliterated name similarity ≥ .85 | 99 | 9,318 | .011 |
| Address similarity ≥ .85 | 89 | 2,248 | .038 |
| Equal core name | 42 | 5,894 | .007 |
| Shared address token with global IDF ≥ 8 | 159 | 18,151 | .009 |
| Shared address number | 138 | 15,320 | .009 |
| Candidate p ≥ .20, name ≥ .85, address ≥ .70 | 29 | 22 | .569 |
| Candidate p ≥ .20, name ≥ .85, shared number | 33 | 17 | .660 |

For missing-address FNs, native-name similarity ≥ .85 covers 35/57 but
exposes 263 missing-address negatives. Strong address or shared-number signals
cannot help targets with absent addresses. Even the p ≥ .40/sibling p ≥ .90
signal would add 94 TPs and 35 FPs, dropping the existing 2,483/2,541 pair
precision to approximately .965. No direct sibling boost is justified.
Full per-pair measurements, including same/opposite source, core-token,
transliteration, rare-token, numeric, and strongest-sibling features, are in
`research_runs/forensic_20260927/sibling_diagnostics.parquet` and its JSON
summary (ignored research artifacts).

## 2. OOF collective feature design and corrected ablation

The clean nested protocol splits the two outer-training folds into disjoint
S1 cohorts **A/B/C**. Stage 1 fits only A, then predicts B, C, and the held-out
outer cohort D. Stage 2 fits only B; C chooses its threshold from a fixed small
grid under an inner precision ≥ .97 requirement; D is evaluated once. The
same A-fitted stage-1 model scores B/C/D, matching the stage-1 probability
distribution across stage-2 fit, calibration, and evaluation. No candidate's
label or S1 identifier is a model feature. This deliberate split reduces the
stage-1 training size to roughly 320 S1s, so comparisons with the 800-S1
baseline are diagnostic; each stage-2 arm is compared to its same-protocol
stage-1 control.

Collective features exclude the candidate itself: strongest and mean top-two
*other* probabilities, other-candidate counts over .5/.7/.9, strongest same
and opposite-source probability, strongest S2/S3 evidence, probability gap,
and candidate-to-sibling native/transliterated name, core, address, and number
similarity. Existing pair features and global target IDF are retained. The
rule arm adds .08 only when another candidate has p ≥ .9, name similarity
≥ .85, and address similarity ≥ .7. The rule and its magnitude were fixed
before outer-fold results; only the final threshold is chosen on C.

| Model | Fold 1 | Fold 2 | Fold 3 | Mean ± std | Outer pair precision | Result |
|---|---:|---:|---:|---:|---|---|
| Same-protocol stage-1 control | .9372 | .9293 | .9182 | .9282 ± .0078 | .9692/.9601/.9728 | Reference for nested comparison |
| Fixed sibling rule | .9377 | .9369 | .9189 | .9311 ± .0087 | .9682/.9691/.9750 | All-fold gain against reduced-data control, but below original .9353 and precision floor in two folds |
| Stage-2 LightGBM | .9237 | .9076 | .9190 | .9168 ± .0067 | .9763/.9505/.9693 | Reject |
| Stage-2 logistic regression | .9167 | .9325 | .9163 | .9218 ± .0075 | .9707/.9619/.9761 | Reject |

| Model | Mean pair recall | Mean singleton accuracy | Mean missing-address F0.5 | Mean cross-script F0.5 | Mean short-name F0.5 |
|---|---:|---:|---:|---:|---:|
| Stage-1 control | .8798 | .8259 | .8832 | .8759 | .9301 |
| Fixed sibling rule | .8794 | .8444 | .8870 | .8812 | .9323 |
| Stage-2 LightGBM | .8736 | .6444 | .8700 | .8902 | .9434 |
| Stage-2 logistic | .8675 | .7926 | .8697 | .8658 | .9230 |

The sibling rule's outer recall is .8912/.8831/.8639, singleton accuracy
.9000/.8333/.8000, missing-address F0.5 .8605/.9212/.8792, cross-script
F0.5 .9089/.8912/.8435, and short-name F0.5 .9610/.9144/.9216.
The same-protocol stage-1 missing-address scores are .8605/.9065/.8824, so
the rule regresses that slice on fold 3. Its blocker recall remains the fixed
K100 .9713 overall. All thresholds, fold confusion counts, and slices are in
`research_runs/forensic_20260927/collective_clean_nested.json`.

## 3. Pairwise K100 target graph

An unlabeled graph was constructed over each S1's **actual 100 candidates**.
Edges require a strong combination of target-target name/transliteration,
address, rare-address-token, and number evidence; connected-component size,
degree, weighted degree, and support from selected high-probability neighbors
were measured. There are 23,411 undirected edges across 800 K100 graphs.

| Graph signal on nonselected candidates | Collective FN coverage | Negative exposure | Incremental precision |
|---|---:|---:|---:|
| Any graph edge | 77 | 26,573 | .003 |
| Edge to selected p ≥ .90 target | 52 | 261 | .166 |
| Candidate p ≥ .20 and high-prob edge | 38 | 33 | .535 |
| Candidate p ≥ .40 and high-prob edge | 28 | 17 | .622 |
| Candidate p ≥ .40 and cross-source support ≥ .8 | 10 | 7 | .588 |

This graph does enrich some missed siblings but covers few and cannot safely
act as a blanket promotion rule. It was not added to the production matcher.
Graph features and the exact edge definition are saved in the ignored
`research_runs/forensic_20260927/target_graph_features.parquet` and JSON.

## 4. Exact structural constraints

In the current development OOF decisions, 1,558 target IDs are candidates for
multiple S1s, but **zero selected targets** have multiple predicted owners.
Greedy, margin-aware, or strong-margin target exclusivity therefore change
**zero pairs** and score exactly the same as no exclusivity. Country conflict
occurs in 3,900/80,000 K100 pairs, zero true candidate links, and zero
selected predictions. An open-set nonempty-string country post-filter also
changes zero current predictions. Neither is promoted as an accuracy gain.
See `research_runs/forensic_20260927/structural_constraints.json`.

## 5. Bounded exact sparse reverse test

The 56 development raw blocker misses were selected for a **diagnostic** query
set using labels. Each target was then scored without labels against every
same-country S1 in the complete 2,206,821-row training S1 corpus. Exact sparse
TF-IDF cosine used target name character trigrams (the 262,144 most frequent
S1 features) and unpruned address word tokens. Similarity was computed by
exact sparse multiplication on that fixed vocabulary;
the country string is an open-set equality partition. The true S1 ranked in
the top 8 for **36/56** queries, top 10 for 37, top 40 for 41, and top 100
for 45. India: 22/38 top 8; US: 14/18 top 8. Peak RSS was ~2.0 GB for the
bounded search, elapsed 85 seconds including indexing and TSV reads.

This is a meaningful **potential** rescue of raw misses, unlike the prior
FAISS ANN result, but is not a new blocker recall. A top-8 route over all
10,320,219 target records could propose up to ~82.6M target-to-S1 pairs before
deduplication. The complete label-free target scan, runtime, candidate union,
K100 capping, and matcher response have not been measured. No route was added.
Details are in `research_runs/forensic_20260927/sparse_reverse_bounded.json`.

A subsequent label-free throughput profile used the first 50 S2 and 50 S3
targets per training country, searched against **all** 883,188 India or
1,323,633 US S1s, and measured full exact sparse multiplication on the same
fixed TF-IDF vocabulary. India averaged **0.266 s/query**, with 574k nonzero
S1 scores per query; US averaged **0.323 s/query**, with 463k nonzero scores.
Peak RSS was 1.77 GB. Multiplying these sample rates by the complete
country-specific target counts gives an approximate **35.9 days of sequential
query time** for all 10.32M targets, before writing, merging, and capping
pairs. Even ideal 12-core scaling would remain about three days. The sample
is small and not a precise runtime forecast, but it is a concrete local
runtime bottleneck. The full scan was therefore not started with this naive
sparse-multiplication implementation. See the ignored
`research_runs/forensic_20260927/sparse_reverse_profile.json`. A bounded
top-N sparse kernel or stronger query-feature pruning would need a new profile
and a recall check against the 56 misses before another full scan is justified.

**Follow-up runtime gate.** The Apache-2.0 `sparse-dot-topn` kernel was
installed into `/tmp` and verified to return the same top-8 IDs and scores as
SciPy on representative label-free queries. On a larger, fixed 4,000-target
sample (1,000 S2 and 1,000 S3 from each training country), eight-thread
search averaged **0.00990 s/India target** and **0.01146 s/US target**, with
peak RSS **2.34 GB**. At those measured rates, the complete 4,133,346 India
and 6,186,873 US training targets imply approximately **31.1 hours of search
time** for this *two-view* name/address scorer alone. The requested native,
compact, transliterated, address, and combined views would add work. Query
pruning to six name and four address TF-IDF terms lowered the bounded
top-8 diagnostic rescue from **36/56 to 25/56**, while mean per-query time
remained about **0.258 s** with ordinary sparse multiplication. The full-pool
label-free scan was not completed, so all full-pool union/K-cap/matcher metrics
remain unmeasured. Profiling files are the ignored
`research_runs/forensic_20260927/sparse_reverse_topn_profile_1000.json` and
`sparse_reverse_pruned_bounded.json`.

The closer Ayan-style `max_df=0.05` variant removes terms present in more
than 5% of same-country S1s. It retains **34/56** bounded top-8 diagnostic
misses (36/56 without this cutoff). On 4,000 label-free target samples, its
eight-thread top-N kernel averages **0.00447 s/India target** and **0.00892
s/US target**. Multiplying by all training target counts gives **20.46 hours
of search alone** for the native-name/address two-view scorer, before the
requested compact/transliterated views, normalization, output serialization,
union, and K100 capping. Peak measured RSS was **2.09 GB**. Four of five
representative US queries matched SciPy's top-8 IDs exactly; the fifth had
equal cutoff scores but different tied IDs, while the score sets matched.
This is a runtime limitation, not a RAM failure. There is no accessible GPU
driver on this host. The full label-free target scan remains unrun; bounded
query ranks must not be reported as blocker recall. See the ignored
`sparse_reverse_bounded_maxdf05.json` and
`sparse_reverse_topn_profile_1000_maxdf05.json` artifacts.

## 6. Public comparison and the apparent 0.97+ gap

The public numbers below are author reports, not scores reproduced here. A
larger labeled training cohort and a stronger reverse retrieval path are major
structural differences from our 800-S1 development training setup.

| Pipeline | Candidate recall | Validation target pool / split | Threshold | Reverse / dictionary / stage 2 / exclusivity / sibling support | Reported macro F0.5 | Directly comparable? |
|---|---:|---|---|---|---:|---|
| `mru` | .9713 K100 dev; .9682 separate confirmation | Full 10.32M; grouped S1; new disjoint confirmation | Inner grouped OOF | FAISS reverse rejected; no learned dictionary; no selected stage 2; no effective collisions | .9353 dev; .9516 predeclared confirmation control | Reference |
| [Sid](https://github.com/Sid-techweb/AmazonML-New) | .987 author local | Full 10.3M; S1 80/20 grouped | Local held-out sweep; later sealed v2 report | Target→S1 retrieval; fold-fit Indic dictionary; no selected sibling stage 2; best-owner target assignment | .9783 local v2; .970441 user-reported leaderboard | Full-pool design is credible, but different cohort/scale and local threshold selection prevent direct comparison |
| [Ayan](https://github.com/AyanAhmedKhan/amazon-ml-challenge) | ~.989 author local | Full target pool; S1 grouped fold 0 holdout | OOF/ES after protocol correction | Sparse reverse; native→Latin maps; OOF stage 2; exclusivity; sibling support | .98703 local E-6 | Full-pool grouped methodology is relevant, but [validation strategy](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/VALIDATION_STRATEGY.md) discloses maps fit on all train pairs including validation, so local score is optimistic until that is removed |
| [Akash](https://github.com/Akash-bardia/amazon-ml-challenge-2026) | .9648 author local | [Training script](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/train.py) adds all validation true targets to a 300k-distractor pool; grouped S1 IDs | Selects S2/S3 thresholds on the reported validation set | No sparse reverse or learned Indic map; exclusivity; source thresholds; no stage 2 | .9761 local | No: reduced pool includes validation positives and threshold is selected on reported set |

Sid's full-pool retrieval and fold-fit dictionary are credible ideas, though
its README distinguishes local .9783 from the user-reported .970441
leaderboard. Ayan's sparse reverse concept is supported by our bounded miss
test; its exact local score is not leakage-free as documented by its own
validation note. Ayan also reports expected-F below its threshold baseline,
matching our negative ablation. Sid reports a sibling-feature run that fell to
.933 and was withdrawn; sibling support is not automatically beneficial.
Akash's reported validation is not usable as a benchmark for our full-pool
pipeline.

## 7. Decision

**Best trustworthy development configuration remains the existing K100 50/50
IDF + numeric blend**: .9339/.9403/.9316 folds, .9353 mean. The strongest
observed untouched result remains its predeclared .9516 confirmation control;
the frozen selected policy is .9496. No new model here meets the promotion
requirements: meaningful improvement over .9353, precision ≥ .97, missing
address stability, singleton stability, and no material fold regression.
No new confirmation was opened and no pipeline commit was made.

The next concrete experiment is a retrieval representation that removes common
posting work without losing the bounded rescues, followed by another runtime
gate before any full-target scan. Only after the full label-free scan and K100 comparison can the 36/56 bounded
rescues be counted as real blocker gains. If pursued, keep `candidate_pairs.tsv`
identical to the union actually passed to the matcher and maintain France
through open-set country equality.
