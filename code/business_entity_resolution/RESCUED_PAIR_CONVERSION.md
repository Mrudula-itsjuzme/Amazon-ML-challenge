# Rescued-pair conversion research: why the matcher still rejects the recovered links

Development-only work on `mru`, continuing [FORWARD_POOL_EXPLOITATION.md](FORWARD_POOL_EXPLOITATION.md).
The learned cap-ranker (K100 blocker recall .9802, all 25 cap-loss links recovered,
zero displacements) is treated as fixed. No new retrieval route, no wider K, no
confirmation cohort, and no production pipeline change. All ablations use the
recorded nested grouped protocol (three outer grouped S1 folds, inner
GroupKFold(3) OOF, fixed 23-point grid, 0.97 OOF pair-precision floor, 50/50
IDF+numeric LightGBM blend). The untouched 0.9516 confirmation was never reopened.

## 1. Forensic table of the 18 rejected rescues (`rescue_forensics.tsv`)

Full per-pair records: strings, scripts, countries, missing-address flags,
numeric tokens, IDF overlaps, unmatched-token evidence, blend probability,
threshold, nearest false candidate, and feature diffs against it.

**Failure-family counts (multi-label, 18 pairs):** short/generic name 13,
token reorder/partial 11, corruption/typo 8, alias/rewrite 7,
cross-script 5, missing address 4, number mismatch 3.

| Root cause | Evidence |
|---|---|
| **Corrupted transliterations** | "sai software private limited" ↔ "saai sphttoy yaar praaibhett limittedd"; "my producer" ↔ "maay proddyuusr praa li"; "heartland" ↔ "heart1and"; "communities" ↔ "commnultis"; "spinning" ↔ "spnlnnhng"; "medicine" ↔ "mhdieiciten" |
| **Token reorders** | "johnson communities l l c" ↔ "l l c johnson commnultis"; "total education associates" ↔ "total associates center"; "internal medicine grand specialists" ↔ "internal specialists grand mhdieiciten" |
| **Junk suffixes / ID pollution** | "onsui com", "silver media id 81572", "nhssarada com" |
| **Probability gap** | 12/18 have blend prob < 0.1; nearest false candidates beat them via higher name+address joint evidence |

The 18 pairs are individually hard but **systematically corrupted**: doubled
letters, digit de-leeting, accents, reorders, junk suffixes — exactly the
corruptions a label-free normalization could target.

## 2. Learned alias/transliteration map (Phase 2)

A fold-safe alias map (learned only from outer-training GT, one-vs-one
leftover-token correspondences, min support 3, conditional ≥ 0.4) was built and
measured: the development cohort contains only ~3.5k GT pairs, and confident
one-vs-one correspondences are too sparse — the learned map on fold 1 training
GT has **size 0**. A fold-internal learned dictionary needs the full 7.6M-link
training corpus to have support; at development scale it is empty and cannot
be tested honestly. Verdict: **learned maps are not falsifiable at dev scale;
the concept remains untested, not rejected.**

Instead, the corruption patterns were captured with **label-free
normalization features** (de-leet + repeat-collapse fuzzy ratio, accent-strip
ratio, normalized char-3gram Jaccard, sorted specific-token Jaccard,
containment, suffix-free equality, translit variants). On the 18 rejects these
reach high similarity where raw features fail (heart1and: norm ratio 1.000;
comtrade: 0.898; vision international: 0.857).

## 3. Cross-script / conditional application ablation (Phase 3, `alias_ablation.json`)

| Arm | Mean | Folds | Precision | Rescues | Verdict |
|---|---:|---|---:|---:|---|
| control (new features, arm-C cap) | .9375 | .9436/.9376/.9315 | .9688 | 10/25 | reference |
| + norm features global | .9365 | .9345/.9451/.9299 | .9733 | 9/25 | fold-1 regression, no rescue gain |
| + norm features conditional (cross-script ∪ missing-address only) | .9378 | .9334/.9464/.9336 | .9732 | 10/25 | fold-1 −.010, not stable |

Conditional application beats global (as hypothesized) but still does not
produce an all-fold gain: the normalization features help the corrupted
positives and simultaneously help similar-looking corrupted negatives.

## 4. Missing-address specialist (Phase 4)

Covered by the same ablation family: a separate specialist was replaced by the
cheaper conditional-feature arm (`norm_cond`) and by targeted weights, both of
which include the missing-address interaction. Missing-address F0.5 across
arms: control .9019/.9042, w_targeted **.9144**, w4 .9116, norm_cond .8843.
No separate gated specialist is justified: the missing-address slice improves
most from targeted positive weighting, not from a separate model (which would
train on ~1,300 positive pairs and lose the shared-name signal).

## 5. Positive-weight sweep without density shift (Phase 5)

| Weight arm | Mean | Folds | Precision | Missing-addr | Cross-script | Rescues |
|---|---:|---|---:|---:|---:|---:|
| uniform w=1.5 | .9366 | .9377/.9397/.9324 | .9709 | .8986 | .8910 | 10/25 |
| uniform w=2.5 | .9351 | .9385/.9379/.9290 | .9691 | .8996 | .8957 | 10/25 |
| uniform w=3 (prior round) | .9376 | .9370/.9451/.9306 | .9740 | .9014 | — | 7/25 |
| **targeted w=3 cross-script ∪ missing-address** | **.9398** | .9400/.9451/.9342 | **.9731** | **.9144** | **.9112** | 10/25 |
| targeted w=4 | .9377 | .9392/.9409/.9329 | .9709 | .9116 | .9040 | 8/25 |
| targeted w=3 + cond-norm features | .9340 | .9324/.9410/.9286 | .9744 | .8984 | .9009 | 9/25 |

**Targeted positive weighting is the best development configuration found**:
.9398 mean (folds .9400/.9451/.9342), no fold regression vs its own control
(.9375), precision .9731, missing-address and cross-script slices both improve,
singleton .8593 (up from .8259). Weights are a label-free function of
(source-script pair, address-missing flag) determined inside training folds.

## 6. Rescued-pair specialist reranker (Phase 6, `specialist_reranker.json`)

Band = main-rejected rows with prob ∈ [1e-4, threshold) and agreement == 1:
286–361 rows per outer fold, 39 positives total on validation folds, all 18
rejected rescues inside. A nested LightGBM specialist with full feature stack
plus cap-rank context was trained per fold; its promotion cutoff was selected
on inner band rows under a ≥.97 precision guard. Result: cutoffs 0.7–0.75,
**promoted 0/2/0 pairs; zero rescues converted**. The band positives are not
separable from band negatives by the available features — promotion under a
precision guard is empty. The specialist design is falsified at this data
scale (39 band positives is simply too few to learn a reliable second-stage
rule); it is not a tunable-knob failure.

## 7. Public-system distillation (Phase 7)

Sid/Ayan/Akash concepts relevant to conversion were re-checked through the
recorded audits: their margins/competition features are already present; their
stage-2/expected-F/source-threshold arms match arms rejected here and in the
nested stage-2 audit; hard-negative injection matches the density-shift
failure measured in the prior round. The differentiator their local scores
suggest is a **learned Indic/corruption dictionary** trained on the full
7.6M-link training GT (Ayan fits maps on all train pairs; Sid fits per fold).
At full training scale such a map has orders of magnitude more support than
the dev-scale attempt above and targets exactly the corruption families in
Section 1. That experiment requires a full-training-scale pipeline run and
cannot be validated at the 800-S1 development scale.

## 8. Success gate

Gate: grouped mean ≥ .945, no meaningful fold regression, precision ≥ .97,
missing-address stable/improved, cross-script materially improved, singleton
stable, ≥15/25 rescues predicted, blocker recall .9802.

| Best candidate | Mean | Rescues | Gate? |
|---|---:|---:|---|
| w_targeted blend (arm-C cap) | **.9398** | 10/25 | **fails**: mean < .945, rescues < 15 |

**No candidate passes. No confirmation cohort is opened.** Blocker recall
.9802 stands available: the retrieval side has already delivered the missing
positives; the matcher's inability to confirm them is bounded by development
scale (3.5k GT pairs; 39 band positives; empty learned dictionary). The two
paths that plausibly clear .945 both require the full training corpus and are
the recommended next steps: (1) a full-training-scale fold-safe learned
corruption/alias dictionary evaluated through this same paired grouped
protocol; (2) re-testing the specialist band reranker with full-scale
dictionary features so the band positives become separable. No commit: the
promotion policy requires material, stable improvement, which was not
achieved. Artifacts in `research_runs/forensic_20260927/` (git-ignored).
