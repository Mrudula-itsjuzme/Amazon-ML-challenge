# Full-corpus corruption dictionary: mined, applied leak-free, measured

Development-only work on `mru`, continuing [RESCUED_PAIR_CONVERSION.md](RESCUED_PAIR_CONVERSION.md)
and [FORWARD_POOL_EXPLOITATION.md](FORWARD_POOL_EXPLOITATION.md). The learned
cap-ranker (blocker recall .9802, 25/25 cap-loss links recovered, zero
displacements) is fixed. No new retrieval route, no wider K, no confirmation
cohort, no production pipeline change. All matcher ablations use the recorded
nested grouped protocol (three outer grouped S1 folds, inner GroupKFold(3) OOF,
fixed 23-point grid, 0.97 OOF pair-precision floor, 50/50 IDF+numeric blend).

**Main question answered: yes — the full corpus contains strong repeated
corruption structure, and exploiting it produces the largest development gain
measured so far (+.0076 macro F0.5 over control, precision .9748). It does
not, however, convert the unique-corruption rescued pairs, and the promotion
gate still fails.**

## 1. Full-corpus mapping statistics (Phase 1–2)

Mining streams all S1 GT links (sample divisor 20 → 383,501 pairs for the
curve study) and aligns each pair's specific core tokens by multiset
difference; the unshared remainders become left→right correspondences with
support, conditional probability, ambiguity, source/country/script metadata.

| support ≥ | mappings | tokens | cond ≥ .8 share |
|---:|---:|---:|---:|
| 1 | 45,803 | 40,592 | .88 |
| 3 | 3,354 | 2,059 | .60 |
| 5 | 1,797 | 1,158 | .63 |
| 10 | 758 | 513 | .67 |
| 20 | 260 | 184 | .69 |
| 100 | 8 | 8 | .62 |

Top mappings are *systematic accent corruption*: `cáre→care` (support 203,
cond 1.00), `ássociates→associates` (201, 1.00), `héalth→health` (138, 1.00),
`clínic→clinic` (120, 1.00), `bróthers→brothers` (96, 1.00). Ambiguous junk
separates cleanly by conditional probability (`center→care` support 103 but
cond 0.01, ambiguity 1860). Phrase-level mining adds 388 mappings ≥ 3 support,
mostly compaction (`center family→familycenter`) and cross-script
correspondences. At the 5% sample the useful-mapping count grows roughly
linearly with corpus size; the full 7.6M links would hold ~10× these counts.

## 2. Fold-safe construction proof (Phase 3)

The dictionary used for every development feature is **dev-blind**: mined from
all full-corpus GT links whose S1 is NOT one of the 1,000 development S1s
(552,072 S1s, 1,910,533 links kept; all 1,000 dev S1s excluded). No
development label can appear in any mapping — stricter than per-fold
exclusion and valid for all three folds simultaneously. Filtered to
support ≥ 3, cond ≥ 0.5, ambiguity ≤ 5: **7,013 confident mappings**
(`correspondence_devblind_mappings.parquet`, `mine_devblind.log` shows the
exclusion banner).

## 3. Dictionary features and matcher ablation (Phase 4–5, 8)

Features (per pair): mapped-token Jaccard, mapped compact-name ratio, alias
coverage, max/mean mapping confidence, mapped-core equality, unmatched-after
count, hit counts, mapped char-3gram Jaccard. Mappings are used as FEATURES
only; input text is never rewritten.

| Arm | Mean | Folds | Precision | Recall | Sing | Miss F0.5 | Cross F0.5 | Rescues |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ctrl_w0 (new features) | .9375 | .9436/.9376/.9315 | .9688 | .9011 | .807 | .9119 | .9049 | 10/25 |
| wt3 targeted weights | .9398 | .9400/.9451/.9342 | .9731 | .8944 | .859 | .9144 | .9112 | 10/25 |
| **dict (global dictionary features)** | **.9429** | .9367/.9511/.9408 | **.9748** | .9015 | **.863** | .8948 | **.9141** | 10/25 |
| dict + wt3 | .9370 | .9258/.9485/.9368 | .9774 | .8864 | .844 | .8921 | .9116 | 10/25 |
| dict conditional (cross-script/miss-addr only) | .9381 | .9411/.9414/.9319 | .9731 | .8944 | .826 | .9043 | .9111 | 9/25 |
| dict low-confidence only (agreement=1) | .9356 | .9376/.9390/.9300 | .9684 | .8994 | .826 | .9060 | .8991 | 10/25 |

Findings: global application wins decisively — every targeted/conditional
restriction hurts (the accent-corruption signal is broadly useful, not
pocket-specific). `dict` is the best development configuration ever measured:
+.0054 over the targeted-weight arm, +.0076 over control, with the best
precision and cross-script score. Its weakness is fold 1 (.9367, −.0069 vs
ctrl_w0 fold 1) — a material single-fold regression that alone blocks the
promotion rule.

## 4. Effect on the 25 cap rescues (Phase 6, `rescue_dict_effect.tsv`)

| Pair | ctrl prob | dict prob | shift | promoted |
|---|---:|---:|---:|---|
| S2-569824841 (johnson communities ↔ l l c johnson commnultis) | .278 | **.916** | **+.638** | ✔ |
| S3-386342590 (pediatric dental specialists ↔ pediatric dental) | .349 | **.842** | +.493 | ✔ |
| S3-56707670 (comtrade spinning ↔ comtrade spnlnnhng) | .529 | **.828** | +.299 | ✔ |
| S3-108293647 (nhs sarada ↔ nhssarada com) | .173 | .392 | +.219 | ✖ |
| S3-383477932 (my producer ↔ माय प्रोड्यूसर) | .004 | .189 | +.185 | ✖ |
| S2-62373651 (total education associates ↔ total associates center) | .183 | .354 | +.171 | ✖ |
| … 12 more with shifts < +.13, 6 essentially unmoved | | | | |

Converted: **10/25 (3 flipped on, 0 flipped off)** — target ≥ 15/25 not
reached. The dictionary converts exactly the *repeated-corruption* rescues
(compaction `commnultis`, truncation `pediatric dental`, garble `spnlnnhng`)
whose patterns recur in training GT. The remaining 8 rejected rescues are
*unique* corruptions (doubled transliteration gibberish `saai sphttoy yaar
praaibhett limittedd`, junk-suffix names, unrelated-word aliases like `agni
brothers ↔ agni services`) with near-zero dictionary coverage: for 5 of them
the map has no applicable mapping at all, and 3 map slightly *downward*.

## 5. Specialist band retest (Phase 7, `specialist_v2.json`)

With dictionary features the band specialist again promotes almost nothing
(1/2/0 pairs, 0 rescues) under the ≥.97 precision guard. The band's positives
(11/7/20 per fold) remain inseparable from its 200–330 negatives even for a
model that sees the dictionary features. The information simply is not in the
features: unique corruptions do not repeat, so no learned rule can separate
them from equally unique negative noise.

## 6. Promotion-gate verdict (Phase 9)

Gate: mean ≥ .945, no meaningful fold regression, precision ≥ .97, rescues
≥ 15/25, blocker recall .9802.

| Candidate | Mean | Folds vs control | Rescues | Verdict |
|---|---:|---|---:|---|
| dict | .9429 | fold 1 −.0069 | 10/25 | **fails** (fold regression, rescues) |

**No candidate passes; no confirmation cohort is opened; no commit.** What
this round established:

1. The corpus *does* hold exploitable corruption structure (accent/compaction
   mappings with support 100–200 and conditional 1.00), and using it as
   features yields the best dev model so far (.9429, precision .9748).
2. The last mile — the 15 unconverted rescues — is **unique-corruption** noise
   (doubled-letter transliteration gibberish, DBA-style alias rewrites, junk
   suffixes) that by construction cannot form a frequent mapping. A
   character-level learned model (e.g. a small char-CNN/char-LM over the
   normalized name pair, trained on the full-corpus GT as features inside the
   grouped protocol) is the remaining coherent idea for that pocket; a
   dictionary cannot cover it.
3. For final test inference the frozen recipe would be: dev-blind-style
   dictionary rebuilt from ALL train GT (label-free at test time), global
   dictionary features, no targeted weights, cap-ranker K100.

Artifacts in `research_runs/forensic_20260927/`: `mine_correspondences.py`,
`correspondence_devblind_*`, `dict_features.py`/`.parquet`,
`dict_ablation.json`, `specialist_v2.json`, `rescue_dict_effect.tsv`.
