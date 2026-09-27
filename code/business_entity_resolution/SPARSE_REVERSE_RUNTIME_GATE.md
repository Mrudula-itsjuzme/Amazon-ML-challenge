# Sparse reverse retrieval: bounded rescue and full-pool runtime gate

This is development-only research on `mru`. The matcher, forward blocker,
candidate cap, and frozen confirmation policy were not changed. No confirmation
cohort was opened. The true S1 was used to **select** the 56 missed-link
diagnostic queries and to score their ranks afterward; no label entered any
TF-IDF fit or search. Every diagnostic query searched the complete
2,206,821-S1 training corpus within its matching open-set country string.

## What is measured

| Search variant on 56 development raw misses | True S1 top 3 | Top 5 | Top 8 | Top 10 | Full target scan? |
|---|---:|---:|---:|---:|---|
| Native-name character trigrams + address words, full query | 34 | 35 | **36** | 37 | No |
| Same two views, `max_df=0.05` country-specific IDF | — | — | **34** | 34 | No |
| Keep six name and four address query terms | — | — | **25** | 26 | No |

The complete pair IDs, country, score, and rank for each of the 56 links are
in `research_runs/forensic_20260927/sparse_reverse_bounded.json` and
`sparse_reverse_bounded_maxdf05.json`. These counts are **bounded diagnostic
rank coverage**, not newly rescued full-pool blocker links. A target-centric
route would have to retrieve and reverse edges for **all** targets before
candidate union and capping can be evaluated.

## Runtime and RAM

The target samples for throughput were label-free: the first 1,000 S2 and
1,000 S3 records from each training country, 4,000 total. The index still
contained all same-country S1s. The `sparse-dot-topn` kernel (Apache-2.0),
installed only in `/tmp`, kept eight results per target with eight CPU
threads. Representative top-8 scores matched SciPy multiplication; one US
query chose different IDs at an equal-score cutoff tie.

| Representation | India, 883,188 S1 | US, 1,323,633 S1 | Measured peak RSS | Extrapolated search time for all 10,320,219 train targets |
|---|---:|---:|---:|---:|
| Full TF-IDF terms, SciPy sparse multiplication | .266 s/target | .323 s/target | 1.77 GB | ~35.9 days sequential |
| Full TF-IDF terms, bounded top-N kernel | .00990 s/target | .01146 s/target | 2.34 GB | ~31.1 hours with eight threads |
| Drop terms in >5% of same-country S1, bounded top-N | **.00447 s/target** | **.00892 s/target** | **2.09 GB** | **~20.46 hours with eight threads** |

The extrapolation multiplies the measured per-country rate by 4,133,346 India
and 6,186,873 US targets. It excludes target normalization, compact-name and
transliterated-name views, output serialization, candidate union, and K100
capping. It is a planning estimate from 4,000 sampled targets, not an
executed 10.32M-target scan. RAM is not the limiting factor; repeated sparse
posting work is. The host exposes no usable NVIDIA GPU driver.

The requested full scan was **not completed** because this two-view search
alone requires about 20.5 hours at measured throughput; the requested five
views would take longer. Running the 56 label-selected queries through a
complete S1 index does not satisfy the full-pool label-free retrieval rule.
No reverse-only recall, union recall, candidate-growth distribution, cap
improvement, rescue taxonomy, or matcher effect is claimed.

## Valid forward baseline and missing full-pool comparisons

These forward numbers are from the 800-S1 grouped development cohort with
2,828 true links and the complete 10,320,219-target pool. The raw union has
56 misses. K100 retains 2,747 true links and loses 25 more after raw union.

| Retrieval | Raw true links | Raw recall | K40 | K60 | K80 | K100 | K150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing forward routes | 2,772 | .9802 | .8893 | .9558 | .9678 | .9714 | .9760 |
| Sparse reverse alone | **unmeasured** | **unmeasured** | — | — | — | — | — |
| Forward ∪ sparse reverse | **unmeasured** | **unmeasured** | **unmeasured** | **unmeasured** | **unmeasured** | **unmeasured** | **unmeasured** |

At k=8, a full scan could emit up to **82,561,752** reverse target→S1
edges before deduplication. The actual per-S1 growth and how many survive
K100 are unknown. Neither a cap ranker nor reverse matcher features were
trained, because the route has not passed the full-pool retrieval gate.

## Public algorithm comparison

| Approach | Sparse representation and weights | Reverse k, partition | Union and competition | Relevance here |
|---|---|---|---|---|
| [Ayan blocking](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/src/ber/blocking.py), [stage](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/modal_app/stage_block.py) | Normalized name `char_wb` 3-grams, compact `char` 3-grams, address words; combined name/address TF-IDF; high-document-frequency terms removed | Target→S1 combined view top 8, partitioned by each country string | Forward views plus reverse union, per-view ranks and target competition | The `max_df=0.05` bounded variant was profiled, but only two views and no complete scan |
| [Sid retrieval](https://github.com/Sid-techweb/AmazonML-New/blob/main/business_entity_resolution/src/retrieval.py), [candidates](https://github.com/Sid-techweb/AmazonML-New/blob/main/business_entity_resolution/src/candidates.py) | Compact name char trigrams and address word TF-IDF; GPU CountSketch approximate top 50 followed by exact sparse rerank | Target→S1 top 10, open-set country partition | Best-owner target decision and candidate margin features | No usable GPU here; prior FAISS/SVD reverse had a different representation and rescued zero raw links |
| This bounded test | Raw-name char trigrams plus address words, equal 0.5/0.5 score; no learned transliteration map | Target→all same-country S1; tested top 3/5/8/10 only for diagnostic queries | No full union or cap yet | 36/56 potential top-8 raw misses, 34/56 with document-frequency cutoff |

The public code provides ideas, not transferable score estimates. Ayan's
reported validation includes normalization maps trained with validation-fold
links; Sid's local threshold sweep and different training scale also preclude
an apples-to-apples comparison with the frozen `mru` confirmation result.

## Decision

**No retrieval route is promoted.** The bounded result is strong enough to
justify a future faster full scan, but not to count as blocker improvement.
The current selected matcher and 0.9516 predeclared confirmation reference
remain unchanged. Another confirmation is not justified. No commit is made.
The next implementation must first reduce posting work while preserving the
36/56 bounded top-8 signal and demonstrate an acceptable full-target runtime;
then it can scan all targets without labels, union edges, compare k=3/5/8/10
and K40–150, and proceed to matcher ablations only if K100 improves.

## 2026-09-27 runtime-gate follow-up

The requested full-truth country check was repeated on all **7,638,365**
training links: **zero** normalized country-string mismatches. Country
partitioning therefore remains a generic open-set equality rule, with no
country names embedded in retrieval logic. The existing 20.46-hour profile
**already** partitions by country and **already** uses one combined sparse
name/address matrix. Neither change supplies an additional speedup over that
measurement. The existing two-view query is weighted 0.5/0.5; a `char_wb`
name variant and other weights have not been shown to pass the runtime gate.

An inverted-postings prototype was tested against all same-country S1 records.
It selected the 3, 5, or 8 rarest TF-IDF terms from **each** of native-name
trigrams and address words, unioned their postings, and rescored those S1s
with the same combined cosine score. Selection and scoring were label-free.
The 56 historical misses were consulted only after ranking to measure bounded
diagnostic top-8 coverage. Throughput used 200 label-free targets per country;
the full S1 country index was used in every case.

| Terms per view | India diagnostic | US diagnostic | Total / 56 | India seconds/target | US seconds/target | Projected 10.32M search |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 18/38 | 13/18 | **31/56** | .00545 | .00695 | **18.2 h** |
| 5 | 20/38 | 12/18 | **32/56** | .01743 | .02042 | **55.1 h** |
| 8 | 20/38 | 13/18 | **33/56** | .03112 | .03329 | **93.0 h** |

These projections include per-query postings union and exact rescore but
exclude full-target reading, serialization, and forward union. In this
implementation, wider postings increase candidate work faster than they
recover diagnostic links. The fastest variant misses the required 32/56
minimum and still projects to 18.2 hours, triple the six-hour gate. This
specific inverted approach is rejected. No full-target scan, reverse blocker
claim, matcher experiment, or new confirmation follows from this result.

The prototype and per-query diagnostic outcomes are in
`research_runs/forensic_20260927/sparse_reverse_inverted_gate.py` and its
JSON output. It is a bounded research instrument, not an inference route.

### Phase and thread profile of the existing sparse kernel

One full same-country S1 index was fitted for each country. Each timing below
used 2,000 label-free target queries per country (1,000 S2 and 1,000 S3),
with the same matrices reused across thread counts. The per-target times
measure sparse top-8 multiplication and sorting; output serialization is
excluded. The host reports 8 physical cores and 12 available logical CPUs.

| Threads | India ms/target | US ms/target | Projected full search | Effective cores, India / US |
|---:|---:|---:|---:|---:|
| 1 | 7.94 | 19.71 | 43.0 h | 1.00 / 1.00 |
| 4 | 4.42 | 10.05 | 22.4 h | 3.85 / 3.92 |
| 8 | 4.27 | 8.70 | 19.9 h | 7.61 / 7.36 |
| 12 | 4.29 | 8.55 | 19.6 h | 8.78 / 8.39 |

Index preparation took 16.1 s name vocabulary/TF-IDF plus 14.4 s address
vocabulary/TF-IDF for India, and 22.4 s plus 12.7 s for US. The library
combines vocabulary construction with CSR creation, so those cannot be
separated accurately without changing its implementation. Transforming the
2,000 target rows took 0.08 s in India and 0.06 s in US. CSR concatenation,
sorting, and transpose took 1.1 s and 2.6 s respectively. Iterating the
in-memory top-8 output took about 0.001 s per country. Peak RSS was 2.32 GB.
No disk serialization was part of the benchmark, so its cost remains
unmeasured; it would increase the full-scan time. Native names were passed
directly to the vectorizer, whose lowercase and Unicode accent normalization
is included in the fit/transform timings. No separate normalization pass was
performed. The source breakdown had 1,000 S2 and 1,000 S3 queries per
country; the timed kernel was pooled within country and was not measured
separately by source.

The kernel uses most available physical CPU capacity at eight threads. Twelve
threads improve the whole-scan projection by only about 0.2 h and remain far
above the six-hour gate. The CPU multiplication, especially the US partition,
is the measured bottleneck. A new posting algorithm or substantially different
index is required; neither existing combined-view retrieval nor thread tuning
solves it. The measured gate fails, so no checkpointed production scanner or
10.32M-target run was launched. The JSON phase profile is at
`research_runs/forensic_20260927/sparse_reverse_thread_profile.json`.
