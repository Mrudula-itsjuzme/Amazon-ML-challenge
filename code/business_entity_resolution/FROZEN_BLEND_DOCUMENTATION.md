# Business Entity Resolution methodology

## Methodology

The pipeline normalizes Unicode business names and addresses while retaining
their native scripts. It matches each Source 1 record to zero or more Source 2
and Source 3 records. Country is treated as an open-set string feature, so
France and other unseen values remain eligible.

## Candidate generation

The forward blocker scans the complete target pool without ground-truth
access. It uses native-name character and word similarity, address character
similarity, numeric overlap, rare tokens, transliterated-name similarity, and
core-name similarity. It retains the top 50 candidates per route, unions route
edges, and selects at most 100 per Source 1 using route agreement and
reciprocal rank. No true link is injected or protected using labels. The
submitted `candidate_pairs.tsv` is the exact K100 set scored by the matcher.
The full test set is processed in disjoint Source 1 shards; every shard scans
all Source 2 and Source 3 targets. The merge checks all Source 1 records appear
exactly once.

## Matcher

Two LightGBM classifiers use pairwise Unicode-safe name, transliteration,
address, numeric, route-rank, route-score, relative-ambiguity, and global
target-token IDF features. The second model adds numeric and source-context
interactions. Both train on the same 800 Source 1 development groups and
their genuinely retrieved K100 candidates. Probabilities are averaged 50/50;
matches with blend probability at least 0.55 are selected. The threshold was
chosen on grouped development data. Singleton output is an empty ID list.

## Validation and limitations

Development grouped by `source1_entity_id`. A predeclared 50/50 blend control
observed macro entity F0.5 **0.9516**, pair precision **0.9819**, and pair
recall **0.8980** on one separate, previously untouched 500-S1 confirmation
cohort. The frozen selected source-floor policy scored 0.9496 on that cohort;
the control was not promoted on the basis of its confirmation score. Choosing
the control for submission after that observation is a deployment choice, not
a new independent validation. No test or leaderboard F0.5 is known.

The training target pool contained 10,320,219 records; the test target pool
contains 9,969,589. The development K100 blocker recalled about 0.9714 of
true links. Blocker recall is a retrieval measure, not matcher accuracy.
Training ground truth is used only for matcher fitting and evaluation, not for
candidate generation. No external business lookup, geocoding, or enrichment
service is used. Models and dependencies are within the challenge's stated
license and size limits.
