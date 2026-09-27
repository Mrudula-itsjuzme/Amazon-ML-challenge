What we did in Trial 5
	• Problem: Previous blocker produced ~300M candidate pairs with only ~1.7% pair precision.
	• Goal: High recall without candidate explosion.
	• Controlled testing
		○ Run on 5K S1 sample, seed 32.
		○ Final cap: 200 candidates/S1.
	• Indexes built
		○ Exact: name, core name, transliteration, address.
		○ Token: name + address with frequency tracking.
		○ Postal code.
		○ Address numbers.
		○ Name prefixes.
		○ Character 3-grams.
	• Major change: intersections
		○ Common tokens are no longer discarded.
		○ Rare tokens can retrieve alone.
		○ Common tokens are only useful through A ∩ B style intersections.
		○ This prevents "private" / "road" etc. from exploding candidates.
	• Additional blocking routes
		○ Postal + name
		○ Number + name
		○ Number + address token
		○ Rare address tokens
		○ Name/address intersections
		○ Prefix matching
		○ Character n-grams
	• Fuzzy matching
		○ RapidFuzz is only applied to candidates already retrieved by other routes.
		○ Never fuzzy-search the entire target dataset.
	• Multi-route evidence
		○ Every candidate records its retrieval channels.
		○ Candidates found through multiple independent routes get higher priority.
	• Final pruning
		○ If >200 candidates, rank using route strength + multiple-route evidence.
		○ Keep top 200.
	• Efficiency
		○ Character indexes are cached.
		○ Output is written in batches of 5K.
Core idea:
millions of targets → selective blocking/intersections → ≤200 candidates → expensive matching later
And next step is diagnostics: measure recall per route and candidate-count distribution before blindly tuning more stuff.



======================================================================
BLOCKER CANDIDATES — 02_generate_candidate_keys.py
======================================================================
Input: C:\Users\ksupr\Desktop\KS\Projects\S5\amazon_ml\Amazon-ML-challenge\trial2\candidates\train_candidate_pairs.tsv
Chunk size: 500,000
Processed chunk 1 | Predictions: 500,000 | TP: 8,227
Processed chunk 2 | Predictions: 988,346 | TP: 16,257

Results
----------------------------------------------------------------------
Ground-truth entities: 5,000
Ground-truth true pairs: 17,085
Predicted pairs: 988,346
TP: 16,257
FP: 972,089
FN: 828

Pair precision: 0.016449
Pair recall: 0.951536
Pair F0.5: 0.020472
Candidate recall: 0.951536
Macro entity F0.5: 0.020583
No-match entities: 262
No-match accuracy: 0.000000

Mean candidates/entity: 197.67
Median candidates/entity: 200.00
95th percentile: 200.00
99th percentile: 200.00
Max candidates/entity: 200

Entities with missed true matches: 636
Missed true pairs: 828

(datasci) C:\Users\ksupr\Desktop\KS\Projects\S5\amazon_ml\Amazon-ML-challenge\trial2>

For seed 32

Tried modifying 02 to improve recall
