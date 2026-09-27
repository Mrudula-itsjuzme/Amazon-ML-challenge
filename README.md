Training
<br><br>
python 01_preprocess.py <br>
python 02_generate_candidates.py --split train <br>
python 03_filter_candidates.py --split train <br>
python 04_evaluate.py <br>




You'll get:<br>

processed/train_*.tsv <br>
        ↓
candidates/train_candidate_pairs.tsv <br>
        ↓
filtered/train_final_candidate_pairs.tsv <br>
        ↓
metrics <br>






Test
<br><br>
python 02_generate_candidates.py --split test <br>
python 03_filter_candidates.py --split test <br>
python 05_make_submission.py <br>


<br><br>
giving:<br>

submission/matching_results.tsv<br>


02 — Candidate Blocker / Retrieval

Purpose:
Take each S1 entity and reduce ~10.3M possible S2+S3 targets to at most 200 plausible candidates, while losing as few real matches as possible.

What we used:

Deterministic/string-based blocking — no ML model
Exact indexes
Token inverted indexes
Token intersections
Prefix blocking
Character trigram indexes
Numeric evidence
Postal/name combinations
Address/name intersections
Bounded fuzzy retrieval using RapidFuzz
Retrieval-route provenance


Purpose:
Take those ~200 candidates per S1 entity and answer:

“Is this particular S1–target pair actually the same entity?”

This is where the ML model comes in.

Model

We are using:

LightGBM binary classifier
Main feature families

Name similarity

name_ratio
name_wratio
name_token_set
name_token_sort
name_token_jaccard
name length ratio
shared token counts

Core-name similarity

core_name_ratio
token set/sort
etc.

Transliteration similarity

translit_ratio
transliterated token similarities

Address similarity

address_ratio
address_wratio
address_token_set
address_token_sort
address token Jaccard
containment
shared token count

Numeric evidence

number overlap/Jaccard
shared numbers
number presence

Secondary fields

postal code agreement
country agreement
missingness indicators

Structural features

name length difference/ratio
address length difference
token-count differences

Cross-feature interactions
For example:

name_address_mean
name_address_min
name_address_product

So the model can distinguish:

“good name AND good address”

from

“good name BUT terrible address.”

Retrieval provenance is also fed into 03
