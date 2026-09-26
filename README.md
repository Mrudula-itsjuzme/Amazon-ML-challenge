Training

python 01_preprocess.py
python 02_generate_candidates.py --split train
python 03_filter_candidates.py --split train
python 04_evaluate.py




You'll get:

processed/train_*.tsv
        ↓
candidates/train_candidate_pairs.tsv
        ↓
filtered/train_final_candidate_pairs.tsv
        ↓
metrics






Test

python 02_generate_candidates.py --split test
python 03_filter_candidates.py --split test
python 05_make_submission.py



giving:

submission/matching_results.tsv
