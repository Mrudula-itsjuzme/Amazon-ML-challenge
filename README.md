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
