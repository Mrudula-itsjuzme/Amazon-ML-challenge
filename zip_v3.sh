#!/bin/bash
while pgrep -f smart_fast_match.py > /dev/null; do
  sleep 2
done
zip smart_submission.zip output/matching_results_v3.tsv output/candidate_pairs_v3.tsv
echo "Zipped smart_submission.zip!"
