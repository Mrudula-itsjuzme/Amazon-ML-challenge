#!/bin/bash
while [ ! -f output/matching_results.tsv ]; do
  sleep 5
done
# Wait for the python script to finish writing (wait for process to end)
while pgrep -f mega_fast_match.py > /dev/null; do
  sleep 5
done
zip fast_submission_final.zip output/matching_results.tsv output/candidate_pairs.tsv
echo "Done zipping!"
