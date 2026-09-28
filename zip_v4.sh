#!/bin/bash
while pgrep -f smart_fast_match_v2.py > /dev/null; do
  sleep 2
done
zip safe_submission.zip output/matching_results_v4.tsv
echo "Zipped safe_submission.zip!"
