#!/usr/bin/env bash
# Run on Linux, a Colab terminal, or a cloud VM from the repository root.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$python_bin" -m venv .venv
  . .venv/bin/activate
  python_bin=python
  "$python_bin" -m pip install -r code/business_entity_resolution/requirements.txt
fi
test -f student_resource/dataset/test/test_source1.tsv
test -f student_resource/dataset/test/test_source2.tsv
test -f student_resource/dataset/test/test_source3.tsv
test -f research_runs/global_target_token_df/name_df.parquet
test -f research_runs/global_target_token_df/address_df.parquet
test -f code/business_entity_resolution/frozen_blend_policy.json
run_dir=research_runs/submission_exact_shards
if [[ ! -f "$run_dir/shards.json" ]]; then
  "$python_bin" code/business_entity_resolution/12_exact_sharded_submission.py prepare \
    --run-dir "$run_dir" --shard-count 174
fi
if [[ ! -f "$run_dir/frozen_blend.joblib" ]]; then
  "$python_bin" code/business_entity_resolution/08_frozen_blend_inference.py \
    --train-only --model-cache "$run_dir/frozen_blend.joblib"
fi
"$python_bin" - <<'PY'
import hashlib, json
from pathlib import Path
import joblib
root=Path.cwd()
config=root/'code/business_entity_resolution/frozen_blend_policy.json'
cache=root/'research_runs/submission_exact_shards/frozen_blend.joblib'
policy=json.loads(config.read_text())
for filename, expected in policy['code_hashes'].items():
    path=config.parent/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, path
for filename, expected in policy['df_hashes'].items():
    path=root/policy['global_df_dir']/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, path
assert joblib.load(cache)['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
shards=json.loads((root/'research_runs/submission_exact_shards/shards.json').read_text())
assert shards['shard_count']==174 and shards['source_count']==1732544
print('Bootstrap PASS: frozen policy, data statistics, model cache, and 174 shards')
PY
