# Exact K100 blend submission runbook

This path keeps the seven forward routes, K100 cap ordering, IDF and numeric
features, model seed, 50/50 blend, and development threshold from the
predeclared control. It never uses test labels. The 0.9516 macro F0.5 was one
training confirmation observation; no test score is known.

## Inputs and prepared state

- `student_resource/dataset/{train,test}/*_source*.tsv`
- `student_resource/dataset/train/train_ground_truth.tsv`
- `research_runs/global_target_token_df/{manifest.json,name_df.parquet,address_df.parquet}`
- `research_runs/ber_fullpool_1000_offset300_aug1/` frozen development artifacts
- `code/business_entity_resolution/frozen_blend_policy.json`

`research_runs/submission_exact_shards/shards.json` already records 174 source
shards covering **1,732,544** test S1 rows, 9,620–10,191 rows per shard.
The S1 file SHA-256 is checked again at merge. Recreate the shards if that
source file changes:

```bash
python code/business_entity_resolution/12_exact_sharded_submission.py prepare \
  --run-dir research_runs/submission_exact_shards --shard-count 174
```

Train and cache the two frozen models once:

```bash
python code/business_entity_resolution/08_frozen_blend_inference.py \
  --train-only --model-cache research_runs/submission_exact_shards/frozen_blend.joblib
```

## Distributed execution

For the current Colab arrangement, upload the two laptop files
`research_runs/colab_frozen_worker_input.tar.zst` and its `.sha256` sidecar to
`MyDrive/amazon_exact_transfer/`, then run
`exact_frozen_shard_colab.ipynb` in a High-RAM CPU runtime. The notebook first
benchmarks production shard 0 using the complete target pool. The bundle
contains only the provided test dataset, exact pipeline code, frozen model,
unlabeled target-token statistics, prepared shard definitions, and commit ID.
It contains no external business data or smoke submission file. The extracted
bundle passed `bootstrap_distributed.sh` on the laptop.

Colab's `15_colab_checkpoint.py` writes verified phase archives and SHA-256
sidecars to Drive. If the runtime disconnects, rerun the notebook: completed
phases restore from Drive. Incomplete phases rerun. After a completed shard,
download its three phase archives and sidecars, then import on the laptop:

```bash
python code/business_entity_resolution/16_import_colab_checkpoints.py \
  --archive-dir /path/to/downloaded/completed_shards --shard-index 0
```

Repeat for distinct shard indices only after the real shard 0 runtime and RAM
have been measured. The notebook prints CPU, I/O, target throughput, and a
projection for 1, 2, 4, 8, 16, and 32 sessions against 11:59 PM IST on
the day it runs. Adjust the deadline date if the portal shows another day.
The projection includes a planning allowance for transfer and
merge; it is not a guarantee of Colab session availability.

Copy the provided challenge dataset, this repository code, frozen token-DF
artifacts, `shards.json`, all 174 `sources/*.tsv`, and the frozen model cache
to each worker under the same relative paths. These are the only data inputs;
do not upload external business data. On each Linux, Colab, or cloud VM worker:

```bash
INSTALL_DEPS=1 bash code/business_entity_resolution/bootstrap_distributed.sh
python code/business_entity_resolution/14_run_distributed_shard.py \
  --shard-index 0 --num-shards 174 \
  --run-dir research_runs/submission_exact_shards --workers 4
```

Assign each shard index to exactly one worker at a time. A stopped worker can
rerun the same command: completed retrieval, augmentation, and inference phases
are skipped; incomplete phases are recomputed. Shards do not read another
shard's outputs. Each reads the complete test S2/S3 files and applies the
unchanged frozen retrieval and matcher. A completed worker returns its
`base/NNNNN/`, `aug/NNNNN/`, and `pred/NNNNN/` directories. Transfer these
directories using any file transfer available to the operator, preserving
relative paths and the `completion_manifest.json`; verify its two TSV SHA-256
checksums after transfer. The manifest is created last, after validation.
No partial or smoke TSV is a submission file.

For transfer, archive only a completed shard, using its five-digit index:

```bash
SHARD=00000
tar -C research_runs/submission_exact_shards -czf "shard_${SHARD}.tar.gz" \
  "base/${SHARD}" "aug/${SHARD}" "pred/${SHARD}" "metrics/${SHARD}"
sha256sum "shard_${SHARD}.tar.gz" > "shard_${SHARD}.tar.gz.sha256"
# On the merge machine, after transferring both files:
sha256sum -c "shard_${SHARD}.tar.gz.sha256"
tar -C research_runs/submission_exact_shards -xzf "shard_${SHARD}.tar.gz"
```

Parquet phase files remain compressed internally; gzip also compresses shard
TSVs. The merge rechecks each output file against its completion manifest.

The wrapper records elapsed time, sampled process-tree peak resident memory,
CPU time, and disk I/O per phase. It also records the frozen model and policy
hashes, dataset hashes, git commit, source count, and output checksums.
`sampled_peak_rss_bytes` is an observation sampled at 0.5-second intervals,
so a very brief peak can be missed. Full per-shard timings should determine
worker capacity; the 10k-target probe below is only a lower bound.

## Exact split checks completed

- Six test S1 records versus two disjoint three-record partitions, over the
  same 200-target fixture: forward K100 pairs and raw route audit matched
  exactly, as did augmented K100 pairs and audit.
- Five S1 records from the existing full-target smoke: two disjoint source
  partitions produced byte-identical candidate and match TSVs after restoring
  original S1 order. The sealed merge reproduced the unsplit TSVs byte for
  byte (500 candidates, 9 matches).
- The frozen candidate scoring is per S1; each shard preserves source order,
  and merge restores the original test S1 order. No shard reads another's data.

For each shard ID from `0` through `173`, the wrapper runs the following
steps. They can also be invoked individually for diagnosis. Each retrieval
step scans **all 9,969,589 test targets**.

```bash
SHARD=0
python code/business_entity_resolution/12_exact_sharded_submission.py retrieve \
  --run-dir research_runs/submission_exact_shards --shard-id "$SHARD" \
  --workers 4 --target-chunk 50000
python code/business_entity_resolution/12_exact_sharded_submission.py augment \
  --run-dir research_runs/submission_exact_shards --shard-id "$SHARD" \
  --workers 4 --target-chunk 50000
python code/business_entity_resolution/12_exact_sharded_submission.py infer \
  --run-dir research_runs/submission_exact_shards --shard-id "$SHARD"
```

Only when every shard is complete, merge and validate. The merge refuses
missing/duplicate S1s, candidate lists over K100, and matches outside the
actual inference candidates. It never publishes partial TSVs under final
names.

```bash
python code/business_entity_resolution/12_exact_sharded_submission.py merge \
  --run-dir research_runs/submission_exact_shards --output-dir output
python code/business_entity_resolution/09_validate_submission_local.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
python code/business_entity_resolution/13_package_submission.py \
  --output-dir output --zip team_name_submission.zip
```

The organizer's `utils/validate_submission.py` is not present in this checkout;
obtain it from the challenge materials and require its PASS before packaging.
The local validator is an additional gate and does not substitute for it.
Upload only `output/matching_results.tsv` to the leaderboard and
include both TSVs in the final package. Never upload a shard or smoke file.

## Runtime and resource gate

On this 15 GiB host, a five-S1 full-target smoke required multiple full
target passes for base and augmented retrieval. Even an unrealistically
constant 14 minutes per shard would require over 40 hours for 174 serial
shards; actual 10k-S1 shard scoring is much more expensive. The host currently
has around 2 GiB available and substantial swap use from other workloads.
An additional bounded probe using 9,997 S1 records and only **10,000** of
9,969,589 targets took **124.4 seconds** for base retrieval and peaked at
**4,379,412 KiB RSS**. Thus the 174 base shards alone require over six hours
serially even under the impossible assumption that scanning the remaining
9.96M targets per shard costs nothing. Augmentation and model inference add
further work. This is a runtime and available-memory bottleneck, not a code
formatting issue.
The exact route is therefore prepared for parallel or larger-memory execution
but **has not produced full-test outputs here**. Do not claim submission
readiness until the merge, validator, and package gates pass.

No representative full-target production shard has completed on this host.
The bounded 124.4-second, 10k-target measurement yields only these impossible
best-case floors for base retrieval: 4 workers >=1.50 hours, 8 workers >=0.75
hours, 16 workers >=0.38 hours, and 32 workers >=0.19 hours. They assume the
remaining 9.96 million targets, augmentation, inference, transfer, and merge
take zero time. Once a full shard completes on adequate RAM, estimate with
`ceil(174/workers) * measured_shard_seconds` plus transfer/merge time. The
minimum worker count before the deadline cannot be determined until the
full-shard runtime and exact deadline are known.

The copied package layout was exercised with a five-S1 full-target retrieval
fixture and the cached frozen model. Its `matching_results.tsv` and
`candidate_pairs.tsv` were byte-for-byte identical to the same fixture run
from the working tree. This verifies packaging and cached inference only; the
fixture is not a full submission.

The separate SQLite FTS preblock was tested against the 800-S1 grouped
development population and rejected: its fastest rare-term variant retrieved
only 1,056/2,828 true links at K100 (0.3734), compared with 0.9714 for the
retained forward blocker. No FTS edges enter this submission path.
