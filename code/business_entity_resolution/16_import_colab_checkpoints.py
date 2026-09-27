"""Verify and import completed Colab shard archives on the laptop."""
import argparse
import importlib
import json
from pathlib import Path

restore_module = importlib.import_module('15_colab_checkpoint')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir', type=Path, required=True,
                   help='Local directory containing transferred NNNNN_phase.tar.gz and .sha256 files')
    p.add_argument('--run-dir', type=Path, default=ROOT/'research_runs/submission_exact_shards')
    p.add_argument('--shard-index', type=int, required=True)
    a = p.parse_args()
    run = a.run_dir.resolve()
    info = json.loads((run/'shards.json').read_text())
    if not 0 <= a.shard_index < info['shard_count']:
        p.error('Invalid shard index')
    shard = f'{a.shard_index:05d}'
    for phase in ('retrieve','augment','infer'):
        if not restore_module.restore(run, a.archive_dir.resolve(), phase, shard):
            p.error(f'Missing verified {phase} archive for shard {shard}')
    completion = run/'pred'/shard/'completion_manifest.json'
    if not completion.exists():
        p.error('Imported shard has no final completion manifest')
    record = json.loads(completion.read_text())
    worker = importlib.import_module('14_run_distributed_shard')
    checks = worker.validate_shard(run, a.shard_index,
                                   info['shard_sizes'][a.shard_index], record['target_count'])
    if record['completion'] is not True or checks != record['checksums']:
        p.error('Imported shard failed completion/checksum validation')
    print(f'IMPORTED COMPLETE shard {shard}: {record.get("runtime_seconds")} seconds')


if __name__ == '__main__':
    main()
