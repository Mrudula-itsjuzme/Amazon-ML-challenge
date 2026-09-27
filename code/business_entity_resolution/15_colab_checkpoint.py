"""Run an exact shard, checkpointing each completed phase to mounted Drive."""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
worker = importlib.import_module('14_run_distributed_shard')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def phase_paths(phase, shard):
    kind = {'retrieve': 'base', 'augment': 'aug', 'infer': 'pred'}[phase]
    return [f'{kind}/{shard}', f'metrics/{shard}/{phase}.json']


def restore(run, remote, phase, shard):
    archive = remote / f'{shard}_{phase}.tar.gz'
    check = archive.with_suffix(archive.suffix + '.sha256')
    if not (archive.exists() and check.exists()):
        return False
    if check.read_text().strip() != sha(archive):
        raise ValueError(f'Drive checkpoint checksum mismatch: {archive}')
    with tarfile.open(archive, 'r:gz') as tar:
        allowed = set(phase_paths(phase, shard))
        for member in tar.getmembers():
            if member.name not in allowed and not any(member.name.startswith(p + '/') for p in allowed):
                raise ValueError(f'Unexpected archive member: {member.name}')
            if member.issym() or member.islnk() or '..' in Path(member.name).parts:
                raise ValueError(f'Unsafe archive member: {member.name}')
        tar.extractall(run, filter='data')
    return True


def checkpoint(run, remote, phase, shard):
    remote.mkdir(parents=True, exist_ok=True)
    archive = remote / f'{shard}_{phase}.tar.gz'
    temp = remote / f'{shard}_{phase}.tar.gz.partial'
    paths = phase_paths(phase, shard)
    with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
        for rel in paths:
            if not (run / rel).exists():
                raise ValueError(f'Missing checkpoint input: {run / rel}')
            tar.add(run / rel, arcname=rel, recursive=True)
    digest = sha(temp)
    os.replace(temp, archive)
    check = archive.with_suffix(archive.suffix + '.sha256')
    tmp_check = check.with_suffix(check.suffix + '.partial')
    tmp_check.write_text(digest + '\n')
    os.replace(tmp_check, check)
    print(f'DRIVE CHECKPOINT {phase}: {archive} {digest}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--num-shards', type=int, default=174)
    p.add_argument('--drive-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, default=ROOT/'research_runs/submission_exact_shards')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--target-chunk', type=int, default=50000)
    a = p.parse_args()
    run = a.run_dir.resolve()
    remote = a.drive_dir.resolve()
    shard = f'{a.shard_index:05d}'
    info = json.loads((run/'shards.json').read_text())
    if a.num_shards != info['shard_count'] or not 0 <= a.shard_index < a.num_shards:
        p.error('Shard assignment differs from prepared partition')
    if not str(remote).startswith('/content/drive/'):
        p.error('Drive directory must be under /content/drive/ in Colab')
    for phase in ('retrieve', 'augment', 'infer'):
        paths = phase_paths(phase, shard)
        marker = run / paths[0] / ('inference_manifest.json' if phase=='infer' else 'test_retrieval_run.json')
        metric = run / paths[1]
        if not (marker.exists() and metric.exists()):
            restore(run, remote, phase, shard)
        if not (marker.exists() and metric.exists()):
            if marker.exists():
                measure = {'seconds': None, 'reused_without_metrics': True,
                           'sampled_peak_rss_bytes': 0}
            else:
                command = [sys.executable, str(HERE/'12_exact_sharded_submission.py'), phase,
                           '--run-dir', str(run), '--shard-id', str(a.shard_index),
                           '--workers', str(a.workers), '--target-chunk', str(a.target_chunk)]
                measure = worker.run_phase(command)
            if not marker.exists():
                raise ValueError(f'{phase} returned without complete phase marker')
            metric.parent.mkdir(parents=True, exist_ok=True)
            metric.write_text(json.dumps(measure, indent=2))
        checkpoint(run, remote, phase, shard)
    subprocess.run([sys.executable, str(HERE/'14_run_distributed_shard.py'),
                    '--shard-index', str(a.shard_index), '--num-shards', str(a.num_shards),
                    '--run-dir', str(run), '--workers', str(a.workers),
                    '--target-chunk', str(a.target_chunk)], check=True)
    checkpoint(run, remote, 'infer', shard)
    complete = json.loads((run/'pred'/shard/'completion_manifest.json').read_text())
    print(json.dumps({'shard_index': a.shard_index, 'runtime_seconds': complete['runtime_seconds'],
                      'peak_rss_gib': round(complete['sampled_peak_rss_bytes']/2**30, 3),
                      'phases': complete['phases'], 'drive_dir': str(remote)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
