"""Run one frozen, exact test shard with restartable phases and a sealed manifest."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_phase(command):
    start = time.monotonic()
    proc = subprocess.Popen(command, start_new_session=True)
    peak_rss = 0
    cpu = {}
    read_bytes = {}
    write_bytes = {}
    while proc.poll() is None:
        try:
            family = [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)
            rss = 0
            for child in family:
                try:
                    rss += child.memory_info().rss
                    usage = child.cpu_times()
                    cpu[child.pid] = max(cpu.get(child.pid, 0.0), usage.user + usage.system)
                    io = child.io_counters()
                    read_bytes[child.pid] = max(read_bytes.get(child.pid, 0), io.read_bytes)
                    write_bytes[child.pid] = max(write_bytes.get(child.pid, 0), io.write_bytes)
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    pass
            peak_rss = max(peak_rss, rss)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.5)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return {'seconds': round(time.monotonic() - start, 2),
            'sampled_peak_rss_bytes': peak_rss, 'sampled_cpu_seconds': round(sum(cpu.values()), 2),
            'sampled_read_bytes': sum(read_bytes.values()), 'sampled_write_bytes': sum(write_bytes.values())}


def validate_shard(run, index, expected_count, target_count):
    folder = run / 'pred' / f'{index:05d}'
    source = run / 'sources' / f'{index:05d}.tsv'
    with source.open(newline='') as handle:
        expected = [row['entity_id'] for row in csv.DictReader(handle, delimiter='\t')]
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise ValueError('Shard source count/uniqueness mismatch')
    retrieval = json.loads((run / 'aug' / f'{index:05d}' / 'test_retrieval_run.json').read_text())
    inference = json.loads((folder / 'inference_manifest.json').read_text())
    if retrieval['target_pool_size'] != target_count or retrieval['labels_used_for_retrieval'] is not False:
        raise ValueError('Incomplete or label-dependent target retrieval')
    if inference['source1_count'] != expected_count or inference['target_pool_size'] != target_count:
        raise ValueError('Inference population mismatch')
    files = [folder / 'candidate_pairs.tsv', folder / 'matching_results.tsv']
    with files[0].open(newline='') as cfile, files[1].open(newline='') as mfile:
        candidates = csv.DictReader(cfile, delimiter='\t')
        matches = csv.DictReader(mfile, delimiter='\t')
        if candidates.fieldnames != ['source1_entity_id', 'candidate_entity_ids'] or matches.fieldnames != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Bad output columns')
        count = 0
        for sid, c, m in zip(expected, candidates, matches, strict=True):
            if sid != c['source1_entity_id'] or sid != m['source1_entity_id']:
                raise ValueError('Shard output S1 ordering mismatch')
            ids = c['candidate_entity_ids'].split(',') if c['candidate_entity_ids'] else []
            selected = m['matched_entity_ids'].split(',') if m['matched_entity_ids'] else []
            if len(ids) > 100 or len(ids) != len(set(ids)) or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
                raise ValueError(f'Bad candidate/match IDs for {sid}')
            count += 1
    if count != expected_count:
        raise ValueError('Incomplete shard output')
    return {p.name: digest(p) for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--num-shards', type=int, default=174)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'research_runs/submission_exact_shards')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--target-chunk', type=int, default=50000)
    args = p.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / 'shards.json').read_text())
    index = args.shard_index
    if args.num_shards != manifest['shard_count'] or not 0 <= index < args.num_shards:
        p.error('Shard index/count differs from frozen shard partition')
    source = run / 'sources' / f'{index:05d}.tsv'
    model = run / 'frozen_blend.joblib'
    config = HERE / 'frozen_blend_policy.json'
    target_dir = ROOT / 'student_resource/dataset/test'
    target_files = [target_dir / f'test_source{i}.tsv' for i in (2, 3)]
    target_count = sum(sum(1 for _ in path.open()) - 1 for path in target_files)
    target_checks = {path.name: digest(path) for path in target_files}
    if target_count != 9969589 or digest(ROOT / 'student_resource/dataset/test/test_source1.tsv') != manifest['test_source1_sha256']:
        p.error('Test dataset differs from prepared partition')
    if not model.exists():
        p.error('Frozen model cache missing')
    import joblib
    payload = joblib.load(model)
    if payload['config_sha256'] != digest(config):
        p.error('Model cache and policy differ')
    final = run / 'pred' / f'{index:05d}' / 'completion_manifest.json'
    if final.exists():
        old = json.loads(final.read_text())
        checks = validate_shard(run, index, manifest['shard_sizes'][index], target_count)
        if old.get('completion') is not True or old.get('checksums') != checks or old.get('model_sha256') != digest(model):
            raise ValueError('Existing completed shard failed verification')
        print(f'Shard {index} already complete and verified', flush=True)
        return
    metrics = {}
    for step in ('retrieve', 'augment', 'infer'):
        metrics_path = run / 'metrics' / f'{index:05d}' / f'{step}.json'
        marker = (run / ('pred' if step == 'infer' else 'aug' if step == 'augment' else 'base') /
                  f'{index:05d}' / ('inference_manifest.json' if step == 'infer' else 'test_retrieval_run.json'))
        if metrics_path.exists() and marker.exists():
            metrics[step] = json.loads(metrics_path.read_text())
            continue
        if marker.exists():
            metrics[step] = {'seconds': None, 'reused_without_metrics': True,
                             'sampled_peak_rss_bytes': 0}
            continue
        cmd = [sys.executable, str(HERE / '12_exact_sharded_submission.py'), step,
               '--run-dir', str(run), '--shard-id', str(index),
               '--workers', str(args.workers), '--target-chunk', str(args.target_chunk)]
        metrics[step] = run_phase(cmd)
        if not marker.exists():
            raise ValueError(f'{step} ended without its phase marker')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temp_metrics = metrics_path.with_suffix('.json.partial')
        temp_metrics.write_text(json.dumps(metrics[step], indent=2))
        os.replace(temp_metrics, metrics_path)
    checks = validate_shard(run, index, manifest['shard_sizes'][index], target_count)
    commit_file = ROOT / 'frozen_git_commit.txt'
    if commit_file.exists():
        commit = commit_file.read_text().strip()
    else:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = 'unavailable'
    if {path.name: digest(path) for path in target_files} != target_checks:
        raise ValueError('Test target files changed during shard execution')
    timed = [x['seconds'] for x in metrics.values()]
    result = {'completion': True, 'shard_index': index, 'num_shards': args.num_shards,
              'source1_count': manifest['shard_sizes'][index], 'target_count': target_count,
              'source_sha256': digest(source), 'target_sha256': target_checks,
              'model_sha256': digest(model), 'config_sha256': digest(config),
              'git_commit': commit,
              'runtime_seconds': round(sum(timed), 2) if all(x is not None for x in timed) else None,
              'sampled_peak_rss_bytes': max(x['sampled_peak_rss_bytes'] for x in metrics.values()),
              'phases': metrics, 'checksums': checks}
    temp = final.with_suffix('.json.partial')
    temp.write_text(json.dumps(result, indent=2))
    os.replace(temp, final)
    print(f'COMPLETE shard {index}: {result["runtime_seconds"]} sec measured', flush=True)


if __name__ == '__main__':
    main()
