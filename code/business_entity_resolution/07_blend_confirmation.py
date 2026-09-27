"""Once-only full-pool confirmation of a frozen K100 IDF/numeric blend."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load_module('entity_validation', '03_validate.py')
match = validation.match
numeric_module = load_module('entity_numeric_context', '06_numeric_context.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_retrieval(folder, expected_sources):
    run = json.loads((folder / 'train_retrieval_run.json').read_text())
    if (run['target_pool_size'] != 10_320_219 or
            run['source_count'] != expected_sources or
            run['labels_used_for_retrieval'] is not False):
        raise ValueError(f'Invalid label-free full-pool retrieval: {folder}')
    return run


def pair_features(pairs, source, target, df_dir):
    base = match.build_features(pairs, source, target).drop(
        columns=['name_native_idf', 'address_rare', 'number_weighted'])
    idf = match.global_target_idf_features(pairs, source, target, df_dir)
    numeric = numeric_module.numeric_context_features(pairs, base, source, target)
    return pd.concat([base, idf], axis=1), numeric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-config', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config_path = args.frozen_config.resolve()
    config = json.loads(config_path.read_text())
    dev_dir = ROOT / config['development_dir']
    confirm_base = ROOT / config['confirmation_base_dir']
    confirm_dir = ROOT / config['confirmation_dir']
    df_dir = ROOT / config['global_df_dir']
    data_dir = ROOT / config['data_dir']
    output = confirm_dir / 'frozen_blend_confirmation.json'
    marker = confirm_dir / 'frozen_blend_confirmation.started.json'
    if output.exists() or marker.exists():
        parser.error('This confirmation has already been opened; refusing a repeat')
    if config['policy'] != 'idf_numeric_half_idf_top_ge_05' or config['cap'] != 100:
        raise ValueError('Frozen policy changed')
    for filename, key in config['code_hashes'].items():
        if sha256(HERE / filename) != key:
            raise ValueError(f'Frozen code changed: {filename}')
    for filename, key in config['df_hashes'].items():
        if sha256(df_dir / filename) != key:
            raise ValueError(f'Global target DF changed: {filename}')
    new_development_report = ROOT / config['new_development_dir'] / 'newdev_fixed_rule_comparison.json'
    if sha256(new_development_report) != config['new_development_report_sha256']:
        raise ValueError('Frozen disjoint-development evidence changed')
    if tuple(match.ROUTES) != tuple(config['routes']):
        raise ValueError('Retrieval route set changed')
    source, target, _ = match.load_inputs(dev_dir, 'train')
    confirm_source, confirm_target, _ = match.load_inputs(confirm_dir, 'train')
    if len(source) != 1000 or len(confirm_source) != 500:
        raise ValueError('Unexpected cohort size')
    old_a, old_b, old_locked = validation.sealed_partitions(source)
    train_ids = old_a | old_b
    if len(train_ids) != 800 or len(old_locked) != 200:
        raise ValueError('Training partition changed')
    new_ids = set(confirm_source.entity_id)
    for folder in config['excluded_cohort_dirs']:
        older = pd.read_parquet(ROOT / folder / 'train_s1.parquet')
        if new_ids & set(older.entity_id):
            raise ValueError(f'Confirmation overlaps opened cohort {folder}')
    if new_ids & set(source.entity_id):
        raise ValueError('Confirmation overlaps training cohort')
    check_retrieval(dev_dir, 1000)
    check_retrieval(confirm_dir, 500)
    base_run = check_retrieval(confirm_base, 500)
    if base_run['sample_offset'] != config['confirmation_sample_offset']:
        raise ValueError('Confirmation source offset changed')
    old_audit = pd.read_parquet(dev_dir / 'train_route_audit.parquet')
    old_audit = old_audit[old_audit.source1_entity_id.isin(train_ids)]
    new_audit = pd.read_parquet(confirm_dir / 'train_route_audit.parquet')
    train_pairs = validation.subset_pairs(old_audit, match.ROUTES, 100)
    new_pairs = validation.subset_pairs(new_audit, match.ROUTES, 100)
    if len(train_pairs) != 80000 or len(new_pairs) != 50000:
        raise ValueError('Unexpected retrieved K100 candidate counts')
    X, numeric = pair_features(train_pairs, source, target, df_dir)
    new_X, new_numeric = pair_features(new_pairs, confirm_source, confirm_target, df_dir)
    saved_base = pd.read_parquet(dev_dir / 'development_k100_features.parquet')
    saved_idf = pd.read_parquet(dev_dir / 'development_k100_global_idf_features.parquet')
    saved_numeric = pd.read_parquet(dev_dir / 'development_k100_numeric_source_interactions_features.parquet')
    if (list(X.columns) != list(new_X.columns) or
            list(numeric.columns) != list(new_numeric.columns) or
            not np.allclose(X.to_numpy(), pd.concat([saved_base, saved_idf], axis=1).to_numpy()) or
            not np.allclose(numeric.to_numpy(), saved_numeric.to_numpy())):
        raise ValueError('Frozen feature schema or development feature parity failed')
    print(f'Preflight: {len(train_pairs)} train and {len(new_pairs)} confirmation K100 pairs; '
          f'{len(X.columns)} IDF and {len(numeric.columns)} numeric features', flush=True)
    if args.preflight:
        return
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'status': 'opened_once', 'config_sha256': sha256(config_path)}, handle)
    train_truth = match.parse_truth(data_dir / 'train/train_ground_truth.tsv', train_ids)
    confirm_truth = match.parse_truth(data_dir / 'train/train_ground_truth.tsv', new_ids)
    y = np.fromiter((p.candidate_entity_id in train_truth.get(p.source1_entity_id, set())
                     for p in train_pairs.itertuples(index=False)), dtype=np.int8)
    idf_model = match.model(config['model_seed'])
    idf_model.fit(X, y)
    numeric_model = match.model(config['model_seed'])
    numeric_model.fit(pd.concat([X, numeric], axis=1), y)
    idf_probability = idf_model.predict_proba(new_X)[:, 1]
    blended_probability = .5 * idf_probability + .5 * numeric_model.predict_proba(
        pd.concat([new_X, new_numeric], axis=1))[:, 1]
    top_idf = pd.Series(idf_probability).groupby(new_pairs.source1_entity_id.reset_index(drop=True)).transform('max').to_numpy()
    selected_probability = np.where(top_idf < config['min_source_top_idf'], 0.0, blended_probability)
    diagnostic_target = validation.hydrate_truth_for_diagnostics(
        confirm_target, confirm_truth, data_dir,
        confirm_dir / 'train_truth_targets_diagnostics.parquet')
    results = {}
    for name, probabilities, threshold in (
        ('idf_baseline', idf_probability, config['idf_threshold']),
        ('half_blend', blended_probability, config['blend_threshold']),
        ('selected_blend_with_source_floor', selected_probability, config['blend_threshold']),
    ):
        metrics, slices, failures = validation.evaluate(
            new_ids, new_pairs, probabilities, threshold, confirm_truth,
            confirm_source, diagnostic_target, full_target_pool=True)
        results[name] = {'metrics': metrics, 'slices': slices, 'failures': dict(failures)}
        print(name, metrics, flush=True)
    report = {
        'status': 'completed_once', 'selected_policy': config['policy'],
        'config_sha256': sha256(config_path),
        'cohort': {'entities': 500, 'true_links': sum(map(len, confirm_truth.values())),
                   'pairs': len(new_pairs), 'positives_in_pairs': int(sum(
                       p.candidate_entity_id in confirm_truth.get(p.source1_entity_id, set())
                       for p in new_pairs.itertuples(index=False))),
                   'target_pool_size': 10_320_219,
                   'candidate_recall_at_k': validation.retrieval_k_report(new_audit, confirm_truth, new_ids)},
        'results': results,
        'feature_importance': {
            'idf': sorted(zip(X.columns, idf_model.feature_importances_), key=lambda row: -row[1])[:25],
            'numeric': sorted(zip(pd.concat([X, numeric], axis=1).columns,
                                  numeric_model.feature_importances_), key=lambda row: -row[1])[:25],
        },
    }
    output.write_text(json.dumps(validation.json_safe(report), indent=2, allow_nan=False))
    print(f'Wrote once-only confirmation {output}', flush=True)


if __name__ == '__main__':
    main()
