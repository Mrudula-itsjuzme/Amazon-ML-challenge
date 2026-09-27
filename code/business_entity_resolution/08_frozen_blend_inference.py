"""Reproduce the predeclared 50/50 IDF/numeric control on retrieved test pairs.

This consumes the exact K100 pair table from 01_pipeline.py. By default it
refuses partial Source 1 test coverage; --allow-partial is only for smoke runs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


match = module('ber_match_for_submission', '02_match.py')
validation = module('ber_validate_for_submission', '03_validate.py')
numeric = module('ber_numeric_for_submission', '06_numeric_context.py')


def features(pairs, source, target, df_dir):
    base = match.build_features(pairs, source, target).drop(
        columns=['name_native_idf', 'address_rare', 'number_weighted'])
    idf = match.global_target_idf_features(pairs, source, target, df_dir)
    context = numeric.numeric_context_features(pairs, base, source, target)
    x = pd.concat([base, idf], axis=1)
    return x, pd.concat([x, context], axis=1)


def write_rows(path, ids, mapping, column):
    with path.open('w') as handle:
        handle.write('source1_entity_id\t' + column + '\n')
        for entity_id in ids:
            handle.write(entity_id + '\t' + ','.join(mapping.get(entity_id, [])) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--config', type=Path, default=HERE/'frozen_blend_policy.json')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--model-cache', type=Path)
    p.add_argument('--train-only', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['cap'] != 100 or config['blend_threshold'] != .55:
        p.error('Unexpected frozen blend configuration')
    dev_dir = ROOT/config['development_dir']
    df_dir = ROOT/config['global_df_dir']
    for filename, expected in config['code_hashes'].items():
        if hashlib.sha256((HERE/filename).read_bytes()).hexdigest() != expected:
            p.error(f'Frozen component changed: {filename}')
    for filename, expected in config['df_hashes'].items():
        if hashlib.sha256((df_dir/filename).read_bytes()).hexdigest() != expected:
            p.error(f'Frozen token statistics changed: {filename}')
    config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.model_cache and args.model_cache.exists():
        payload=joblib.load(args.model_cache)
        if payload['config_sha256']!=config_hash:
            p.error('Cached model belongs to a different frozen config')
        idf_model,numeric_model=payload['idf_model'],payload['numeric_model']
    else:
        raw = pd.read_parquet(dev_dir/'train_route_audit.parquet')
        train_source, train_target, _ = match.load_inputs(dev_dir, 'train')
        train_ids = set.union(*validation.sealed_partitions(train_source)[:2])
        raw = raw[raw.source1_entity_id.isin(train_ids)]
        train_pairs = validation.subset_pairs(raw, match.ROUTES, 100)
        if len(train_ids) != 800 or len(train_pairs) != 80000:
            p.error('Frozen training cohort or K100 candidates differ')
        train_truth = match.parse_truth(ROOT/config['data_dir']/'train/train_ground_truth.tsv', train_ids)
        y = np.fromiter((r.candidate_entity_id in train_truth[r.source1_entity_id]
                         for r in train_pairs.itertuples(index=False)), dtype=np.int8)
        x, xn = features(train_pairs, train_source, train_target, df_dir)
        saved_base = pd.read_parquet(dev_dir/'development_k100_features.parquet')
        saved_idf = pd.read_parquet(dev_dir/'development_k100_global_idf_features.parquet')
        saved_numeric = pd.read_parquet(dev_dir/'development_k100_numeric_source_interactions_features.parquet')
        if (list(x.columns) != list(pd.concat([saved_base, saved_idf], axis=1).columns)
                or not np.allclose(x.to_numpy(), pd.concat([saved_base, saved_idf], axis=1).to_numpy())
                or not np.allclose(xn.iloc[:, len(x.columns):].to_numpy(), saved_numeric.to_numpy())):
            p.error('Training feature parity with frozen confirmation failed')
        idf_model = match.model(config['model_seed']).fit(x, y)
        numeric_model = match.model(config['model_seed']).fit(xn, y)
        if args.model_cache:
            args.model_cache.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'config_sha256':config_hash,'idf_model':idf_model,
                         'numeric_model':numeric_model},args.model_cache)
        del train_source, train_target, train_pairs, raw, x, xn, y
    if args.train_only:
        print('Frozen blend models ready',flush=True)
        return
    if not args.input_dir or not args.output_dir:
        p.error('--input-dir and --output-dir are required for inference')

    source, target, pairs = match.load_inputs(args.input_dir, 'test')
    if pairs.duplicated(['source1_entity_id', 'candidate_entity_id']).any():
        p.error('Duplicate candidate pairs')
    if not set(pairs.source1_entity_id).issubset(set(source.entity_id)):
        p.error('Candidate has an unknown Source 1 ID')
    full_count = sum(1 for _ in (ROOT/config['data_dir']/'test/test_source1.tsv').open())-1
    if not args.allow_partial and len(source) != full_count:
        p.error(f'Partial S1 population: {len(source)} of {full_count}; refusing submission output')
    manifest = args.input_dir/'test_retrieval_run.json'
    if not manifest.exists():
        p.error('Missing label-free full-pool retrieval manifest')
    retrieval = json.loads(manifest.read_text())
    expected_target = sum(sum(1 for _ in (ROOT/config['data_dir']/f'test/test_source{i}.tsv').open())-1
                          for i in (2, 3))
    if retrieval['target_pool_size'] != expected_target or retrieval['labels_used_for_retrieval'] is not False:
        p.error('Candidate pool is not from the complete, label-free test target universe')
    if pairs.groupby('source1_entity_id').size().max() > 100:
        p.error('Candidate cap exceeds frozen K100')
    x, xn = features(pairs, source, target, df_dir)
    probabilities = .5*idf_model.predict_proba(x)[:,1] + .5*numeric_model.predict_proba(xn)[:,1]
    chosen = pairs.loc[probabilities >= config['blend_threshold']]
    candidates = pairs.groupby('source1_entity_id', sort=False).candidate_entity_id.agg(list).to_dict()
    matches = chosen.groupby('source1_entity_id', sort=False).candidate_entity_id.agg(list).to_dict()
    ids = source.entity_id.tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir/'candidate_pairs.tsv', ids, candidates, 'candidate_entity_ids')
    write_rows(args.output_dir/'matching_results.tsv', ids, matches, 'matched_entity_ids')
    (args.output_dir/'inference_manifest.json').write_text(json.dumps({
        'model':'predeclared_half_blend_control', 'threshold':config['blend_threshold'],
        'source1_count':len(source), 'expected_source1_count':full_count,
        'candidate_pairs':len(pairs), 'matches':len(chosen),
        'target_pool_size':expected_target, 'partial':bool(args.allow_partial)}, indent=2))
    print(f'Wrote {len(ids)} S1 rows, {len(pairs)} candidates, {len(chosen)} matches', flush=True)


if __name__ == '__main__':
    main()
