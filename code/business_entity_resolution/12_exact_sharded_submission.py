"""Resume-safe shards of the frozen full-pool forward retrieval and blend.

Each shard scans the complete S2/S3 target pool with the unchanged functions
from 01_pipeline.py. No target labels enter retrieval. A merged output is
published only after every test S1 appears exactly once.
"""
import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('ber_exact_forward',HERE/'01_pipeline.py')
forward=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=forward
spec.loader.exec_module(forward)
TEST=ROOT/'student_resource/dataset/test'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def shard_path(run,kind,index):
    return run/kind/f'{index:05d}'


def prepare(run,count):
    manifest=run/'shards.json'
    if manifest.exists():raise ValueError(f'Existing shards manifest: {manifest}')
    folder=run/'sources';folder.mkdir(parents=True,exist_ok=True)
    sizes=[0]*count
    for chunk in pd.read_csv(TEST/'test_source1.tsv',sep='\t',dtype=str,
                             keep_default_na=False,chunksize=100000):
        bucket=(pd.util.hash_pandas_object(chunk.entity_id,index=False).to_numpy()%count)
        for index in pd.unique(bucket):
            selected=chunk.loc[bucket==index]
            path=folder/f'{int(index):05d}.tsv'
            selected.to_csv(path,sep='\t',index=False,mode='a',header=not path.exists())
            sizes[int(index)]+=len(selected)
    expected=sum(1 for _ in (TEST/'test_source1.tsv').open())-1
    if sum(sizes)!=expected or min(sizes)==0:raise ValueError('Incomplete/empty S1 shard')
    manifest.write_text(json.dumps({'source_count':expected,'shard_count':count,
                                    'shard_sizes':sizes,'assignment':'pandas_hash_modulo',
                                    'test_source1_sha256':sha256(TEST/'test_source1.tsv')},indent=2))
    print(f'prepared {count} shards for {expected} S1',flush=True)


def save_retrieval(folder,s1,target,pairs,audit,population,phase,base=None):
    folder.mkdir(parents=True,exist_ok=True)
    prefix=folder/'test'
    s1.to_parquet(f'{prefix}_s1.parquet',index=False)
    target.to_parquet(f'{prefix}_target.parquet',index=False)
    pairs.to_parquet(f'{prefix}_pairs.parquet',index=False)
    audit.to_parquet(f'{prefix}_route_audit.parquet',index=False)
    (folder/'test_retrieval_run.json').write_text(json.dumps({
        'target_pool_size':population,'source_count':len(s1),
        'candidate_pair_count':len(pairs),'raw_pair_count':len(audit),
        'route_top':50,'cap':100,'phase':phase,'augmented_from':str(base) if base else None,
        'labels_used_for_retrieval':False},indent=2))
    print(f'{phase}: {len(s1)} S1, {len(pairs)} K100 pairs, {population} targets',flush=True)


def retrieve(run,index,workers,chunk):
    source=run/'sources'/f'{index:05d}.tsv'
    out=shard_path(run,'base',index)
    if (out/'test_retrieval_run.json').exists():
        print(f'base shard {index} complete; skip',flush=True);return
    s1=forward.read_source(source)
    target,pairs,population,audit=forward.stream_generate(
        s1,TEST,'test',chunk_size=chunk,batch=64,route_top=50,cap=100,
        audit_retrieval=True,workers=workers)
    save_retrieval(out,s1,target,pairs,audit,population,'base')


def augment(run,index,workers,chunk):
    base=shard_path(run,'base',index)
    out=shard_path(run,'aug',index)
    if (out/'test_retrieval_run.json').exists():
        print(f'aug shard {index} complete; skip',flush=True);return
    if not (base/'test_retrieval_run.json').exists():raise ValueError('Base retrieval missing')
    s1=pd.read_parquet(base/'test_s1.parquet')
    target=pd.read_parquet(base/'test_target.parquet')
    audit=pd.read_parquet(base/'test_route_audit.parquet')
    target,pairs,audit,population=forward.augment_retrieval(
        s1,target,audit,TEST,'test',chunk_size=chunk,batch=64,
        route_top=50,cap=100,workers=workers)
    save_retrieval(out,s1,target,pairs,audit,population,'aug',base)


def infer(run,index):
    inp=shard_path(run,'aug',index)
    out=shard_path(run,'pred',index)
    if (out/'inference_manifest.json').exists():
        print(f'prediction shard {index} complete; skip',flush=True);return
    if not (inp/'test_retrieval_run.json').exists():raise ValueError('Augmented retrieval missing')
    subprocess.run([sys.executable,str(HERE/'08_frozen_blend_inference.py'),
                    '--input-dir',str(inp),'--output-dir',str(out),
                    '--model-cache',str(run/'frozen_blend.joblib'),'--allow-partial'],check=True)


def merge(run,output,allow_partial=False):
    from importlib import import_module
    seal=import_module('14_run_distributed_shard')
    manifest=json.loads((run/'shards.json').read_text())
    expected_order={}
    with (TEST/'test_source1.tsv').open(newline='') as handle:
        for position,row in enumerate(csv.DictReader(handle,delimiter='\t')):
            sid=row['entity_id']
            if sid in expected_order:raise ValueError(f'Duplicate S1 in test input: {sid}')
            expected_order[sid]=position
    expected=set(expected_order)
    if len(expected)!=manifest['source_count']:raise ValueError('Test S1 count changed')
    if sha256(TEST/'test_source1.tsv')!=manifest['test_source1_sha256']:
        raise ValueError('Test S1 file changed')
    if not allow_partial:
        missing=[i for i in range(manifest['shard_count'])
                 if not (shard_path(run,'pred',i)/'completion_manifest.json').exists()]
        if missing:raise ValueError(f'Missing {len(missing)} prediction shards; first: {missing[0]}')
    target_files=[TEST/f'test_source{i}.tsv' for i in (2,3)]
    actual_target_count=sum(sum(1 for _ in path.open())-1 for path in target_files)
    actual_target_hashes={path.name:sha256(path) for path in target_files}
    identity=None
    for index in range(manifest['shard_count']):
        final=shard_path(run,'pred',index)/'completion_manifest.json'
        if not final.exists():continue
        record=json.loads(final.read_text())
        checks=seal.validate_shard(run,index,manifest['shard_sizes'][index],record['target_count'])
        if record['completion'] is not True or record['shard_index']!=index or \
                record['num_shards']!=manifest['shard_count'] or record['checksums']!=checks:
            raise ValueError(f'Invalid completion manifest: {index}')
        current=(record['model_sha256'],record['config_sha256'],
                 tuple(sorted(record['target_sha256'].items())),record['target_count'])
        if record['target_count']!=actual_target_count or record['target_sha256']!=actual_target_hashes:
            raise ValueError(f'Shard target universe differs from merge input: {index}')
        if identity is None:identity=current
        elif identity!=current:raise ValueError(f'Shard model/config/target mismatch: {index}')
        if seal.digest(run/'frozen_blend.joblib')!=record['model_sha256'] or \
                seal.digest(HERE/'frozen_blend_policy.json')!=record['config_sha256']:
            raise ValueError(f'Local model or policy differs from shard {index}')
    output.mkdir(parents=True,exist_ok=True)
    match_tmp=output/'matching_results.tsv.partial'
    cand_tmp=output/'candidate_pairs.tsv.partial'
    if (output/'matching_results.tsv').exists() or (output/'candidate_pairs.tsv').exists():
        raise ValueError('Final output already exists; refusing overwrite')
    seen=set();pairs=0;links=0
    with ExitStack() as stack:
        mf=stack.enter_context(match_tmp.open('w',newline=''))
        cf=stack.enter_context(cand_tmp.open('w',newline=''))
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n')
        cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mw.writerow(('source1_entity_id','matched_entity_ids'))
        cw.writerow(('source1_entity_id','candidate_entity_ids'))
        streams=[]
        for index in range(manifest['shard_count']):
            folder=shard_path(run,'pred',index)
            if not (folder/'completion_manifest.json').exists():
                if allow_partial:continue
                raise ValueError(f'Missing prediction shard {index}')
            info=json.loads((folder/'inference_manifest.json').read_text())
            if info['source1_count']!=manifest['shard_sizes'][index]:
                raise ValueError(f'S1 count mismatch in shard {index}')
            mh=stack.enter_context((folder/'matching_results.tsv').open(newline=''))
            ch=stack.enter_context((folder/'candidate_pairs.tsv').open(newline=''))
            mr=csv.DictReader(mh,delimiter='\t');cr=csv.DictReader(ch,delimiter='\t')
            def ordered_rows(matches, candidates):
                previous=-1
                for m,c in zip(matches,candidates,strict=True):
                    sid=m['source1_entity_id']
                    if sid not in expected_order:raise ValueError(f'Unknown S1: {sid}')
                    position=expected_order[sid]
                    if position<=previous:raise ValueError(f'Out-of-order shard S1: {sid}')
                    previous=position
                    yield position,m,c
            streams.append(ordered_rows(mr,cr))
        for _,m,c in heapq.merge(*streams,key=lambda item:item[0]):
            sid=m['source1_entity_id']
            if sid!=c['source1_entity_id'] or sid in seen:
                raise ValueError(f'Duplicate/misaligned S1: {sid}')
            candidates=c['candidate_entity_ids'].split(',') if c['candidate_entity_ids'] else []
            matches=m['matched_entity_ids'].split(',') if m['matched_entity_ids'] else []
            if len(candidates)>100 or len(candidates)!=len(set(candidates)) or \
                    len(matches)!=len(set(matches)) or not set(matches)<=set(candidates):
                raise ValueError(f'Invalid candidate or match list: {sid}')
            seen.add(sid);pairs+=len(candidates);links+=len(matches)
            mw.writerow((sid,','.join(matches)));cw.writerow((sid,','.join(candidates)))
    if not allow_partial and seen!=expected:
        raise ValueError(f'Incomplete output: {len(seen)} of {len(expected)} S1')
    if allow_partial:
        print(f'partial merge checked {len(seen)} S1; final files not published',flush=True)
        return
    validator=HERE/'09_validate_submission_local.py'
    subprocess.run([sys.executable,str(validator),'--matching',str(match_tmp),
                    '--candidate',str(cand_tmp),'--test-dir',str(TEST)],check=True)
    match_tmp.replace(output/'matching_results.tsv')
    cand_tmp.replace(output/'candidate_pairs.tsv')
    (output/'submission_manifest.json').write_text(json.dumps({
        'source1_count':len(seen),'candidate_pairs':pairs,'predicted_links':links,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'shard_count':manifest['shard_count'],
        'shard_manifest_sha256':{str(i):sha256(shard_path(run,'pred',i)/'completion_manifest.json')
                                  for i in range(manifest['shard_count'])},
        'matching_sha256':sha256(output/'matching_results.tsv'),
        'candidate_sha256':sha256(output/'candidate_pairs.tsv'),
        'model_sha256':identity[0], 'config_sha256':identity[1],
        'retrieval':'unchanged full-pool forward K100, sharded by S1',
        'matcher':'frozen predeclared 50/50 IDF and numeric blend'},indent=2))
    print(f'published {len(seen)} S1, {pairs} candidates, {links} links',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['prepare','retrieve','augment','infer','merge'])
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--shard-count',type=int,default=174)
    p.add_argument('--shard-id',type=int)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--target-chunk',type=int,default=50000)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--allow-partial-merge',action='store_true')
    a=p.parse_args()
    if a.step=='prepare':prepare(a.run_dir,a.shard_count)
    elif a.step=='merge':
        if not a.output_dir:p.error('--output-dir required')
        merge(a.run_dir,a.output_dir,a.allow_partial_merge)
    else:
        if a.shard_id is None:p.error('--shard-id required')
        info=json.loads((a.run_dir/'shards.json').read_text())
        if not 0<=a.shard_id<info['shard_count']:p.error('Invalid shard ID')
        if a.step=='retrieve':retrieve(a.run_dir,a.shard_id,a.workers,a.target_chunk)
        elif a.step=='augment':augment(a.run_dir,a.shard_id,a.workers,a.target_chunk)
        else:infer(a.run_dir,a.shard_id)


if __name__=='__main__':main()
