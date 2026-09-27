"""Retrospective development-only evaluation of completed label-free FTS edges."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ber_match_eval',HERE/'02_match.py')
match=importlib.util.module_from_spec(spec);spec.loader.exec_module(match)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run=json.loads(a.retrieval_json.read_text())
    ids={r['source1_entity_id'] for r in run['rows']}
    truth=match.parse_truth(ROOT/'student_resource/dataset/train/train_ground_truth.tsv',ids)
    total=sum(len(v) for v in truth.values())
    cuts=(10,20,40,60,80,100,150)
    counts={k:0 for k in cuts};raw=0;candidate_counts=[]
    misses=[]
    for row in run['rows']:
        sid=row['source1_entity_id'];ordered=row['raw'];actual=truth[sid]
        if len(ordered)!=len(set(ordered)):
            raise ValueError(f'Duplicate candidates for {sid}')
        candidate_counts.append(len(ordered))
        present=actual & set(ordered)
        raw+=len(present)
        for k in cuts:counts[k]+=len(actual & set(ordered[:k]))
        for tid in actual-set(ordered[:100]):
            misses.append({'source1_entity_id':sid,'candidate_entity_id':tid,
                           'raw_present':tid in present,
                           'raw_rank':ordered.index(tid)+1 if tid in present else None})
    report={'source_entities':len(ids),'true_links':total,'raw_recall':raw/total,
            'raw_true_links':raw,'recall_at_k':{str(k):counts[k]/total for k in cuts},
            'true_links_at_k':counts,'mean_raw_candidates':sum(candidate_counts)/len(candidate_counts),
            'elapsed_seconds':run['elapsed_seconds'],'misses_k100':misses}
    a.output.write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='misses_k100'})

if __name__=='__main__':main()
