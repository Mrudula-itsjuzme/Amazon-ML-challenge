"""Disk-backed, label-free forward retrieval gate for a bounded S1 cohort.

The index is built from every S2/S3 row in one split. Truth is consulted only
by the optional retrospective development report, after retrieval finishes.
This is an experimental route and does not replace the frozen K100 blocker.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
from unidecode import unidecode

from importlib import util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = util.spec_from_file_location('ber_forward', HERE/'01_pipeline.py')
forward = util.module_from_spec(spec)
spec.loader.exec_module(forward)


def connection(path):
    db = sqlite3.connect(path)
    db.execute('pragma journal_mode=off')
    db.execute('pragma synchronous=off')
    db.execute('pragma cache_size=-32768')
    db.execute('pragma temp_store=memory')
    return db


def build(db, split, chunk_size, max_rows=None):
    db.execute('create table if not exists records(id text primary key, name text, address text, country text)')
    db.execute("create virtual table if not exists docs using fts5(name, address, translit, core, content='')")
    db.execute('create table if not exists meta(key text primary key, value text)')
    if db.execute("select value from meta where key='complete'").fetchone():
        raise ValueError('Index already complete')
    start=time.monotonic(); count=0
    for source in (2,3):
        path=ROOT/'student_resource/dataset'/split/f'{split}_source{source}.tsv'
        for frame in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=chunk_size):
            if max_rows is not None:
                frame=frame.head(max_rows-count)
            if frame.empty:break
            rows=[];fts=[]
            for r in frame.itertuples(index=False):
                name=forward.norm(r.business_name)
                address=forward.norm(r.business_address)
                trans=forward.norm(unidecode(r.business_name))
                legal=' '.join(forward.LEGAL.get(t,t) for t in name.split())
                core=' '.join(t for t in legal.split() if t not in forward.SUFFIXES)
                rows.append((r.entity_id,r.business_name,r.business_address,forward.norm(r.country)))
                fts.append((name,address,trans,core))
            db.executemany('insert into records(id,name,address,country) values(?,?,?,?)',rows)
            first=db.execute('select last_insert_rowid()').fetchone()[0]-len(rows)+1
            db.executemany('insert into docs(rowid,name,address,translit,core) values(?,?,?,?,?)',
                           ((first+i,*row) for i,row in enumerate(fts)))
            count+=len(rows)
            db.commit()
            if count%500000<chunk_size:
                print(f'indexed {count:,} targets in {time.monotonic()-start:.0f}s',flush=True)
            if max_rows is not None and count>=max_rows:break
        if max_rows is not None and count>=max_rows:break
    db.execute('create index if not exists country_idx on records(country)')
    db.execute('insert into meta(key,value) values(?,?)',('complete',json.dumps({'split':split,'rows':count})))
    db.commit()
    print(f'complete {count:,} rows in {time.monotonic()-start:.0f}s',flush=True)


def terms(text, limit=5):
    # FTS5 unicode61 is word-based. Quote terms to avoid parser operators.
    values=[x.replace('"','') for x in text.split() if len(x)>=2]
    return list(dict.fromkeys(values))[:limit]


def query_route(db, column, text, country, limit, rare_terms):
    words=terms(text)
    if not words:return []
    df={word:db.execute('select doc from vocab where term=?',(word,)).fetchone() for word in words}
    words=sorted(words,key=lambda word:(df[word][0] if df[word] else 10**9,word))[:rare_terms]
    expression=' OR '.join(f'{column}:"{word}"' for word in words)
    sql=('select records.id, rank as score from docs '
         'join records on records.rowid=docs.rowid '
         'where docs match ? and records.country=? order by rank limit ?')
    return db.execute(sql,(expression,country,limit)).fetchall()


def diagnostic(db, source_path, limit, output, rare_terms):
    source=pd.read_parquet(source_path)
    db.execute("create virtual table if not exists vocab using fts5vocab(docs, row)")
    rows=[];start=time.monotonic()
    routes=(('name','name_native'),('address','address_native'),
            ('translit','name_translit'),('core','name_core'))
    for i,r in enumerate(source.itertuples(index=False),1):
        evidence={}
        for route, field in routes:
            text=getattr(r,field)
            if route in ('name','address','core') and any(ord(c)>127 for c in text):
                continue
            for rank,(tid,score) in enumerate(query_route(db,route,text,r.country_norm,limit,rare_terms),1):
                evidence.setdefault(tid,{})[route]=(rank,float(score))
        ordered=sorted(evidence,key=lambda tid:(-len(evidence[tid]),
                       -sum(1/(rank+1) for rank,_ in evidence[tid].values()),tid))
        rows.append({'source1_entity_id':r.entity_id,'raw':ordered,'k100':ordered[:100]})
        if i%10==0:print(f'queried {i:,} S1 in {time.monotonic()-start:.1f}s',flush=True)
    output.write_text(json.dumps({'elapsed_seconds':time.monotonic()-start,
                                  'source_count':len(source),'route_top':limit,'rows':rows}))
    print(f'wrote {output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--build',action='store_true')
    p.add_argument('--source-parquet',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--route-top',type=int,default=150)
    p.add_argument('--rare-terms',type=int,default=2)
    p.add_argument('--chunk-size',type=int,default=50000)
    p.add_argument('--max-rows',type=int,help='Tiny index fixture only; never a full-pool result')
    a=p.parse_args()
    db=connection(a.db)
    if a.build:build(db,a.split,a.chunk_size,a.max_rows)
    if a.source_parquet:
        if not a.output:p.error('--output required with --source-parquet')
        diagnostic(db,a.source_parquet,a.route_top,a.output,a.rare_terms)


if __name__=='__main__':main()
