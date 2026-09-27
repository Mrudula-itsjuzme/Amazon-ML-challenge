"""Build the final challenge zip only after complete output validation."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=(
    '01_pipeline.py','02_match.py','03_validate.py','06_numeric_context.py',
    '07_blend_confirmation.py',
    '08_frozen_blend_inference.py','09_validate_submission_local.py',
    '12_exact_sharded_submission.py','13_package_submission.py',
    '14_run_distributed_shard.py','bootstrap_distributed.sh')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--zip',type=Path,required=True)
    a=p.parse_args()
    if a.zip.exists():p.error('Package already exists; refusing overwrite')
    output=a.output_dir
    manifest=output/'submission_manifest.json'
    if not manifest.exists():p.error('Complete submission manifest is missing')
    report=json.loads(manifest.read_text())
    full_count=sum(1 for _ in (ROOT/'student_resource/dataset/test/test_source1.tsv').open())-1
    if report['source1_count']!=full_count or full_count!=1732544:
        p.error('Incomplete or unexpected full test S1 population')
    matching=output/'matching_results.tsv'
    candidate=output/'candidate_pairs.tsv'
    subprocess.run([sys.executable,str(HERE/'09_validate_submission_local.py'),
                    '--matching',str(matching),'--candidate',str(candidate),
                    '--test-dir',str(ROOT/'student_resource/dataset/test')],check=True)
    official=ROOT/'utils/validate_submission.py'
    if not official.exists():
        p.error('Official utils/validate_submission.py is missing; obtain it and require PASS')
    subprocess.run([sys.executable,str(official),'--matching',str(matching),
                    '--candidate',str(candidate),
                    '--test-dir',str(ROOT/'student_resource/dataset/test')],check=True)
    model=ROOT/'research_runs/submission_exact_shards/frozen_blend.joblib'
    if not model.exists():p.error('Frozen model artifact missing')
    df=ROOT/'research_runs/global_target_token_df'
    files={
        'output/matching_results.tsv':matching,
        'output/candidate_pairs.tsv':candidate,
        'output/submission_manifest.json':manifest,
        'Documentation_template.md':HERE/'FROZEN_BLEND_DOCUMENTATION.md',
        'README.md':HERE/'README.md',
        'code/business_entity_resolution/README.md':HERE/'README.md',
        'code/business_entity_resolution/requirements.txt':HERE/'requirements.txt',
        'code/business_entity_resolution/SUBMISSION_RUNBOOK.md':HERE/'SUBMISSION_RUNBOOK.md',
        'code/business_entity_resolution/frozen_blend_policy.json':HERE/'frozen_blend_policy.json',
        'research_runs/submission_exact_shards/frozen_blend.joblib':model,
    }
    for name in SOURCE:
        files[f'code/business_entity_resolution/{name}']=HERE/name
        files[f'code/business_entity_resolution/src/{name}']=HERE/name
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py','.md','.sh','.json','.txt','.ipynb'):
            files.setdefault(f'code/business_entity_resolution/{path.name}',path)
    for name in ('manifest.json','name_df.parquet','address_df.parquet'):
        files[f'research_runs/global_target_token_df/{name}']=df/name
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.zip.with_name(a.zip.name+'.partial')
    if temporary.exists():p.error(f'Partial zip exists: {temporary}')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3,allowZip64=True) as archive:
        for arcname,path in files.items():
            if not path.exists():p.error(f'Missing package input: {path}')
            archive.write(path,arcname)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('Zip CRC verification failed')
    temporary.replace(a.zip)
    print(f'Validated package: {a.zip}',flush=True)


if __name__=='__main__':main()
