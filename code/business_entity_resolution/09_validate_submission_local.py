"""Strict local TSV format validator; does not compute a leaderboard score."""
import argparse
import csv
from pathlib import Path


def ids(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['entity_id', 'business_name', 'business_address', 'country']:
            raise ValueError(f'Unexpected source schema: {path}')
        return {row['entity_id'] for row in reader}


def rows(path, value_column, expected, targets):
    result = {}
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['source1_entity_id', value_column]:
            raise ValueError(f'Wrong TSV columns: {path}')
        for line, row in enumerate(reader, 2):
            sid = row['source1_entity_id']
            if sid not in expected or sid in result:
                raise ValueError(f'Unknown/duplicate S1 at {path}:{line}: {sid}')
            value = row[value_column]
            values = value.split(',') if value else []
            if len(values) != len(set(values)) or any(x not in targets for x in values):
                raise ValueError(f'Duplicate/unknown target at {path}:{line}')
            result[sid] = set(values)
    if result.keys() != expected:
        raise ValueError(f'Missing {len(expected-set(result))} S1 rows in {path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matching', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    args = parser.parse_args()
    source = ids(args.test_dir/'test_source1.tsv')
    target = ids(args.test_dir/'test_source2.tsv') | ids(args.test_dir/'test_source3.tsv')
    candidate = rows(args.candidate, 'candidate_entity_ids', source, target)
    matching = rows(args.matching, 'matched_entity_ids', source, target)
    bad = [sid for sid in source if not matching[sid] <= candidate[sid]]
    if bad:
        raise ValueError(f'{len(bad)} S1 rows contain matches outside candidate pairs')
    print(f'LOCAL PASS: {len(source)} S1 rows, all matches among valid candidates')


if __name__ == '__main__':
    main()
