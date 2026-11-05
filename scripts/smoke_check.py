"""Quick reproduction checks. No result CSV in results/ is overwritten.
Run from repository root: python scripts/smoke_check.py
Uses temporary files; verifies all 9 datasets, AD ZIP, one 1-seed iForest
run, and a very small OptIForest execution. Not a full benchmark rerun.
"""
from __future__ import annotations
import csv
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

def run(*args):
    print('RUN:', ' '.join(str(x) for x in args), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)

def main():
    archive = ROOT / 'OptIForest-main/data/ad.csv.zip'
    expected_hash = '7dfd4264c9aa385ed998df2f4c2eb7a175c0aed816eff495d17933f31df8e832'
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['ad.csv'], 'Unexpected AD ZIP structure'
        b = z.read('ad.csv')
    assert len(b) == 45_919_116, 'Unexpected AD byte count'
    assert hashlib.sha256(b).hexdigest() == expected_hash, 'Original AD dataset checksum mismatch'
    print('PASS: original AD ZIP size and SHA-256', flush=True)
    csvfiles = sorted((ROOT / 'data/generated').glob('transactions_*.csv'))
    assert len(csvfiles) == 9, f'Expected 9 generated files, got {len(csvfiles)}'
    for f in csvfiles:
        run('scripts/validate_transactions.py', f.relative_to(ROOT))
    import pandas as pd
    x = pd.read_csv(ROOT / 'data/generated/transactions_point_p05_seed42.csv')
    tiny = pd.concat([x[x.label == 0].head(160), x[x.label == 1].head(40)], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix='comp8240_smoke_') as tmp:
        t = Path(tmp)
        data = t / 'tiny.csv'
        tiny.to_csv(data, index=False)
        result_if = t / 'iforest.csv'
        result_opt = t / 'optiforest.csv'
        # Use a real generated dataset for comparable iForest smoke metrics.
        run('scripts/run_baseline.py', '--data', 'data/generated/transactions_point_p01_seed42.csv', '--runs', 1, '--output', result_if)
        with result_if.open(newline='') as h:
            row = next(csv.DictReader(h))
        assert abs(float(row['average_precision']) - 0.846692) < 0.00001, 'Baseline AP mismatch'
        # A very small execution to check OptIForest API and NumPy patch.
        run('scripts/run_optiforest_synthetic.py', '--data', data, '--runs', 1, '--trees', 3, '--threshold', 100, '--output', result_opt)
        assert result_opt.is_file(), 'No OptIForest result saved'
    print('PASS: 9 synthetic datasets, original AD archive, iForest result and OptIForest API smoke test')
    print('Note: smoke tests do NOT certify 15-run benchmark reproducibility or remove the upstream numerical warning.')

if __name__ == '__main__':
    main()
