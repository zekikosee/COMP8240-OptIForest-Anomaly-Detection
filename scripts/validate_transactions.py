import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from generate_transactions import FEATURES,generate

p=argparse.ArgumentParser();p.add_argument('csv');a=p.parse_args()
path=Path(a.csv);df=pd.read_csv(path);meta=json.loads(path.with_suffix('.json').read_text())
assert list(df.columns)==FEATURES+['label'],'Unexpected feature/label columns'
assert len(df)==meta['rows'],'Incorrect row count'
assert not df.isna().any().any(),'Missing values'
assert df.label.isin([0,1]).all(),'Invalid labels'
assert int(df.label.sum())==round(meta['rows']*meta['requested_prevalence']),'Anomaly prevalence incorrect'
assert df.hour.between(0,23).all() and df.merchant_risk.between(0,1).all() and df.device_familiarity.between(0,1).all(),'Invalid bounds'
assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],'CSV checksum mismatch'
expected=generate(meta['rows'],meta['requested_prevalence'],meta['anomaly_type'],meta['seed'])
assert df.shape==expected.shape and __import__('numpy').allclose(df[FEATURES].to_numpy(),expected[FEATURES].to_numpy(),rtol=1e-8,atol=1e-6),'Not reproducible from seed'
assert df.label.equals(expected.label),'Labels differ from seed'
print(f'PASS: {path.name}, {len(df)} rows, {int(df.label.sum())} anomalies, no missing values, reproducible, checksum verified')
