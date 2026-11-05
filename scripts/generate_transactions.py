"""Reproducible simulated transaction data, NOT real financial records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['amount','hour','account_age_days','distance_km','transactions_24h','merchant_risk','device_familiarity','geographic_velocity']

def generate(n=10000, prevalence=.05, anomaly='point', seed=42):
    if n < 20 or not 0 < prevalence < 1: raise ValueError('n>=20 and 0<prevalence<1 required')
    if anomaly not in {'point','contextual','collective'}: raise ValueError('invalid anomaly type')
    rng=np.random.default_rng(seed)
    profile=rng.choice(3, n, p=[.55,.30,.15])
    amount=np.clip(rng.lognormal(np.log(np.array([35,90,200])[profile]),.65),.01,1e5)
    hour=(rng.normal(np.array([13,17,11])[profile],3.5).round().astype(int))%24
    age=rng.integers(30,3500,n)
    distance=rng.exponential(np.array([6.,12.,25.])[profile])
    frequency=np.clip(rng.poisson(np.array([2.,4.,6.])[profile])+1,1,1000)
    risk=rng.beta(2,7,n)
    familiarity=rng.beta(8,2,n)
    velocity=np.clip(distance / rng.uniform(.2,4.,n),0,1e5)
    count=round(n*prevalence)
    idx=rng.choice(n,count,replace=False)
    label=np.zeros(n,dtype=int); label[idx]=1
    if anomaly=='point':
        amount[idx]*=rng.uniform(15,35,count)
        velocity[idx]+=rng.uniform(500,1500,count)
    elif anomaly=='contextual':
        # Individually plausible values that are suspicious in combination.
        hour[idx]=rng.integers(0,5,count)
        distance[idx]=rng.uniform(100,220,count)
        familiarity[idx]=rng.uniform(0,.12,count)
        velocity[idx]=distance[idx]/rng.uniform(.2,.6,count)
    else:
        # Group bursts: linked synthetic events. No customer identifiers are stored.
        hour[idx]=rng.integers(1,4,count)
        frequency[idx]=rng.integers(30,65,count)
        amount[idx]=rng.uniform(80,150,count)
        velocity[idx]=rng.uniform(150,300,count)
    df=pd.DataFrame(dict(zip(FEATURES,[amount,hour,age,distance,frequency,risk,familiarity,velocity])))
    df['label']=label
    return df

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=10000)
    p.add_argument('--prevalence',type=float,default=.05)
    p.add_argument('--anomaly',choices=['point','contextual','collective'],default='point')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',default='data/generated');a=p.parse_args()
    df=generate(a.n,a.prevalence,a.anomaly,a.seed)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    stem=f'transactions_{a.anomaly}_p{int(a.prevalence*100):02d}_seed{a.seed}'
    csv=out/(stem+'.csv');df.to_csv(csv,index=False,float_format='%.10g')
    meta={'simulation':True,'method':'custom educational synthetic generator','seed':a.seed,'rows':len(df),'features':FEATURES,'anomaly_type':a.anomaly,'requested_prevalence':a.prevalence,'actual_prevalence':float(df.label.mean()),'label_counts':{str(k):int(v) for k,v in df.label.value_counts().items()},'sha256':hashlib.sha256(csv.read_bytes()).hexdigest()}
    (out/(stem+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print('Created',csv,'rows=',len(df),'anomalies=',int(df.label.sum()))
if __name__=='__main__':main()
