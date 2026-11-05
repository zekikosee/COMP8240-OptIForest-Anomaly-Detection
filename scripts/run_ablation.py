"""OptIForest controlled branch/threshold ablation (COMP8240).

Run from OptIForest_Starter:
python scripts/run_ablation.py --data OptIForest-main/data/ad.csv --thresholds 403 --branches 0 2 3 --runs 1

Original AD is never scaled; synthetic data is RobustScaled using features only.
Records each completed configuration/seed immediately for safe resuming.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "OptIForest-main"))
from detectors import OptIForest


def main():
    parser = argparse.ArgumentParser(description="Run branch/threshold sensitivity experiments")
    parser.add_argument("--data", default="OptIForest-main/data/ad.csv")
    parser.add_argument("--branches", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument("--thresholds", nargs="+", type=int, default=[403])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--output", default="results/ablation_ad.csv")
    args = parser.parse_args()
    if args.runs < 1 or args.trees < 1 or any(t < 1 for t in args.thresholds):
        parser.error("runs, trees, thresholds must be positive")
    if any(b not in (0, 2, 3) for b in args.branches):
        parser.error("branches must be 0, 2 or 3")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if data_path.name.startswith("transactions_"):
        df = pd.read_csv(data_path)
        if "label" not in df.columns:
            raise ValueError("Expected 'label' column in synthetic dataset")
        X = df.drop(columns=["label"]).to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        X = RobustScaler().fit_transform(X)
        scaling = "RobustScaler"
    else:
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy(dtype=int)
        scaling = "none"
    if not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Check finite numeric features and binary labels")

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "scaling", "branch", "threshold", "trees", "seed", "auc_roc", "average_precision", "fit_seconds", "score_seconds"]
    existing = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("dataset") == data_path.name and row.get("scaling") == scaling and int(row["trees"]) == args.trees):
                    existing.add((int(row["branch"]), int(row["threshold"]), int(row["seed"])))
    total = len(args.branches) * len(args.thresholds) * args.runs
    print(f"Dataset: {data_path.name} | shape={X.shape} | scaling={scaling} | target runs={total}", flush=True)
    print(f"Output: {out}", flush=True)
    count = 0
    for branch in args.branches:
        for threshold in args.thresholds:
            for seed in range(args.runs):
                if (branch, threshold, seed) in existing:
                    print(f"SKIP branch={branch}, threshold={threshold}, seed={seed} (already saved)", flush=True)
                    continue
                np.random.seed(seed)
                random.seed(seed)
                print(f"START branch={branch}, threshold={threshold}, seed={seed}", flush=True)
                model = OptIForest("L2OPT", args.trees, threshold, branch)
                start = time.perf_counter()
                model.fit(X)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                score = -np.asarray(model.decision_function(X)).ravel()
                score_seconds = time.perf_counter() - start
                record = {
                    "dataset": data_path.name, "scaling": scaling, "branch": branch,
                    "threshold": threshold, "trees": args.trees, "seed": seed,
                    "auc_roc": roc_auc_score(y, score),
                    "average_precision": average_precision_score(y, score),
                    "fit_seconds": fit_seconds, "score_seconds": score_seconds,
                }
                with out.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(record)
                count += 1
                print(f"DONE AUC={record['auc_roc']:.6f} AP={record['average_precision']:.6f} fit={fit_seconds:.1f}s score={score_seconds:.1f}s", flush=True)
    print(f"Finished: {count} new run(s). Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
