
"""Run OptIForest on synthetic transaction datasets."""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# Project directories
ROOT = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT / "OptIForest-main"

if not OPT_DIR.exists():
    raise FileNotFoundError(
        f"Original OptIForest repository not found: {OPT_DIR}"
    )

sys.path.insert(0, str(OPT_DIR))

from detectors import OptIForest

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--threshold", type=int, default=403)
parser.add_argument("--branch", type=int, default=0)
parser.add_argument("--trees", type=int, default=100)
parser.add_argument(
    "--output",
    default="results/optiforest_synthetic.csv"
)

args = parser.parse_args()

if args.runs < 1 or args.trees < 1:
    raise ValueError("Runs and trees must be positive integers.")

# Load synthetic dataset
data_path = Path(args.data)

if not data_path.is_absolute():
    data_path = ROOT / data_path

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

X = df.drop(columns=["label"]).astype(float).to_numpy()
y = df["label"].astype(int).to_numpy()

if not np.isfinite(X).all():
    raise ValueError("Features contain missing or invalid values.")

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Labels must contain both 0 and 1.")

# Preprocessing: fit without labels
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("\n========== DATASET INFORMATION ==========")
print("Dataset:", data_path.name)
print("Transactions:", len(df))
print("Features:", X.shape[1])
print("Normal transactions:", int((y == 0).sum()))
print("Anomalies:", int((y == 1).sum()))
print("RobustScaler: Applied")
print("Number of runs:", args.runs)
print("Number of trees:", args.trees)
print("Threshold:", args.threshold)
print("Branch:", args.branch)

records = []

# Run experiments
for seed in range(args.runs):

    print(f"\nStarting run {seed + 1}/{args.runs}", flush=True)

    np.random.seed(seed)
    random.seed(seed)

    model = OptIForest(
        "L2OPT",
        args.trees,
        args.threshold,
        args.branch
    )

    start = time.perf_counter()
    model.fit(X_scaled)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.decision_function(X_scaled)
    score_seconds = time.perf_counter() - start

    # OptIForest uses lower scores for anomalies
    anomaly_scores = -np.asarray(predictions).ravel()

    auc_roc = roc_auc_score(y, anomaly_scores)
    ap = average_precision_score(y, anomaly_scores)

    record = {
        "seed": seed,
        "auc_roc": auc_roc,
        "average_precision": ap,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds
    }

    records.append(record)

    print(f"AUC-ROC: {auc_roc:.6f}", flush=True)
    print(f"Average Precision: {ap:.6f}", flush=True)
    print(f"Training time: {fit_seconds:.2f} seconds")
    print(f"Scoring time: {score_seconds:.2f} seconds")

    # Save after every completed run
    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(records)
    results.to_csv(output_path, index=False)

# Final summary
results = pd.DataFrame(records)

print("\n========== FINAL RESULTS ==========")

print(results.mean(numeric_only=True).to_string())

if len(results) > 1:
    print("\nStandard deviation:")
    print(
        results[
            ["auc_roc", "average_precision"]
        ].std().to_string()
    )

print(f"\nResults saved to: {output_path}")
print("Experiment completed successfully!")
