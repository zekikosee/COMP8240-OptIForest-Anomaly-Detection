
"""Run Isolation Forest on original or synthetic datasets.

Original AD dataset: no scaling.
Synthetic transaction datasets: RobustScaler.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler


# Command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    default="OptIForest-main/data/ad.csv",
    help="Path to dataset CSV"
)

parser.add_argument(
    "--runs",
    type=int,
    default=15,
    help="Number of experimental runs"
)

parser.add_argument(
    "--output",
    default="results/iforest_baseline.csv",
    help="Output CSV file"
)

args = parser.parse_args()

if args.runs < 1:
    raise ValueError("Number of runs must be at least 1.")


# Load dataset
data_path = Path(args.data)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

# Synthetic datasets have a named 'label' column.
# Original AD data has no column headers.
is_synthetic = data_path.name.startswith("transactions_")

if is_synthetic:
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Synthetic dataset needs a label column.")

    X = df.drop(columns=["label"]).astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()

    # Apply the same preprocessing as OptIForest.
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    print("Synthetic dataset detected.")
    print("RobustScaler: Applied")

else:
    df = pd.read_csv(data_path, header=None)

    X = df.iloc[:, :-1].astype(float).to_numpy()
    y = df.iloc[:, -1].astype(int).to_numpy()

    print("Original dataset detected.")
    print("RobustScaler: Not applied")


# Validate features and labels
if not np.isfinite(X).all():
    raise ValueError("Features contain missing or non-finite values.")

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Labels must contain both 0 and 1.")


print("\n========== DATASET INFORMATION ==========")
print(f"Dataset: {data_path.name}")
print(f"Total rows: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Normal samples: {(y == 0).sum()}")
print(f"Anomalies: {(y == 1).sum()}")
print(f"Number of runs: {args.runs}")


# Run experiments
records = []

for seed in range(args.runs):

    model = IsolationForest(
        n_estimators=100,
        max_samples=512,
        random_state=seed,
        n_jobs=-1
    )

    start = time.perf_counter()
    model.fit(X)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = -model.score_samples(X)
    score_seconds = time.perf_counter() - start

    record = {
        "seed": seed,
        "auc_roc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds
    }

    records.append(record)

    print(f"\nRun {seed + 1}/{args.runs}")
    print(f"AUC-ROC: {record['auc_roc']:.6f}")
    print(f"Average Precision: {record['average_precision']:.6f}")
    print(f"Training time: {fit_seconds:.3f} seconds")
    print(f"Scoring time: {score_seconds:.3f} seconds")


# Save results
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(records)
results_df.to_csv(output_path, index=False)


# Final summary
print("\n========== FINAL RESULTS ==========")

metrics = [
    "auc_roc",
    "average_precision",
    "fit_seconds",
    "score_seconds"
]

print("\nMean:")
print(results_df[metrics].mean().to_string())

print("\nStandard deviation:")
print(results_df[metrics].std().to_string())

print(f"\nResults saved to: {output_path}")
print("Experiment completed successfully!")
