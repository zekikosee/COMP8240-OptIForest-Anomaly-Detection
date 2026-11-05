# COMP8240 OptIForest Final Project — Zeki Kose (48718459)

Research project: **Reproducing and Extending OptIForest for Anomaly Detection**. Includes research code, nine seeded synthetic datasets and their JSON checksum metadata, benchmark scripts, recorded CSV experiment results, and a draft ACL-compatible LaTeX report.

**Important provenance:** `OptIForest-main/` originates from [xiagll/OptIForest](https://github.com/xiagll/OptIForest). It is included for replication, not claimed as original work. See `PATCH_NOTES.md`. `report/main.pdf` is an offline two-column preview; for final submission, import `report/main.tex` and `report/figures/` into the official ACL Overleaf template and recompile. Replace **[ADD PUBLIC GITHUB URL BEFORE SUBMISSION]** in `report/main.tex` with the actual public repository link before final submission.

## Setup (Windows CMD)

From this project root, activate an existing Python venv or create one:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Python 3.12 was used in original experiments. Requirements contain minimum-version ranges, not an exact lockfile. Environment/version drift remains a reproduction limitation.

## Run original AD benchmark

```cmd
cd OptIForest-main
python demo.py -StringA ad -IntB 403 -IntC 0
cd ..
python scripts\run_baseline.py --data OptIForest-main\data\ad.csv --runs 15 --output results\ad_iforest_baseline.csv
```

The upstream `demo.py` performs 15 repetitions without saving each replicate; the reported replicated AD mean/std were retained in terminal notes rather than in the latest result ZIP. The 15-row `ad_iforest_baseline.csv` is available. Do not confuse these two sources.

## Synthetic dataset generation and validation

```cmd
python scripts\generate_transactions.py --n 10000 --prevalence 0.01 --anomaly point --seed 42 --output-dir data\generated
python scripts\validate_transactions.py data\generated\transactions_point_p01_seed42.csv
```

Repeat for anomaly types `point`, `contextual`, `collective` and prevalence levels `0.01`, `0.05`, `0.10`. All nine accompanying CSVs/JSONs are included and passed row-count, checksum, reproducibility and missing-data checks in the final package build. They are NOT real bank records.

## New data experiment examples

```cmd
python scripts\run_optiforest_synthetic.py --data data\generated\transactions_point_p01_seed42.csv --runs 3 --output results\point_p01_optiforest_3runs.csv
python scripts\run_baseline.py --data data\generated\transactions_point_p01_seed42.csv --runs 15 --output results\point_p01_iforest.csv
python scripts\run_ablation.py --data OptIForest-main\data\ad.csv --branches 0 2 3 --thresholds 403 --runs 3 --output results\ablation_ad.csv
```

**Compute warning:** low threshold values can require unusually long computation; threshold 7 had no completed run. The original code may issue an invalid-division runtime warning.

## Result interpretation

- Results are stored as per-run CSVs, with `seed`, `auc_roc`, `average_precision`, `fit_seconds` and `score_seconds`.
- Original AD OptIForest 15-run aggregate: AUC 0.7741166942 (std 0.0057122459), AP 0.4351114702 (std 0.0293811097); direct demo console evidence rather than archived seed rows.
- Synthetic iForest has 15 model seeds per dataset; synthetic OptIForest has 1 model seed per dataset, plus 3 seeds for Point 1% and Point 5%.
- Generator seed 42 is fixed across the nine scenarios. Experiments are unsupervised and transductive; labels enter evaluation only.
- `report/synthetic_summary.csv`, `branch_summary.csv`, `threshold_summary.csv` are compiled from actual CSV files.
- The repository has not been published to GitHub by this package-generation step; make it public and insert its URL into the report before submission.

## Obtaining the original AD dataset

The 45.9 MB original `OptIForest-main/data/ad.csv` is intentionally excluded from the browser-upload package. Obtain the file from the original upstream repository https://github.com/xiagll/OptIForest and place it under `OptIForest-main/data/ad.csv` before running AD replication or ablations. Preserve the source attribution and comply with the upstream project's redistribution terms.

## GitHub upload

The archive is structured with files at its root. Extract it before using GitHub's **Add file → Upload files** interface. Keep the repository **Public** to meet assessment requirements. When publishing, update the placeholder in the report to the actual URL, recompile the report PDF and upload the final PDF.

Note: the research source incorporates third-party upstream implementation; confirm applicable licensing/permissions before redistribution.
