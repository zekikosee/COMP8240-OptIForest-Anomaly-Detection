# COMP8240 — Reproducing and Extending OptIForest for Anomaly Detection

**Student:** Zeki Kose (48718459)  
**Course:** COMP8240, Macquarie University  
**Public repository:** https://github.com/zekikosee/COMP8240-OptIForest-Anomaly-Detection

This research project partially reproduces **OptIForest** on its original **AD** benchmark and extends the evaluation with nine seeded synthetic transaction datasets. It compares OptIForest with scikit-learn Isolation Forest, studies branching and cut-threshold choices, and reports ROC-AUC, average precision (AP), fitting time, and scoring time.

## Project structure

| Path | Contents |
| --- | --- |
| `OptIForest-main/` | Third-party OptIForest implementation (upstream: [xiagll/OptIForest](https://github.com/xiagll/OptIForest)); limited NumPy compatibility edits noted in `PATCH_NOTES.md` |
| `scripts/` | Dataset generator and validator; Isolation Forest baseline; OptIForest synthetic tests; branch/threshold ablations |
| `data/generated/` | Nine synthetic CSV datasets and JSON metadata/checksums |
| `results/` | Recorded experimental metrics for individual model seeds |
| `report/` | LaTeX manuscript, bibliography, figures, summaries and archived report PDF |
| `OptIForest-main/data/ad.csv.zip` | Compressed copy of the original AD benchmark (upstream attribution below); extract before running AD experiments |
| `DATASET_SETUP.md` | Instructions to extract and verify the original AD dataset |
| `PATCH_NOTES.md` | Attribution, compatibility modifications and unresolved numerical warning |
| `requirements.txt` | Python package dependencies |

## Environment setup (Windows CMD)

Python **3.12** was used for the experiments. From the repository root:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead. The dependency file supplies compatible version **ranges**, not a fully frozen environment; precise results and timings can depend on package versions and hardware.

## Required AD data — compressed archive included

The original AD benchmark is provided in compressed form as `OptIForest-main/data/ad.csv.zip` because the uncompressed 45.9 MB file exceeds GitHub's browser upload limit. Before running any AD experiment, extract it to `OptIForest-main/data/ad.csv`.

**Windows:** Open `OptIForest-main/data/ad.csv.zip`, choose **Extract All**, and ensure the resulting `ad.csv` is in `OptIForest-main/data/` (not an extra subfolder).

**Cross-platform from the repository root:**

```bash
python -m zipfile -e OptIForest-main/data/ad.csv.zip OptIForest-main/data
```

Then check its integrity (optional, Python 3.12):

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('OptIForest-main/data/ad.csv'); print('bytes:',p.stat().st_size,'SHA256:',hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected file size: **45,919,116 bytes**. Expected SHA-256: **`7dfd4264c9aa385ed998df2f4c2eb7a175c0aed816eff495d17933f31df8e832`**.

The AD benchmark contains **3,279 observations**, **1,555 features**, and a binary label. It originates from the third-party [OptIForest repository](https://github.com/xiagll/OptIForest), not from this student's synthetic transaction generator. The dataset is redistributed here solely for academic reproducibility with attribution; consult the original publisher's terms before further use or distribution. If the archive is unavailable or redistribution is disallowed, obtain it directly from upstream instead.

See `DATASET_SETUP.md` for the same instructions.

## Reproduce original AD experiments

Run the upstream demo from its own folder (15 repetitions):

```cmd
cd OptIForest-main
python demo.py -StringA ad -IntB 403 -IntC 0
cd ..
```

Run the scikit-learn baseline from the repository root:

```cmd
python scripts\run_baseline.py --data OptIForest-main\data\ad.csv --runs 15 --output results\ad_iforest_baseline.csv
```

The original **OptIForest AD 15-run aggregate** was preserved from console output, rather than as 15 seed-indexed OptIForest rows in the archived `results/` files. Reported mean ROC-AUC: **0.7741167** (SD **0.0057122**); mean AP: **0.4351115** (SD **0.0293811**). Do not mistake these aggregate values for a raw 15-run CSV. For the original AD benchmark, the features were **not scaled**.

## Synthetic datasets and validation

Nine synthetic datasets (3 anomaly types × 3 prevalence levels) contain 10,000 records each, with 8 numerical features and a `label` column; no genuine customer data is used. The datasets use a **generator seed of 42**. Each has a JSON sidecar containing provenance metadata and a SHA-256 checksum.

For example, to generate and validate Point 1%:

```cmd
python scripts\generate_transactions.py --n 10000 --prevalence 0.01 --anomaly point --seed 42 --output-dir data\generated
python scripts\validate_transactions.py data\generated\transactions_point_p01_seed42.csv
```

Repeat for `--anomaly` in `point`, `contextual`, `collective` and `--prevalence` in `0.01`, `0.05`, `0.10`; all nine saved CSV/JSON pairs are included. The generation/validation checks test internal consistency, **not real-world fraud realism**.

## Baseline and OptIForest experiments

Run these examples from the repository root, after installing dependencies:

```cmd
python scripts\run_baseline.py --data data\generated\transactions_point_p01_seed42.csv --runs 15 --output results\point_p01_iforest.csv
python scripts\run_optiforest_synthetic.py --data data\generated\transactions_point_p01_seed42.csv --runs 3 --output results\point_p01_optiforest_3runs.csv
python scripts\run_ablation.py --data OptIForest-main\data\ad.csv --branches 0 2 3 --thresholds 403 --runs 3 --output results\ablation_ad.csv
```

**Be aware:** rerunning the baseline script or synthetic OptIForest script may replace the selected output CSV. Use a new `--output` path to preserve archived evidence. The ablation script skips completed dataset/configuration/seed combinations in an existing output CSV.

For synthetic datasets, the two models receive **RobustScaler-transformed** features; labels are used **only for evaluation**. Fitting and scoring are performed on the same dataset (unsupervised, transductive protocol) rather than on a future-data holdout. Isolation Forest uses 100 estimators and a subsample size of 512. The primary OptIForest experiments use L2OPT with 100 trees, threshold 403 and branch 0.

## How to interpret recorded results

- `results/` contains seed-indexed CSV metrics: `seed`, `auc_roc`, `average_precision`, `fit_seconds` and `score_seconds`; ablations also record branch/threshold configuration.
- Isolation Forest has **15 model seeds** on each of the nine synthetic datasets. OptIForest has at least **one seed** on each, and **three seeds** on Point 1% and Point 5%.
- The nine single-seed model comparisons in the report use **matched model seed 0**. These are exploratory results, not evidence of statistical superiority across populations.
- Generator seed 42 is fixed across all scenarios; repeated model seeds **do not** measure variation across independently generated datasets.
- The AD branch study compared 0, 2 and 3 at threshold 403 for seeds 0–2. Thresholds 20, 55, 148, 403 and 512 each completed at seed 0. **Threshold 7 was attempted but interrupted twice**, with no score recorded.
- The original OptIForest code may emit an `invalid value encountered in scalar divide` warning. Its numerical implications have **not** been fully audited. Low-threshold settings may be very slow; see the saved times before rerunning.

## Manuscript and figures

`report/` contains the report LaTeX source, its bibliography and figure PDFs, plus an archived report PDF. The **submitted** report should be the final PDF exported from the official ACL Overleaf template with the public GitHub URL in its footnote and reproducibility statement. The archived PDF in this repository should not be assumed to be the latest submitted version unless it has been separately checked.

## Attribution and research integrity

The OptIForest algorithm and `OptIForest-main/` are **third-party work**, not original student-written code; see upstream https://github.com/xiagll/OptIForest and `PATCH_NOTES.md`. Our Python wrappers, synthetic generator and result summaries are research artifacts developed for this class project. The two NumPy `np.mat` to `np.asmatrix` compatibility adjustments are described in `PATCH_NOTES.md`; no claim is made that every path has been fully regression-tested. No personal records, API keys or private data should be committed to this public repository.

## Fast independent smoke check (no expensive OptIForest training)

Run `python scripts/smoke_check.py` to check all nine synthetic datasets, the original AD archive checksum, an iForest baseline and a very small OptIForest execution without overwriting archived experiment results. The OptIForest numerical warning remains unresolved.

From the repository root, after installing requirements and extracting the AD archive:

```bash
python scripts/validate_transactions.py data/generated/transactions_point_p05_seed42.csv
python scripts/run_baseline.py --data data/generated/transactions_point_p01_seed42.csv --runs 1 --output results/smoke_iforest.csv
```

The first command should print `PASS`. The second should write a new CSV with `seed`, `auc_roc`, `average_precision`, `fit_seconds`, and `score_seconds`. Results in `results/smoke_iforest.csv` are smoke-test artifacts, not part of the manuscript's reported 15-seed experiments. For a more thorough test, use the original commands above.

**Scope:** The compressed original AD benchmark is included, but the upstream licence/dataset redistribution permissions have not been independently verified. Original source: https://github.com/xiagll/OptIForest.
