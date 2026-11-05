# Original AD benchmark: setup and provenance

The original OptIForest AD benchmark comes from the upstream public project:
https://github.com/xiagll/OptIForest

This GitHub repository includes a compressed copy at `OptIForest-main/data/ad.csv.zip`. Uncompressed file size is **45,919,116 bytes**, which exceeds the GitHub web-upload limit.

## Unpack (required before running AD experiments)

From the repository root:

```bash
python -m zipfile -e OptIForest-main/data/ad.csv.zip OptIForest-main/data
```

The resulting dataset must exist at:

`OptIForest-main/data/ad.csv`

If using Windows File Explorer, right-click the ZIP and choose Extract All, then make sure the CSV is placed directly in `OptIForest-main/data/`.

## Verify

- File name: `ad.csv`
- Size: **45,919,116 bytes**
- SHA-256: **`7dfd4264c9aa385ed998df2f4c2eb7a175c0aed816eff495d17933f31df8e832`**
- Expected data: **3,279 observations, 1,555 numeric features, and a binary class label**.

## Attribution / usage

The file is an existing original OptIForest benchmark, not synthetic research data created for this assignment. It is attributed to the original authors and their repository. Review upstream rights and any relevant benchmark terms before redistribution; if needed, download the same source file directly from upstream rather than using this copy.
