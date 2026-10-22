# Reproducibility package

This package recomputes the manuscript results from raw daily OHLCV data. It does not treat the values printed in the manuscript as verified inputs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_reproduction.py --config config.json --output outputs
```

If Yahoo Finance download is unavailable, place one CSV per asset in `data/` using names such as `SPY.csv` and `BTC_USD.csv`. Required columns are `Date,Open,High,Low,Close,Volume`.

## Generated evidence

- `outputs/predictions.csv`: date-level actuals, forecasts, losses, models and fold IDs.
- `outputs/fold_metrics.csv`: fold boundaries, sample sizes, MAE and Rank IC.
- `outputs/stationary_bootstrap_tests.csv`: one-sided stationary-bootstrap tests against GARCH, persistence and LSTM.
- LightGBM ablations are named `without_har`, `without_range`, `without_volume`, `without_momentum`, `without_vol_of_vol`, and `without_technical`.

## Reproducibility controls

- Seed: 42.
- Chronological rolling folds only; no random split.
- Five observations are purged from the end of every training window because their forward targets overlap the test boundary.
- Every feature is backward-looking.
- The LSTM scaler is fitted only on the training fold.
- Raw downloaded CSVs are cached and should be archived with checksums.

## Publication archive

This repository is intended to remain **private during peer review and before publication**. Do not include the manuscript source or PDF here at this stage.

Before submission, run the pipeline and commit the generated fold-level outputs. Preserve the raw-data snapshot and its checksums in a restricted archive when redistribution terms permit. After acceptance/publication, create a versioned Zenodo or OSF release, switch the archival access setting to open, and replace the manuscript's repository placeholder with the persistent DOI/URL.
