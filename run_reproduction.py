#!/usr/bin/env python3
"""Public reconstruction for the BCT article; not its missing original run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import make_features

ASSETS = ("SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "BTC-USD")
REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def validate_data(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans:
        raise ValueError("Data must have a valid DatetimeIndex")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted ascending")
    if set(REQUIRED) - set(raw.columns):
        raise ValueError("Required columns: Date,Open,High,Low,Close,Volume")
    raw = raw[REQUIRED].astype(float)
    if raw.empty or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Data must be nonempty and finite; missing bars are not imputed")
    if (raw[REQUIRED[:4]] <= 0).any().any() or (raw.Volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((raw.High < raw[["Open", "Close", "Low"]].max(axis=1)) |
            (raw.Low > raw[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("OHLC bars are internally inconsistent")
    return raw


def load_data(ticker: str, cfg: dict) -> tuple[pd.DataFrame, Path]:
    path = Path(cfg["data_dir"]) / f"{ticker.replace('-', '_')}.csv"
    if not path.exists():
        if not cfg.get("allow_download", False):
            raise FileNotFoundError(f"Supply {path}; see data/README.md. Downloads require --download.")
        import yfinance as yf
        raw = yf.download(ticker, start=cfg["start"], end=cfg["end"],
                          auto_adjust=False, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = validate_data(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(path, index_label="Date")
    raw = pd.read_csv(path, index_col="Date", parse_dates=True)
    raw = validate_data(raw)
    raw = raw.loc[(raw.index >= cfg["start"]) & (raw.index < cfg["end"])]
    if raw.empty:
        raise ValueError(f"No rows in requested interval for {ticker}")
    return raw, path


def folds(n: int, train: int, test: int, step: int, horizon: int):
    if min(train, test, step, horizon) <= 0 or train <= horizon:
        raise ValueError("Positive windows and train > horizon are required")
    for k, start in enumerate(range(0, n - train - test + 1, step)):
        boundary = start + train
        yield k, np.arange(start, boundary - horizon), np.arange(boundary, boundary + test)


def select_lgb(x, y, cfg, horizon, seed):
    from lightgbm import LGBMRegressor, early_stopping
    params = dict(cfg["lightgbm"])
    common = dict(random_state=seed, verbosity=-1, n_jobs=1, deterministic=True, force_col_wise=True)
    v = int(cfg["inner_val_window"])
    boundary = len(x) - v
    if boundary - horizon < 20 or v < 5:
        raise ValueError("Insufficient observations for purged inner validation")
    selector = LGBMRegressor(**params, **common)
    selector.fit(x.iloc[:boundary - horizon], y.iloc[:boundary - horizon],
                 eval_set=[(x.iloc[boundary:], y.iloc[boundary:])],
                 callbacks=[early_stopping(int(cfg["early_stopping_rounds"]), verbose=False)])
    params["n_estimators"] = int(selector.best_iteration_ or params["n_estimators"])
    final = LGBMRegressor(**params, **common).fit(x, y)
    return final, params["n_estimators"]


def garch_predictions(train_returns, update_returns, horizon, annualization):
    """Fit through the prior cutoff, update with r_t, then forecast t+1...t+H."""
    from arch import arch_model
    values = np.asarray(train_returns, dtype=float) * 100
    fit = arch_model(values, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    if fit.convergence_flag:
        raise RuntimeError("GARCH optimization did not converge")
    omega, alpha, beta = [float(fit.params[p]) for p in ("omega", "alpha[1]", "beta[1]")]
    variance = float(fit.conditional_volatility[-1] ** 2)
    previous = float(values[-1])
    out = []
    for current in np.asarray(update_returns, dtype=float) * 100:
        variance = omega + alpha * previous ** 2 + beta * variance
        next_variance = omega + alpha * current ** 2 + beta * variance
        path = []
        for _ in range(horizon):
            path.append(next_variance)
            next_variance = omega + (alpha + beta) * next_variance
        out.append(np.sqrt(max(float(np.mean(path)), 0)) / 100 * np.sqrt(annualization))
        previous = float(current)
    return np.asarray(out)


def unique_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Retain the earliest fitted fold at each asset/model/origin for date-level loss."""
    return (pred.sort_values(["asset", "date", "model", "fold"])
            .drop_duplicates(["asset", "date", "model"], keep="first"))


def evaluate(cfg: dict, output: Path):
    h = int(cfg["horizon"])
    if h < 2:
        raise ValueError("Sample-standard-deviation target requires horizon >= 2")
    if not cfg["assets"] or set(cfg["assets"]) - set(ASSETS):
        raise ValueError("Use the six assets of the BCT paper")
    predictions, metrics, inputs = [], [], []
    for ticker in cfg["assets"]:
        raw, path = load_data(ticker, cfg)
        annual = 365 if ticker == "BTC-USD" else 252
        features, targets, naive = make_features(raw, h, annual)
        # Trim only the leading warm-up. Gaps inside the feature timeline fail visibly.
        finite = np.isfinite(features.to_numpy()).all(axis=1)
        if not finite.any():
            raise ValueError(f"No valid feature rows for {ticker}")
        start = int(np.flatnonzero(finite)[0])
        if not finite[start:].all():
            raise ValueError(f"Nonfinite interior features for {ticker}; review data and zero-volume bars")
        x = features.iloc[start:]
        y = targets.reindex(x.index)
        ret = np.log(raw.Close).diff()
        ends = pd.Series(raw.index, index=raw.index).shift(-h)
        asset_folds = 0
        for k, train, test in folds(len(x), int(cfg["train_window"]),
                                    int(cfg["test_window"]), int(cfg["step"]), h):
            candidate_test = test
            test = test[np.isfinite(y.iloc[test].to_numpy())]
            if len(test) < 3:
                continue
            cutoff = x.index[candidate_test[0] - 1]
            if not (ends.loc[x.index[train]] <= cutoff).all():
                raise AssertionError("Training label extends beyond information cutoff")
            model, trees = select_lgb(x.iloc[train], y.iloc[train], cfg, h, int(cfg["seed"]) + k)
            gtrain = ret.loc[(ret.index >= x.index[train[0]]) & (ret.index <= cutoff)].dropna()
            update = ret.loc[x.index[candidate_test[0]]:x.index[test[-1]]]
            gp = pd.Series(garch_predictions(gtrain, update, h, annual), index=update.index)
            forecasts = {
                "lightgbm": np.maximum(model.predict(x.iloc[test]), 1e-8),
                "garch": gp.loc[x.index[test]].to_numpy(),
                "persistence": naive.loc[x.index[test]].to_numpy(),
            }
            truth = y.iloc[test].to_numpy()
            for name, values in forecasts.items():
                if not np.isfinite(values).all():
                    raise RuntimeError(f"Nonfinite predictions: {ticker}/{k}/{name}")
                ic = float(spearmanr(truth, values).statistic) if np.ptp(values) > 0 and np.ptp(truth) > 0 else np.nan
                metrics.append(dict(asset=ticker, fold=k, model=name,
                                    train_start=x.index[train[0]], train_end=x.index[train[-1]],
                                    train_label_end=ends.loc[x.index[train[-1]]], information_cutoff=cutoff,
                                    test_start=x.index[test[0]], test_end=x.index[test[-1]],
                                    n_test=len(test), mae=float(np.mean(np.abs(truth - values))),
                                    rank_ic=ic, selected_trees=trees if name == "lightgbm" else None))
                for date, actual, forecast in zip(x.index[test], truth, values):
                    predictions.append(dict(date=date, target_end=ends.loc[date], asset=ticker,
                                            fold=k, model=name, actual=float(actual),
                                            prediction=float(forecast), absolute_error=float(abs(actual - forecast))))
            asset_folds += 1
        if not asset_folds:
            raise ValueError(f"Insufficient rows for one full fold: {ticker}")
        inputs.append(dict(asset=ticker, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           first_date=str(raw.index[0].date()), last_date=str(raw.index[-1].date()),
                           n_rows=len(raw), annualization=annual, folds=asset_folds))
        print(f"{ticker}: {asset_folds} folds, {len(raw)} input bars", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pred, fm = pd.DataFrame(predictions), pd.DataFrame(metrics)
    pred.to_csv(output / "predictions.csv", index=False)
    fm.to_csv(output / "fold_metrics.csv", index=False)
    unique_predictions(pred).groupby(["asset", "model"]).absolute_error.agg(["count", "mean"]).to_csv(output / "unique_date_mae.csv")
    versions = {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "arch", "statsmodels"]}
    meta = dict(status="new_reconstruction_not_original_paper_outputs", config=cfg, inputs=inputs,
                python=platform.python_version(), packages=versions,
                forecast_origin="after_close", evaluation="overlapping_fold_metrics; earliest_fold_per_date_for_date_losses")
    (output / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--output", default="outputs/reconstruction")
    p.add_argument("--download", action="store_true", help="Fetch current vendor data if a local file is absent")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg["allow_download"] = args.download
    evaluate(cfg, Path(args.output))


if __name__ == "__main__":
    main()
