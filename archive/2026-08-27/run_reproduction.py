#!/usr/bin/env python3
"""Leakage-aware volatility-forecasting reproduction pipeline.

The script downloads adjusted daily OHLCV data, constructs causal features,
runs purged walk-forward evaluation, and writes date-level predictions and
fold-level metrics. Optional LSTM and ablation runs are controlled by config.
No manuscript result is hard-coded into the computed outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


FEATURE_GROUPS: Dict[str, List[str]] = {
    "har": ["har_daily", "har_weekly", "har_monthly"],
    "range": ["high_low_ratio", "close_open_ratio", "overnight_gap", "parkinson_vol"],
    "volume": ["volume_ma_ratio", "volume_trend", "volume_rv_cross"],
    "momentum": ["mom_5", "mom_10", "mom_21", "rsi_5", "rsi_10", "rsi_21"],
    "vol_of_vol": ["vol_of_vol", "vol_of_vol_norm"],
    "technical": ["price_ma_ratio_5", "price_ma_ratio_10", "price_ma_ratio_21", "price_ma_ratio_50", "bb_width"],
}


def make_features(raw: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = raw.copy().sort_index()
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    close, open_, high, low, volume = (df[c].astype(float) for c in ["Close", "Open", "High", "Low", "Volume"])
    ret = np.log(close).diff()
    annual = np.sqrt(252.0)
    x = pd.DataFrame(index=df.index)
    for w in [5, 10, 21, 42, 63]:
        x[f"rv_{w}"] = ret.rolling(w).std(ddof=1) * annual
    x["har_daily"] = x["rv_5"].shift(1)
    x["har_weekly"] = x["rv_21"].shift(1)
    x["har_monthly"] = x["rv_63"].shift(1)
    x["rv_skew_21"] = ret.rolling(21).skew()
    x["rv_kurt_21"] = ret.rolling(21).kurt()
    x["high_low_ratio"] = high / low
    x["close_open_ratio"] = close / open_
    x["overnight_gap"] = (open_ / close.shift(1) - 1).abs()
    x["parkinson_vol"] = np.sqrt(np.log(high / low).pow(2) / (4 * np.log(2))) * annual
    x["volume_ma_ratio"] = volume / volume.rolling(21).mean()
    x["volume_trend"] = volume / volume.shift(1)
    x["volume_rv_cross"] = x["volume_ma_ratio"] * x["rv_21"]
    for w in [5, 10, 21]:
        x[f"mom_{w}"] = close.pct_change(w)
        x[f"rsi_{w}"] = rsi(close, w)
    x["vol_of_vol"] = x["rv_21"].rolling(63).std(ddof=1)
    x["vol_of_vol_norm"] = x["vol_of_vol"] / x["rv_21"]
    for w in [5, 10, 21, 50]:
        x[f"price_ma_ratio_{w}"] = close / close.rolling(w).mean()
    x["bb_width"] = 2 * close.rolling(20).std(ddof=1) / close.rolling(20).mean()
    for base in [5, 21]:
        for lag in [1, 2, 5]:
            x[f"rv_{base}_lag{lag}"] = x[f"rv_{base}"].shift(lag)

    # y_t uses only r_{t+1},...,r_{t+H}; the reverse/rolling/reverse form
    # aligns the future window with date t without leaking it into features.
    future_rv = ret.shift(-1).iloc[::-1].rolling(horizon).std(ddof=1).iloc[::-1] * annual
    naive = ret.rolling(horizon).std(ddof=1) * annual
    return x.replace([np.inf, -np.inf], np.nan), future_rv, naive


def load_data(ticker: str, start: str, end: str, data_dir: Path) -> pd.DataFrame:
    local = data_dir / f"{ticker.replace('-', '_')}.csv"
    if local.exists():
        df = pd.read_csv(local, index_col=0, parse_dates=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(f"{local} is absent and yfinance is not installed") from exc
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    data_dir.mkdir(parents=True, exist_ok=True)
    df[["Open", "High", "Low", "Close", "Volume"]].to_csv(local)
    return df[["Open", "High", "Low", "Close", "Volume"]]


@dataclass
class Fold:
    fold: int
    train: np.ndarray
    test: np.ndarray


def folds(n: int, train_window: int, test_window: int, step: int, purge: int) -> Iterable[Fold]:
    k = 0
    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        yield Fold(k, np.arange(start, train_end - purge), np.arange(train_end, train_end + test_window))
        start += step
        k += 1


def lgb_model(params: dict, seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=1, **params)


def lstm_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, cfg: dict, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    set_seed(seed)
    seq = int(cfg["sequence_length"])
    if len(x_train) <= seq or len(x_test) <= seq:
        return np.full(len(x_test), np.nan)
    mu, sd = x_train.mean(0), x_train.std(0)
    sd[sd == 0] = 1
    tr = (x_train - mu) / sd
    combined = np.vstack([x_train[-seq:], x_test])
    te = (combined - mu) / sd

    def windows(a: np.ndarray, targets: np.ndarray | None = None):
        xx = np.stack([a[i - seq:i] for i in range(seq, len(a))])
        if targets is None:
            return xx
        return xx, targets[seq:]

    xw, yw = windows(tr, y_train)
    xt = windows(te)
    device = torch.device("cpu")
    tx = torch.tensor(xw, dtype=torch.float32, device=device)
    ty = torch.tensor(yw[:, None], dtype=torch.float32, device=device)
    vx = torch.tensor(xt, dtype=torch.float32, device=device)

    class Net(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.rnn = nn.LSTM(n_features, int(cfg["hidden_size"]), batch_first=True)
            self.out = nn.Linear(int(cfg["hidden_size"]), 1)

        def forward(self, z):
            z, _ = self.rnn(z)
            return self.out(z[:, -1])

    net = Net(tx.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.HuberLoss()
    net.train()
    for _ in range(int(cfg["epochs"])):
        order = torch.randperm(len(tx))
        for s in range(0, len(tx), int(cfg["batch_size"])):
            idx = order[s:s + int(cfg["batch_size"])]
            opt.zero_grad(); loss = loss_fn(net(tx[idx]), ty[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(vx).cpu().numpy().ravel()


def garch_filter_forecast(train_returns: np.ndarray, test_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Fit Student-t GARCH(1,1), then update variance causally through test dates."""
    from arch import arch_model

    scaled = 100 * train_returns[np.isfinite(train_returns)]
    fit = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    omega, alpha, beta = (float(fit.params[k]) for k in ["omega", "alpha[1]", "beta[1]"])
    variance = float(fit.conditional_volatility[-1] ** 2)
    previous = float(scaled[-1])
    out = []
    for r in test_returns:
        variance = omega + alpha * previous ** 2 + beta * variance
        path, v = [], variance
        for _ in range(horizon):
            path.append(v)
            v = omega + (alpha + beta) * v
        out.append(np.sqrt(np.mean(path)) / 100 * np.sqrt(252))
        if np.isfinite(r):
            previous = 100 * float(r)
    return np.asarray(out)


def stationary_bootstrap_p(loss_advantage: np.ndarray, reps: int, mean_block: int, seed: int) -> float:
    d = np.asarray(loss_advantage, float)
    d = d[np.isfinite(d)]
    observed = d.mean()
    centered = d - observed
    rng = np.random.default_rng(seed)
    n, p = len(d), 1 / mean_block
    boot = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, dtype=int); idx[0] = rng.integers(n)
        for i in range(1, n):
            idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
        boot[b] = centered[idx].mean()
    return float((1 + np.sum(boot >= observed)) / (reps + 1))


def evaluate(cfg: dict, output: Path) -> None:
    seed = int(cfg["seed"]); set_seed(seed)
    prediction_rows, fold_rows = [], []
    for ticker in cfg["assets"]:
        raw = load_data(ticker, cfg["start"], cfg["end"], Path(cfg["data_dir"]))
        x, y, naive = make_features(raw, int(cfg["horizon"]))
        returns = np.log(raw["Close"]).diff().reindex(x.index)
        valid_features = x.notna().all(axis=1)
        x = x.loc[valid_features]; y = y.reindex(x.index); naive = naive.reindex(x.index); returns = returns.reindex(x.index)
        variants = {"full": list(x.columns)}
        for group, columns in FEATURE_GROUPS.items():
            variants[f"without_{group}"] = [c for c in x.columns if c not in columns]

        for fold in folds(len(x), int(cfg["train_window"]), int(cfg["test_window"]), int(cfg["step"]), int(cfg["horizon"])):
            train_ok = np.isfinite(y.iloc[fold.train].to_numpy())
            test_ok = np.isfinite(y.iloc[fold.test].to_numpy())
            train_idx = fold.train[train_ok]; test_idx = fold.test[test_ok]
            if len(test_idx) < 10:
                continue
            actual = y.iloc[test_idx].to_numpy()
            base_pred = naive.iloc[test_idx].to_numpy()
            garch_pred = garch_filter_forecast(returns.iloc[train_idx].to_numpy(), returns.iloc[test_idx].to_numpy(), int(cfg["horizon"]))
            model_predictions: Dict[str, np.ndarray] = {"naive": base_pred, "garch": garch_pred}
            for name, columns in variants.items():
                model = lgb_model(cfg["lightgbm"], seed + fold.fold)
                model.fit(x.iloc[train_idx][columns], y.iloc[train_idx])
                model_predictions[f"lightgbm_{name}"] = model.predict(x.iloc[test_idx][columns])
            if cfg.get("run_lstm", True):
                model_predictions["lstm"] = lstm_predict(
                    x.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy(),
                    x.iloc[test_idx].to_numpy(), cfg["lstm"], seed + fold.fold,
                )
            for model_name, pred in model_predictions.items():
                ok = np.isfinite(actual) & np.isfinite(pred)
                mae = float(np.mean(np.abs(actual[ok] - pred[ok])))
                ic = float(spearmanr(actual[ok], pred[ok]).statistic) if ok.sum() > 2 else np.nan
                fold_rows.append({"asset": ticker, "fold": fold.fold, "model": model_name,
                                  "train_end": x.index[train_idx[-1]], "test_start": x.index[test_idx[0]],
                                  "test_end": x.index[test_idx[-1]], "n_test": int(ok.sum()), "mae": mae, "rank_ic": ic})
                for date, truth, forecast in zip(x.index[test_idx][ok], actual[ok], pred[ok]):
                    prediction_rows.append({"date": date, "asset": ticker, "fold": fold.fold,
                                            "model": model_name, "actual": truth, "prediction": forecast,
                                            "absolute_error": abs(truth - forecast)})

    output.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(prediction_rows)
    fm = pd.DataFrame(fold_rows)
    pred.to_csv(output / "predictions.csv", index=False)
    fm.to_csv(output / "fold_metrics.csv", index=False)
    if not pred.empty:
        tests = []
        for asset in pred.asset.unique():
            wide = pred[pred.asset == asset].pivot_table(index=["date", "fold"], columns="model", values="absolute_error", aggfunc="mean")
            for benchmark in ["garch", "naive", "lstm"]:
                if benchmark in wide and "lightgbm_full" in wide:
                    d = wide[benchmark] - wide["lightgbm_full"]
                    tests.append({"asset": asset, "benchmark": benchmark, "n": d.notna().sum(),
                                  "mean_loss_advantage": d.mean(),
                                  "spa_like_stationary_bootstrap_p": stationary_bootstrap_p(d.to_numpy(), int(cfg["spa_reps"]), int(cfg["spa_mean_block"]), seed)})
        pd.DataFrame(tests).to_csv(output / "stationary_bootstrap_tests.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    evaluate(cfg, Path(args.output))


if __name__ == "__main__":
    main()
