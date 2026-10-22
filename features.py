"""Causal daily-bar features, adapted from the archived first-paper scaffold."""
from typing import Tuple
import numpy as np
import pandas as pd

def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)


def make_features(raw: pd.DataFrame, horizon: int, annualization: int = 252) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = raw.copy().sort_index()
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    close, open_, high, low, volume = (df[c].astype(float) for c in ["Close", "Open", "High", "Low", "Volume"])
    ret = np.log(close).diff()
    annual = np.sqrt(float(annualization))
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

