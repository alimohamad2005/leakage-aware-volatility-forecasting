"""Run all three models on synthetic bars; no market-data claims are made."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from run_reproduction import evaluate


def synthetic(n=450):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    return pd.DataFrame(dict(Open=close * 0.998, High=close * 1.015,
                             Low=close * 0.985, Close=close,
                             Volume=rng.integers(10000, 100000, n)),
                        index=pd.date_range("2020-01-01", periods=n, freq="B", name="Date"))


def run(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    synthetic().to_csv(base / "SPY.csv")
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg.update(assets=["SPY"], start="2020-01-01", end="2023-01-01", data_dir=str(base),
               train_window=220, test_window=42, step=42, inner_val_window=42)
    cfg["lightgbm"].update(n_estimators=20, min_child_samples=10)
    evaluate(cfg, base / "results")
    return base / "results"


if __name__ == "__main__":
    print("Synthetic validation only. Results do not reproduce the article.")
    run(ROOT / "tmp/smoke")
