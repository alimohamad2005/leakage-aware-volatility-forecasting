"""Inference for newly reconstructed runs; original paper p-values are not inputs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_reproduction import unique_predictions


def analyze(folder: Path, reps: int = 10000, block_size: int = 10):
    from arch.bootstrap import SPA
    import statsmodels.api as sm
    pred = pd.read_csv(folder / "predictions.csv", parse_dates=["date"])
    fm = pd.read_csv(folder / "fold_metrics.csv")
    unique = unique_predictions(pred)
    spa_rows, nw_rows = [], []
    for asset, group in unique.groupby("asset"):
        wide = group.pivot(index="date", columns="model", values="absolute_error").sort_index()
        for benchmark in ["garch", "persistence"]:
            pair = wide[[benchmark, "lightgbm"]].dropna()
            if len(pair) < 20:
                raise ValueError("At least 20 paired dates are required for inference")
            test = SPA(pair[benchmark].to_numpy(), pair[["lightgbm"]].to_numpy(),
                       reps=reps, block_size=block_size, bootstrap="stationary", studentize=True, seed=42)
            test.compute()
            spa_rows.append(dict(asset=asset, benchmark=benchmark, n_unique_dates=len(pair),
                                 bootstrap_reps=reps, mean_block=block_size,
                                 mean_loss_advantage=float((pair[benchmark] - pair.lightgbm).mean()),
                                 **{f"spa_p_{k}": float(v) for k, v in test.pvalues.items()}))
        ic = fm.loc[(fm.asset == asset) & (fm.model == "lightgbm")].sort_values("fold").rank_ic.dropna()
        if len(ic) >= 3:
            fit = sm.OLS(ic.to_numpy(), np.ones((len(ic), 1))).fit(cov_type="HAC",
                         cov_kwds={"maxlags": min(10, len(ic) - 1)}, use_t=True)
            annual = 365 if asset == "BTC-USD" else 252
            sd = float(ic.std(ddof=1))
            nw_rows.append(dict(asset=asset, folds=len(ic), mean_ic=float(ic.mean()), sd_ic=sd,
                                annualized_fold_ic_ratio=float(ic.mean()/sd*np.sqrt(annual/21)) if sd else None,
                                nw_t=float(fit.tvalues[0]), nw_p_two_sided=float(fit.pvalues[0]),
                                nw_maxlags=min(10, len(ic)-1)))
    pd.DataFrame(spa_rows).to_csv(folder / "spa_tests.csv", index=False)
    pd.DataFrame(nw_rows).to_csv(folder / "nw_ic_tests.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="outputs/reconstruction")
    p.add_argument("--reps", type=int, default=10000)
    args = p.parse_args()
    analyze(Path(args.input), args.reps)
