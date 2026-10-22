"""Check arithmetic of rounded published summaries, without claiming reproduction."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    with (ROOT / "reported/paper_metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    calculated = []
    for r in rows:
        lgb = float(r["mae_lightgbm"])
        annual = 365 if r["asset"] == "BTC-USD" else 252
        calculated.append(dict(asset=r["asset"],
            reduction_garch_pct=100 * (1 - lgb / float(r["mae_garch"])),
            reduction_persistence_pct=100 * (1 - lgb / float(r["mae_persistence"])),
            ir_from_rounded_components=float(r["mean_ic"]) / float(r["sd_ic"]) * math.sqrt(annual / 21)))
    return dict(status="arithmetic_only_not_independent_reproduction",
                source="https://doi.org/10.56632/bct.2026.5201", tables=[6, 7, 8],
                rounding_note="Ratios from rounded printed values can differ from printed ratios.",
                mean_ic=sum(float(r["mean_ic"]) for r in rows)/len(rows),
                mean_positive_ic_pct=sum(float(r["positive_ic_pct"]) for r in rows)/len(rows),
                mean_reduction_garch_pct=sum(r["reduction_garch_pct"] for r in calculated)/len(rows),
                mean_reduction_persistence_pct=sum(r["reduction_persistence_pct"] for r in calculated)/len(rows),
                mean_ir_from_rounded_components=sum(r["ir_from_rounded_components"] for r in calculated)/len(rows),
                assets=calculated)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
