# Published summaries

`paper_metrics.csv` transcribes rounded values from Tables 6, 7, and 8 of the published BCT article: https://doi.org/10.56632/bct.2026.5201.

These are **paper-reported aggregate values**. They are not date-level predictions, a recovered experiment, or computed outputs from this repository. The six asset rows are retained without an artificial average row. `tools/audit_reported_results.py` computes summary arithmetic from these rounded numbers; small differences from the article's printed ratios can occur.

IR means the annualized fold-IC ratio, using `sqrt(252/21)` for the five exchange-traded instruments and `sqrt(365/21)` for Bitcoin. It is not an investment portfolio information ratio.
