# Reproducibility status

**Current status: a documented research companion with a tested reconstruction. Exact recovery of the original experiment is pending.**

## Evidence inventory

| Item | Available | Meaning |
| --- | --- | --- |
| Published article and DOI | Yes, journal links | Primary source for title, authors, methods, and reported tables |
| Historical code/configuration | Yes, immutable snapshot | A recoverable scaffold associated with the first-paper settings |
| Original raw market-data snapshot | No | Contemporary downloads cannot recover an identical input history |
| Original date-level forecasts/losses | No | Archived `predictions.csv` contains a header only |
| Original fold metrics | No | Archived `fold_metrics.csv` contains a header only |
| Original bootstrap outputs | No | Archived bootstrap CSV contains a header only |
| Published rounded summaries | Yes | Transcribed into `reported/paper_metrics.csv` |
| Arithmetic audit of summaries | Yes | Ratios and means can be checked without fitting a model |
| Software validation | Yes | Unit contracts and a deterministic synthetic three-model integration run |
| Exact numerical reproduction | Pending | Requires original input/output evidence and reconciliation |

## Reconstruction versus archive

The public runner is a new reconstruction derived from the historical feature code. It is deliberately separate from the unmodified snapshot. It uses the six original instruments, the original outer window settings, asset-specific annualization, chronological inner validation with label purging, and three published comparator families.

The historical scaffold has known differences from the final article: 252-day annualization for Bitcoin, no implemented inner early stopping, GARCH state updates that skip the purge interval, optional LSTM/ablation runs outside the final paper's scope, and a pairwise centered bootstrap labeled “SPA-like” rather than Hansen's SPA. The public runner addresses these software issues; changing them means the runner cannot be represented as the exact original experiment.

Original inner-validation length, stopping iterations, input snapshots, detailed GARCH conventions, complete bootstrap configuration, and date-level forecasts remain unrecovered. The public defaults and inference conventions are documented in `METHOD.md`; they are explicit reconstruction choices.

## How to close the gap

1. Recover the original six input histories, including source/adjustment metadata and hashes.
2. Recover original per-date predictions and complete fold boundaries if available.
3. Run the public pipeline against frozen inputs; archive configuration and installed environment.
4. Reconcile target alignment, annualization, folds, model selection, loss aggregation, IC, and inference with the published tables.
5. Publish populated evidence with clear provenance before claiming an exact numerical reproduction.

Synthetic tests do not validate the published 58.0% improvement or any published significance claim. No new market-data result has been asserted in this initial public package.
