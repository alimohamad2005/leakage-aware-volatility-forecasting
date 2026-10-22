# Verification status

## Verified from the supplied manuscript tables

- The cross-asset mean Rank IC is 0.1632 after averaging the six printed asset values, consistent with the reported 0.163.
- The mean positive-fold rate is 74.38%, consistent with 74.4%.
- The arithmetic mean of asset-level MAE reductions is 57.99% versus GARCH and 30.61% versus persistence, consistent with 58.0% and 30.6%.

## Not independently verifiable from the supplied files

- Date-level predictions and losses.
- The 77 equity folds and 131 BTC folds.
- Newey-West statistics.
- Hansen SPA p-values.
- SHAP stability and regime-conditioned statistics.

Those items require the raw OHLCV snapshot and the original date-level/fold-level outputs. The executable pipeline generates the required audit trail, including an LSTM comparator and feature-group ablations. Until it is run on the archived data snapshot, the numerical claims remain manuscript-reported rather than independently reproduced.
