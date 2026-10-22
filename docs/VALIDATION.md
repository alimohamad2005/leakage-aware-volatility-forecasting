# Validation of the initial public package

Validated locally on 22 October 2026 with Python 3.12.14. Direct package versions are recorded in `requirements-tested.txt`.

Seven tests pass:

1. Mutating future prices leaves earlier features unchanged.
2. The target uses exactly the next five returns and the requested calendar factor.
3. Every retained outer training label ends before the test boundary.
4. Duplicate dates and internally inconsistent OHLC bars fail explicitly.
5. Overlapping forecasts are reduced to one origin per asset/model/date under the declared policy.
6. Published summary arithmetic is labeled separately from independent reproduction.
7. All three models execute on synthetic data; inference runs, output boundaries are checked, and a repeated run yields identical predictions.

The integration test uses 450 synthetic bars, three folds, and only 20 bootstrap replications for fast software verification. The normal analysis default remains 10,000 replications. This check cannot establish the accuracy or statistical significance of the article's market-data results.

The 14 archived files match their original Git blob identities. The public package has a separate SHA-256 inventory. Header and method SVGs are rendered locally for visual review. GitHub Actions repeats software checks when the repository receives a push; local success alone is not a claim that hosted CI has passed.
