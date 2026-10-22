import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from features import make_features
from run_reproduction import folds, unique_predictions, validate_data
from tools.audit_reported_results import audit
from tools.smoke_run import run, synthetic


class ResearchContracts(unittest.TestCase):
    def test_future_prices_do_not_change_past_features(self):
        raw = synthetic()
        changed = raw.copy()
        changed.iloc[300:, :4] *= 1.8
        x, y, _ = make_features(raw, 5)
        xx, yy, _ = make_features(changed, 5)
        pd.testing.assert_frame_equal(x.iloc[:300], xx.iloc[:300])
        pd.testing.assert_series_equal(y.iloc[:295], yy.iloc[:295])
        self.assertNotAlmostEqual(y.iloc[299], yy.iloc[299])
        self.assertEqual(len(x.columns), 36)

    def test_forward_target_exact_support_and_calendar(self):
        raw = synthetic()
        _, y, naive = make_features(raw, 5, 365)
        ret = np.log(raw.Close).diff()
        self.assertAlmostEqual(y.iloc[150], ret.iloc[151:156].std(ddof=1) * np.sqrt(365))
        self.assertAlmostEqual(naive.iloc[150], ret.iloc[146:151].std(ddof=1) * np.sqrt(365))
        self.assertTrue(y.iloc[-5:].isna().all())

    def test_every_training_target_is_before_test(self):
        f = list(folds(1000, 756, 63, 21, 5))
        self.assertEqual(len(f), 9)
        for _, train, test in f:
            self.assertLess(train[-1] + 5, test[0])
            self.assertEqual(len(train), 751)

    def test_invalid_bars_and_duplicate_dates_fail(self):
        raw = synthetic()
        duplicate = pd.concat([raw.iloc[:1], raw])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_data(duplicate)
        raw.iloc[20, raw.columns.get_loc("Low")] = raw.Close.iloc[20] * 2
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_data(raw)

    def test_overlapping_fold_losses_have_one_origin(self):
        df = pd.DataFrame([dict(asset="SPY", date="2020-01-01", model="lightgbm", fold=2, prediction=2),
                           dict(asset="SPY", date="2020-01-01", model="lightgbm", fold=1, prediction=1)])
        out = unique_predictions(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0].prediction, 1)

    def test_published_arithmetic_is_separate_from_reproduction(self):
        a = audit()
        self.assertIn("not_independent_reproduction", a["status"])
        self.assertAlmostEqual(a["mean_ic"], 0.1631666666667)
        self.assertAlmostEqual(a["mean_reduction_garch_pct"], 58.0, delta=0.05)
        self.assertAlmostEqual(a["mean_reduction_persistence_pct"], 30.6, delta=0.05)


class SyntheticIntegration(unittest.TestCase):
    def test_three_models_and_inference_on_synthetic_data(self):
        from analyze_outputs import analyze
        with tempfile.TemporaryDirectory() as folder:
            out = run(Path(folder))
            pred = pd.read_csv(out / "predictions.csv")
            metrics = pd.read_csv(out / "fold_metrics.csv")
            self.assertEqual(set(pred.model), {"lightgbm", "garch", "persistence"})
            self.assertTrue(np.isfinite(pred[["actual", "prediction", "absolute_error"]]).all().all())
            self.assertEqual(pred.groupby(["fold", "model"]).size().nunique(), 1)
            self.assertTrue((pd.to_datetime(metrics.train_label_end) <= pd.to_datetime(metrics.information_cutoff)).all())
            self.assertIn("not_original", json.loads((out / "run_metadata.json").read_text())["status"])
            analyze(out, reps=20, block_size=5)
            stats = pd.read_csv(out / "spa_tests.csv")
            self.assertEqual(len(stats), 2)
            self.assertTrue(stats.spa_p_consistent.between(0, 1).all())
            first = pred.copy()
            run(Path(folder))
            pd.testing.assert_frame_equal(first, pd.read_csv(out / "predictions.csv"))


if __name__ == "__main__":
    unittest.main()
