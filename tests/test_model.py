"""
Unit tests for TimesFMForecaster.
Verifies model instantiation, multi-horizon inference, quantile ordering,
and backtest metrics.
"""

import unittest
import numpy as np
from engine.model_wrapper import TimesFMForecaster


class TestTimesFMForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecaster = TimesFMForecaster()

    def test_model_status(self):
        self.assertIsNotNone(self.forecaster.model_status)
        self.assertIn("TimesFM", self.forecaster.model_status)

    def test_forecast_series(self):
        # Generate dummy 100 days of price data
        np.random.seed(42)
        base = 150.0
        prices = (base + np.cumsum(np.random.normal(0.2, 1.5, 120))).tolist()

        res = self.forecaster.forecast_series(prices, horizon=14)
        self.assertEqual(res["horizon"], 14)
        self.assertEqual(len(res["predictions"]), 14)
        self.assertIn("trend", res)
        self.assertIn(res["trend"], ["BULLISH", "BEARISH", "NEUTRAL"])

        # Check quantile ordering: q10 <= q25 <= q50 <= q75 <= q90
        for step in res["predictions"]:
            self.assertLessEqual(step["q10"], step["q25"])
            self.assertLessEqual(step["q25"], step["q50"])
            self.assertLessEqual(step["q50"], step["q75"])
            self.assertLessEqual(step["q75"], step["q90"])

    def test_backtest_evaluation(self):
        np.random.seed(42)
        prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(100)]
        dates = [f"2026-01-{(i%28)+1:02d}" for i in range(100)]

        bt = self.forecaster.run_backtest(prices, dates, test_days=15)
        self.assertEqual(bt["test_days"], 15)
        self.assertTrue(bt["mae"] >= 0)
        self.assertTrue(bt["rmse"] >= 0)
        self.assertTrue(bt["mape"] >= 0)
        self.assertTrue(0 <= bt["directional_accuracy_pct"] <= 100)
        self.assertEqual(len(bt["comparison"]), 15)


if __name__ == "__main__":
    unittest.main()
