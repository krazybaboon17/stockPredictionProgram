"""
Unit and Integration Tests for Stock Data Engine & TimesFM Model Wrapper.
Authored by: AI Tester
"""

import unittest
import numpy as np
from engine.data_loader import StockDataEngine, POPULAR_TICKERS
from engine.model_wrapper import TimesFMForecaster


class TestStockDataEngine(unittest.TestCase):
    def setUp(self):
        self.engine = StockDataEngine()

    def test_popular_tickers_structure(self):
        self.assertGreaterEqual(len(POPULAR_TICKERS), 5)
        for t in POPULAR_TICKERS:
            self.assertIn("symbol", t)
            self.assertIn("name", t)
            self.assertIn("sector", t)

    def test_get_stock_data_valid_ticker(self):
        data = self.engine.get_stock_data("AAPL", period="1y")
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn("history", data)
        self.assertGreaterEqual(len(data["history"]), 30)

        latest = data["history"][-1]
        for key in ["date", "open", "high", "low", "close", "volume", "sma_20", "rsi", "macd", "bb_upper", "bb_lower"]:
            self.assertIn(key, latest, f"Missing indicator key: {key}")

    def test_technical_indicators_sanity(self):
        data = self.engine.get_stock_data("NVDA", period="1y")
        history = data["history"]
        latest = history[-1]

        # RSI must be bounded between 0 and 100
        self.assertGreaterEqual(latest["rsi"], 0.0)
        self.assertLessEqual(latest["rsi"], 100.0)

        # Bollinger Bands upper >= lower
        self.assertGreaterEqual(latest["bb_upper"], latest["bb_lower"])

        # High >= Low
        self.assertGreaterEqual(latest["high"], latest["low"])

    def test_search_tickers(self):
        res = self.engine.search_tickers("apple")
        self.assertTrue(any(t["symbol"] == "AAPL" for t in res))

        res_custom = self.engine.search_tickers("CUSTOMX")
        self.assertTrue(any("CUSTOMX" in t["symbol"] for t in res_custom))


class TestTimesFMForecaster(unittest.TestCase):
    def setUp(self):
        self.forecaster = TimesFMForecaster()

    def test_singleton_instance(self):
        f2 = TimesFMForecaster()
        self.assertIs(self.forecaster, f2)

    def test_forecast_series_quantiles(self):
        # 60 days of synthetic prices
        base = 150.0
        prices = [base * (1 + 0.005 * i + 0.01 * np.sin(i)) for i in range(60)]
        horizon = 14

        res = self.forecaster.forecast_series(prices, horizon=horizon)
        self.assertIn("predictions", res)
        self.assertEqual(len(res["predictions"]), horizon)
        self.assertIn("trend", res)
        self.assertIn(res["trend"], ["BULLISH", "BEARISH", "NEUTRAL"])

        # Validate non-crossing quantile bounds: q10 <= q25 <= q50 <= q75 <= q90
        for p in res["predictions"]:
            self.assertLessEqual(p["q10"], p["q25"] + 1e-5)
            self.assertLessEqual(p["q25"], p["q50"] + 1e-5)
            self.assertLessEqual(p["q50"], p["q75"] + 1e-5)
            self.assertLessEqual(p["q75"], p["q90"] + 1e-5)
            self.assertGreater(p["q10"], 0)

    def test_backtest_execution(self):
        prices = [100.0 + i * 0.5 + 2.0 * np.sin(i / 3.0) for i in range(80)]
        dates = [f"2026-01-{(i % 28) + 1:02d}" for i in range(80)]

        bt = self.forecaster.run_backtest(prices, dates, test_days=10)
        self.assertIn("mae", bt)
        self.assertIn("rmse", bt)
        self.assertIn("mape", bt)
        self.assertIn("directional_accuracy_pct", bt)
        self.assertEqual(len(bt["comparison"]), 10)
        self.assertGreaterEqual(bt["mae"], 0.0)


class TestRiskEngine(unittest.TestCase):
    def test_calculate_risk_metrics(self):
        from engine.risk_engine import RiskEngine
        current_price = 150.0
        hist_prices = [140.0 + i * 0.2 for i in range(50)]
        forecast_predictions = [
            {"date": f"2026-11-{i+1:02d}", "predicted": 150.0 + i * 0.5, "q10": 145.0 + i * 0.3}
            for i in range(14)
        ]
        metrics = RiskEngine.calculate_risk_metrics(current_price, hist_prices, forecast_predictions)
        self.assertIn("var_95_pct", metrics)
        self.assertIn("cvar_95_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("risk_tier", metrics)
        self.assertTrue(metrics["var_95_pct"] >= 0)
        self.assertTrue(metrics["cvar_95_pct"] >= metrics["var_95_pct"])


if __name__ == "__main__":
    unittest.main()

