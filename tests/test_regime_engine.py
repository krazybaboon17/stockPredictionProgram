"""
Unit and integration tests for RegimeEngine and /api/regime/{ticker}.
"""

import unittest
from engine.regime_engine import RegimeEngine
from engine.data_loader import StockDataEngine


class TestRegimeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_engine = StockDataEngine()

    def test_classify_regime_valid_ticker(self):
        data = self.data_engine.get_stock_data("NVDA", period="1y")
        res = RegimeEngine.classify_regime(data)

        self.assertEqual(res["ticker"], "NVDA")
        self.assertIn("regime", res)
        self.assertIn(res["regime"], [
            "Bullish Expansion",
            "Bearish Breakdown",
            "Overbought Exhaustion",
            "Squeeze Consolidation"
        ])
        self.assertIn("regime_color", res)
        self.assertTrue(res["regime_color"].startswith("#"))
        self.assertTrue(0.0 <= res["confidence_score"] <= 1.0)
        self.assertIn("metrics", res)
        self.assertIn("rsi", res["metrics"])
        self.assertIn("macd_hist", res["metrics"])
        self.assertIn("bb_width_pct", res["metrics"])
        self.assertIn("trend_direction", res["metrics"])
        self.assertIn("summary", res)

    def test_insufficient_history_fallback(self):
        sparse_data = {
            "ticker": "SPARSE",
            "current_price": 100.0,
            "history": [{"date": "2026-10-01", "close": 100.0}]
        }
        res = RegimeEngine.classify_regime(sparse_data)
        self.assertEqual(res["regime"], "Squeeze Consolidation")
        self.assertEqual(res["confidence_score"], 0.50)

    def test_empty_data_handling(self):
        res = RegimeEngine.classify_regime({})
        self.assertEqual(res["ticker"], "UNKNOWN")
        self.assertEqual(res["regime"], "Squeeze Consolidation")


if __name__ == "__main__":
    unittest.main()
