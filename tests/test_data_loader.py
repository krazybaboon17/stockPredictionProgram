"""
Unit tests for StockDataEngine.
Verifies data fetching, indicator generation, and fallback behavior.
"""

import unittest
from engine.data_loader import StockDataEngine, POPULAR_TICKERS


class TestStockDataEngine(unittest.TestCase):
    def setUp(self):
        self.engine = StockDataEngine(cache_ttl_seconds=60)

    def test_popular_tickers_list(self):
        self.assertTrue(len(POPULAR_TICKERS) >= 5)
        symbols = [t["symbol"] for t in POPULAR_TICKERS]
        self.assertIn("NVDA", symbols)
        self.assertIn("AAPL", symbols)

    def test_search_tickers(self):
        results = self.engine.search_tickers("NVDA")
        self.assertTrue(any(r["symbol"] == "NVDA" for r in results))

        results_broad = self.engine.search_tickers("Tech")
        self.assertTrue(len(results_broad) > 0)

    def test_stock_data_structure(self):
        data = self.engine.get_stock_data("NVDA", period="1y")
        self.assertEqual(data["ticker"], "NVDA")
        self.assertIn("current_price", data)
        self.assertIn("history", data)
        self.assertTrue(len(data["history"]) >= 30)

        # Check technical indicator keys in history
        latest = data["history"][-1]
        for key in ["date", "open", "high", "low", "close", "volume", "sma_20", "sma_50", "rsi", "macd", "bb_upper", "bb_lower"]:
            self.assertIn(key, latest)

    def test_rsi_bounds(self):
        data = self.engine.get_stock_data("AAPL", period="1y")
        for point in data["history"]:
            self.assertTrue(0 <= point["rsi"] <= 100, f"RSI out of bounds: {point['rsi']}")


if __name__ == "__main__":
    unittest.main()
