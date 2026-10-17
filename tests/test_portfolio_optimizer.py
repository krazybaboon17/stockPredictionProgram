"""
Unit tests for PortfolioOptimizer and Markowitz Efficient Frontier.
"""

import unittest
import numpy as np
from engine.portfolio_optimizer import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.04)
        np.random.seed(123)
        # Create synthetic price series for 3 assets
        days = 100
        t1 = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, days)))
        t2 = 50.0 * np.exp(np.cumsum(np.random.normal(0.0008, 0.012, days)))
        t3 = 200.0 * np.exp(np.cumsum(np.random.normal(0.0015, 0.020, days)))

        self.price_series = {
            "NVDA": t1.tolist(),
            "AAPL": t2.tolist(),
            "MSFT": t3.tolist()
        }
        self.forecasted_returns = {
            "NVDA": 0.25,
            "AAPL": 0.15,
            "MSFT": 0.20
        }

    def test_portfolio_optimization_three_assets(self):
        results = self.optimizer.optimize_portfolio(
            tickers=["NVDA", "AAPL", "MSFT"],
            price_series=self.price_series,
            forecasted_returns=self.forecasted_returns
        )

        self.assertEqual(len(results["tickers"]), 3)
        self.assertIn("max_sharpe", results)
        self.assertIn("min_volatility", results)
        self.assertIn("equal_weight", results)
        self.assertIn("frontier_points", results)

        # Check weights sum to ~1.0
        max_sharpe_weights = results["max_sharpe"]["weights"]
        total_w = sum(max_sharpe_weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

        # Check Sharpe ratio ordering
        self.assertGreaterEqual(
            results["max_sharpe"]["sharpe_ratio"],
            results["min_volatility"]["sharpe_ratio"] - 0.1
        )

        # Check minimum volatility property
        self.assertLessEqual(
            results["min_volatility"]["annualized_volatility_pct"],
            results["max_sharpe"]["annualized_volatility_pct"] + 0.1
        )

    def test_single_asset_fallback(self):
        single_price = {"NVDA": self.price_series["NVDA"]}
        single_ret = {"NVDA": 0.25}
        results = self.optimizer.optimize_portfolio(
            tickers=["NVDA"],
            price_series=single_price,
            forecasted_returns=single_ret
        )
        self.assertEqual(results["tickers"], ["NVDA"])
        self.assertEqual(results["max_sharpe"]["weights"]["NVDA"], 1.0)


if __name__ == "__main__":
    unittest.main()
