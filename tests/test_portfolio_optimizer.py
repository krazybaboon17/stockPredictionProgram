"""
Unit and integration tests for PortfolioOptimizer (TASK-PM-006 / TASK-TEST-009).
Validates Markowitz Efficient Frontier optimization, weight bounds, and Sharpe ratio logic.
"""

import unittest
import numpy as np
from engine.portfolio_optimizer import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        # Synthetic 60-day price series for 3 assets
        np.random.seed(42)
        days = 60
        returns_a = np.random.normal(0.001, 0.015, days)
        returns_b = np.random.normal(0.0005, 0.010, days)
        returns_c = np.random.normal(0.002, 0.025, days)

        self.price_history = {
            "NVDA": list(100.0 * np.exp(np.cumsum(returns_a))),
            "AAPL": list(150.0 * np.exp(np.cumsum(returns_b))),
            "MSFT": list(300.0 * np.exp(np.cumsum(returns_c)))
        }

        self.forecasted_returns = {
            "NVDA": 0.25,   # 25% projected return over horizon
            "AAPL": 0.12,   # 12% projected return over horizon
            "MSFT": 0.18    # 18% projected return over horizon
        }
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.045)

    def test_weights_sum_to_one(self):
        result = self.optimizer.optimize_portfolio(
            tickers=["NVDA", "AAPL", "MSFT"],
            price_series=self.price_history,
            forecasted_returns=self.forecasted_returns,
            horizon=30
        )

        weights = result["optimal_portfolio"]["weights"]
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3,
                               msg="Optimal portfolio weights must sum to 1.0")

        min_weights = result["min_variance_portfolio"]["weights"]
        self.assertAlmostEqual(sum(min_weights.values()), 1.0, places=3)

        eq_weights = result["equal_weight_portfolio"]["weights"]
        self.assertAlmostEqual(sum(eq_weights.values()), 1.0, places=3)

    def test_long_only_non_negative_weights(self):
        result = self.optimizer.optimize_portfolio(
            tickers=["NVDA", "AAPL", "MSFT"],
            price_series=self.price_history,
            forecasted_returns=self.forecasted_returns,
            horizon=30
        )

        weights = result["optimal_portfolio"]["weights"]
        for ticker, w in weights.items():
            self.assertGreaterEqual(w, -1e-4, f"Weight for {ticker} cannot be negative: {w}")

    def test_sharpe_ratio_superiority_over_equal_weight(self):
        result = self.optimizer.optimize_portfolio(
            tickers=["NVDA", "AAPL", "MSFT"],
            price_series=self.price_history,
            forecasted_returns=self.forecasted_returns,
            horizon=30
        )

        opt_sharpe = result["optimal_portfolio"]["sharpe_ratio"]
        eq_sharpe = result["equal_weight_portfolio"]["sharpe_ratio"]

        self.assertGreaterEqual(opt_sharpe, eq_sharpe - 1e-2,
                                "Optimized portfolio Sharpe must be >= Equal Weight Sharpe")

    def test_efficient_frontier_structure(self):
        result = self.optimizer.optimize_portfolio(
            tickers=["NVDA", "AAPL", "MSFT"],
            price_series=self.price_history,
            forecasted_returns=self.forecasted_returns,
            horizon=30
        )

        frontier = result.get("frontier", [])
        self.assertTrue(len(frontier) >= 5, "Frontier should contain at least 5 sample points")

        for point in frontier:
            self.assertIn("volatility_pct", point)
            self.assertIn("return_pct", point)
            self.assertIn("sharpe", point)

    def test_insufficient_tickers_raises_error(self):
        with self.assertRaises(ValueError):
            self.optimizer.optimize_portfolio(
                tickers=["NVDA"],
                price_series={"NVDA": self.price_history["NVDA"]},
                forecasted_returns={"NVDA": 0.25},
                horizon=30
            )


if __name__ == "__main__":
    unittest.main()
