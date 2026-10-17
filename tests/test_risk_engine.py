"""
Unit tests for RiskEngine and Value-at-Risk calculations.
"""

import unittest
from engine.risk_engine import RiskEngine


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.current_price = 150.0
        # Simulated 60 days of historical prices
        self.historical_prices = [
            140.0 + (i * 0.2) + ((-1) ** i * 0.5) for i in range(60)
        ]
        # Simulated 30-day forecast predictions with quantiles
        self.forecast_predictions = []
        p = 150.0
        for i in range(30):
            p += 0.3
            self.forecast_predictions.append({
                "date": f"2026-11-{i+1:02d}",
                "predicted": round(p, 2),
                "q10": round(p * 0.94, 2),
                "q25": round(p * 0.97, 2),
                "q50": round(p, 2),
                "q75": round(p * 1.03, 2),
                "q90": round(p * 1.06, 2),
            })

    def test_risk_metrics_calculation(self):
        metrics = RiskEngine.calculate_risk_metrics(
            current_price=self.current_price,
            historical_prices=self.historical_prices,
            forecast_predictions=self.forecast_predictions
        )

        self.assertIn("var_95_pct", metrics)
        self.assertIn("var_95_amount", metrics)
        self.assertIn("var_99_pct", metrics)
        self.assertIn("var_99_amount", metrics)
        self.assertIn("cvar_95_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("risk_tier", metrics)

        # 99% VaR should strictly exceed 95% VaR
        self.assertGreater(metrics["var_99_pct"], metrics["var_95_pct"])
        self.assertGreater(metrics["var_99_amount"], metrics["var_95_amount"])
        # CVaR (Expected Shortfall) should be >= 95% VaR
        self.assertGreaterEqual(metrics["cvar_95_pct"], metrics["var_95_pct"])
        # Volatility should be positive
        self.assertGreater(metrics["annualized_volatility_pct"], 0)

    def test_invalid_price_raises_error(self):
        with self.assertRaises(ValueError):
            RiskEngine.calculate_risk_metrics(
                current_price=0,
                historical_prices=self.historical_prices,
                forecast_predictions=self.forecast_predictions
            )


if __name__ == "__main__":
    unittest.main()
