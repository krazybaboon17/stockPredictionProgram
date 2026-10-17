"""
Risk Engine for TimesFM Stock Prediction Platform.
Calculates Value-at-Risk (VaR), Expected Shortfall (CVaR),
Sharpe Ratio, and downside volatility from TimesFM quantile forecasts.
"""

import math
from typing import Dict, Any, List
import numpy as np


class RiskEngine:
    """
    Computes probabilistic risk metrics combining historical data
    and TimesFM foundation model quantile distributions.
    """

    @staticmethod
    def calculate_risk_metrics(
        current_price: float,
        historical_prices: List[float],
        forecast_predictions: List[Dict[str, Any]],
        risk_free_rate: float = 0.045
    ) -> Dict[str, Any]:
        """
        Computes VaR, CVaR, Drawdown, and Sharpe metrics.
        """
        if current_price <= 0:
            raise ValueError("Current price must be positive.")

        if len(historical_prices) < 10:
            hist_array = np.array([current_price] * 10, dtype=np.float64)
        else:
            hist_array = np.array(historical_prices, dtype=np.float64)

        # Historical daily log returns
        returns = np.diff(np.log(hist_array))
        daily_vol = float(np.std(returns)) if len(returns) > 0 else 0.015
        annualized_vol = daily_vol * math.sqrt(252)

        horizon = len(forecast_predictions) if forecast_predictions else 30
        horizon_sqrt = math.sqrt(horizon)

        # 1. Parametric VaR (Horizon-scaled)
        var_95_pct = 1.6449 * daily_vol * horizon_sqrt
        var_99_pct = 2.3263 * daily_vol * horizon_sqrt

        var_95_dollar = round(current_price * var_95_pct, 2)
        var_99_dollar = round(current_price * var_99_pct, 2)

        # 2. TimesFM Quantile Downside (from Q10 trajectory)
        q10_prices = [p.get("q10", p.get("predicted", current_price)) for p in forecast_predictions]
        min_q10 = min(q10_prices) if q10_prices else current_price
        timesfm_q10_loss_pct = max(0.0, (current_price - min_q10) / current_price)

        # 3. Expected Shortfall (CVaR at 95%)
        # For normal distribution: CVaR_alpha = mu + sigma * phi(z_alpha) / (1 - alpha)
        # For standard 95%, phi(1.645) / 0.05 ~ 2.063
        cvar_95_pct = 2.0627 * daily_vol * horizon_sqrt
        cvar_95_dollar = round(current_price * cvar_95_pct, 2)

        # 4. Maximum Drawdown along expected trajectory
        pred_prices = [p.get("predicted", current_price) for p in forecast_predictions]
        all_pred = [current_price] + pred_prices
        peak = all_pred[0]
        max_drawdown = 0.0
        for p in all_pred:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # 5. Projected Annualized Return & Sharpe Ratio
        if forecast_predictions:
            final_pred = forecast_predictions[-1].get("predicted", current_price)
            horizon_return = (final_pred - current_price) / current_price
            annualized_return = horizon_return * (252 / max(horizon, 1))
        else:
            annualized_return = 0.0

        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = round(excess_return / max(annualized_vol, 1e-4), 2)

        # 6. Risk Level Categorization
        if annualized_vol < 0.20 and var_95_pct < 0.08:
            risk_tier = "LOW RISK"
            risk_color = "emerald"
        elif annualized_vol < 0.35 and var_95_pct < 0.15:
            risk_tier = "MODERATE RISK"
            risk_color = "amber"
        elif annualized_vol < 0.55:
            risk_tier = "ELEVATED RISK"
            risk_color = "purple"
        else:
            risk_tier = "HIGH VOLATILITY"
            risk_color = "rose"

        return {
            "horizon_days": horizon,
            "annualized_volatility_pct": round(annualized_vol * 100, 2),
            "projected_annualized_return_pct": round(annualized_return * 100, 2),
            "sharpe_ratio": sharpe_ratio,
            "var_95_pct": round(var_95_pct * 100, 2),
            "var_95_amount": var_95_dollar,
            "var_99_pct": round(var_99_pct * 100, 2),
            "var_99_amount": var_99_dollar,
            "cvar_95_pct": round(cvar_95_pct * 100, 2),
            "cvar_95_amount": cvar_95_dollar,
            "timesfm_downside_q10_pct": round(timesfm_q10_loss_pct * 100, 2),
            "max_drawdown_risk_pct": round(max_drawdown * 100, 2),
            "risk_tier": risk_tier,
            "risk_color": risk_color
        }
