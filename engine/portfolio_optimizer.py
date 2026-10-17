"""
Portfolio Allocation and Modern Portfolio Theory (MPT) Optimizer.
Computes Markowitz Efficient Frontier, Maximum Sharpe Ratio allocation,
and Minimum Volatility weights using TimesFM forecasted returns and asset covariance.
"""

import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer that combines Google TimesFM return projections
    with historical covariance matrices to derive optimal asset allocations.
    """

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(
        self,
        tickers: List[str],
        price_series: Optional[Dict[str, List[float]]] = None,
        historical_price_matrix: Optional[Dict[str, List[float]]] = None,
        forecasted_returns: Optional[Dict[str, float]] = None,
        horizon: int = 30,
        num_simulations: int = 4000
    ) -> Dict[str, Any]:
        """
        Calculates optimal portfolios and generates Efficient Frontier curves.

        Args:
            tickers: List of ticker symbols (2 to 10 assets).
            price_series: Map of ticker -> list of historical close prices.
            historical_price_matrix: Alias for price_series.
            forecasted_returns: Map of ticker -> expected annualized return (e.g. 0.18 for 18%).
            horizon: Investment horizon in trading days.
            num_simulations: Number of Monte Carlo Dirichlet weight allocations.

        Returns:
            Dict containing max_sharpe, min_volatility, equal_weight allocations,
            and efficient frontier scatter points.
        """
        prices_map = price_series if price_series is not None else (historical_price_matrix or {})
        returns_map = forecasted_returns or {}

        valid_tickers = [t for t in tickers if t in prices_map and len(prices_map[t]) >= 30]
        n_assets = len(valid_tickers)

        if n_assets < 2:
            single = valid_tickers[0] if valid_tickers else (tickers[0] if tickers else "UNKNOWN")
            exp_ret = returns_map.get(single, 0.10)
            return {
                "tickers": [single],
                "max_sharpe": {
                    "weights": {single: 1.0},
                    "expected_return_pct": round(exp_ret * 100, 2),
                    "annualized_volatility_pct": 20.0,
                    "sharpe_ratio": round((exp_ret - self.risk_free_rate) / 0.20, 2)
                },
                "min_volatility": {
                    "weights": {single: 1.0},
                    "expected_return_pct": round(exp_ret * 100, 2),
                    "annualized_volatility_pct": 20.0,
                    "sharpe_ratio": round((exp_ret - self.risk_free_rate) / 0.20, 2)
                },
                "equal_weight": {
                    "weights": {single: 1.0},
                    "expected_return_pct": round(exp_ret * 100, 2),
                    "annualized_volatility_pct": 20.0,
                    "sharpe_ratio": round((exp_ret - self.risk_free_rate) / 0.20, 2)
                },
                "frontier_points": [],
                "summary": f"Single asset allocation for {single} (100% weight)."
            }

        # Align series lengths
        min_len = min(len(prices_map[t]) for t in valid_tickers)
        prices = np.array([prices_map[t][-min_len:] for t in valid_tickers], dtype=float)

        # Compute log daily returns
        returns_matrix = np.diff(np.log(prices), axis=1)

        # Annualized Covariance Matrix (252 trading days)
        cov_matrix = np.cov(returns_matrix) * 252.0

        # Expected Returns vector
        exp_returns_vec = np.array([returns_map.get(t, 0.12) for t in valid_tickers], dtype=float)

        # Generate random long-only portfolio weights using Dirichlet distribution
        np.random.seed(42)  # Deterministic repeatability
        weights_sim = np.random.dirichlet(np.ones(n_assets), size=num_simulations)

        # Include standard boundary allocations (single asset, equal weight)
        identity_weights = np.eye(n_assets)
        equal_w = np.ones((1, n_assets)) / n_assets
        all_weights = np.vstack([weights_sim, identity_weights, equal_w])

        # Vectorized portfolio return and volatility calculations
        port_returns = np.dot(all_weights, exp_returns_vec)
        # Variance = w * Cov * w^T
        port_var = np.einsum('ij,jk,ik->i', all_weights, cov_matrix, all_weights)
        port_vols = np.sqrt(np.maximum(port_var, 1e-8))

        # Sharpe Ratios
        sharpe_ratios = (port_returns - self.risk_free_rate) / np.maximum(port_vols, 1e-6)

        # 1. Max Sharpe Portfolio
        max_sharpe_idx = int(np.argmax(sharpe_ratios))
        max_sharpe_w = all_weights[max_sharpe_idx]

        # 2. Min Volatility Portfolio
        min_vol_idx = int(np.argmin(port_vols))
        min_vol_w = all_weights[min_vol_idx]

        # 3. Equal Weight Portfolio
        eq_w = np.ones(n_assets) / n_assets
        eq_ret = float(np.dot(eq_w, exp_returns_vec))
        eq_vol = float(np.sqrt(np.dot(eq_w.T, np.dot(cov_matrix, eq_w))))
        eq_sharpe = (eq_ret - self.risk_free_rate) / max(eq_vol, 1e-6)

        def _format_weights(w_arr: np.ndarray) -> Dict[str, float]:
            return {t: round(float(w_arr[i]), 4) for i, t in enumerate(valid_tickers)}

        # Sample Efficient Frontier Curve Points for charting
        # Sort simulated portfolios by volatility and extract upper envelope
        frontier_points = []
        vol_bins = np.linspace(np.min(port_vols), np.max(port_vols), num=25)
        for i in range(len(vol_bins) - 1):
            mask = (port_vols >= vol_bins[i]) & (port_vols < vol_bins[i + 1])
            if np.any(mask):
                max_ret_in_bin = np.max(port_returns[mask])
                best_idx = np.where(mask & (port_returns == max_ret_in_bin))[0][0]
                frontier_points.append({
                    "volatility_pct": round(float(port_vols[best_idx]) * 100, 2),
                    "return_pct": round(float(port_returns[best_idx]) * 100, 2),
                    "sharpe": round(float(sharpe_ratios[best_idx]), 2)
                })

        # Narrative Summary
        top_asset = valid_tickers[int(np.argmax(max_sharpe_w))]
        top_weight_pct = max_sharpe_w[int(np.argmax(max_sharpe_w))] * 100
        summary = (
            f"Optimized allocation across {n_assets} assets delivers a maximum Sharpe ratio of "
            f"{sharpe_ratios[max_sharpe_idx]:.2f} (Expected Return: {port_returns[max_sharpe_idx]*100:.1f}%, Volatility: {port_vols[max_sharpe_idx]*100:.1f}%). "
            f"The primary driver is {top_asset} at {top_weight_pct:.1f}% allocation weight."
        )

        # Pairwise Correlation Matrix
        stds = np.sqrt(np.diag(cov_matrix))
        outer_stds = np.outer(stds, stds)
        corr_matrix = cov_matrix / np.maximum(outer_stds, 1e-8)
        correlation_dict = {
            t1: {t2: round(float(corr_matrix[i, j]), 3) for j, t2 in enumerate(valid_tickers)}
            for i, t1 in enumerate(valid_tickers)
        }

        max_sharpe_dict = {
            "name": "Maximum Sharpe Portfolio",
            "weights": _format_weights(max_sharpe_w),
            "expected_return_pct": round(float(port_returns[max_sharpe_idx]) * 100, 2),
            "annualized_volatility_pct": round(float(port_vols[max_sharpe_idx]) * 100, 2),
            "volatility_pct": round(float(port_vols[max_sharpe_idx]) * 100, 2),
            "sharpe_ratio": round(float(sharpe_ratios[max_sharpe_idx]), 2)
        }
        min_vol_dict = {
            "name": "Minimum Volatility Portfolio",
            "weights": _format_weights(min_vol_w),
            "expected_return_pct": round(float(port_returns[min_vol_idx]) * 100, 2),
            "annualized_volatility_pct": round(float(port_vols[min_vol_idx]) * 100, 2),
            "volatility_pct": round(float(port_vols[min_vol_idx]) * 100, 2),
            "sharpe_ratio": round(float(sharpe_ratios[min_vol_idx]), 2)
        }
        eq_weight_dict = {
            "name": "Equal Weight Benchmark (1/N)",
            "weights": _format_weights(eq_w),
            "expected_return_pct": round(eq_ret * 100, 2),
            "annualized_volatility_pct": round(eq_vol * 100, 2),
            "volatility_pct": round(eq_vol * 100, 2),
            "sharpe_ratio": round(eq_sharpe, 2)
        }

        return {
            "tickers": valid_tickers,
            "horizon_days": horizon,
            "risk_free_rate_pct": round(self.risk_free_rate * 100, 2),
            "max_sharpe": max_sharpe_dict,
            "optimal_portfolio": max_sharpe_dict,
            "min_volatility": min_vol_dict,
            "min_variance_portfolio": min_vol_dict,
            "equal_weight": eq_weight_dict,
            "equal_weight_portfolio": eq_weight_dict,
            "frontier_points": frontier_points,
            "frontier": frontier_points,
            "correlation_matrix": correlation_dict,
            "summary": summary
        }
