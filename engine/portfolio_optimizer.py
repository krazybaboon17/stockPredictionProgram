"""
Portfolio Allocation and Modern Portfolio Theory (MPT) Optimizer for TimesFM Platform.
Calculates Markowitz Efficient Frontier, Maximum Sharpe Ratio, Minimum Volatility,
and Equal-Weight portfolios using forecasted asset returns and empirical covariance.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger("engine.portfolio_optimizer")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PortfolioOptimizer:
    """Computes optimal asset allocation and Markowitz efficient frontier."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = float(risk_free_rate)

    def calculate_returns_and_covariance(
        self, price_series: Dict[str, List[float]]
    ) -> tuple[List[str], np.ndarray, np.ndarray]:
        """
        Aligns price histories, computes daily log/pct returns, and annualizes covariance.
        Returns (valid_tickers, daily_returns_matrix, annualized_cov_matrix).
        """
        tickers = list(price_series.keys())
        if len(tickers) < 2:
            raise ValueError("At least 2 tickers are required for portfolio optimization.")

        # Determine minimum length across series
        min_len = min(len(prices) for prices in price_series.values())
        if min_len < 10:
            raise ValueError("Insufficient historical price data (at least 10 observations required per ticker).")

        # Trim each series to the most recent min_len observations
        trimmed = {ticker: np.array(prices[-min_len:], dtype=np.float64) for ticker, prices in price_series.items()}

        # Compute percentage returns
        returns_list = []
        for ticker in tickers:
            prices = trimmed[ticker]
            if np.any(prices <= 0):
                raise ValueError(f"Invalid non-positive price detected for ticker {ticker}")
            pct_returns = np.diff(prices) / prices[:-1]
            returns_list.append(pct_returns)

        returns_matrix = np.column_stack(returns_list)  # Shape (T-1, N)
        cov_matrix = np.cov(returns_matrix, rowvar=False) * 252.0  # Annualized covariance

        # Ensure 2D covariance matrix even for 2 assets
        if cov_matrix.ndim == 0:
            cov_matrix = cov_matrix.reshape(1, 1)

        # Numerical stabilization of covariance matrix
        cov_matrix += np.eye(len(tickers)) * 1e-6

        return tickers, returns_matrix, cov_matrix

    def calculate_expected_returns(
        self,
        tickers: List[str],
        returns_matrix: np.ndarray,
        forecasted_returns: Optional[Dict[str, float]] = None,
        horizon: int = 30
    ) -> np.ndarray:
        """
        Calculates annualized expected returns combining TimesFM forecasts and historical drift.
        """
        historical_annual_mean = np.mean(returns_matrix, axis=0) * 252.0

        if not forecasted_returns:
            return historical_annual_mean

        # Annualize TimesFM forecast returns: ret * (252 / horizon)
        scale_factor = 252.0 / max(horizon, 1)
        expected_returns = []
        for i, ticker in enumerate(tickers):
            if ticker in forecasted_returns and forecasted_returns[ticker] is not None:
                # Blend 70% TimesFM forecast with 30% historical momentum
                forecast_annual = float(forecasted_returns[ticker]) * scale_factor
                blended = 0.70 * forecast_annual + 0.30 * historical_annual_mean[i]
                expected_returns.append(blended)
            else:
                expected_returns.append(historical_annual_mean[i])

        return np.array(expected_returns, dtype=np.float64)

    def portfolio_metrics(
        self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Computes portfolio (expected_return, volatility, sharpe_ratio).
        """
        weights = np.array(weights, dtype=np.float64)
        exp_return = float(np.sum(weights * mean_returns))
        variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        volatility = float(np.sqrt(max(variance, 1e-8)))
        sharpe = float((exp_return - self.risk_free_rate) / max(volatility, 1e-8))
        return exp_return, volatility, sharpe

    def _optimize_scipy(
        self, mean_returns: np.ndarray, cov_matrix: np.ndarray, objective: str = "sharpe", target_return: Optional[float] = None
    ) -> np.ndarray:
        """Optimizes weights using SciPy SLSQP solver."""
        n_assets = len(mean_returns)
        init_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if objective == "sharpe":
            def loss(w):
                ret, vol, _ = self.portfolio_metrics(w, mean_returns, cov_matrix)
                # Maximize Sharpe ratio <=> minimize negative Sharpe
                return -(ret - self.risk_free_rate) / max(vol, 1e-8)
        elif objective == "volatility":
            def loss(w):
                _, vol, _ = self.portfolio_metrics(w, mean_returns, cov_matrix)
                return vol
        elif objective == "target_return" and target_return is not None:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w * mean_returns) - target_return})
            def loss(w):
                _, vol, _ = self.portfolio_metrics(w, mean_returns, cov_matrix)
                return vol
        else:
            raise ValueError(f"Unknown objective {objective}")

        result = minimize(
            loss,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-7}
        )

        weights = result.x if result.success else init_weights
        weights = np.clip(weights, 0.0, 1.0)
        return weights / np.sum(weights)

    def _optimize_pure_numpy(
        self, mean_returns: np.ndarray, cov_matrix: np.ndarray, objective: str = "sharpe"
    ) -> np.ndarray:
        """Fallback Dirichlet Monte Carlo simulation optimizer when SciPy is not installed."""
        n_assets = len(mean_returns)
        n_samples = 15000
        # Sample random simplex weights from Dirichlet distribution
        samples = np.random.dirichlet(np.ones(n_assets), size=n_samples)

        # Include equal-weight and pure-asset boundaries
        pure_assets = np.eye(n_assets)
        equal_weight = np.ones((1, n_assets)) / n_assets
        candidates = np.vstack([samples, pure_assets, equal_weight])

        returns = np.dot(candidates, mean_returns)
        variances = np.einsum("ij,jk,ik->i", candidates, cov_matrix, candidates)
        volatilities = np.sqrt(np.maximum(variances, 1e-8))
        sharpes = (returns - self.risk_free_rate) / np.maximum(volatilities, 1e-8)

        if objective == "sharpe":
            best_idx = np.argmax(sharpes)
        else:
            best_idx = np.argmin(volatilities)

        best_weights = candidates[best_idx]
        return best_weights / np.sum(best_weights)

    def optimize_max_sharpe(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Finds portfolio weights that maximize the Sharpe ratio."""
        if HAS_SCIPY:
            return self._optimize_scipy(mean_returns, cov_matrix, objective="sharpe")
        return self._optimize_pure_numpy(mean_returns, cov_matrix, objective="sharpe")

    def optimize_min_volatility(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Finds portfolio weights that minimize portfolio volatility."""
        if HAS_SCIPY:
            return self._optimize_scipy(mean_returns, cov_matrix, objective="volatility")
        return self._optimize_pure_numpy(mean_returns, cov_matrix, objective="volatility")

    def generate_efficient_frontier(
        self, mean_returns: np.ndarray, cov_matrix: np.ndarray, num_points: int = 16
    ) -> List[Dict[str, float]]:
        """Generates points along the Markowitz Efficient Frontier curve."""
        min_vol_weights = self.optimize_min_volatility(mean_returns, cov_matrix)
        min_ret, _, _ = self.portfolio_metrics(min_vol_weights, mean_returns, cov_matrix)
        max_ret = float(np.max(mean_returns))

        if max_ret <= min_ret:
            max_ret = min_ret + 0.05

        target_returns = np.linspace(min_ret, max_ret, num=num_points)
        frontier_points = []

        if HAS_SCIPY:
            for target in target_returns:
                try:
                    w = self._optimize_scipy(mean_returns, cov_matrix, objective="target_return", target_return=target)
                    ret, vol, sharpe = self.portfolio_metrics(w, mean_returns, cov_matrix)
                    frontier_points.append({
                        "return_pct": round(ret * 100, 2),
                        "volatility_pct": round(vol * 100, 2),
                        "sharpe": round(sharpe, 3)
                    })
                except Exception:
                    continue

        if not frontier_points:
            # Fallback Monte Carlo frontier envelope
            n_assets = len(mean_returns)
            samples = np.random.dirichlet(np.ones(n_assets), size=8000)
            returns = np.dot(samples, mean_returns)
            variances = np.einsum("ij,jk,ik->i", samples, cov_matrix, samples)
            volatilities = np.sqrt(np.maximum(variances, 1e-8))

            bins = np.linspace(min(returns), max(returns), num=num_points)
            for i in range(len(bins) - 1):
                mask = (returns >= bins[i]) & (returns < bins[i + 1])
                if np.any(mask):
                    best_sample_idx = np.where(mask)[0][np.argmin(volatilities[mask])]
                    ret = returns[best_sample_idx]
                    vol = volatilities[best_sample_idx]
                    sharpe = (ret - self.risk_free_rate) / max(vol, 1e-8)
                    frontier_points.append({
                        "return_pct": round(float(ret) * 100, 2),
                        "volatility_pct": round(float(vol) * 100, 2),
                        "sharpe": round(float(sharpe), 3)
                    })

        # Ensure points are sorted by volatility
        frontier_points.sort(key=lambda p: p["volatility_pct"])
        return frontier_points

    def optimize_portfolio(
        self,
        tickers: List[str],
        price_series: Dict[str, List[float]],
        forecasted_returns: Optional[Dict[str, float]] = None,
        horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Executes end-to-end portfolio optimization returning optimal, min-vol, and equal-weight portfolios.
        """
        valid_tickers, returns_matrix, cov_matrix = self.calculate_returns_and_covariance(price_series)
        mean_returns = self.calculate_expected_returns(valid_tickers, returns_matrix, forecasted_returns, horizon)

        # 1. Optimal Maximum Sharpe Portfolio
        opt_weights = self.optimize_max_sharpe(mean_returns, cov_matrix)
        opt_ret, opt_vol, opt_sharpe = self.portfolio_metrics(opt_weights, mean_returns, cov_matrix)

        # 2. Minimum Volatility Portfolio
        min_weights = self.optimize_min_volatility(mean_returns, cov_matrix)
        min_ret, min_vol, min_sharpe = self.portfolio_metrics(min_weights, mean_returns, cov_matrix)

        # 3. Equal-Weight Portfolio (1/N)
        eq_weights = np.ones(len(valid_tickers)) / len(valid_tickers)
        eq_ret, eq_vol, eq_sharpe = self.portfolio_metrics(eq_weights, mean_returns, cov_matrix)

        # 4. Correlation Matrix
        stds = np.sqrt(np.diag(cov_matrix))
        outer_stds = np.outer(stds, stds)
        corr_matrix = cov_matrix / np.maximum(outer_stds, 1e-8)
        correlation_dict = {
            t1: {t2: round(float(corr_matrix[i, j]), 3) for j, t2 in enumerate(valid_tickers)}
            for i, t1 in enumerate(valid_tickers)
        }

        # 5. Efficient Frontier
        frontier = self.generate_efficient_frontier(mean_returns, cov_matrix)

        return {
            "tickers": valid_tickers,
            "horizon_days": horizon,
            "risk_free_rate_pct": round(self.risk_free_rate * 100, 2),
            "asset_metrics": {
                ticker: {
                    "expected_return_pct": round(float(mean_returns[i]) * 100, 2),
                    "annual_volatility_pct": round(float(np.sqrt(cov_matrix[i, i])) * 100, 2),
                    "individual_sharpe": round(float((mean_returns[i] - self.risk_free_rate) / max(np.sqrt(cov_matrix[i, i]), 1e-8)), 3)
                }
                for i, ticker in enumerate(valid_tickers)
            },
            "optimal_portfolio": {
                "name": "Maximum Sharpe Portfolio",
                "weights": {ticker: round(float(opt_weights[i]), 4) for i, ticker in enumerate(valid_tickers)},
                "expected_return_pct": round(opt_ret * 100, 2),
                "volatility_pct": round(opt_vol * 100, 2),
                "sharpe_ratio": round(opt_sharpe, 3)
            },
            "min_variance_portfolio": {
                "name": "Minimum Volatility Portfolio",
                "weights": {ticker: round(float(min_weights[i]), 4) for i, ticker in enumerate(valid_tickers)},
                "expected_return_pct": round(min_ret * 100, 2),
                "volatility_pct": round(min_vol * 100, 2),
                "sharpe_ratio": round(min_sharpe, 3)
            },
            "equal_weight_portfolio": {
                "name": "Equal Weight Benchmark (1/N)",
                "weights": {ticker: round(float(eq_weights[i]), 4) for i, ticker in enumerate(valid_tickers)},
                "expected_return_pct": round(eq_ret * 100, 2),
                "volatility_pct": round(eq_vol * 100, 2),
                "sharpe_ratio": round(eq_sharpe, 3)
            },
            "frontier": frontier,
            "correlation_matrix": correlation_dict
        }
