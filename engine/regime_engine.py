"""
Technical Market Regime & Momentum Classification Engine.
Analyzes RSI, MACD histogram, Bollinger Band squeeze, and moving average momentum
to classify the market regime into one of four standard states:
- Bullish Expansion
- Bearish Breakdown
- Overbought Exhaustion
- Squeeze Consolidation
"""

import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeEngine:
    """
    Automated Technical Regime and Momentum Screener.
    Combines trend strength, momentum oscillators, and volatility compression
    to identify institutional market phases.
    """

    COLOR_MAP = {
        "Bullish Expansion": "#10b981",       # Emerald green
        "Bearish Breakdown": "#ef4444",       # Crimson red
        "Overbought Exhaustion": "#f59e0b",   # Amber / warning orange
        "Squeeze Consolidation": "#6366f1",   # Indigo / compression blue
    }

    @staticmethod
    def classify_regime(stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies current market regime from processed stock history.
        
        Args:
            stock_data: Dict returned by StockDataEngine.get_stock_data containing history and metrics.

        Returns:
            Dict containing ticker, current_price, regime, regime_color, confidence_score, metrics, summary.
        """
        ticker = stock_data.get("ticker", "UNKNOWN").upper()
        history = stock_data.get("history", [])
        current_price = float(stock_data.get("current_price", 0.0))

        if not history or len(history) < 20:
            return {
                "ticker": ticker,
                "current_price": current_price,
                "regime": "Squeeze Consolidation",
                "regime_color": RegimeEngine.COLOR_MAP["Squeeze Consolidation"],
                "confidence_score": 0.50,
                "metrics": {
                    "rsi": 50.0,
                    "macd_hist": 0.0,
                    "bb_width_pct": 0.0,
                    "trend_direction": "Neutral",
                    "price_to_sma20_pct": 0.0,
                    "price_to_sma50_pct": 0.0,
                },
                "summary": "Insufficient historical points for high-conviction regime classification; defaulting to neutral consolidation."
            }

        latest = history[-1]
        prev = history[-2] if len(history) >= 2 else latest

        rsi = float(latest.get("rsi", 50.0))
        macd_hist = float(latest.get("macd_hist", 0.0))
        prev_macd_hist = float(prev.get("macd_hist", 0.0))
        sma_20 = float(latest.get("sma_20", current_price))
        sma_50 = float(latest.get("sma_50", current_price))
        bb_upper = float(latest.get("bb_upper", current_price * 1.05))
        bb_lower = float(latest.get("bb_lower", current_price * 0.95))

        # Calculate Bollinger Band Bandwidth %
        bb_bandwidth = (bb_upper - bb_lower) / max(sma_20, 1e-4) * 100.0

        # Check historical bandwidth to detect volatility squeeze (bandwidth below 20th percentile)
        recent_bandwidths = []
        for p in history[-30:]:
            upper = float(p.get("bb_upper", 0.0))
            lower = float(p.get("bb_lower", 0.0))
            sma = float(p.get("sma_20", 1.0))
            if sma > 0:
                recent_bandwidths.append((upper - lower) / sma * 100.0)

        median_bandwidth = float(np.median(recent_bandwidths)) if recent_bandwidths else bb_bandwidth
        is_bandwidth_compressed = bb_bandwidth < (median_bandwidth * 0.75) or bb_bandwidth < 4.5

        # Price relatives
        price_to_sma20 = ((current_price - sma_20) / max(sma_20, 1e-4)) * 100.0
        price_to_sma50 = ((current_price - sma_50) / max(sma_50, 1e-4)) * 100.0

        # Classification Logic
        # 1. Overbought Exhaustion: RSI > 72 or (RSI > 68 and MACD histogram decaying from peak)
        if rsi >= 72.0 or (rsi >= 68.0 and macd_hist < prev_macd_hist and macd_hist > 0):
            regime = "Overbought Exhaustion"
            confidence = min(0.96, 0.65 + (rsi - 68.0) * 0.02 + (0.05 if macd_hist < prev_macd_hist else 0.0))
            summary = (
                f"{ticker} is flashing overbought exhaustion signals with RSI at {rsi:.1f}. "
                f"Momentum expansion is decelerating as price trades {price_to_sma20:+.1f}% above its 20-day SMA, "
                "suggesting heightened risk of mean-reversion pullbacks."
            )

        # 2. Squeeze Consolidation: compressed volatility bandwidth and rangebound RSI (42 - 58)
        elif is_bandwidth_compressed or (44.0 <= rsi <= 56.0 and abs(price_to_sma20) < 1.8):
            regime = "Squeeze Consolidation"
            confidence = min(0.94, 0.70 + (0.15 if is_bandwidth_compressed else 0.05))
            summary = (
                f"{ticker} is in a coiled squeeze consolidation phase with Bollinger bandwidth tightened to {bb_bandwidth:.2f}%. "
                f"RSI is neutral at {rsi:.1f}, indicating an impending directional volatility breakout as energy accumulates."
            )

        # 3. Bullish Expansion: Price above SMA20 & SMA50, positive MACD histogram, constructive RSI
        elif current_price >= sma_20 and macd_hist > -0.05 and rsi >= 50.0:
            regime = "Bullish Expansion"
            confidence = min(0.95, 0.68 + (0.10 if current_price > sma_50 else 0.0) + (0.10 if macd_hist > prev_macd_hist else 0.0))
            summary = (
                f"{ticker} is sustaining an aggressive bullish expansion regime. "
                f"Price holds {price_to_sma20:+.1f}% above the 20-day SMA with positive MACD momentum ({macd_hist:+.3f}) "
                f"and healthy RSI expansion at {rsi:.1f}."
            )

        # 4. Bearish Breakdown: Price below SMA20, negative MACD histogram, sub-50 RSI
        else:
            regime = "Bearish Breakdown"
            confidence = min(0.95, 0.68 + (0.12 if current_price < sma_50 else 0.0) + (0.08 if rsi < 40.0 else 0.0))
            summary = (
                f"{ticker} is trending in a bearish breakdown regime. "
                f"Price trades {price_to_sma20:+.1f}% below its 20-day SMA, with negative MACD histogram momentum ({macd_hist:+.3f}) "
                f"and depressed RSI ({rsi:.1f}) reflecting seller dominance."
            )

        trend_direction = "Bullish" if current_price > sma_50 else ("Bearish" if current_price < sma_50 else "Neutral")

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "regime": regime,
            "regime_color": RegimeEngine.COLOR_MAP[regime],
            "confidence_score": round(confidence, 2),
            "metrics": {
                "rsi": round(rsi, 2),
                "macd_hist": round(macd_hist, 4),
                "bb_width_pct": round(bb_bandwidth, 2),
                "trend_direction": trend_direction,
                "price_to_sma20_pct": round(price_to_sma20, 2),
                "price_to_sma50_pct": round(price_to_sma50, 2),
            },
            "summary": summary
        }
