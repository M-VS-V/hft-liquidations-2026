"""
Heuristic benchmark: EWMA liquidation pressure → raw score.

The non-ML baseline that the ML model must beat.
Score = -(same-side liquidation pressure), so higher = safer trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def strategy_raw_score(features: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    """
    Heuristic raw score: higher = better trade (higher expected pnl, keep it).

    Simplest version: score = -(same-side liq pressure at preferred halflife).
    Uses direction-relative features (same_side_liq_ewm_*), not absolute buy/sell.

    Returns float64 array of length len(features).
    """
    raise NotImplementedError
