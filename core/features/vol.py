"""
Volatility and regime-context features.

Conditioning layer: modulates how dangerous a given level of liq pressure is
given the current market regime.
"""

from __future__ import annotations

import pandas as pd


def compute_vol_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    windows_s: tuple[float, ...] = (60.0, 300.0),
    rank_window: int = 1000,
) -> pd.DataFrame:
    """
    Compute volatility and regime features for each trade.

    Output columns:
        rolling_vol_{w}s       — std of log-returns over window (annualized or raw)
        spread_bps_rank_{n}    — rolling rank percentile of spread (0-1)
        depth_rank_{n}         — rolling rank percentile of top-of-book size (0-1)
        vol_rank_{n}           — rolling rank percentile of volatility (0-1)
    """
    raise NotImplementedError
