"""
Book / L1 features from Binance BBO stream.

Computes features from the last BBO update strictly before each trade:
  - spread (in bps)
  - top-of-book imbalance
  - microprice deviation from mid
  - depth dynamics (bid/ask amount changes over rolling windows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _last_asof(
    query_ts: np.ndarray,
    ref_ts: np.ndarray,
    ref_vals: np.ndarray,
) -> np.ndarray:
    """
    For each query_ts, return the value from ref_vals at the last ref_ts < query_ts.
    Strictly causal (allow_exact_matches=False equivalent).
    Returns NaN where no prior reference exists.
    """
    raise NotImplementedError


def compute_book_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    windows_s: tuple[float, ...] = (1.0, 10.0),
) -> pd.DataFrame:
    """
    Compute book features for each trade.

    Output columns:
        spread_bps      — (ask - bid) / mid * 10000
        imbalance       — (bid_amt - ask_amt) / (bid_amt + ask_amt)
        microprice_dev  — (microprice - mid) / mid * 10000
        bid_amount      — last bid size before trade (absolute, pre-direction-transform)
        ask_amount      — last ask size before trade (absolute, pre-direction-transform)
        bid_amount_delta_{w}s — change in bid size over window
        ask_amount_delta_{w}s — change in ask size over window

    Absolute BID/ASK columns will be converted to direction-relative form
    by core.transforms.direction before reaching the model.
    """
    raise NotImplementedError
