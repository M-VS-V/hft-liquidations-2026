"""
Forward markout computation.

For each trade i and horizon tau, compute m_i(tau) = the forward-fill mid
at time t_i + tau (the last BBO mid with timestamp <= t_i + tau).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND: int = 1_000_000
TAUS: tuple[int, ...] = (30, 120, 300)


def add_mid(bbo: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns to BBO frame:
        mid        = (bid_price + ask_price) / 2
        microprice = (bid_price * ask_amount + ask_price * bid_amount) / (bid_amount + ask_amount)
    """
    raise NotImplementedError


def compute_markout(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    taus: tuple[int, ...] = TAUS,
) -> pd.DataFrame:
    """
    For each trade and each tau, compute the forward-fill mid at t_i + tau.

    Implementation:
        For each tau:
          1. lookup_ts = trades.timestamp + tau * US_PER_SECOND
          2. asof-join (backward) lookup_ts against bbo.timestamp → mid value
          3. Where lookup_ts > max(bbo.timestamp), mid = NaN, edge_{tau} = True

    Adds columns to trades:
        mid_{tau}  : float64 (NaN on edge trades)
        edge_{tau} : bool (True = trade falls off BBO range, excluded from scoring)
    """
    raise NotImplementedError
