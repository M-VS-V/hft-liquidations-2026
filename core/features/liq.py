"""
Liquidation pressure features.

Primary signal for cascade detection. Computes EWMA of liquidation notional
across venues (Binance, Bybit) and sides (buy, sell), plus intensity metrics
(event count, time since last liq).

Bybit timestamps must be pre-shifted (+200ms) before calling these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ewma_event_state_at(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    event_vals: np.ndarray,
    halflife_us: float,
) -> np.ndarray:
    """
    For each query timestamp, compute the EWMA state of an event stream
    using only events strictly before the query time.

    The EWMA decays as exp(-(t_query - t_event) / tau) where tau = halflife_us / ln(2).

    Parameters
    ----------
    query_ts   : sorted int64 array of trade timestamps (microseconds)
    event_ts   : sorted int64 array of event timestamps (microseconds, shifted if Bybit)
    event_vals : float64 array of event values (e.g. notional = price * amount)
    halflife_us: EWMA half-life in microseconds

    Returns
    -------
    float64 array of length len(query_ts), the EWMA state at each query time.
    """
    raise NotImplementedError


def _liq_event_count(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    window_us: int,
) -> np.ndarray:
    """
    Count liquidation events in [query_ts - window_us, query_ts) for each query.
    Uses searchsorted for O(n log n) performance.
    """
    raise NotImplementedError


def _time_since_last_liq(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
) -> np.ndarray:
    """
    Microseconds since the last liquidation event strictly before each query time.
    Returns np.inf where no prior event exists.
    """
    raise NotImplementedError


def compute_liq_features(
    trades: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
    halflives_s: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    """
    Compute all liquidation pressure features for each trade.

    Output columns (for each venue, side, halflife):
        liq_ewma_{venue}_{side}_{hl}s     — EWMA of liq notional
        liq_count_{venue}_{side}_{window}s — event count in window
        liq_time_since_{venue}_{side}      — seconds since last liq

    All values are in absolute buy/sell terms. Direction-relative conversion
    (same_side / opp_side) is handled by core.transforms.direction.
    """
    raise NotImplementedError
