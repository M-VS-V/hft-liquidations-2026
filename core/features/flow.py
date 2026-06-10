"""
Signed trade flow features.

Leading indicator of cascade onset: aggressive directional flow building
before liquidations fire. Computed from the Binance trades stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_sum_count(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    event_vals: np.ndarray,
    window_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each query_ts, compute the sum and count of event_vals
    in the window [query_ts - window_us, query_ts).

    Uses searchsorted(side='left') to ensure strict causality:
    events at exactly query_ts are excluded.
    """
    raise NotImplementedError


def compute_flow_features(
    trades: pd.DataFrame,
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    """
    Compute signed taker flow features for each trade.

    Output columns (for each window):
        signed_volume_{w}s     — Σ s_j * min(notional_j, 100k) over window
        total_volume_{w}s      — Σ min(notional_j, 100k) over window
        taker_imbalance_{w}s   — signed_volume / total_volume (in [-1, 1])
        trade_count_{w}s       — number of trades in window

    These are in absolute signed terms. Direction-relative conversion
    (same_side_flow = s_i * signed_volume) is in core.transforms.direction.
    """
    raise NotImplementedError
