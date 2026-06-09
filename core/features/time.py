"""
Time features.

Time-of-day, day-of-week, and time-to-known-event encodings.
"""

from __future__ import annotations

import pandas as pd


def compute_time_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time features for each trade.

    Output columns:
        hour_utc          — hour of day (0-23) from timestamp
        minute_utc        — minute of hour (0-59)
        time_to_funding_s — seconds until next Binance funding (00:00, 08:00, 16:00 UTC)
    """
    raise NotImplementedError
