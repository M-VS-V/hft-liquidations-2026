"""Synthetic fixtures so the tests do not depend on the multi-GB dataset."""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaw_features import MarketContext  # noqa: E402

S = 1_000_000


@pytest.fixture
def frames():
    """Small, hand-checkable trades / bbo / liquidation frames (timestamps in µs)."""
    trades = pl.DataFrame(
        {
            "timestamp": [10 * S, 50 * S, 100 * S, 200 * S],
            "side": ["buy", "sell", "buy", "sell"],
            "price": [100.0, 101.0, 102.0, 103.0],
            "amount": [1.0, 2.0, 1.0, 1.0],
        }
    )
    bbo = pl.DataFrame(
        {
            "timestamp": [0, 40 * S, 90 * S, 150 * S, 250 * S],
            "bid_price": [99.0, 100.0, 101.0, 102.0, 103.0],
            "bid_amount": [10.0, 10.0, 10.0, 10.0, 10.0],
            "ask_price": [101.0, 102.0, 103.0, 104.0, 105.0],
            "ask_amount": [10.0, 5.0, 10.0, 10.0, 10.0],
        }
    )
    liq_binance = pl.DataFrame(
        {
            "timestamp": [5 * S, 45 * S, 95 * S],
            "side": ["buy", "sell", "buy"],
            "price": [100.0, 101.0, 102.0],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    # Bybit raw ts; context shifts these +200 ms.
    liq_bybit = pl.DataFrame(
        {
            "timestamp": [30 * S, 80 * S],
            "side": ["sell", "buy"],
            "price": [100.5, 101.5],
            "amount": [5.0, 5.0],
        }
    )
    return trades, bbo, liq_binance, liq_bybit


@pytest.fixture
def ctx(frames):
    return MarketContext.from_frames(*frames)
