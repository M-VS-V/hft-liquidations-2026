"""
Data loading and mandatory preprocessing.

Responsible for:
  - reading parquet files per symbol
  - casting timestamps to int64 microseconds
  - shifting Bybit liquidation timestamps by +200ms
  - sorting all frames by timestamp
  - optional date-range filtering for train/val splits
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Symbol = Literal["btcusdt", "ethusdt"]

US_PER_SECOND: int = 1_000_000
BYBIT_LAG_US: int = 200_000  # +200 ms

SPLIT_RANGES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train":      (pd.Timestamp("2025-12-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")),
    "validation": (pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
}


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw parquet frame:
      - ensure timestamp is int64
      - lowercase string columns (side, ticker)
      - sort by timestamp ascending
    Idempotent.
    """
    raise NotImplementedError


def load_data_with_required_preprocess(
    data_dir: str,
    symbol: Symbol,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 4 frames for one symbol and run mandatory preprocessing.

    Steps:
      1. Read parquet files for the given symbol.
      2. If split is given, filter to the date range BEFORE shifting Bybit.
      3. Shift liq_bybit.timestamp += BYBIT_LAG_US.
      4. Sort each frame by timestamp ascending.
      5. For liq data, include a lookback buffer (~5 min before split start)
         to avoid cold-start on EWMA features.

    Returns (trades, bbo, liq_binance, liq_bybit).
    liq_bybit timestamps are ALREADY shifted.
    """
    raise NotImplementedError


def compute_num_days(trades: pd.DataFrame) -> float:
    """
    Number of distinct calendar dates (UTC) spanned by the trades frame.
    Uses trades["timestamp"] (int64 microseconds).
    """
    raise NotImplementedError


def detect_symbol(trades: pd.DataFrame) -> Symbol:
    """
    Infer symbol from the ticker column.
    'perp:btcusdt' -> 'btcusdt', 'perp:ethusdt' -> 'ethusdt'.
    Raises ValueError on mixed or unknown tickers.
    """
    raise NotImplementedError
