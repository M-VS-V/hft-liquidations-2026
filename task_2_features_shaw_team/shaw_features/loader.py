"""Data loading helpers for the liquidation feature pipeline (Shaw Team).

The competition ships four parquet tables per symbol:

    binance_trades/perp_<sym>.parquet        timestamp, ticker, side, price, amount
    binance_booktickers/perp_<sym>.parquet   timestamp, ticker, bid_price, bid_amount,
                                             ask_price, ask_amount
    binance_liquidations/perp_<sym>.parquet  timestamp, ticker, side, price, amount
    bybit_liquidations/<sym>.parquet         timestamp, ticker, side, price, amount

`timestamp` is int64 microseconds since the UNIX epoch (UTC) on every table.

Trades are huge (4e8 - 7e8 rows), so they are never loaded whole here: callers either
restrict to a contiguous time window (`load_window`) or take a deterministic systematic
sample (`sample_trades`).  Liquidation and BBO tables are small enough to read in full.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

TRADE_COLS = ["timestamp", "side", "price", "amount"]
BBO_COLS = ["timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount"]
LIQ_COLS = ["timestamp", "side", "price", "amount"]


def find_data_root(explicit: str | None = None) -> Path:
    """Resolve the directory that holds ``binance_trades/`` etc.

    Priority: explicit argument -> ``SHAW_DATA_ROOT`` env var -> walk up from this file
    looking for ``liquidation_task/data``.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("SHAW_DATA_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "liquidation_task" / "data"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate liquidation_task/data; pass data_root explicitly "
        "or set SHAW_DATA_ROOT."
    )


def _trades_path(root: Path, sym: str) -> Path:
    return root / "binance_trades" / f"perp_{sym}usdt.parquet"


def _bbo_path(root: Path, sym: str) -> Path:
    return root / "binance_booktickers" / f"perp_{sym}usdt.parquet"


def _binance_liq_path(root: Path, sym: str) -> Path:
    return root / "binance_liquidations" / f"perp_{sym}usdt.parquet"


def _bybit_liq_path(root: Path, sym: str) -> Path:
    return root / "bybit_liquidations" / f"{sym}usdt.parquet"


def load_window(
    sym: str,
    ts_min: int,
    ts_max: int,
    data_root: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Load every table for ``sym`` restricted to ``[ts_min, ts_max)`` (microseconds).

    Returns a dict with keys ``trades``, ``bbo``, ``liq_binance``, ``liq_bybit``.
    Liquidation/BBO frames are clipped to the same window so the example stays
    self-consistent; in production they would be supplied whole.
    """
    root = find_data_root(data_root)

    def scan(path: Path, cols: list[str]) -> pl.DataFrame:
        return (
            pl.scan_parquet(path)
            .select(cols)
            .filter((pl.col("timestamp") >= ts_min) & (pl.col("timestamp") < ts_max))
            .sort("timestamp")
            .collect()
        )

    return {
        "trades": scan(_trades_path(root, sym), TRADE_COLS),
        "bbo": scan(_bbo_path(root, sym), BBO_COLS),
        "liq_binance": scan(_binance_liq_path(root, sym), LIQ_COLS),
        "liq_bybit": scan(_bybit_liq_path(root, sym), LIQ_COLS),
    }


def sample_trades(sym: str, n: int, data_root: str | None = None) -> pl.DataFrame:
    """Deterministic systematic sample of ``n`` trades spread across the whole table.

    A systematic (every-k-th-row) sample preserves the calendar coverage of the data,
    which matters because the regime shifts between the train and validation months.
    It is reproducible without an RNG seed.
    """
    root = find_data_root(data_root)
    path = _trades_path(root, sym)
    total = pq.ParquetFile(path).metadata.num_rows
    if n >= total:
        return pl.read_parquet(path, columns=TRADE_COLS).sort("timestamp")
    stride = max(total // n, 1)
    return (
        pl.scan_parquet(path)
        .select(TRADE_COLS)
        .gather_every(stride)
        .sort("timestamp")
        .collect()
    )


def load_supporting(sym: str, data_root: str | None = None) -> dict[str, pl.DataFrame]:
    """Load the full BBO and both liquidation tables for ``sym`` (no trades)."""
    root = find_data_root(data_root)
    return {
        "bbo": pl.read_parquet(_bbo_path(root, sym), columns=BBO_COLS).sort("timestamp"),
        "liq_binance": pl.read_parquet(_binance_liq_path(root, sym), columns=LIQ_COLS).sort(
            "timestamp"
        ),
        "liq_bybit": pl.read_parquet(_bybit_liq_path(root, sym), columns=LIQ_COLS).sort(
            "timestamp"
        ),
    }


def trades_path(sym: str, data_root: str | None = None) -> str:
    return str(_trades_path(find_data_root(data_root), sym))
