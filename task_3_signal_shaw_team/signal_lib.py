"""Liquidation-signal trade filter (Shaw Team) — task 3.

Builds the markout labels and causal features for the Binance maker-fill filter and
evaluates the competition score

    Score(tau) = PnL_kept(tau) - PnL_all(tau)

subject to KeptTurnoverPerDay >= 500_000.  Markout, the +200 ms Bybit shift and all
causal window sums are reused from ``task_2_features_shaw_team`` (MarketContext), so the
labels here are identical to the validated feature infrastructure.

The data is hundreds of millions of trades, so evaluation runs day-by-day and accumulates
the exact weighted sums needed for the score over a whole period (no sampling for the
reported numbers).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

# reuse the task-2 feature infrastructure (MarketContext, +200ms shift, causal windows)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task_2_features_shaw_team"))
from shaw_features.context import MarketContext  # noqa: E402

DATA = ROOT / "liquidation_task_dataset" / "data"
TAUS = (30, 120, 300)
US = 1_000_000
DAY_US = 86_400 * US
TURNOVER_FLOOR = 500_000.0


def us_of(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() * 1e6)


def _paths(sym: str):
    return (
        DATA / "binance_trades" / f"perp_{sym}usdt.parquet",
        DATA / "binance_booktickers" / f"perp_{sym}usdt.parquet",
        DATA / "binance_liquidations" / f"perp_{sym}usdt.parquet",
        DATA / "bybit_liquidations" / f"{sym}usdt.parquet",
    )


def _load_liq(sym: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    _, _, lb, lby = _paths(sym)
    cols = ["timestamp", "side", "price", "amount"]
    return (
        pl.read_parquet(lb, columns=cols).sort("timestamp"),
        pl.read_parquet(lby, columns=cols).sort("timestamp"),
    )


def build_table(sym: str, t0: int, t1: int, liq_b: pl.DataFrame, liq_by: pl.DataFrame,
                stride: int = 1) -> dict[str, np.ndarray]:
    """Markout labels + causal features for every (optionally every-`stride`-th) trade in [t0,t1)."""
    tpath, bpath, _, _ = _paths(sym)

    bbo = (
        pl.scan_parquet(bpath)
        .select("timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount")
        .filter((pl.col("timestamp") >= t0) & (pl.col("timestamp") < t1 + 300 * US))
        .sort("timestamp")
        .collect()
    )
    tr = pl.scan_parquet(tpath).select("timestamp", "side", "price", "amount").filter(
        (pl.col("timestamp") >= t0) & (pl.col("timestamp") < t1)
    )
    if stride > 1:
        tr = tr.gather_every(stride)
    trades = tr.sort("timestamp").collect()
    if trades.height == 0:
        return {}

    # liquidation lookback: keep events from 300s before the window
    lb = liq_b.filter((pl.col("timestamp") >= t0 - 300 * US) & (pl.col("timestamp") < t1))
    lby = liq_by.filter((pl.col("timestamp") >= t0 - 301 * US) & (pl.col("timestamp") < t1))

    ctx = MarketContext.from_frames(trades, bbo, lb, lby)

    price = ctx.trade_price
    side = ctx.trade_side  # +1 taker buy (maker sell), -1 taker sell (maker buy)
    notional = price * trades["amount"].to_numpy().astype(np.float64)
    w = np.minimum(notional, 100_000.0)

    out = {"w": w, "side": side, "ts": ctx.trade_ts}
    for tau in TAUS:
        m = ctx.bbo_at("mid", tau * US)  # NaN past last quote -> excluded
        pnl = -side * (m - price) / price * 1e4 + 0.5
        out[f"pnl{tau}"] = pnl  # NaN where markout undefined

    # causal features (all read [t-W, t) only)
    for win in (30, 120, 300):
        buy = ctx.liq_buy_notional("combined", win)
        sell = ctx.liq_sell_notional("combined", win)
        out[f"signed{win}"] = buy - sell           # >0 up-pressure, <0 down-pressure
        out[f"absliq{win}"] = buy + sell
    out["cnt30"] = ctx.liq_count("combined", 30)
    out["tsl"] = ctx.time_since_liq("combined")
    out["spread"] = ctx.bbo_at("spread_bps", 0)
    out["imb"] = ctx.bbo_at("imbalance", 0)
    mid_now = ctx.bbo_at("mid", 0)
    mid_5s = ctx.bbo_at("mid", -5 * US)
    out["ret5"] = (mid_now - mid_5s) / mid_5s * 1e4
    return out


def iter_days(t0: int, t1: int):
    d = t0
    while d < t1:
        yield d, min(d + DAY_US, t1)
        d += DAY_US
