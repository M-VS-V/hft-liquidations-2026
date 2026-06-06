"""Reusable EDA computations for the liquidation-filter dataset (Shaw Team, task 1).

The functions here are deliberately framework-light: each takes already-loaded Polars frames
(or a parquet path for the streaming aggregations) and returns plain numbers / small frames,
so the companion notebook can focus on plotting and narrative.

Covers the four questions from the assignment:

* data quality - NaNs, duplicate / non-monotonic timestamps, bad prices & sides, crossed
  books, time gaps (and whether a gap is a market feature or a collection artefact);
* markout - maker PnL in bps at tau in {30, 120, 300}s, weighted average and distribution;
* liquidations - time-clustering into cascades and cross-exchange overlap;
* turnover constraint - minimum share of clipped turnover needed to keep >= $500k/day.

Conventions (from description.md): timestamps are int64 microseconds UTC; trade ``side`` is
the taker side (``buy`` => maker sold); maker PnL uses a forward-filled mid at ``t + tau`` and a
+0.5 bps rebate; trade weight is ``min(notional, 100_000)``; Bybit liquidations are shifted
+200 ms before being compared with Binance times.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

TAUS_S = (30, 120, 300)
REBATE_BPS = 0.5
NOTIONAL_CLIP = 100_000.0
MIN_TURNOVER_PER_DAY = 500_000.0
BYBIT_LAG_US = 200_000

TRAIN_START = int(datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp() * 1e6)
VAL_START = int(datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp() * 1e6)
VAL_END = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1e6)


def us_to_datetime(ts_us: int) -> datetime:
    return datetime.fromtimestamp(ts_us / 1e6, tz=timezone.utc)


# --------------------------------------------------------------------------- #
# 1. Data quality
# --------------------------------------------------------------------------- #
def nan_report(df: pl.DataFrame) -> dict[str, int]:
    """Per-column NaN/null count (null for any dtype, NaN for floats)."""
    out = {}
    for col in df.columns:
        nulls = df[col].null_count()
        nans = 0
        if df[col].dtype in (pl.Float32, pl.Float64):
            nans = int(df[col].is_nan().sum())
        out[col] = int(nulls) + nans
    return out


def timestamp_health(ts: np.ndarray) -> dict:
    """Monotonicity and duplicate diagnostics for a timestamp array (microseconds)."""
    diffs = np.diff(ts)
    return {
        "n": int(len(ts)),
        "is_sorted": bool(np.all(diffs >= 0)),
        "n_decreasing": int((diffs < 0).sum()),
        "n_duplicate_ts": int((diffs == 0).sum()),
    }


def trade_anomalies(trades: pl.DataFrame, ret_threshold: float = 0.01) -> dict:
    """Bad prices/amounts/sides and large tick-to-tick returns in the trades frame."""
    price = trades["price"].to_numpy()
    amount = trades["amount"].to_numpy()
    sides = set(trades["side"].unique().to_list())
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.abs(np.diff(price) / price[:-1])
    return {
        "price_min": float(price.min()),
        "price_max": float(price.max()),
        "n_nonpositive_price": int((price <= 0).sum()),
        "n_nonpositive_amount": int((amount <= 0).sum()),
        "side_domain": sorted(sides),
        "n_unexpected_side": int(trades.filter(~pl.col("side").is_in(["buy", "sell"])).height),
        f"n_tick_jumps_gt_{ret_threshold:.0%}": int((ret > ret_threshold).sum()),
    }


def bbo_quality(bbo: pl.DataFrame) -> dict:
    """Crossed/locked books and spread distribution (bps)."""
    bid = bbo["bid_price"].to_numpy()
    ask = bbo["ask_price"].to_numpy()
    mid = (bid + ask) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_bps = (ask - bid) / mid * 1e4
    return {
        "n_crossed_bid_gt_ask": int((bid > ask).sum()),
        "n_locked_bid_eq_ask": int((bid == ask).sum()),
        "spread_bps_median": float(np.nanmedian(spread_bps)),
        "spread_bps_p99": float(np.nanpercentile(spread_bps, 99)),
        "spread_bps_max": float(np.nanmax(spread_bps)),
        "n_negative_spread": int((spread_bps < 0).sum()),
    }


def time_gaps(ts: np.ndarray, big_gap_s: float = 10.0) -> dict:
    """Inter-event gap statistics (seconds) and the count of gaps above ``big_gap_s``."""
    gaps_s = np.diff(ts) / 1e6
    return {
        "median_s": float(np.median(gaps_s)),
        "p99_s": float(np.percentile(gaps_s, 99)),
        "p999_s": float(np.percentile(gaps_s, 99.9)),
        "max_s": float(gaps_s.max()),
        f"n_gaps_gt_{big_gap_s:g}s": int((gaps_s > big_gap_s).sum()),
    }


# --------------------------------------------------------------------------- #
# 2. Markout
# --------------------------------------------------------------------------- #
def _asof_mid(bbo_ts: np.ndarray, bbo_mid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Forward-filled mid at query times ``q`` (NaN before first / after last quote)."""
    idx = np.searchsorted(bbo_ts, q, side="right") - 1
    out = np.where(idx >= 0, bbo_mid[np.clip(idx, 0, len(bbo_mid) - 1)], np.nan)
    out[q < bbo_ts[0]] = np.nan
    out[q > bbo_ts[-1]] = np.nan
    return out


def compute_markout(
    trades: pl.DataFrame, bbo: pl.DataFrame, taus_s: tuple[int, ...] = TAUS_S
) -> dict:
    """Per-trade maker PnL (bps) and weighted summaries at each horizon.

    Returns ``{"per_trade": {tau: pnl_array}, "weight": w, "summary": pl.DataFrame}``.
    ``summary`` holds PnL_all (weighted mean over trades with a defined markout), the share
    of trades excluded because ``t + tau`` exceeds the available BBO, and PnL std.
    """
    trades = trades.sort("timestamp")
    bbo = bbo.sort("timestamp")
    t = trades["timestamp"].to_numpy()
    p = trades["price"].to_numpy()
    s = np.where((trades["side"] == "buy").to_numpy(), 1.0, -1.0)
    notional = p * trades["amount"].to_numpy()
    w = np.minimum(notional, NOTIONAL_CLIP)

    bts = bbo["timestamp"].to_numpy()
    bmid = ((bbo["bid_price"] + bbo["ask_price"]) / 2.0).to_numpy()

    per_trade = {}
    rows = []
    for tau in taus_s:
        m = _asof_mid(bts, bmid, t + tau * 1_000_000)
        pnl = -s * (m - p) / p * 1e4 + REBATE_BPS
        per_trade[tau] = pnl
        valid = ~np.isnan(pnl)
        wv, pv = w[valid], pnl[valid]
        pnl_all = float(np.sum(wv * pv) / np.sum(wv)) if wv.sum() > 0 else float("nan")
        rows.append(
            {
                "tau_s": tau,
                "pnl_all_bps": pnl_all,
                "pnl_std_bps": float(np.nanstd(pv)),
                "n_valid": int(valid.sum()),
                "excluded_share": float((~valid).mean()),
            }
        )
    return {"per_trade": per_trade, "weight": w, "summary": pl.DataFrame(rows)}


# --------------------------------------------------------------------------- #
# 3. Liquidation cascades
# --------------------------------------------------------------------------- #
def combined_liquidations(liq_binance: pl.DataFrame, liq_bybit: pl.DataFrame) -> pl.DataFrame:
    """Concatenate both venues with Bybit shifted +200 ms; add ``notional`` and ``venue``."""
    a = liq_binance.with_columns(
        (pl.col("price") * pl.col("amount")).alias("notional"),
        pl.lit("binance").alias("venue"),
    )
    b = liq_bybit.with_columns(
        (pl.col("timestamp") + BYBIT_LAG_US).alias("timestamp"),
        (pl.col("price") * pl.col("amount")).alias("notional"),
        pl.lit("bybit").alias("venue"),
    )
    cols = ["timestamp", "side", "price", "amount", "notional", "venue"]
    return pl.concat([a.select(cols), b.select(cols)]).sort("timestamp")


def detect_cascades(
    liq: pl.DataFrame,
    gap_s: float = 1.0,
    min_events: int = 5,
    min_notional: float = 1_000_000.0,
) -> pl.DataFrame:
    """Time-cluster liquidations into cascades.

    Consecutive events whose inter-arrival gap is below ``gap_s`` join the same cluster;
    a cluster is a *cascade* when it has at least ``min_events`` events and at least
    ``min_notional`` total notional.  Returns one row per cluster with size, notional,
    duration, dominant side and how many venues took part.
    """
    liq = liq.sort("timestamp")
    ts = liq["timestamp"].to_numpy()
    new_cluster = np.concatenate([[0], (np.diff(ts) > gap_s * 1e6).astype(int)])
    cluster_id = np.cumsum(new_cluster)
    liq = liq.with_columns(pl.Series("cluster_id", cluster_id))

    grp = (
        liq.group_by("cluster_id")
        .agg(
            pl.len().alias("n_events"),
            pl.col("notional").sum().alias("total_notional"),
            ((pl.col("timestamp").max() - pl.col("timestamp").min()) / 1e6).alias("duration_s"),
            pl.col("timestamp").min().alias("start_ts"),
            (pl.col("notional").filter(pl.col("side") == "buy").sum()).alias("buy_notional"),
            pl.col("venue").n_unique().alias("n_venues"),
        )
        .with_columns(
            (pl.col("buy_notional") / pl.col("total_notional")).alias("buy_share"),
        )
        .sort("start_ts")
    )
    return grp.filter(
        (pl.col("n_events") >= min_events) & (pl.col("total_notional") >= min_notional)
    )


# --------------------------------------------------------------------------- #
# 4. Turnover constraint
# --------------------------------------------------------------------------- #
def daily_clipped_turnover(trades_path: str) -> pl.DataFrame:
    """Per-day sum of clipped notional ``min(price*amount, 100k)`` via lazy aggregation.

    Streams the (huge) trades parquet without materialising it, so it runs in bounded memory.
    """
    return (
        pl.scan_parquet(trades_path)
        .select(
            (pl.col("timestamp") // 86_400_000_000).alias("day"),
            pl.min_horizontal(pl.col("price") * pl.col("amount"), pl.lit(NOTIONAL_CLIP)).alias(
                "clipped"
            ),
        )
        .group_by("day")
        .agg(pl.col("clipped").sum().alias("clipped_turnover"), pl.len().alias("n_trades"))
        .sort("day")
        .collect(engine="streaming")
    )


def min_turnover_share(daily: pl.DataFrame) -> dict:
    """Minimum share of total clipped turnover that still satisfies $500k/day on average."""
    avg_daily = float(daily["clipped_turnover"].mean())
    share = MIN_TURNOVER_PER_DAY / avg_daily
    return {
        "avg_daily_clipped_turnover_usd": avg_daily,
        "min_turnover_per_day_usd": MIN_TURNOVER_PER_DAY,
        "min_share_to_keep": share,
        "max_filtered_share": 1.0 - share,
    }
