"""Concrete example features for the liquidation-filtering task.

Each feature is a thin causal composition of :class:`MarketContext` accessors, so adding a
new one is a single small class that the pipeline and validators pick up automatically.

The set below is deliberately a *demonstration* of the infrastructure (the task asks for
~10 example features and reusable plumbing, not a tuned model): liquidation pressure over
several windows, its side imbalance and velocity, a recency signal, per-exchange notional,
and BBO context (spread, book imbalance, short mid-return).
"""
from __future__ import annotations

import numpy as np

from .base import BaseFeature
from .context import MarketContext


class LiqNotional(BaseFeature):
    """Total liquidation notional (Binance + Bybit) in the last ``window_s`` seconds."""

    streams = ("combined",)

    def __init__(self, window_s: int):
        self.window_s = window_s
        self.name = f"liq_notional_{window_s}s"
        self.lookback_s = window_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        return ctx.liq_notional("combined", self.window_s)


class LiqCount(BaseFeature):
    """Number of liquidation events in the last ``window_s`` seconds."""

    streams = ("combined",)

    def __init__(self, window_s: int):
        self.window_s = window_s
        self.name = f"liq_count_{window_s}s"
        self.lookback_s = window_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        return ctx.liq_count("combined", self.window_s)


class LiqSideImbalance(BaseFeature):
    """(buy - sell) / (buy + sell) liquidation notional over ``window_s`` seconds.

    Positive ⇒ forced buying dominates (upward pressure); 0 when there is no flow.
    """

    streams = ("combined",)

    def __init__(self, window_s: int):
        self.window_s = window_s
        self.name = f"liq_side_imbalance_{window_s}s"
        self.lookback_s = window_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        buy = ctx.liq_buy_notional("combined", self.window_s)
        sell = ctx.liq_sell_notional("combined", self.window_s)
        total = buy + sell
        out = np.zeros_like(total)
        np.divide(buy - sell, total, out=out, where=total > 0)
        return out


class LiqVelocity(BaseFeature):
    """Short-window notional as a fraction of the long-window notional.

    Near 1 ⇒ the liquidation flow is concentrated in the recent short window (a fresh
    burst); near 0 ⇒ the flow is old.  The ``+1`` floor keeps it finite when there is no
    long-window flow.
    """

    streams = ("combined",)

    def __init__(self, short_s: int = 30, long_s: int = 120):
        self.short_s = short_s
        self.long_s = long_s
        self.name = f"liq_velocity_{short_s}s_{long_s}s"
        self.lookback_s = long_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        short = ctx.liq_notional("combined", self.short_s)
        long = ctx.liq_notional("combined", self.long_s)
        return short / (long + 1.0)


class ExchangeLiqNotional(BaseFeature):
    """Per-exchange liquidation notional over ``window_s`` seconds (``binance``/``bybit``)."""

    def __init__(self, exchange: str, window_s: int):
        assert exchange in ("binance", "bybit")
        self.exchange = exchange
        self.window_s = window_s
        self.streams = (exchange,)
        self.name = f"{exchange}_liq_notional_{window_s}s"
        self.lookback_s = window_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        return ctx.liq_notional(self.exchange, self.window_s)


class TimeSinceLiq(BaseFeature):
    """Seconds since the last liquidation (capped large value when none seen yet)."""

    streams = ("combined",)
    name = "time_since_liq_s"
    lookback_s = 0.0

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        return ctx.time_since_liq("combined")


class BBOSpread(BaseFeature):
    """Relative bid/ask spread in basis points at the trade time.

    A trade that arrives before the first observed quote has no book information; we emit
    the neutral value 0.0 there so the feature stays NaN-free (in production the BBO stream
    starts before the trades and this edge does not occur for interior trades).
    """

    streams = ("bbo",)
    name = "bbo_spread_bps"
    lookback_s = 0.0

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        x = ctx.bbo_at("spread_bps", offset_us=0)
        return np.where(np.isfinite(x), x, 0.0)


class BBOImbalance(BaseFeature):
    """Order-book imbalance (bid_amount - ask_amount)/(sum) at the trade time.

    Neutral 0.0 before the first observed quote (see :class:`BBOSpread`).
    """

    streams = ("bbo",)
    name = "bbo_imbalance"
    lookback_s = 0.0

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        x = ctx.bbo_at("imbalance", offset_us=0)
        return np.where(np.isfinite(x), x, 0.0)


class MidReturn(BaseFeature):
    """Mid-price return (bps) over the trailing ``window_s`` seconds, ending at the trade."""

    streams = ("bbo",)

    def __init__(self, window_s: int = 5):
        self.window_s = window_s
        self.name = f"mid_return_{window_s}s_bps"
        self.lookback_s = window_s

    def calculate(self, ctx: MarketContext) -> np.ndarray:
        now = ctx.bbo_at("mid", offset_us=0)
        past = ctx.bbo_at("mid", offset_us=-self.window_s * 1_000_000)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (now - past) / past * 1e4
        return np.where(np.isfinite(ret), ret, 0.0)


def default_features() -> list[BaseFeature]:
    """The 12-feature example set used in the demo and tests."""
    return [
        LiqNotional(30),
        LiqNotional(120),
        LiqNotional(300),
        LiqCount(30),
        LiqSideImbalance(30),
        LiqVelocity(30, 120),
        ExchangeLiqNotional("binance", 30),
        ExchangeLiqNotional("bybit", 30),
        TimeSinceLiq(),
        BBOSpread(),
        BBOImbalance(),
        MidReturn(5),
    ]
