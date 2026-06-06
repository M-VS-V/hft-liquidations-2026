"""MarketContext - the single boundary between Polars frames and feature code.

Construction (`from_frames`) does all of the heavy, one-off preparation:

* shifts Bybit liquidation timestamps forward by ``BYBIT_LAG_US`` (+200 ms) to model the
  cross-exchange delay before any look-up against Binance trade times;
* sorts each liquidation stream and precomputes prefix sums of notional / buy-notional /
  sell-notional, so any windowed sum is two ``searchsorted`` calls and a subtraction;
* derives the BBO series (mid, spread in bps, book imbalance) once.

Features never touch the raw frames - they only call the causal accessor methods below, which
guarantees they read data at or before the trade time (Bybit at trade time minus the 200 ms it
has already been shifted by).  ``truncate_at`` rebuilds a one-trade context limited to the data
available at ``t`` and powers the differential no-look-ahead validator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import windows

BYBIT_LAG_US = 200_000  # +200 ms cross-exchange availability delay


@dataclass
class _Stream:
    """A time-sorted liquidation stream with prefix sums for fast window queries."""

    ts: np.ndarray
    prefix_notional: np.ndarray
    prefix_buy: np.ndarray
    prefix_sell: np.ndarray

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> "_Stream":
        df = df.sort("timestamp")
        ts = df["timestamp"].to_numpy()
        notional = (df["price"] * df["amount"]).to_numpy().astype(np.float64)
        is_buy = (df["side"] == "buy").to_numpy()
        buy = np.where(is_buy, notional, 0.0)
        sell = np.where(~is_buy, notional, 0.0)
        return cls(
            ts=ts,
            prefix_notional=windows.build_prefix(notional),
            prefix_buy=windows.build_prefix(buy),
            prefix_sell=windows.build_prefix(sell),
        )

    def slice_before(self, t: int) -> "_Stream":
        k = int(np.searchsorted(self.ts, t, side="left"))
        return _Stream(
            ts=self.ts[:k],
            prefix_notional=self.prefix_notional[: k + 1],
            prefix_buy=self.prefix_buy[: k + 1],
            prefix_sell=self.prefix_sell[: k + 1],
        )


@dataclass
class MarketContext:
    trade_ts: np.ndarray
    trade_price: np.ndarray
    trade_side: np.ndarray  # +1 taker buy (maker sell), -1 taker sell (maker buy)
    streams: dict[str, _Stream] = field(default_factory=dict)
    bbo_ts: np.ndarray = field(default=None)
    bbo_series: dict[str, np.ndarray] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------
    @staticmethod
    def _trade_arrays(trades: pl.DataFrame):
        trades = trades.sort("timestamp")
        ts = trades["timestamp"].to_numpy()
        price = trades["price"].to_numpy().astype(np.float64)
        side = np.where((trades["side"] == "buy").to_numpy(), 1.0, -1.0)
        return ts, price, side

    @classmethod
    def from_frames(
        cls,
        trades: pl.DataFrame,
        bbo: pl.DataFrame,
        liq_binance: pl.DataFrame,
        liq_bybit: pl.DataFrame,
        bybit_lag_us: int = BYBIT_LAG_US,
    ) -> "MarketContext":
        ts, price, side = cls._trade_arrays(trades)

        bybit_shifted = liq_bybit.with_columns(
            (pl.col("timestamp") + bybit_lag_us).alias("timestamp")
        )
        combined = pl.concat(
            [
                liq_binance.select("timestamp", "side", "price", "amount"),
                bybit_shifted.select("timestamp", "side", "price", "amount"),
            ]
        )
        streams = {
            "binance": _Stream.from_frame(liq_binance),
            "bybit": _Stream.from_frame(bybit_shifted),
            "combined": _Stream.from_frame(combined),
        }

        bbo = bbo.sort("timestamp")
        bts = bbo["timestamp"].to_numpy()
        bid_p = bbo["bid_price"].to_numpy().astype(np.float64)
        ask_p = bbo["ask_price"].to_numpy().astype(np.float64)
        bid_a = bbo["bid_amount"].to_numpy().astype(np.float64)
        ask_a = bbo["ask_amount"].to_numpy().astype(np.float64)
        mid = (bid_p + ask_p) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            spread_bps = (ask_p - bid_p) / mid * 1e4
            imbalance = (bid_a - ask_a) / (bid_a + ask_a)
        bbo_series = {"mid": mid, "spread_bps": spread_bps, "imbalance": imbalance}

        return cls(
            trade_ts=ts,
            trade_price=price,
            trade_side=side,
            streams=streams,
            bbo_ts=bts,
            bbo_series=bbo_series,
        )

    # ---- shape --------------------------------------------------------------
    @property
    def n_trades(self) -> int:
        return len(self.trade_ts)

    # ---- causal accessors used by features ----------------------------------
    def liq_notional(self, stream: str, window_s: int) -> np.ndarray:
        st = self.streams[stream]
        return windows.window_sum(
            st.ts, st.prefix_notional, self.trade_ts, window_s * 1_000_000
        )

    def liq_buy_notional(self, stream: str, window_s: int) -> np.ndarray:
        st = self.streams[stream]
        return windows.window_sum(
            st.ts, st.prefix_buy, self.trade_ts, window_s * 1_000_000
        )

    def liq_sell_notional(self, stream: str, window_s: int) -> np.ndarray:
        st = self.streams[stream]
        return windows.window_sum(
            st.ts, st.prefix_sell, self.trade_ts, window_s * 1_000_000
        )

    def liq_count(self, stream: str, window_s: int) -> np.ndarray:
        st = self.streams[stream]
        return windows.window_count(st.ts, self.trade_ts, window_s * 1_000_000)

    def time_since_liq(self, stream: str, default_s: float = 1e9) -> np.ndarray:
        return windows.time_since_last(self.streams[stream].ts, self.trade_ts, default_s)

    def bbo_at(self, series: str, offset_us: int = 0) -> np.ndarray:
        """Forward-filled BBO ``series`` at ``trade_ts + offset_us``.

        ``offset_us <= 0`` reads the current/past quote (always observable); a positive
        offset is a future look-up (markout) and returns NaN past the last quote.
        """
        q = self.trade_ts + offset_us
        return windows.asof_prev(
            self.bbo_ts, self.bbo_series[series], q, allow_future=offset_us <= 0
        )

    # ---- helpers for validation / streaming ---------------------------------
    def truncate_at(self, i: int) -> "MarketContext":
        """One-trade context for trade ``i`` keeping only data a causal feature may read.

        Liquidation streams are cut strictly before ``t`` (an event at exactly ``t`` is
        therefore *absent*, so a feature that illegally reads it changes value and is
        flagged); BBO keeps ``<= t`` because the quote at ``t`` is observable.
        """
        t = int(self.trade_ts[i])
        streams = {name: st.slice_before(t) for name, st in self.streams.items()}
        kb = int(np.searchsorted(self.bbo_ts, t, side="right"))
        bbo_series = {k: v[:kb] for k, v in self.bbo_series.items()}
        return MarketContext(
            trade_ts=self.trade_ts[i : i + 1],
            trade_price=self.trade_price[i : i + 1],
            trade_side=self.trade_side[i : i + 1],
            streams=streams,
            bbo_ts=self.bbo_ts[:kb],
            bbo_series=bbo_series,
        )

    def with_trades(self, trades: pl.DataFrame) -> "MarketContext":
        """Reuse the prepared (heavy) streams/BBO with a different batch of trades."""
        ts, price, side = self._trade_arrays(trades)
        return MarketContext(
            trade_ts=ts,
            trade_price=price,
            trade_side=side,
            streams=self.streams,
            bbo_ts=self.bbo_ts,
            bbo_series=self.bbo_series,
        )
