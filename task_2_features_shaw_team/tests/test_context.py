import numpy as np

from shaw_features import BYBIT_LAG_US, MarketContext

S = 1_000_000


def test_bybit_lag_applied(frames):
    trades, bbo, liq_binance, liq_bybit = frames
    ctx = MarketContext.from_frames(trades, bbo, liq_binance, liq_bybit)
    raw = liq_bybit["timestamp"].to_numpy()
    shifted = ctx.streams["bybit"].ts
    assert np.array_equal(shifted, np.sort(raw) + BYBIT_LAG_US)


def test_combined_stream_merges_both_exchanges(frames):
    trades, bbo, liq_binance, liq_bybit = frames
    ctx = MarketContext.from_frames(trades, bbo, liq_binance, liq_bybit)
    assert len(ctx.streams["combined"].ts) == len(liq_binance) + len(liq_bybit)


def test_trade_arrays_sorted_and_signed(ctx):
    assert np.all(np.diff(ctx.trade_ts) >= 0)
    # trades: buy, sell, buy, sell -> +1,-1,+1,-1
    assert list(ctx.trade_side) == [1.0, -1.0, 1.0, -1.0]


def test_bbo_at_current_quote_observable(ctx):
    # mid is forward-filled and present at every trade time (first quote at ts=0).
    mid = ctx.bbo_at("mid", offset_us=0)
    assert np.all(np.isfinite(mid))
