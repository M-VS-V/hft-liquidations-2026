import numpy as np

from shaw_features import FeatureSet, default_features
from shaw_features.features import LiqNotional, LiqSideImbalance, TimeSinceLiq

S = 1_000_000


def test_default_set_runs_and_aligns(ctx):
    fs = FeatureSet(default_features())
    df = fs.generate(ctx)
    assert df.height == ctx.n_trades
    assert set(fs.names).issubset(df.columns)


def test_liq_notional_value(ctx):
    # trade at t=50s; combined liq before 50s: binance buy@5s (100*10=1000),
    # binance sell@45s (101*20=2020), bybit sell@30.2s (100.5*5=502.5).
    f = LiqNotional(120)
    val = f.calculate(ctx)[1]
    assert abs(val - (1000.0 + 2020.0 + 502.5)) < 1e-6


def test_side_imbalance_in_range(ctx):
    v = LiqSideImbalance(30).calculate(ctx)
    assert np.all(v >= -1.0) and np.all(v <= 1.0)


def test_time_since_liq_nonnegative(ctx):
    v = TimeSinceLiq().calculate(ctx)
    assert np.all(v >= 0)
