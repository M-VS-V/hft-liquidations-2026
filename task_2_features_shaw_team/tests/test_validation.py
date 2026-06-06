import numpy as np

from shaw_features import (
    default_features,
    validate_feature,
    validate_feature_set,
)
from shaw_features.base import BaseFeature


class FutureLiqCount(BaseFeature):
    """Deliberately look-ahead feature: counts liquidations in [t, t + window).

    It bypasses the causal accessors and reads future events, so the differential
    validator must flag it.  Used only to prove the validator works.
    """

    name = "future_liq_count"
    lookback_s = 0.0
    streams = ("combined",)

    def __init__(self, window_s: int = 60):
        self.window_us = window_s * 1_000_000

    def calculate(self, ctx):
        ts = ctx.streams["combined"].ts
        hi = np.searchsorted(ts, ctx.trade_ts + self.window_us, side="right")
        lo = np.searchsorted(ts, ctx.trade_ts, side="left")
        return (hi - lo).astype(np.float64)


def test_default_features_all_valid(ctx):
    report = validate_feature_set(default_features(), ctx)
    assert report["ok"].all(), report


def test_no_nan_no_inf(ctx):
    for row in validate_feature_set(default_features(), ctx).iter_rows(named=True):
        assert row["n_nan"] == 0
        assert row["n_inf"] == 0
        assert row["aligned"]


def test_validator_catches_lookahead(ctx):
    res = validate_feature(FutureLiqCount(60), ctx)
    assert res["no_lookahead_ok"] is False
    assert res["lookahead_violations"] > 0
    assert res["ok"] is False
