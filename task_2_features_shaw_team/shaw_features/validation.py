"""Feature validators.

The task requires four guarantees per feature:

1. **no NaN**            - ``check_finite``
2. **no inf**            - ``check_finite``
3. **correct alignment** - ``check_alignment`` (one value per trade, same order)
4. **no forward-looking**- ``check_no_lookahead``

The look-ahead check is *differential*: for a sample of probe trades it recomputes the
feature on a context truncated to the data available at the trade time (``ctx.truncate_at``)
and requires the value to match the full-data computation.  If a feature peeks at a future
liquidation or quote, the truncated value differs and the check fails.  It is a strong
empirical test rather than a formal proof (a feature that caches state across calls could
fool it, and on huge data a leak confined to a handful of rows needs enough probes to be
hit - raise ``n_probe``).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .base import BaseFeature
from .context import MarketContext


def check_finite(name: str, values: np.ndarray, allow_nan: bool = False) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n_nan = int(np.isnan(v).sum())
    n_inf = int(np.isinf(v).sum())
    ok = n_inf == 0 and (allow_nan or n_nan == 0)
    return {"check": "finite", "ok": ok, "n_nan": n_nan, "n_inf": n_inf}


def check_alignment(name: str, values: np.ndarray, n_trades: int) -> dict:
    ok = values.ndim == 1 and len(values) == n_trades
    return {"check": "alignment", "ok": ok, "len": int(len(values)), "n_trades": n_trades}


def check_no_lookahead(
    feature: BaseFeature,
    ctx: MarketContext,
    n_probe: int = 256,
    seed: int = 0,
    tol: float = 1e-6,
) -> dict:
    """Differential look-ahead test (see module docstring)."""
    full = feature.calculate(ctx)
    n = ctx.n_trades
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= n_probe else rng.choice(n, n_probe, replace=False)

    violations = 0
    max_abs_diff = 0.0
    for i in idx:
        got = feature.calculate(ctx.truncate_at(int(i)))[0]
        ref = full[int(i)]
        both_nan = np.isnan(got) and np.isnan(ref)
        if both_nan:
            continue
        if np.isnan(got) or np.isnan(ref):
            violations += 1
            continue
        diff = abs(float(got) - float(ref))
        if diff > tol:
            violations += 1
            max_abs_diff = max(max_abs_diff, diff)
    return {
        "check": "no_lookahead",
        "ok": violations == 0,
        "n_violations": violations,
        "n_probe": int(len(idx)),
        "max_abs_diff": float(max_abs_diff),
    }


def validate_feature(
    feature: BaseFeature, ctx: MarketContext, allow_nan: bool = False, **kw
) -> dict:
    values = feature.calculate(ctx)
    fin = check_finite(feature.name, values, allow_nan=allow_nan)
    align = check_alignment(feature.name, values, ctx.n_trades)
    look = (
        check_no_lookahead(feature, ctx, **kw)
        if align["ok"]
        else {"ok": False, "n_violations": -1, "max_abs_diff": float("nan")}
    )
    return {
        "feature": feature.name,
        "lookback_s": feature.lookback_s,
        "finite_ok": fin["ok"],
        "n_nan": fin["n_nan"],
        "n_inf": fin["n_inf"],
        "aligned": align["ok"],
        "no_lookahead_ok": look["ok"],
        "lookahead_violations": look["n_violations"],
        "max_abs_diff": look["max_abs_diff"],
        "ok": fin["ok"] and align["ok"] and look["ok"],
    }


def validate_feature_set(
    features: list[BaseFeature], ctx: MarketContext, allow_nan: bool = False, **kw
) -> pl.DataFrame:
    return pl.DataFrame(
        [validate_feature(f, ctx, allow_nan=allow_nan, **kw) for f in features]
    )
