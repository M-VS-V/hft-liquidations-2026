"""Causal window primitives over time-sorted event streams.

Every helper answers a question about a query time ``q`` (a trade timestamp) using only
events strictly **before** ``q``.  The causal window is the half-open interval ``[q - W, q)``:
the upper bound uses ``searchsorted(..., side="left")`` so an event sharing the trade's own
timestamp is *not* counted.  This is what makes the features look-ahead free by construction.

All timestamps are int64 microseconds.
"""
from __future__ import annotations

import numpy as np


def build_prefix(values: np.ndarray) -> np.ndarray:
    """Cumulative sum with a leading zero, so a range sum is ``prefix[hi] - prefix[lo]``."""
    return np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])


def window_sum(
    src_ts: np.ndarray, prefix: np.ndarray, q: np.ndarray, window_us: int
) -> np.ndarray:
    """Sum the values whose timestamps fall in ``[q - window_us, q)`` for each query ``q``.

    ``prefix`` must be ``build_prefix(values)`` of the same stream as ``src_ts``.
    """
    hi = np.searchsorted(src_ts, q, side="left")
    lo = np.searchsorted(src_ts, q - window_us, side="left")
    return prefix[hi] - prefix[lo]


def window_count(src_ts: np.ndarray, q: np.ndarray, window_us: int) -> np.ndarray:
    """Number of events in ``[q - window_us, q)`` for each query ``q``."""
    hi = np.searchsorted(src_ts, q, side="left")
    lo = np.searchsorted(src_ts, q - window_us, side="left")
    return (hi - lo).astype(np.float64)


def time_since_last(
    src_ts: np.ndarray, q: np.ndarray, default_s: float = 1e9
) -> np.ndarray:
    """Seconds between ``q`` and the most recent event strictly before ``q``.

    Returns ``default_s`` when no prior event exists (or the stream is empty).
    """
    if src_ts.size == 0:
        return np.full(len(q), default_s)
    idx = np.searchsorted(src_ts, q, side="left") - 1
    last = np.where(idx >= 0, src_ts[np.clip(idx, 0, len(src_ts) - 1)], np.nan)
    out = (q - last) / 1e6
    return np.where(np.isnan(out), default_s, out)


def asof_prev(
    src_ts: np.ndarray, src_val: np.ndarray, q: np.ndarray, allow_future: bool = False
) -> np.ndarray:
    """Forward-filled value: the last ``src_val`` with ``src_ts <= q`` (NaN before the first).

    With ``allow_future=False`` (default) any ``q`` past the last observed timestamp also
    yields NaN, which is the correct behaviour for a markout label read at ``t + tau``.
    Reading the quote *at* the trade time (offset 0) is always observable, so callers that
    want the current quote should not need that guard.
    """
    if src_ts.size == 0:
        return np.full(len(q), np.nan)
    idx = np.searchsorted(src_ts, q, side="right") - 1
    out = np.where(idx >= 0, src_val[np.clip(idx, 0, len(src_val) - 1)], np.nan).astype(
        np.float64
    )
    out[q < src_ts[0]] = np.nan
    if not allow_future:
        out[q > src_ts[-1]] = np.nan
    return out
