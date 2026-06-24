"""
Final submission function — called on the hidden test.

make_filter() is self-contained: it embeds pre-trained model/params
and calibrates the threshold on the test data via turnover (label-free).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.liq_filter.config import TAUS, FittedPipeline


# Populated by run_train() before submission. Baked into module scope.
FITTED: FittedPipeline | None = None


def make_filter(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """
    FINAL submission function.

    Accepts 4 frames (same schemas as public files; liq_bybit arrives UNSHIFTED).
    Returns { 30: arr_30, 120: arr_120, 300: arr_300 },
    each arr is np.ndarray of length len(trades) with values 0 or 1.

    Internal flow:
      1. Detect symbol, shift Bybit +200ms, sort.
      2. Compute features via DatasetBuilder (causal).
      3. For each tau: raw_score from FITTED model/strategy.
      4. fit_threshold on test data's w_i (label-free).
      5. apply_filter → 0/1 array.
    """
    if FITTED is None:
        raise RuntimeError("FITTED pipeline not set. Call run_train() first.")
    raise NotImplementedError
