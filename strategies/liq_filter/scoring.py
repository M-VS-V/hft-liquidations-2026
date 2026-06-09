"""
Scoring harness for the liquidation filter task.

Computes Score(τ) = PnL_kept(τ) - PnL_all(τ) and related diagnostics.
This is application-specific — core/ does not know about ScoreReport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.liq_filter.config import TAUS


@dataclass
class ScoreReport:
    """Unified report for one (symbol, tau)."""
    tau: int
    score: float             # PnL_kept - PnL_all (main metric)
    pnl_all: float           # baseline, constant
    pnl_kept: float          # NaN if no valid kept trades
    pnl_filtered: float      # NaN if no valid filtered trades
    kept_turnover_per_day: float
    filtered_turnover_per_day: float
    constraint_ok: bool
    n_trades: int
    n_valid: int
    n_kept: int
    n_filtered: int
    n_edge: int


def score_one(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    num_days: float,
    tau: int,
) -> ScoreReport:
    """
    Compute ScoreReport for one tau.

    Formulas (over valid trades only, where pnl is not NaN):
        PnL_all      = Σ w·pnl         / Σ w
        PnL_kept     = Σ (1-f)·w·pnl   / Σ (1-f)·w
        PnL_filtered = Σ f·w·pnl       / Σ f·w
        Score        = PnL_kept - PnL_all

    Division-by-zero: returns NaN for the affected metric.
    """
    raise NotImplementedError


def score_all(
    trades_with_pnl: pd.DataFrame,
    f_by_tau: dict[int, np.ndarray],
    num_days: float,
) -> dict[int, ScoreReport]:
    """Run score_one for all taus. Unified output format."""
    raise NotImplementedError


def reports_to_frame(
    reports: dict[int, ScoreReport],
    symbol: str = "",
    experiment: str = "",
) -> pd.DataFrame:
    """Convert ScoreReport dict to a summary DataFrame for display."""
    raise NotImplementedError
