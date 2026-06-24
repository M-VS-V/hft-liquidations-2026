"""
Training and evaluation orchestration.

Wires together: data loading → targets → features → model/strategy → threshold → scoring.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from strategies.liq_filter.config import (
    TAUS, SYMBOLS, TURNOVER_FLOOR_PER_DAY, FeatureConfig, FittedPipeline,
)
from strategies.liq_filter.scoring import ScoreReport


Symbol = Literal["btcusdt", "ethusdt"]


def run_train(
    data_dir: str,
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    feature_config: FeatureConfig | None = None,
    model_params: dict | None = None,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    verbose: bool = False,
) -> FittedPipeline:
    """
    Train on the train split for the given symbols. Produces a FittedPipeline.

    For each symbol:
      1. load_data_with_required_preprocess(data_dir, symbol, split='train')
      2. add_mid → compute_markout → compute_pnl
      3. make_features via DatasetBuilder
      4. For each tau: train_model or store strategy params

    Threshold is NOT fitted here — calibrated at inference time via fit_threshold.
    """
    raise NotImplementedError


def run_eval(
    data_dir: str,
    split: str,
    fitted: FittedPipeline,
    symbols: tuple[str, ...] = SYMBOLS,
    verbose: bool = False,
) -> dict[str, dict[int, ScoreReport]]:
    """
    Evaluate on the given split using a pre-trained FittedPipeline.

    Returns {symbol: {tau: ScoreReport}}.
    ONE-SHOT evaluation — do NOT iterate on the threshold to improve val Score.
    """
    raise NotImplementedError


def run_experiment(
    data_dir: str,
    name: str = "experiment",
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    model_params: dict | None = None,
    verbose: bool = False,
) -> tuple[FittedPipeline, dict, pd.DataFrame]:
    """
    Full experiment: train on train split, evaluate on both train and validation.
    Returns (fitted_pipeline, raw_reports_dict, summary_dataframe).
    """
    raise NotImplementedError
