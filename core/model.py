"""
Model training and prediction interface.

Decoupled from feature computation. Accepts a feature matrix and targets,
returns a trained model that can predict on new features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series | None = None,
    model_params: dict | None = None,
) -> Any:
    """
    Train a model on features → target.

    Default: LGBMRegressor with Huber loss and sample_weight.
    Returns the fitted model object (supports .predict(features) → np.ndarray).

    Parameters
    ----------
    features      : feature matrix (NaN rows should be pre-dropped by caller)
    target        : regression target (e.g. pnl_{tau})
    sample_weight : per-sample weight (e.g. clipped notional w_i)
    model_params  : override default LightGBM hyperparameters
    """
    raise NotImplementedError


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Run model prediction. Returns float64 array, higher = better predicted trade."""
    raise NotImplementedError
