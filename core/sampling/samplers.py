"""
Sampling strategies: every trade, volume-triggered, time-triggered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Sampler(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        """
        Return a boolean mask of length len(trades).
        True = this trade is a datapoint in the supervised dataset.
        """
        ...


class EveryTrade(Sampler):
    """Use every trade as a datapoint. Default for the liquidation filter task."""

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        return np.ones(len(trades), dtype=bool)


class VolumeThreshold(Sampler):
    """
    Emit a datapoint every time cumulative traded notional exceeds a threshold.
    The emitted row is the trade that crossed the threshold.
    """

    def __init__(self, notional_threshold: float = 100_000.0):
        self.notional_threshold = notional_threshold

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class TimeInterval(Sampler):
    """Emit a datapoint every N seconds (the last trade in each interval)."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
