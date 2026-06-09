"""
Base protocol for feature blocks.

A FeatureBlock is a callable that takes market data frames and trade timestamps,
and returns a DataFrame of feature columns aligned with the trades index.

All implementations must guarantee strict causality: for trade i at time t_i,
the feature values use only data with timestamp < t_i.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd


class FeatureBlock(Protocol):
    """
    Protocol for composable feature blocks.

    Implementations should:
      - accept the relevant subset of (trades, bbo, liq_binance, liq_bybit)
      - return a DataFrame with the same index as trades
      - name columns descriptively (e.g. 'liq_ewma_binance_buy_5s')
      - guarantee strict causality
    """

    @abstractmethod
    def compute(self, trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute feature columns for each trade.

        Parameters
        ----------
        trades : DataFrame with at least [timestamp, side, price, amount]
        **kwargs : additional frames (bbo=, liq_binance=, liq_bybit=)

        Returns
        -------
        DataFrame with same length/index as trades, containing only feature columns.
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of column names this block produces."""
        ...
