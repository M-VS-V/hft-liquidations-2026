"""Common feature interface.

A feature is a small, stateless object that turns a :class:`MarketContext` into one float
per trade.  Subclasses declare ``name`` (output column), ``lookback_s`` (how far back in
seconds the feature looks - used only for documentation / validation reporting) and the
``streams`` they read, then implement :meth:`calculate`.

Keeping features stateless and pure (output depends only on the context) is what lets the
differential validator recompute a single trade on truncated data and compare.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .context import MarketContext


class BaseFeature(ABC):
    name: str = ""
    lookback_s: float = 0.0
    streams: tuple[str, ...] = ()

    @abstractmethod
    def calculate(self, ctx: MarketContext) -> np.ndarray:
        """Return a float array of length ``ctx.n_trades`` aligned to ``ctx.trade_ts``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, lookback_s={self.lookback_s})"
