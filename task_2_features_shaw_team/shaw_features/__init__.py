"""Shaw Team - reusable feature-engineering infrastructure for the liquidation filter task.

Public surface:

    from shaw_features import (
        MarketContext, FeatureSet, BaseFeature, default_features,
        validate_feature, validate_feature_set, loader,
    )
"""
from __future__ import annotations

from . import features, loader, windows
from .base import BaseFeature
from .context import BYBIT_LAG_US, MarketContext
from .features import default_features
from .pipeline import FeatureSet
from .validation import (
    check_alignment,
    check_finite,
    check_no_lookahead,
    validate_feature,
    validate_feature_set,
)

__all__ = [
    "MarketContext",
    "BYBIT_LAG_US",
    "FeatureSet",
    "BaseFeature",
    "default_features",
    "features",
    "loader",
    "windows",
    "check_finite",
    "check_alignment",
    "check_no_lookahead",
    "validate_feature",
    "validate_feature_set",
]
