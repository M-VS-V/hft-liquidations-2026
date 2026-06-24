"""
core.features — Composable feature blocks.

Each module exposes functions that take raw market data frames and produce
feature columns. All features are strictly causal: for a trade at time t_i,
only information with timestamp < t_i is used.
"""

from core.features.base import FeatureBlock
