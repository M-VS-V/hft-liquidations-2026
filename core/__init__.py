"""
core — Reusable market-data pipeline framework.

This package knows NOTHING about maker PnL, ScoreReport, turnover constraints,
or make_filter. It provides composable building blocks for:
  - loading and preprocessing market data
  - computing features from trades, book, and liquidation streams
  - applying transforms (normalization, direction-relative conversion)
  - constructing supervised datasets with pluggable sampling and labeling
"""
