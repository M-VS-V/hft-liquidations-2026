"""
strategies.liq_filter — Liquidation-based maker filter for Binance perps.

Application-specific code: scoring, thresholds, heuristic benchmark,
training orchestration, and the final make_filter submission function.

This package imports from core/ for features, transforms, targets, and data loading.
It does NOT define any feature computation logic of its own.
"""
