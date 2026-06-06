# Task 2 — Feature Engineering Infrastructure (Shaw Team)

Reusable infrastructure to **generate and validate** features for the liquidation-based
trade-filtering task. The emphasis is the plumbing — a common feature interface, a generation
pipeline, and validation that guarantees features are clean, aligned and **causal** — rather
than PnL tuning. Adding a feature is one small class that the pipeline and validators pick up
automatically.

## What's implemented

| Piece | File |
|---|---|
| Feature interface — `calculate(ctx) -> one value per trade` | `shaw_features/base.py` |
| `MarketContext` — the single Polars boundary | `shaw_features/context.py` |
| Causal window primitives over event streams | `shaw_features/windows.py` |
| 12 example features | `shaw_features/features.py` |
| Validators (no NaN, no inf, alignment, no look-ahead) | `shaw_features/validation.py` |
| `FeatureSet` generation pipeline (+ streaming) | `shaw_features/pipeline.py` |
| Data loading / sampling | `shaw_features/loader.py` |
| Unit tests (synthetic, no big data needed) | `tests/` |
| Guided demo on real BTC data | `feature_infra_demo.ipynb` |

## Design

### Causality is structural, not checked after the fact
`MarketContext.from_frames` does the one-off preparation:

- shifts **Bybit liquidations +200 ms** (cross-exchange availability delay) before any look-up;
- sorts each liquidation stream (`binance`, `bybit`, `combined`) and precomputes prefix sums,
  so a windowed sum is two `searchsorted` calls and a subtraction;
- derives the BBO series (mid, spread in bps, book imbalance) once.

Features never touch the raw frames — only the context's causal accessors. The causal window
is the half-open interval **`[t − W, t)`**: the upper bound uses `searchsorted(side="left")`,
so an event at the trade's own timestamp is excluded. Reading the current quote uses an as-of
(forward-filled) look-up at `t`.

### Example feature set (12)
`liq_notional_{30,120,300}s`, `liq_count_30s`, `liq_side_imbalance_30s`,
`liq_velocity_30s_120s`, `binance_liq_notional_30s`, `bybit_liq_notional_30s`,
`time_since_liq_s`, `bbo_spread_bps`, `bbo_imbalance`, `mid_return_5s_bps`.

### Validation
For every feature:

- **finite** — zero NaN, zero inf;
- **alignment** — exactly one value per trade, same order as `trade_ts`;
- **no look-ahead** — a *differential* test: each probe trade is recomputed on a context
  truncated to the data available at its timestamp (`ctx.truncate_at`) and must match the
  full-data value. A feature that peeks at a future liquidation or quote changes and is
  flagged. (It is an empirical guarantee, not a formal proof — raise `n_probe` on huge data.)

## Validation results

On a 3-hour real BTC slice (≈1.6 M trades), **all 12 example features pass every check**
(`finite_ok`, `aligned`, `no_lookahead_ok` all true). The same `FutureLiqCount` probe that
reads `[t, t+60s)` is **rejected** (`no_lookahead_ok = False`, hundreds of violations),
confirming the validator detects look-ahead. See `feature_infra_demo.ipynb` for the full
report.

## Run

```bash
pip install -r requirements.txt
pytest                       # unit tests (synthetic data)

export SHAW_DATA_ROOT=/path/to/liquidation_task/data
jupyter nbconvert --to notebook --execute feature_infra_demo.ipynb
```

### Full-data generation
The trades table is hundreds of millions of rows; the full matrix does not fit in RAM.
`FeatureSet.generate_streaming(trades_path, bbo, liq_binance, liq_bybit, out_path=...)`
prepares the streams once and writes feature batches to disk, so memory stays bounded.

## Status
- **Done:** feature classes + common interface, validators, 12-feature example set, generation
  pipeline (in-memory + streaming), data loader, unit tests, demo notebook on real data.
- **Not in scope here:** model training / PnL optimisation (the filter itself).
