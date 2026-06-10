# Liquidation Signal — Maker Filter on Binance Perps

Passive market-maker fill filter using cross-venue liquidation cascade detection.

## Project structure

```
core/                        # Reusable pipeline framework (shared across strategies)
  features/                  # Composable feature blocks
  transforms/                # Reusable operators (zscore, clamp, direction-relative)
  targets/                   # Markout and PnL computation
  sampling/                  # Sampling strategies (every trade, volume-triggered)
  data.py                    # Data loading and preprocessing
  dataset.py                 # DatasetBuilder orchestrator

strategies/liq_filter/       # Application: liquidation-based maker filter
  config.py                  # Constants, dataclasses, feature/strategy config
  scoring.py                 # ScoreReport, score_one, score_all
  threshold.py               # fit_threshold, apply_filter
  strategy.py                # Heuristic benchmark (EWMA liq pressure)
  train.py                   # run_train, run_eval, run_experiment
  submission.py              # make_filter (final entry point for hidden test)

notebooks/                   # Experiments and analysis
data/                        # Parquet files (not in git)
```

## Quick start

```bash
pip install -r requirements.txt

# Run the benchmark
python -c "
from strategies.liq_filter.train import run_experiment
fitted, reports, df = run_experiment('data', use_ml=False, symbols=('btcusdt',))
print(df)
"
```

## Data

Place parquet files in `data/` following this layout:
```
data/binance_trades/perp_btcusdt.parquet
data/binance_trades/perp_ethusdt.parquet
data/binance_booktickers/perp_btcusdt.parquet
data/binance_booktickers/perp_ethusdt.parquet
data/binance_liquidations/perp_btcusdt.parquet
data/binance_liquidations/perp_ethusdt.parquet
data/bybit_liquidations/btcusdt.parquet
data/bybit_liquidations/ethusdt.parquet
```

All timestamps are int64 microseconds since UNIX epoch (UTC).
