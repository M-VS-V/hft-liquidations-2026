# Task 1 — EDA: Liquidation Signal Dataset (Shaw Team)

Exploratory analysis of the 3-month Binance/Bybit dataset behind the liquidation-based
trade-filtering task, answering the four assignment questions: data quality, markout,
liquidation cascades, and the turnover constraint.

## Files
| File | Purpose |
|---|---|
| `eda.py` | Reusable, framework-light EDA computations (quality checks, markout, cascade detection, turnover) |
| `eda.ipynb` | Executed notebook: tables, plots and written conclusions (BTC) |
| `requirements.txt` | Dependencies |

## How to run
```bash
pip install -r requirements.txt
export SHAW_DATA_ROOT=/path/to/liquidation_task/data   # holds binance_trades/ etc.
jupyter nbconvert --to notebook --execute eda.ipynb
```
The notebook works on 2-day windows for the markout/quality sections (millions of trades) and
streams the full trades table for the daily-turnover aggregation, so it runs in bounded memory.

## Key findings (BTC)

**1. Data quality — clean; the surprises are market features, not corruption.**
No NaNs, no non-positive prices, no unexpected `side` values, no crossed/locked books, no
negative spreads. Two notable properties:
- **Duplicate timestamps** dominate (~0.7–0.8 of trades share a microsecond) — the individual
  fills of one aggressive marketable order swept across price levels, reported at the same
  microsecond. Order is still non-decreasing, so it is a safe event sequence.
- **Time gaps** are tiny (sub-second p99.9, a few seconds max) — quiet tape, not missing data.

**2. Markout — a clear regime shift.**
Weighted maker PnL_all (bps), `min(notional,100k)` weights:

| window | τ=30s | τ=120s | τ=300s |
|---|---|---|---|
| train Dec 1–3 | +0.04 | +0.02 | −0.24 |
| valid Feb 1–3 | −0.22 | −0.33 | −0.16 |

Baseline flow is roughly break-even in train but **toxic across horizons in validation**. The
distribution is near-symmetric around zero with heavy tails that widen with τ; the +0.5 bps
rebate is what keeps the unfiltered baseline near zero. Rules must be validated out-of-sample.

**3. Liquidations & cascades — bursty and cross-exchange.**
342,910 combined events, **$4.205B** total notional. Simple 1-second time-clustering
(≥5 events, ≥$1M) finds **623 cascades**: mean ~127 events, ~$2.8M, ~5.6s long; the largest
reach thousands of events / tens of $M. **~99.5% of cascades are multi-venue** (Binance and
Bybit fire together), and the median `buy_share` is 0 — cascades are typically one-sided
**forced selling**. These cascades are the mechanistic source of the toxic markout.

**4. Turnover constraint — effectively non-binding.**
BTC trades **~$12B/day** of clipped turnover, so the $500k/day floor is only ~**4×10⁻⁵** of it:
we could keep as little as **0.004%** of turnover and still satisfy the constraint. The score,
not the kept volume, is the binding limit — the filter can be aggressive in toxic regimes.

## Implication for the filter
Build features around windowed, **+200 ms-shifted**, combined liquidation flow (notional, side
imbalance, recency) plus BBO context; validate strictly out-of-sample (train ≠ validation);
filter aggressively during cascades since the turnover budget allows it. That feature
infrastructure is implemented in `task_2_features_shaw_team`.
