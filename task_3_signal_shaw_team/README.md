# Task 3 — Liquidation-Signal Trade Filter (Shaw Team)

A trade filter for Binance maker fills built on the liquidation signal, with a **full
out-of-sample evaluation** on the validation month. Score is the competition metric

```
Score(τ) = PnL_kept(τ) − PnL_all(τ)        (bps, higher is better)
constraint: kept clipped-turnover ≥ $500,000 / day
```

## Method

- **Labels / markout** reuse the validated Task-2 `MarketContext`: maker PnL in bps at
  `t+τ` with a forward-filled Binance mid and the +0.5 bps rebate, weight `min(notional,100k)`.
  The pipeline reproduces the Task-1 EDA baseline markouts to ±0.002 bps, so labels are correct.
- **Features** (all causal, `[t−W, t)` only, Bybit shifted +200 ms): combined liquidation
  signed notional / |notional| over 30·120·300 s, liquidation count, time-since-liquidation,
  BBO spread & book imbalance, 5 s mid return, taker side, and the maker-adverse interaction
  `side × signed_liq_flow` (same sign ⇒ maker is run over ⇒ toxic).
- **Model**: one `HistGradientBoostingRegressor` per horizon predicting `pnl_i(τ)`, weighted
  by `w_i`. Trained on **5 two-day windows spanning Dec 2025 – Jan 2026** (train split).
- **Filter (pre-registered, not tuned on validation)**: filter every trade whose predicted
  markout is negative (cutoff = 0). We also report the full Score-vs-cutoff curve.

## Evaluation

Out-of-sample on the **entire validation month (Feb 1–28 2026), every BTC trade** (~210 M),
day-by-day exact accumulation of the weighted sums — no sampling in the reported numbers.

## Results (BTC, validation = Feb 2026, all trades)

Baseline maker flow is toxic (negative `PnL_all`) at every horizon. The filter at the
pre-registered cutoff = 0:

| τ | PnL_all (baseline) | PnL_kept | **Score** | filtered | kept turnover/day |
|---|---:|---:|---:|---:|---:|
| 30 s  | −0.188 | **+0.193** | **+0.381** | 79% | $3.1 B |
| 120 s | −0.174 | −0.292 | −0.118 | 51% | $7.2 B |
| 300 s | −0.156 | +0.073 | **+0.228** | 51% | $7.1 B |

The Score-vs-aggressiveness curve (one Feb pass over a cutoff grid) is monotone for 30 s and
300 s; pushing the cutoff up to +1.0 bps lifts **Score(30 s) to +0.77 and Score(300 s) to
+0.34**, still keeping >$1 B/day. The 120 s horizon is the weak one — only positive
(+0.25) under very aggressive filtering. The **$500k/day turnover constraint is never binding**
(kept turnover stays in the billions), confirming the EDA finding that the score, not volume,
is the limit.

**Headline:** the filter converts the toxic baseline into positive kept markout out-of-sample —
strongest at τ = 30 s, **Score = +0.38 bps** (PnL_all −0.19 → PnL_kept +0.19).

## Run

```bash
export SHAW_DATA_ROOT=...      # or place data at ../liquidation_task_dataset/data
python3 run_eval.py            # writes results.txt
```

`signal_lib.py` builds markout + features per window; `run_eval.py` trains and runs the
full-month evaluation. Full log in `results.txt`.
