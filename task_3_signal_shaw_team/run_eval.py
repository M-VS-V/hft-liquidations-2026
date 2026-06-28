"""Fit the filter on train, evaluate Score(tau) on the full validation month (Shaw Team).

One pass over February accumulates the exact weighted sums for a whole GRID of cutoffs per
horizon, so we see the honest out-of-sample Score-vs-aggressiveness curve instead of a single
cutoff tuned in-sample.  Results are written to results.txt.
"""
from __future__ import annotations

import sys
import numpy as np

from signal_lib import build_table, iter_days, us_of, _load_liq, TAUS, DAY_US

FEATS = ["side", "signed30", "signed120", "signed300", "absliq30", "absliq120",
         "absliq300", "cnt30", "tsl", "spread", "imb", "ret5"]
# candidate cutoffs on predicted markout (bps): keep trade iff pred >= cut
GRID = np.array([-1e9, -1.0, -0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 1.0])

OUT = open("results.txt", "w")
def emit(s=""):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


def design_matrix(t: dict) -> np.ndarray:
    cols = [t[k] for k in FEATS]
    for w in (30, 120, 300):
        cols.append(t["side"] * t[f"signed{w}"])  # maker-adverse interaction
    return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    sym = "btc"
    lb, lby = _load_liq(sym)
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    train_windows = ["2025-12-02", "2025-12-18", "2026-01-05", "2026-01-20", "2026-01-29"]
    emit(f"Building train table (stride={stride}) over {train_windows}")
    parts = []
    for d in train_windows:
        t0 = us_of(d)
        t = build_table(sym, t0, t0 + 2 * DAY_US, lb, lby, stride=stride)
        if t:
            parts.append(t)
            emit(f"  {d}: {len(t['w']):,} trades")
    tr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    Xtr = design_matrix(tr)

    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {}
    emit("\nFitting per-tau HistGradientBoosting markout regressors (weighted by w):")
    for tau in TAUS:
        pnl = tr[f"pnl{tau}"]
        m = np.isfinite(pnl)
        reg = HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                                            learning_rate=0.05, l2_regularization=1.0)
        reg.fit(Xtr[m], pnl[m], sample_weight=tr["w"][m])
        models[tau] = reg
        emit(f"  tau={tau:3d}: fit on {m.sum():,} trades")

    # ---- VALIDATION: full February, all trades, one pass, whole cutoff grid ----
    emit("\nEvaluating on FULL validation month (Feb 1-28), all trades, cutoff grid:")
    v0, v1 = us_of("2026-02-01"), us_of("2026-03-01")
    # per tau: global sum_w, sum_wp ; per cutoff: kept_w, kept_wp
    g = {tau: dict(sw=0.0, swp=0.0,
                   ksw=np.zeros(len(GRID)), kswp=np.zeros(len(GRID))) for tau in TAUS}
    ndays = 0
    for d0, d1 in iter_days(v0, v1):
        t = build_table(sym, d0, d1, lb, lby)
        if not t:
            continue
        ndays += 1
        X = design_matrix(t)
        for tau in TAUS:
            pnl = t[f"pnl{tau}"]
            m = np.isfinite(pnl)
            w, p = t["w"][m], pnl[m]
            pred = models[tau].predict(X[m])
            a = g[tau]
            a["sw"] += w.sum(); a["swp"] += (w * p).sum()
            for j, c in enumerate(GRID):
                keep = pred >= c
                a["ksw"][j] += w[keep].sum()
                a["kswp"][j] += (w[keep] * p[keep]).sum()
        emit(f"  day {ndays:2d} done ({t['ts'].size:,} trades)")

    emit(f"\n==== VALIDATION (BTC, Feb 2026, {ndays} days, ALL trades) ====")
    emit("Score = PnL_kept - PnL_all ; constraint kept_turnover/day >= $500,000\n")
    best = {}
    for tau in TAUS:
        a = g[tau]
        all_ = a["swp"] / a["sw"]
        emit(f"tau={tau}s   PnL_all (baseline) = {all_:+.4f} bps")
        emit(f"  {'cutoff':>8} {'filtered%':>9} {'PnL_kept':>9} {'Score':>8} {'kept$/day':>14}")
        bs = (-1e9, None)
        for j, c in enumerate(GRID):
            ksw, kswp = a["ksw"][j], a["kswp"][j]
            if ksw <= 0:
                continue
            kept = kswp / ksw
            sc = kept - all_
            filt = (a["sw"] - ksw) / a["sw"] * 100
            kpd = ksw / ndays
            ok = "OK" if kpd >= 500_000 else "VIOL"
            tag = f"{c:+.2f}" if c > -1e8 else "none"
            emit(f"  {tag:>8} {filt:>8.1f}% {kept:>+9.4f} {sc:>+8.4f} {kpd:>13,.0f} {ok}")
            if kpd >= 500_000 and sc > bs[0]:
                bs = (sc, c)
        best[tau] = bs
        emit(f"  -> best feasible Score = {bs[0]:+.4f} bps at cutoff {bs[1]:+.2f}\n")

    emit("==== SUMMARY (best feasible out-of-sample Score per horizon) ====")
    for tau in TAUS:
        emit(f"  tau={tau:3d}s : Score = {best[tau][0]:+.4f} bps")
    OUT.close()


if __name__ == "__main__":
    main()
