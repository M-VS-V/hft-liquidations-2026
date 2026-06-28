"""Generate the 1-page best-results PDF for the form (Shaw Team)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPO = "https://github.com/M-VS-V/hft-liquidations-2026"
BRANCH = "main"

styles = getSampleStyleSheet()
h = ParagraphStyle("h", parent=styles["Title"], fontSize=15, spaceAfter=2)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=6)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.2, leading=12.5, spaceAfter=5)
small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.2, leading=11, textColor=colors.HexColor("#333333"))
hh = ParagraphStyle("hh", parent=styles["Heading2"], fontSize=10.5, spaceBefore=6, spaceAfter=3)

doc = SimpleDocTemplate("report.pdf", pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm, topMargin=14*mm, bottomMargin=12*mm)
e = []
e.append(Paragraph("HFT Liquidation Signal — Best Results (Shaw Team)", h))
e.append(Paragraph("Trade filter for Binance maker fills. Metric: Score(τ) = PnL_kept(τ) − PnL_all(τ) (bps, higher is better). "
                   "Out-of-sample on the FULL validation month (Feb 2026, every BTC trade).", sub))

e.append(Paragraph("Method", hh))
e.append(Paragraph(
    "HistGradientBoosting markout regressor (one per horizon), trained on Dec 2025–Jan 2026, "
    "filtering trades with negative predicted markout (cutoff = 0, fixed in advance — not tuned on validation). "
    "Causal features (read [t−W, t) only, Bybit liquidations shifted +200 ms): combined liquidation signed / "
    "absolute notional over 30·120·300 s, liquidation count, time-since-liquidation, BBO spread &amp; book imbalance, "
    "5 s mid-return, taker side, and the maker-adverse interaction side×signed-flow. "
    "Markout labels reproduce the EDA baseline to ±0.002 bps.", body))

e.append(Paragraph("Results — BTC, validation = Feb 2026, ALL trades (~210 M), exact day-by-day accounting", hh))
data = [
    ["τ", "PnL_all (baseline)", "PnL_kept", "Score", "filtered", "kept turnover/day"],
    ["30 s",  "−0.188", "+0.193", "+0.381", "79%", "$3.1 B"],
    ["120 s", "−0.174", "−0.292", "−0.118", "51%", "$7.2 B"],
    ["300 s", "−0.156", "+0.073", "+0.228", "51%", "$7.1 B"],
]
t = Table(data, colWidths=[16*mm, 34*mm, 24*mm, 22*mm, 20*mm, 38*mm])
t.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 8.6),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3b66")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef2f8")]),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3), ("TOPPADDING", (0,0), (-1,-1), 3),
]))
e.append(t)
e.append(Spacer(1, 4))
e.append(Paragraph(
    "<b>Headline:</b> the filter converts the toxic baseline into positive kept markout out-of-sample. "
    "Strongest at τ = 30 s: <b>Score = +0.38 bps</b> (PnL_all −0.19 → PnL_kept +0.19). "
    "The Score-vs-aggressiveness curve is monotone for 30 s &amp; 300 s; a more aggressive cutoff (+1.0 bps) "
    "lifts <b>Score(30 s) to +0.77</b> and <b>Score(300 s) to +0.34</b> while still keeping &gt;$1 B/day. "
    "The $500k/day turnover constraint is never binding (kept turnover stays in the billions).", body))

e.append(Paragraph("Links (GitHub)", hh))
links = (
    f'• Task 3 — signal &amp; full OOS evaluation: <a href="{REPO}/tree/{BRANCH}/task_3_signal_shaw_team">{REPO}/tree/{BRANCH}/task_3_signal_shaw_team</a><br/>'
    f'&nbsp;&nbsp;&nbsp;results log: <a href="{REPO}/blob/{BRANCH}/task_3_signal_shaw_team/results.txt">.../task_3_signal_shaw_team/results.txt</a><br/>'
    f'• Task 2 — feature infrastructure (MarketContext, markout, causal windows): <a href="{REPO}/tree/{BRANCH}/task_2_features_shaw_team">{REPO}/tree/{BRANCH}/task_2_features_shaw_team</a><br/>'
    f'• Task 1 — EDA (baseline markouts, cascades): <a href="{REPO}/tree/{BRANCH}/task_1_eda_shaw_team">{REPO}/tree/{BRANCH}/task_1_eda_shaw_team</a>'
)
e.append(Paragraph(links, small))

doc.build(e)
print("wrote report.pdf")
