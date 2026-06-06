"""FeatureSet - run a list of features into a matrix and validate them.

``generate`` is the in-memory path (a sample or a time window).  ``generate_streaming``
handles the full trades table: the liquidation/BBO streams are prepared once and the trades
are read in batches that reuse them, so memory stays bounded while every windowed feature
still sees the full past stream.  It assumes the trades file is sorted by timestamp and that
features read only the liquidation/BBO streams plus the trade's own fields (true for the
example library).
"""
from __future__ import annotations

import polars as pl

from .base import BaseFeature
from .context import MarketContext
from .validation import validate_feature_set


class FeatureSet:
    def __init__(self, features: list[BaseFeature]):
        self.features = list(features)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def generate(self, ctx: MarketContext) -> pl.DataFrame:
        cols = {"timestamp": ctx.trade_ts}
        for f in self.features:
            cols[f.name] = f.calculate(ctx)
        return pl.DataFrame(cols)

    def validate(self, ctx: MarketContext, **kw) -> pl.DataFrame:
        return validate_feature_set(self.features, ctx, **kw)

    def generate_streaming(
        self,
        trades_path: str,
        bbo: pl.DataFrame,
        liq_binance: pl.DataFrame,
        liq_bybit: pl.DataFrame,
        batch_rows: int = 20_000_000,
        out_path: str | None = None,
    ):
        """Full-data feature generation by streaming the trades table in batches.

        With ``out_path`` each batch is written as it is computed (memory bounded);
        otherwise batches are concatenated and returned, for small inputs.
        """
        import pyarrow.parquet as pq

        empty_trades = pl.read_parquet(trades_path, n_rows=0)
        base = MarketContext.from_frames(empty_trades, bbo, liq_binance, liq_bybit)

        pf = pq.ParquetFile(trades_path)
        writer = None
        parts = []
        for batch in pf.iter_batches(
            batch_size=batch_rows, columns=["timestamp", "side", "price", "amount"]
        ):
            res = self.generate(base.with_trades(pl.from_arrow(batch)))
            if out_path:
                table = res.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
            else:
                parts.append(res)
        if out_path:
            if writer is not None:
                writer.close()
            return out_path
        return pl.concat(parts) if parts else base.with_trades(empty_trades)
