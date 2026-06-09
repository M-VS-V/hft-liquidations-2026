"""
Direction-relative feature transform.

Converts absolute BID/ASK and signed features into same-side / opposite-side
relative to the trade's direction (s_i).

Rule: Dir (s_i) is allowed as its own feature. Every other feature must be
expressed relative to the trade's direction, not in absolute terms.
This is enforced as a pipeline contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def direction_relativize(features: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """
    Convert absolute features to direction-relative form.

    Transformations:
        bid_amount, ask_amount → same_side_depth, opp_side_depth
            same_side = ask_amount where s=+1 (taker bought, hit our ask)
            same_side = bid_amount where s=-1 (taker sold, hit our bid)
        bid_amount_delta_*, ask_amount_delta_* → same_side_depth_delta_*, opp_side_depth_delta_*
        signed_volume_* → same_side_flow_* (= s * signed_volume)
        taker_imbalance_* → same_side_taker_imbalance_* (= s * taker_imbalance)
        liq_ewma_{venue}_buy_*, liq_ewma_{venue}_sell_* → liq_ewma_{venue}_same_*, liq_ewma_{venue}_opp_*

    Parameters
    ----------
    features : DataFrame with absolute feature columns
    s        : int array, +1 (taker buy / maker sell) or -1 (taker sell / maker buy)

    Returns
    -------
    DataFrame with absolute columns replaced by direction-relative columns.
    Columns that are already direction-agnostic (spread_bps, imbalance, vol, etc.)
    pass through unchanged.
    """
    raise NotImplementedError
