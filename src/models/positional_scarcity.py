"""Convert raw predicted value into value-above-replacement, adjusted per position.

Forwards, defensemen, and goalies produce very different raw value totals just
because of their role - comparing them directly would make every defenseman
look worse than a middling forward. Replacement level (conventionally the
bottom quartile of NHL-rostered players at a position) gives each position its
own zero point, so a value of e.g. +2.0 means the same thing regardless of
position: two Point-Shares better than a freely-available replacement.
"""

import pandas as pd

REPLACEMENT_PERCENTILE = 25


def compute_replacement_levels(
    df: pd.DataFrame, position_col: str = "position_group", value_col: str = "predicted_value_total"
) -> dict[str, float]:
    return df.groupby(position_col)[value_col].apply(
        lambda s: s.quantile(REPLACEMENT_PERCENTILE / 100)
    ).to_dict()


def add_value_above_replacement(
    df: pd.DataFrame,
    replacement_levels: dict[str, float],
    position_col: str = "position_group",
    value_col: str = "predicted_value_total",
) -> pd.DataFrame:
    out = df.copy()
    out["replacement_level"] = out[position_col].map(replacement_levels)
    out["value_above_replacement"] = out[value_col] - out["replacement_level"]
    return out
