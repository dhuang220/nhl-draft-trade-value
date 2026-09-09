import pandas as pd
import pytest

from src.models.positional_scarcity import add_value_above_replacement, compute_replacement_levels


def test_replacement_level_is_25th_percentile_per_position():
    df = pd.DataFrame(
        {
            "position_group": ["F"] * 4 + ["D"] * 4,
            "predicted_value_total": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    levels = compute_replacement_levels(df)
    assert levels["F"] == pytest.approx(1.75)
    assert levels["D"] == pytest.approx(17.5)


def test_value_above_replacement_uses_own_position_baseline():
    df = pd.DataFrame({"position_group": ["F", "D"], "predicted_value_total": [5.0, 5.0]})
    levels = {"F": 1.0, "D": 4.0}
    out = add_value_above_replacement(df, levels)
    assert out.loc[0, "value_above_replacement"] == 4.0
    assert out.loc[1, "value_above_replacement"] == 1.0
