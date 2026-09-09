import pandas as pd

from src.models.surplus_value import add_surplus_value, fit_dollar_per_var


def test_bargain_player_has_positive_surplus():
    # A clean market-rate trend across most of the league (one row per VAR level,
    # 0-19), plus one deliberate outlier: huge VAR but paid like a replacement player.
    # With enough well-behaved points, that single outlier shouldn't swing the fitted
    # market rate much - it should just show up as having by far the largest surplus.
    var_levels = list(range(20))
    salaries = [750_000 + 300_000 * v for v in var_levels]
    df = pd.DataFrame({"value_above_replacement": var_levels, "Salary": salaries})
    df = pd.concat(
        [df, pd.DataFrame({"value_above_replacement": [15.0], "Salary": [750_000]})],
        ignore_index=True,
    )

    model = fit_dollar_per_var(df)
    out = add_surplus_value(df, model)

    bargain_row = len(df) - 1
    assert out["surplus_value"].idxmax() == bargain_row
    assert out.loc[bargain_row, "surplus_value"] > out.loc[0, "surplus_value"]
