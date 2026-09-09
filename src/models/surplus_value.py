"""Compare on-ice value to salary: is a player outperforming or underperforming their cap hit?

We calibrate a league-wide "dollars per unit of value above replacement" rate
by regressing actual salaries on VAR, then compare each player's predicted
fair-market salary to what they're actually paid. This is the standard
$/WAR-style framing from sports analytics: it answers "is this player a
bargain or an overpay," which is a different question from "is this player
good" (that's what the value model itself already answers).
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_dollar_per_var(df: pd.DataFrame, var_col: str = "value_above_replacement", salary_col: str = "Salary"):
    """Fit Salary ~ VAR. Keeping the intercept (rather than forcing the line
    through the origin) accounts for the league-minimum salary every roster
    player earns regardless of performance."""
    model = LinearRegression()
    model.fit(df[[var_col]], df[salary_col])
    return model


def add_surplus_value(
    df: pd.DataFrame,
    dollar_model: LinearRegression,
    var_col: str = "value_above_replacement",
    salary_col: str = "Salary",
) -> pd.DataFrame:
    out = df.copy()
    out["fair_market_salary"] = dollar_model.predict(out[[var_col]])
    # Positive surplus = producing more value than the salary implies (a bargain).
    # Negative = paid more than performance justifies (an overpay).
    out["surplus_value"] = out["fair_market_salary"] - out[salary_col]
    return out
