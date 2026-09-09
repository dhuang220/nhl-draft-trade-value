"""Combine the value model, positional scarcity, and surplus value into one table.

Run after train_value_model.py. Produces the final per-player valuation table
used by the dashboard and trade grader.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.models.positional_scarcity import add_value_above_replacement, compute_replacement_levels
from src.models.surplus_value import add_surplus_value, fit_dollar_per_var
from src.models.train_value_model import PLAYER_VALUES_PATH, PROCESSED_DIR

VALUATIONS_PATH = PROCESSED_DIR / "player_valuations_2016_17.csv"
SCARCITY_MODEL_PATH = PROCESSED_DIR / "replacement_levels.joblib"
DOLLAR_MODEL_PATH = PROCESSED_DIR / "dollar_per_var_model.joblib"


def build_valuations() -> pd.DataFrame:
    values = pd.read_csv(PLAYER_VALUES_PATH)

    replacement_levels = compute_replacement_levels(values)
    values = add_value_above_replacement(values, replacement_levels)

    dollar_model = fit_dollar_per_var(values)
    values = add_surplus_value(values, dollar_model)

    joblib.dump(replacement_levels, SCARCITY_MODEL_PATH)
    joblib.dump(dollar_model, DOLLAR_MODEL_PATH)
    values.to_csv(VALUATIONS_PATH, index=False)

    return values


if __name__ == "__main__":
    values = build_valuations()

    print("=== Replacement level by position (25th percentile of predicted value) ===")
    print(compute_replacement_levels(values))

    print(f"\n$ per unit of value above replacement: {fit_dollar_per_var(values).coef_[0]:,.0f}")

    print("\n=== Top 10 bargains (surplus value) ===")
    print(
        values.nlargest(10, "surplus_value")[
            [
                "First Name",
                "Last Name",
                "Position",
                "GP",
                "Salary",
                "predicted_value_total",
                "value_above_replacement",
                "surplus_value",
            ]
        ].to_string()
    )

    print("\n=== Top 10 overpays (negative surplus value) ===")
    print(
        values.nsmallest(10, "surplus_value")[
            [
                "First Name",
                "Last Name",
                "Position",
                "GP",
                "Salary",
                "predicted_value_total",
                "value_above_replacement",
                "surplus_value",
            ]
        ].to_string()
    )
