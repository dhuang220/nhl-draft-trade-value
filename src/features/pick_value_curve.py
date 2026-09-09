"""Empirical draft-pick value curve: expected career value by pick number.

Built from isotonic (monotonic) regression rather than the raw per-pick
average, which is noisy pick-to-pick (a single bust or star at one exact slot
would otherwise create a visible zigzag). Isotonic regression fits the
smoothest non-increasing curve through the data - a real constraint here,
since a later pick should never be worth more than an earlier one on average.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features.draft_features import clean_draft_data

CURVE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "pick_value_curve.joblib"


def fit_pick_value_curve(mature_df: pd.DataFrame) -> IsotonicRegression:
    curve = IsotonicRegression(increasing=False, out_of_bounds="clip")
    curve.fit(mature_df["overall_pick"], mature_df["point_shares"])
    return curve


if __name__ == "__main__":
    raw = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "raw" / "draft_history_raw.csv")
    raw = raw[(raw["year"] >= 2000) & (raw["year"] <= 2020)]
    clean = clean_draft_data(raw)
    mature = clean[clean["is_mature"]]

    curve = fit_pick_value_curve(mature)
    joblib.dump(curve, CURVE_PATH)

    for pick in [1, 5, 15, 32, 64, 100, 150, 200]:
        print(f"Pick #{pick:3d}: expected career Point Shares = {curve.predict([pick])[0]:.2f}")
    print(f"\nSaved curve to {CURVE_PATH}")
