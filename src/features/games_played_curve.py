"""Empirical expected career games-played by pick number, from mature draft classes.

Used to project a still-active (non-mature) player's career value forward onto the
same footing as the pick-value curve: pace (value per 82 games) times a realistic
assumed career length for their draft slot, rather than comparing an unfinished raw
total against a curve built from full careers.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features.draft_features import clean_draft_data

CURVE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "games_played_curve.joblib"


def fit_games_played_curve(mature_df: pd.DataFrame) -> IsotonicRegression:
    curve = IsotonicRegression(increasing=False, out_of_bounds="clip")
    curve.fit(mature_df["overall_pick"], mature_df["games_played"])
    return curve


if __name__ == "__main__":
    raw = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "raw" / "draft_history_raw.csv")
    raw = raw[(raw["year"] >= 2000) & (raw["year"] <= 2020)]
    clean = clean_draft_data(raw)
    mature = clean[clean["is_mature"]]

    curve = fit_games_played_curve(mature)
    joblib.dump(curve, CURVE_PATH)

    for pick in [1, 5, 15, 32, 64, 100, 150, 200]:
        print(f"Pick #{pick:3d}: expected career games played = {curve.predict([pick])[0]:.0f}")
    print(f"\nSaved curve to {CURVE_PATH}")
