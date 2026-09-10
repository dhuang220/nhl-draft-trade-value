"""Clean the draft history dataset and mark which draft classes are safe to train on.

point_shares is NaN for players who never played an NHL game - a real zero
outcome (a bust pick), not a missing value, so it must be filled before use.

Separately, recent draft classes are CENSORED: a player drafted in 2019 hasn't
had time to accumulate a career yet, so their low point_shares reflects "too
early to tell," not "this was a bad pick." Training on censored labels would
teach the model that recent picks are worth less than they really are, so
those rows are flagged and excluded from training (they can still be shown in
the dashboard as unverified predictions).
"""

import pandas as pd

MATURE_CUTOFF_YEAR = 2012  # 10+ NHL seasons elapsed as of this dataset's ~2022 snapshot

FEATURE_COLUMNS = ["overall_pick", "age", "position_group"]


def merge_draft_sources(kaggle_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """Union the Kaggle-sourced 2000-2020 rows with the NHL-API-sourced 2021+ rows
    (recent_df's point_shares is our value model's predicted career value, not the real
    stat - see fetch_recent_draft_data.py). The Kaggle dataset does have 2021-2022 rows,
    but their point_shares/games_played are effectively blank (a pre-career snapshot) -
    live-fetched values replace them here rather than being unioned in."""
    kaggle = kaggle_df[(kaggle_df["year"] >= 2000) & (kaggle_df["year"] <= 2020)]
    recent = recent_df[recent_df["year"] >= 2021]
    return pd.concat([kaggle, recent], ignore_index=True)


def clean_draft_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["games_played"] = out["games_played"].fillna(0)
    out["point_shares"] = out["point_shares"].fillna(0)
    out["position_group"] = out["position"].apply(lambda p: "G" if p == "G" else "Skater")
    out["is_mature"] = out["year"] <= MATURE_CUTOFF_YEAR
    return out


def build_feature_matrix(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df[FEATURE_COLUMNS].copy()
    return pd.get_dummies(df, columns=["position_group"], drop_first=True)
