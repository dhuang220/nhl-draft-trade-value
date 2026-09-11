"""Clean the 2016-17 salary dataset and engineer features for the player value model."""

import pandas as pd

RATE_STATS = ["G", "A", "PIM", "iHF", "iGVA", "iTKA", "iBLK", "CF", "CA", "xGF", "xGA"]

FEATURE_COLUMNS = [
    "age",
    "GP",
    *[f"{stat}_per_gp" for stat in RATE_STATS],
    "was_drafted",
    "draft_round",
    "draft_overall",
]

# Human-readable labels for FEATURE_COLUMNS, for showing a player's value breakdown
# (see trade_grader.py's _value_from_stats) rather than raw column names.
FEATURE_LABELS = {
    "age": "Age",
    "GP": "Games played",
    "G_per_gp": "Goals/game",
    "A_per_gp": "Assists/game",
    "PIM_per_gp": "Penalty minutes/game",
    "iHF_per_gp": "Hits/game",
    "iGVA_per_gp": "Giveaways/game",
    "iTKA_per_gp": "Takeaways/game",
    "iBLK_per_gp": "Blocked shots/game",
    "CF_per_gp": "Shot attempts for/game",
    "CA_per_gp": "Shot attempts against/game",
    "xGF_per_gp": "Expected goals for/game",
    "xGA_per_gp": "Expected goals against/game",
    "was_drafted": "Was drafted",
    "draft_round": "Draft round",
    "draft_overall": "Draft position",
}


def _parse_birth_year(born: str) -> int:
    """'Born' is 'YY-MM-DD'. All players in this dataset were born 1965-2000,
    so any two-digit year above 30 must mean 19XX, otherwise 20XX."""
    yy = int(born.split("-")[0])
    return 1900 + yy if yy > 30 else 2000 + yy


def clean_value_training_data(df: pd.DataFrame, season_end_year: int = 2017) -> pd.DataFrame:
    """Turn the raw Kaggle salary-dataset rows into a model-ready feature table.

    One row in, one row out - every column added here is something we can also
    compute for a live/current player, which is what lets the same formula be
    reused outside this one season later.
    """
    out = df.copy()

    birth_year = out["Born"].apply(_parse_birth_year)
    out["age"] = season_end_year - birth_year

    # DftYr/DftRd/Ovrl are NaN for undrafted free agents - that's a real signal
    # (undrafted players who make the NHL anyway are unusual), not a data error,
    # so we keep it as an explicit flag rather than silently dropping or imputing it.
    out["was_drafted"] = out["DftYr"].notna().astype(int)
    out["draft_round"] = out["DftRd"].fillna(0)
    out["draft_overall"] = out["Ovrl"].fillna(300)  # worse than any real pick

    out["position_group"] = out["Position"].apply(
        lambda p: "D" if p == "D" else ("G" if p == "G" else "F")
    )

    out[["CF", "CA", "xGF", "xGA"]] = out[["CF", "CA", "xGF", "xGA"]].fillna(
        out[["CF", "CA", "xGF", "xGA"]].median()
    )

    # Raw counting stats conflate "how good per opportunity" with "how much did
    # this player play" - a player with more ice time racks up more hits/blocks
    # /goals just from being on the ice longer. Rate stats isolate the former;
    # GP is kept as its own feature so playing-time trust is still represented.
    for stat in RATE_STATS:
        out[f"{stat}_per_gp"] = out[stat] / out["GP"].clip(lower=1)

    return out


def build_feature_matrix(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Select just the model's input columns, in a fixed order."""
    return clean_df[FEATURE_COLUMNS]
