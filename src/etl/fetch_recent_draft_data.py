"""Fetch 2021+ NHL draft classes from the NHL's own draft-picks API and value each pick with
our own trained value model. 2023+ has no Kaggle equivalent at all; 2021-2022 do exist in the
Kaggle dataset but are an unusable pre-career snapshot there (point_shares/games_played are
NaN for essentially the whole 2022 class), so they're replaced with this same live fetch too.

point_shares in the output is NOT the real Hockey-Reference stat the Kaggle-sourced rows
carry - real career Point Shares doesn't exist yet for these mostly-not-yet-NHL players. It's
our composite value model's predicted career value (predicted per-game rate x GP, summed over
every NHL regular season the player has played so far), used as a stand-in so this data drops
into the existing draft_history_raw.csv-shaped pipeline without a separate column or code
path. Named point_shares anyway for that reason - see README's Known limitations.
"""

import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from src.etl.fetch_current_stats import (
    PlayerNotFoundError,
    SeasonNotFoundError,
    _get_json,
    fetch_player_nhl_season_ids,
    fetch_player_season_stats,
)
from src.features.value_features import FEATURE_COLUMNS, RATE_STATS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DRAFT_PICKS_URL = "https://api-web.nhle.com/v1/draft/picks/{year}/all"

PLAYER_PACE_DELAY = 0.35  # seconds between players, to stay well clear of NHL API rate limits


def _feature_medians() -> dict:
    # Same file/columns/fallback trade_grader.py's _season_values_feature_medians uses, kept
    # in sync deliberately so a live season's imputation behaves identically either place.
    cols = ["CF_per_gp", "CA_per_gp", "xGF_per_gp", "xGA_per_gp"]
    raw = pd.read_csv(PROCESSED_DIR / "player_values_2016_17.csv")
    return {c: raw[c].median() if c in raw.columns else 0.0 for c in cols}


def _predicted_season_value(stats: dict, model, medians: dict) -> float:
    # Mirrors trade_grader.py's _value_from_stats row-building exactly - duplicated rather
    # than imported since that's a private method on TradeGrader, which pulls in a pile of
    # unrelated trade-specific data files just to construct.
    if stats.get("GP") in (None, 0) or stats.get("position_group") == "G":
        return 0.0

    row = {
        "age": stats["age"], "GP": stats["GP"], "was_drafted": stats["was_drafted"],
        "draft_round": stats["draft_round"], "draft_overall": stats["draft_overall"],
    }
    for stat in RATE_STATS:
        raw_value = stats.get(stat)
        per_gp_col = f"{stat}_per_gp"
        row[per_gp_col] = raw_value / stats["GP"] if raw_value is not None else medians.get(per_gp_col, 0.0)

    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    predicted_rate = model.predict(X)[0]
    return float(predicted_rate * stats["GP"])


def predicted_career_value(player_name: str, model, medians: dict) -> tuple[float, int]:
    """Predicted career value and total NHL games played so far for one player, summed across
    every NHL regular season they've played. Both are 0 for a pick who hasn't debuted yet or
    whose name doesn't resolve on the NHL API - a real, expected outcome for most recent
    picks, not an error."""
    try:
        season_ids = fetch_player_nhl_season_ids(player_name)
    except PlayerNotFoundError:
        return 0.0, 0

    total_value = 0.0
    total_gp = 0
    for season_id in season_ids:
        try:
            stats = fetch_player_season_stats(player_name, season_id)
        except (PlayerNotFoundError, SeasonNotFoundError):
            continue
        total_gp += stats.get("GP") or 0
        total_value += _predicted_season_value(stats, model, medians)

    return total_value, total_gp


def fetch_recent_draft_data(start_year: int = 2021, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    model = joblib.load(PROCESSED_DIR / "value_model.joblib")
    medians = _feature_medians()

    rows = []
    for year in range(start_year, end_year + 1):
        picks = _get_json(DRAFT_PICKS_URL.format(year=year)).get("picks", [])
        for pick in picks:
            if "firstName" not in pick:
                # A forfeited pick (e.g. 2026 overall #63, VGK) has no player attached at
                # all - lastName is just the literal string "Forfeited". Not a real draft
                # pick, so it's skipped entirely rather than logged as a 0-value player.
                continue
            player = f"{pick['firstName']['default']} {pick['lastName']['default']}"
            value, gp = predicted_career_value(player, model, medians)
            rows.append({
                "year": year,
                "overall_pick": pick["overallPick"],
                "team": pick["teamName"]["default"],
                "player": player,
                "position": pick["positionCode"],
                "games_played": gp,
                "point_shares": value,
            })
            time.sleep(PLAYER_PACE_DELAY)
        print(f"{year}: {len(picks)} picks fetched")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_recent_draft_data()
    out_path = PROCESSED_DIR / "recent_draft_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} picks ({df['year'].min()}-{df['year'].max()}) to {out_path}")
