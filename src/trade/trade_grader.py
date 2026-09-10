"""Grade an NHL trade by summing the value each side acquired.

Every asset is valued through one of four paths, in order of confidence:
1. Season-matched player value - the trade happened in 2016-17, so we have the
   real composite value model output for that player that season, computed
   locally with no network call. High confidence. NOTE: unlike paths 2 and 3,
   this is a single-SEASON total, not a career projection (the precomputed
   table has no draft slot to project from without a fragile extra join) - a
   known, documented scale inconsistency with the rest of this file.
2. Season-specific live value - any other historical trade, or a hypothetical/
   future one. That player's actual stats for the relevant season (or their
   current season, for a hypothetical trade) are pulled from the NHL API and
   run through the same trained value model, then PROJECTED to a career total
   using the same games-played-by-draft-slot curve Draft Explorer uses for its
   pace projections - matching the timescale of a pick's expected career value
   (value_pick), rather than comparing one season of a player against an
   entire career of draft-slot expectation. Medium confidence (missing
   possession/xG inputs are imputed with the training median; undrafted
   players get whatever the curve's lowest-pick estimate is, which likely
   understates them).
3. Career draft-value fallback - the season-specific lookup found no NHL
   record for that player in that season (too old for the league's digitized
   records, a name that doesn't resolve, etc.). Falls back to the player's
   career Point Shares from the draft dataset. Low confidence: it values a
   player as "their whole career," not "how good they were at the moment of
   this specific trade" - a real, documented simplification of last resort.
4. Unmatched - not found anywhere.

A bare cash/future-considerations asset with no dollar amount also lands here.
Unvalued assets are reported separately rather than silently treated as zero -
zero would say "this asset is worthless," which is a different (and false)
claim from "we don't know."
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.etl.fetch_current_stats import (
    PlayerNotFoundError,
    SeasonNotFoundError,
    fetch_current_player_stats,
    fetch_player_season_stats,
    fetch_team_standings,
)
from src.features.value_features import FEATURE_COLUMNS, RATE_STATS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FUTURE_PICK_ANNUAL_DISCOUNT = 0.92  # a pick 3 years out is worth 0.92**3 of one available today
CONDITIONAL_PICK_DISCOUNT = 0.7  # a condition might not be met, so it might not convey at all


def estimate_pick_slot_from_standing(pick_round: int, league_rank: int) -> int:
    """Estimate an overall draft slot for a team's pick in a given round, from that team's
    current league standing (1 = best record, 32 = worst) - used for a hypothetical trade,
    where the trading team is known but the actual future draft order isn't yet.

    This assumes a flat worst-picks-first order every round (e.g. the league's worst team
    gets pick 1, its best team gets pick 32, round 2 repeats the same order starting at 33,
    etc.). That's a real simplification, not the league's actual draft-order rules: it
    ignores the draft lottery's reshuffling among non-playoff teams (the worst record isn't
    guaranteed pick 1) and the reverse-standings tiebreak among playoff teams. It's meant to
    be a directionally-realistic improvement over a single round-wide median, not an exact
    prediction of a future pick number.
    """
    estimated = (pick_round - 1) * 32 + (33 - league_rank)
    return max(estimated, 1)


@dataclass
class AssetValue:
    label: str
    value: float | None
    confidence: str  # "high" | "medium" | "low" | "none"
    source: str
    # Only ever set for a live player (value_player_live already has the headshot
    # in hand from the same API response, so it's free) - historical/fallback
    # valuations have no photo source and leave this None, same graceful-
    # degradation pattern as everything else here.
    image_url: str | None = None


@dataclass
class HypotheticalPick:
    """A future draft pick in a hypothetical trade - overall slot is never known yet, so
    value_pick() falls back to a standings-based estimate for the trading team when their
    current league rank is available, or the median-by-round estimate otherwise."""

    year: int
    round: int
    conditional: bool = False


@dataclass
class TradeSideGrade:
    team: str
    assets: list = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return sum(a.value for a in self.assets if a.value is not None)

    @property
    def unvalued_assets(self) -> list:
        return [a for a in self.assets if a.value is None]


class TradeGrader:
    def __init__(self, team_standings: dict[str, int] | None = None):
        # Optional and lazy: a caller (e.g. the dashboard) can pass standings it already has,
        # otherwise this is fetched from the live API on first actual use (a hypothetical
        # trade that needs a team's rank), not at construction - keeps TradeGrader() itself
        # network-free for callers/tests that only need the 2016-17/career-fallback paths.
        self._team_standings = team_standings

        self._season_values = pd.read_csv(PROCESSED_DIR / "player_valuations_2016_17.csv")
        self._season_values["full_name"] = (
            self._season_values["First Name"] + " " + self._season_values["Last Name"]
        )

        draft = pd.read_csv(RAW_DIR / "draft_history_raw.csv")
        draft["point_shares"] = draft["point_shares"].fillna(0)
        # A handful of names are drafted more than once in this table's history isn't
        # possible (a player is only draft-eligible once), but common names can collide
        # across different real people - take the most recent draft year as the better guess.
        self._draft_career_values = (
            draft.sort_values("year").drop_duplicates("player", keep="last").set_index("player")["point_shares"]
        )

        self._pick_curve = joblib.load(PROCESSED_DIR / "pick_value_curve.joblib")
        self._value_model = joblib.load(PROCESSED_DIR / "value_model.joblib")
        # Same curve Draft Explorer uses to turn a still-active prospect's pace into a
        # career-scale projection - reused here so a live/season player value is on the
        # same career-length scale as a pick's expected value (see _value_from_stats).
        self._games_played_curve = joblib.load(PROCESSED_DIR / "games_played_curve.joblib")
        self._feature_medians = self._season_values_feature_medians()

        draft_by_year = draft[draft["year"].between(2000, 2020)]
        self._median_overall_pick_by_round = (
            draft_by_year.assign(round=((draft_by_year["overall_pick"] - 1) // 31) + 1)
            .groupby("round")["overall_pick"]
            .median()
            .to_dict()
        )

    def _get_team_standings(self) -> dict[str, int]:
        if self._team_standings is None:
            self._team_standings = fetch_team_standings()
        return self._team_standings

    def _season_values_feature_medians(self) -> dict:
        # Only CF/CA/xGF/xGA are ever missing (live players can't supply them) -
        # everything else a caller passes in is required, not imputed.
        cols = ["CF_per_gp", "CA_per_gp", "xGF_per_gp", "xGA_per_gp"]
        raw = pd.read_csv(PROCESSED_DIR / "player_values_2016_17.csv")
        return {c: raw[c].median() if c in raw.columns else 0.0 for c in cols}

    # --- Player valuation ---------------------------------------------------

    def value_player_2016_17(self, name: str) -> AssetValue:
        match = self._season_values[self._season_values["full_name"] == name]
        if not match.empty:
            return AssetValue(name, float(match.iloc[0]["predicted_value_total"]), "high", "2016-17 value model")
        return self.value_player_career_fallback(name)

    def value_player_career_fallback(self, name: str) -> AssetValue:
        if name in self._draft_career_values.index:
            return AssetValue(name, float(self._draft_career_values[name]), "low", "career draft Point Shares")
        return AssetValue(name, None, "none", "unmatched")

    def _value_from_stats(self, name: str, stats: dict, source: str) -> AssetValue:
        if stats.get("GP") in (None, 0) or stats.get("position_group") == "G":
            # No games that season (injured/hasn't debuted/didn't play) or a goalie -
            # out of this model's scope either way (see train_value_model.py docstring).
            return self.value_player_career_fallback(name)

        row = {"age": stats["age"], "GP": stats["GP"], "was_drafted": stats["was_drafted"],
               "draft_round": stats["draft_round"], "draft_overall": stats["draft_overall"]}
        for stat in RATE_STATS:
            raw_value = stats.get(stat)
            per_gp_col = f"{stat}_per_gp"
            row[per_gp_col] = (
                raw_value / stats["GP"] if raw_value is not None else self._feature_medians.get(per_gp_col, 0.0)
            )

        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        predicted_rate = self._value_model.predict(X)[0]

        # Project this one season's rate over a realistic FULL career, rather than
        # returning a single-season total - a pick's value (from value_pick) is already
        # an expected CAREER total, and comparing a season number against a career number
        # would silently make picks look weak against even an average current player
        # (one great season otherwise reads as "worth more than a typical whole career").
        # Undrafted players (draft_overall=300, the sentinel used throughout this project)
        # get whatever the curve's lowest-pick estimate is, which likely understates them -
        # a real limitation of using draft slot as the only length-of-career predictor here.
        benchmark_games = self._games_played_curve.predict([stats["draft_overall"]])[0]
        career_value = float(predicted_rate * benchmark_games)

        return AssetValue(name, career_value, "medium", f"{source} (career-projected)", image_url=stats.get("headshot"))

    def value_player_live(self, name: str) -> AssetValue:
        try:
            stats = fetch_current_player_stats(name)
        except (PlayerNotFoundError, SeasonNotFoundError):
            return self.value_player_career_fallback(name)
        return self._value_from_stats(name, stats, "live NHL API + value model")

    def value_player_season(self, name: str, season_id: int) -> AssetValue:
        """Same idea as value_player_live, but for a specific past NHL season rather than
        whichever is most recent - lets a historical trade outside 2016-17 be valued on that
        player's real form at the time, instead of falling straight to career totals."""
        try:
            stats = fetch_player_season_stats(name, season_id)
        except (PlayerNotFoundError, SeasonNotFoundError):
            return self.value_player_career_fallback(name)
        return self._value_from_stats(name, stats, f"NHL API season stats ({season_id}) + value model")

    # --- Pick valuation -------------------------------------------------------

    def value_pick(
        self,
        pick_year: float,
        pick_round: float,
        pick_overall: float,
        is_conditional: bool,
        trade_year: int,
        team_rank: int | None = None,
    ) -> AssetValue:
        """team_rank (1-32, 1 = best record) is only meaningful for a hypothetical trade,
        where the trading team is known - grade_historical_trade never passes it, since we
        don't know a past trade's contemporaneous standings-implied pick value."""
        label = f"{int(pick_year) if pd.notna(pick_year) else '?'} round {int(pick_round) if pd.notna(pick_round) else '?'} pick"

        source = "pick value curve"
        if pd.notna(pick_overall):
            slot = pick_overall
        elif pd.notna(pick_round):
            if team_rank is not None:
                slot = estimate_pick_slot_from_standing(int(pick_round), team_rank)
                source = "pick value curve (standings-based slot estimate)"
            else:
                slot = self._median_overall_pick_by_round.get(int(pick_round))
                if slot is None:
                    return AssetValue(label, None, "none", "unknown round")
        else:
            return AssetValue(label, None, "none", "no pick detail")

        base_value = float(self._pick_curve.predict([slot])[0])

        years_out = (pick_year - trade_year) if pd.notna(pick_year) else 0
        discount = FUTURE_PICK_ANNUAL_DISCOUNT ** max(years_out, 0)
        if is_conditional:
            discount *= CONDITIONAL_PICK_DISCOUNT

        confidence = "medium" if pd.notna(pick_overall) else "low"
        return AssetValue(label, base_value * discount, confidence, source)

    # --- Trade-level grading ----------------------------------------------

    def grade_historical_trade(self, trade_rows: pd.DataFrame) -> dict[str, TradeSideGrade]:
        season = trade_rows["season"].iloc[0]
        trade_year = int(str(trade_rows["season"].iloc[0]).split("-")[0])
        grades: dict[str, TradeSideGrade] = {}

        for side, side_rows in trade_rows.groupby("side"):
            team = side_rows["acquiring_team"].iloc[0]
            grade = TradeSideGrade(team=team)
            for _, row in side_rows.iterrows():
                if row["type"] == "player":
                    if season == "2016-17":
                        # Already have the trained model's exact output for this
                        # season locally - no need to hit the live API for it.
                        asset = self.value_player_2016_17(row["name"])
                    else:
                        season_id = trade_year * 10000 + (trade_year + 1)
                        asset = self.value_player_season(row["name"], season_id)
                elif row["type"] == "pick":
                    asset = self.value_pick(
                        row["pick_year"], row["pick_round"], row["pick_overall"], bool(row["is_conditional"]), trade_year
                    )
                else:
                    asset = AssetValue(row["raw"], None, "none", row["subtype"])
                grade.assets.append(asset)
            grades[side] = grade

        return grades

    def grade_hypothetical_trade(
        self,
        side_a: list[str | HypotheticalPick],
        side_b: list[str | HypotheticalPick],
        team_a: str | None = None,
        team_b: str | None = None,
        trade_year: int | None = None,
    ) -> dict[str, TradeSideGrade]:
        trade_year = trade_year or date.today().year
        grades = {}
        for label, assets, team in [("side_a", side_a, team_a), ("side_b", side_b, team_b)]:
            grade = TradeSideGrade(team=team or label)
            # Only hit the standings lookup (which fetches live on first use) when a real
            # team was actually given - a caller that doesn't care about team identity
            # shouldn't pay for a network call it doesn't need.
            team_rank = self._get_team_standings().get(team) if team else None
            for asset in assets:
                if isinstance(asset, HypotheticalPick):
                    grade.assets.append(
                        self.value_pick(
                            asset.year, asset.round, None, asset.conditional, trade_year, team_rank=team_rank
                        )
                    )
                else:
                    grade.assets.append(self.value_player_live(asset))
            grades[label] = grade
        return grades


def format_grade(grades: dict[str, TradeSideGrade]) -> str:
    lines = []
    for side, grade in grades.items():
        lines.append(f"{grade.team}: {grade.total_value:+.2f} total value")
        for asset in grade.assets:
            value_str = f"{asset.value:+.2f}" if asset.value is not None else "UNVALUED"
            lines.append(f"  [{asset.confidence:6s}] {asset.label:30s} {value_str}  ({asset.source})")
    totals = [g.total_value for g in grades.values()]
    if len(totals) == 2:
        lines.append(f"\nValue differential: {totals[0] - totals[1]:+.2f}")
    return "\n".join(lines)
