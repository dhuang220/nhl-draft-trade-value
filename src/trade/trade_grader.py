"""Grade an NHL trade by summing the value each side acquired.

Every asset is valued through one of four paths, in order of confidence:
1. Season-matched player value - the trade happened in 2016-17, so we have the
   real composite value model output for that player that season, computed
   locally with no network call. High confidence.
2. Season-specific live value - any other historical trade, or a hypothetical/
   future one. That player's actual stats for the relevant season (or their
   current season, for a hypothetical trade) are pulled from the NHL API and
   run through the same trained value model. Medium confidence (missing
   possession/xG inputs are imputed with the training median).
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
)
from src.features.value_features import FEATURE_COLUMNS, RATE_STATS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FUTURE_PICK_ANNUAL_DISCOUNT = 0.92  # a pick 3 years out is worth 0.92**3 of one available today
CONDITIONAL_PICK_DISCOUNT = 0.7  # a condition might not be met, so it might not convey at all


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
    """A future draft pick in a hypothetical trade - overall slot is never known
    yet, so value_pick() always falls back to the median-by-round estimate."""

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
    def __init__(self):
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
        self._feature_medians = self._season_values_feature_medians()

        draft_by_year = draft[draft["year"].between(2000, 2020)]
        self._median_overall_pick_by_round = (
            draft_by_year.assign(round=((draft_by_year["overall_pick"] - 1) // 31) + 1)
            .groupby("round")["overall_pick"]
            .median()
            .to_dict()
        )

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
        return AssetValue(
            name, float(predicted_rate * stats["GP"]), "medium", source, image_url=stats.get("headshot"),
        )

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

    def value_pick(self, pick_year: float, pick_round: float, pick_overall: float, is_conditional: bool, trade_year: int) -> AssetValue:
        label = f"{int(pick_year) if pd.notna(pick_year) else '?'} round {int(pick_round) if pd.notna(pick_round) else '?'} pick"

        if pd.notna(pick_overall):
            slot = pick_overall
        elif pd.notna(pick_round):
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
        return AssetValue(label, base_value * discount, confidence, "pick value curve")

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
        trade_year: int | None = None,
    ) -> dict[str, TradeSideGrade]:
        trade_year = trade_year or date.today().year
        grades = {}
        for label, assets in [("side_a", side_a), ("side_b", side_b)]:
            grade = TradeSideGrade(team=label)
            for asset in assets:
                if isinstance(asset, HypotheticalPick):
                    grade.assets.append(
                        self.value_pick(asset.year, asset.round, None, asset.conditional, trade_year)
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
