"""Fetch NHL player stats from the public api-web.nhle.com API - both "right now" (for
valuing hypothetical/future trades) and any specific past season (for valuing historical
trades outside the 2016-17 training season), so the Trade Grader isn't limited to a single
season's worth of high-confidence player values.

The NHL API has no equivalent to MoneyPuck's possession/expected-goals numbers - CF, CA, xGF,
and xGA always come back None here. Callers should impute those (e.g. with the training-set
median, same as clean_value_training_data does) rather than treat None as zero.
"""

import re
import time
from collections import Counter

import requests

SEARCH_URL = "https://search.d3.nhle.com/api/v1/search/player"
LANDING_URL = "https://api-web.nhle.com/v1/player/{player_id}/landing"
REALTIME_URL = "https://api.nhle.com/stats/rest/en/skater/realtime"
STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team_abbrev}/current"


class PlayerNotFoundError(Exception):
    pass


class SeasonNotFoundError(Exception):
    pass


def _get_json(url: str, params: dict | None = None, retries: int = 3, delay: float = 0.5) -> dict:
    """Retries with backoff on a 429 (the NHL API returns an HTML challenge page, not JSON,
    when rate limited) - needed once the Trade Grader started making several live requests
    per historical trade instead of just a couple per hypothetical one."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(delay * (2**attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


_POSITION_SUFFIX_RE = re.compile(r"^(.*) \(([A-Z]{1,2})\)$")


def search_player_id(player_name: str) -> int:
    """Resolve a full player name to an NHL player ID via the search-index API.

    Only an exact (case-insensitive) name match is accepted - the search endpoint does fuzzy
    matching and happily returns an unrelated player (e.g. "Wayne Gretzky" -> "Wayne Savage")
    instead of nothing, so a loose match here would silently value the wrong player.

    player_name can optionally carry a disambiguating " (POS)" suffix, as produced by
    fetch_rosters_by_team/fetch_all_current_players for two players who share an exact full
    name (e.g. Vancouver currently rosters two active "Elias Pettersson"s - a D and a C).
    When present, it's parsed off and used to additionally filter candidates by position,
    since the name match alone can't tell them apart. Ordinary names without the suffix
    behave exactly as before.
    """
    suffix_match = _POSITION_SUFFIX_RE.match(player_name)
    lookup_name, position = (suffix_match.group(1), suffix_match.group(2)) if suffix_match else (player_name, None)

    results = _get_json(SEARCH_URL, params={"culture": "en-us", "limit": 10, "q": lookup_name})
    matches = [r for r in results if r["name"].lower() == lookup_name.lower()]
    if position:
        matches = [r for r in matches if r.get("positionCode") == position]
    if not matches:
        raise PlayerNotFoundError(f"No exact player match found for '{player_name}'")

    # Prefer the active player when the name is shared with someone retired/inactive.
    active_matches = [r for r in matches if r["active"]]
    chosen = active_matches[0] if active_matches else matches[0]
    return int(chosen["playerId"])


def _season_end_year(season_id: int) -> int:
    """Season IDs are 'YYYYZZZZ' (e.g. 20252026) - the end year is what value_features.py's
    age formula (season_end_year - birth_year) expects."""
    return int(str(season_id)[4:])


def _build_stats(player_id: int, landing: dict, season_id: int, season_rows: list[dict]) -> dict:
    """season_rows is EVERY regular-season NHL row for this season - usually one, but a
    player traded mid-season gets a separate row per team, which must be summed rather than
    just reading row 0 (that silently drops the games played after the trade)."""
    draft = landing.get("draftDetails")
    was_drafted = draft is not None
    # 0 / 300 match the undrafted-player encoding clean_value_training_data.py uses for the
    # training set, so this dict can feed the same model without extra imputation.
    draft_round = draft["round"] if was_drafted else 0
    draft_overall = draft["overallPick"] if was_drafted else 300

    birth_year = int(landing["birthDate"].split("-")[0])
    age = _season_end_year(season_id) - birth_year

    position = landing.get("position")
    position_group = "D" if position == "D" else ("G" if position == "G" else "F")

    games_played = sum(r.get("gamesPlayed") or 0 for r in season_rows)

    stats = {
        "player_id": player_id,
        "headshot": landing.get("headshot"),
        "age": age,
        "GP": games_played,
        "G": sum(r.get("goals") or 0 for r in season_rows),
        "A": sum(r.get("assists") or 0 for r in season_rows),
        "PIM": sum(r.get("pim") or 0 for r in season_rows),
        "iHF": None,
        "iGVA": None,
        "iTKA": None,
        "iBLK": None,
        "position_group": position_group,
        "was_drafted": int(was_drafted),
        "draft_round": draft_round,
        "draft_overall": draft_overall,
        "CF": None,
        "CA": None,
        "xGF": None,
        "xGA": None,
    }

    # Hits/giveaways/takeaways/blocks live in a separate "realtime" report. It already
    # combines a mid-season trade into one row (unlike seasonTotals above), but also
    # includes a separate playoffs row with no type field to distinguish them by - so the
    # regular-season row is identified by matching its games played against the total
    # already computed from seasonTotals, rather than assuming row order. Only covers
    # skaters (goalies aren't in it) and only seasons the league has actually tracked this
    # for (older seasons just come back empty, tolerated the same way CF/CA/xGF/xGA are).
    if position_group != "G" and games_played:
        rt_rows = _get_json(
            REALTIME_URL, params={"cayenneExp": f"seasonId={season_id} and playerId={player_id}"}
        ).get("data") or []
        rt = next((r for r in rt_rows if r.get("gamesPlayed") == games_played), None)
        if rt:
            stats["iHF"] = rt.get("hits")
            stats["iGVA"] = rt.get("giveaways")
            stats["iTKA"] = rt.get("takeaways")
            stats["iBLK"] = rt.get("blockedShots")

    return stats


def fetch_current_player_stats(player_name: str) -> dict:
    """Look up a player by name and return the FEATURE_COLUMNS fields the NHL API can supply,
    for their most recent season with data. Raises PlayerNotFoundError if the name doesn't
    resolve."""
    player_id = search_player_id(player_name)
    landing = _get_json(LANDING_URL.format(player_id=player_id))

    featured = landing.get("featuredStats") or {}
    season_id = featured.get("season")
    if not season_id:
        raise SeasonNotFoundError(f"'{player_name}' has no current-season NHL data")
    season_row = (featured.get("regularSeason") or {}).get("subSeason") or {}

    return _build_stats(player_id, landing, season_id, [season_row])


def fetch_player_season_stats(player_name: str, season_id: int) -> dict:
    """Same as fetch_current_player_stats, but for a specific past NHL season (e.g. 20162017
    for the 2016-17 season) rather than whichever season is most recent - lets the Trade
    Grader value a historical trade using that player's real form at the time, not just their
    career total. Raises PlayerNotFoundError if the name doesn't resolve, or
    SeasonNotFoundError if this player has no NHL regular-season row for that season (they
    weren't in the league yet, were in the minors/juniors that year, etc.)."""
    player_id = search_player_id(player_name)
    landing = _get_json(LANDING_URL.format(player_id=player_id))

    season_rows = landing.get("seasonTotals") or []
    matches = [
        s for s in season_rows
        if s.get("season") == season_id and s.get("leagueAbbrev") == "NHL" and s.get("gameTypeId") == 2
    ]
    if not matches:
        raise SeasonNotFoundError(f"'{player_name}' has no NHL regular-season row for {season_id}")

    return _build_stats(player_id, landing, season_id, matches)


def fetch_player_nhl_season_ids(player_name: str) -> list[int]:
    """Every regular-season NHL season_id (YYYYZZZZ) this player has a seasonTotals row for,
    oldest first - lets a caller enumerate exactly which seasons to pull via
    fetch_player_season_stats instead of guessing. Raises PlayerNotFoundError if the name
    doesn't resolve; an empty list (not an error) just means no NHL games played yet."""
    player_id = search_player_id(player_name)
    landing = _get_json(LANDING_URL.format(player_id=player_id))
    season_rows = landing.get("seasonTotals") or []
    season_ids = {
        s["season"] for s in season_rows
        if s.get("leagueAbbrev") == "NHL" and s.get("gameTypeId") == 2
    }
    return sorted(season_ids)


def _fetch_standings() -> list[dict]:
    """Raw standings rows, shared by every function below that needs them so a caller using
    more than one of those functions doesn't trigger a redundant request each."""
    return _get_json(STANDINGS_URL)["standings"]


# Historical name -> current name, for franchises that relocated/rebranded within this
# project's 2000-2026 data window (draft/trade records correctly use the name a team had
# at the time, which won't match fetch_team_logos()'s current-team-only dict otherwise).
# Older relocations (e.g. Quebec Nordiques -> Colorado Avalanche, 1995) predate this
# project's data and aren't covered - the graceful "no logo" fallback still applies there.
_RELOCATED_TEAM_ALIASES = {
    "Phoenix Coyotes": "Utah Mammoth",
    "Arizona Coyotes": "Utah Mammoth",
    "Atlanta Thrashers": "Winnipeg Jets",
}


def _add_relocated_team_aliases(logos: dict[str, str]) -> dict[str, str]:
    logos = dict(logos)
    for old_name, current_name in _RELOCATED_TEAM_ALIASES.items():
        if current_name in logos:
            logos[old_name] = logos[current_name]
    return logos


def fetch_team_logos() -> dict[str, str]:
    """Full team name -> logo URL, for every current NHL team, plus known historical
    aliases (see _RELOCATED_TEAM_ALIASES). Older/defunct team names not in that alias map
    (e.g. 'Quebec Nordiques') simply won't be in this dict - callers should treat a missing
    key as "no logo available" rather than an error, the same graceful-degradation pattern
    used everywhere else in this project."""
    logos = {team["teamName"]["default"]: team["teamLogo"] for team in _fetch_standings()}
    return _add_relocated_team_aliases(logos)


def fetch_team_standings() -> dict[str, int]:
    """Full team name -> current leagueSequence (1 = best record in the league, 32 = worst),
    for every current NHL team. Used as a proxy for where a team's future draft pick will
    land - see estimate_pick_slot_from_standing in trade_grader.py."""
    return {team["teamName"]["default"]: team["leagueSequence"] for team in _fetch_standings()}


def _fetch_rosters_raw() -> dict[str, list[dict]]:
    """Full team name -> list of raw player dicts (firstName/lastName/positionCode/id, as
    returned by the roster API) for that team's current roster - shared by the two public
    functions below so the 32-team loop only happens once."""
    rosters = {}
    for team in _fetch_standings():
        roster = _get_json(ROSTER_URL.format(team_abbrev=team["teamAbbrev"]["default"]))
        rosters[team["teamName"]["default"]] = roster["forwards"] + roster["defensemen"] + roster["goalies"]
        time.sleep(0.3)
    return rosters


def _disambiguate_duplicate_names(players: list[dict]) -> list[str]:
    """Full display names for a list of player dicts (firstName/lastName/positionCode).

    Two different players can share an exact full name (e.g. Vancouver currently rosters two
    active "Elias Pettersson"s - a defenseman and a center) - a plain name string would then
    be ambiguous, or silently collapse the two into one entry if used as a set/dict key
    upstream. Only names that actually collide within this list get a " (POS)" suffix
    appended to disambiguate them; every unique name is left exactly as-is.
    """
    names = [f"{p['firstName']['default']} {p['lastName']['default']}" for p in players]
    counts = Counter(names)
    return [
        f"{name} ({player['positionCode']})" if counts[name] > 1 else name
        for name, player in zip(names, players)
    ]


def fetch_rosters_by_team() -> dict[str, list[str]]:
    """Full team name -> sorted list of full player names on that team's current roster.
    Disambiguated against collisions within that same team's roster only (see
    _disambiguate_duplicate_names) - a name shared across two DIFFERENT teams doesn't need
    disambiguating here, since each team's list is already unambiguous on its own."""
    return {team: sorted(_disambiguate_duplicate_names(players)) for team, players in _fetch_rosters_raw().items()}


def fetch_all_current_players() -> list[str]:
    """Every player on a current NHL roster, for populating a search/select input that can
    only produce valid names (rather than free text a user could mistype). Disambiguated
    league-wide (see _disambiguate_duplicate_names) - a plain sorted(set(...)) over names
    alone would silently collapse two different players sharing an exact full name (even
    across two different teams) down to one entry, losing the other."""
    all_players = [player for players in _fetch_rosters_raw().values() for player in players]
    return sorted(set(_disambiguate_duplicate_names(all_players)))


if __name__ == "__main__":
    for name in ["Connor McDavid", "Auston Matthews", "Cale Makar", "Wayne Gretzky"]:
        try:
            print(name, "->", fetch_current_player_stats(name))
        except PlayerNotFoundError as e:
            print(name, "-> NOT FOUND:", e)

    print()
    print("Martin Hanzal, 2016-17 season ->", fetch_player_season_stats("Martin Hanzal", 20162017))
