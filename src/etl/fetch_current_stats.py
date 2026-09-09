"""Fetch current-season stats for a live NHL player from the NHL's public api-web.nhle.com
API, so the Trade Grader can value hypothetical trades built from today's rosters, not just
the 2016-17 training season.

The NHL API has no equivalent to MoneyPuck's possession/expected-goals numbers - CF, CA, xGF,
and xGA always come back None here. Callers should impute those (e.g. with the training-set
median, same as clean_value_training_data does) rather than treat None as zero.
"""

import time

import requests

SEARCH_URL = "https://search.d3.nhle.com/api/v1/search/player"
LANDING_URL = "https://api-web.nhle.com/v1/player/{player_id}/landing"
REALTIME_URL = "https://api.nhle.com/stats/rest/en/skater/realtime"
STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team_abbrev}/current"


class PlayerNotFoundError(Exception):
    pass


def search_player_id(player_name: str) -> int:
    """Resolve a full player name to an NHL player ID via the search-index API.

    Only an exact (case-insensitive) name match is accepted - the search endpoint does fuzzy
    matching and happily returns an unrelated player (e.g. "Wayne Gretzky" -> "Wayne Savage")
    instead of nothing, so a loose match here would silently value the wrong player.
    """
    resp = requests.get(
        SEARCH_URL, params={"culture": "en-us", "limit": 10, "q": player_name}, timeout=10
    )
    resp.raise_for_status()
    matches = [r for r in resp.json() if r["name"].lower() == player_name.lower()]
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


def fetch_current_player_stats(player_name: str) -> dict:
    """Look up a player by name and return the FEATURE_COLUMNS fields the NHL API can supply.

    Raises PlayerNotFoundError if the name doesn't resolve. Fields the API can't supply
    (CF, CA, xGF, xGA - see module docstring) come back as None rather than a guessed value.
    """
    player_id = search_player_id(player_name)

    resp = requests.get(LANDING_URL.format(player_id=player_id), timeout=10)
    resp.raise_for_status()
    landing = resp.json()

    draft = landing.get("draftDetails")
    was_drafted = draft is not None
    # 0 / 300 match the undrafted-player encoding clean_value_training_data.py uses for the
    # training set, so this dict can feed the same model without extra imputation.
    draft_round = draft["round"] if was_drafted else 0
    draft_overall = draft["overallPick"] if was_drafted else 300

    featured = landing.get("featuredStats") or {}
    season_id = featured.get("season")
    reg_season = (featured.get("regularSeason") or {}).get("subSeason") or {}

    birth_year = int(landing["birthDate"].split("-")[0])
    age = _season_end_year(season_id) - birth_year if season_id else None

    position = landing.get("position")
    position_group = "D" if position == "D" else ("G" if position == "G" else "F")

    stats = {
        "player_id": player_id,
        "age": age,
        "GP": reg_season.get("gamesPlayed"),
        "G": reg_season.get("goals"),
        "A": reg_season.get("assists"),
        "PIM": reg_season.get("pim"),
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

    # Hits/giveaways/takeaways/blocks live in a separate "realtime" report, and it only covers
    # skaters (goalies aren't in it), so skip the lookup for goalies.
    if position_group != "G" and season_id:
        rt_resp = requests.get(
            REALTIME_URL,
            params={"cayenneExp": f"seasonId={season_id} and playerId={player_id}"},
            timeout=10,
        )
        rt_resp.raise_for_status()
        rt_rows = rt_resp.json().get("data") or []
        if rt_rows:
            rt = rt_rows[0]
            stats["iHF"] = rt.get("hits")
            stats["iGVA"] = rt.get("giveaways")
            stats["iTKA"] = rt.get("takeaways")
            stats["iBLK"] = rt.get("blockedShots")

    return stats


def _get_json(url: str, retries: int = 3, delay: float = 0.5) -> dict:
    """32 back-to-back team-roster requests trips the NHL API's rate limiting (a 429
    with an HTML challenge page, not JSON) - a small delay plus backoff on 429 keeps
    fetch_all_current_players() reliable without needing to slow every other call
    in this module that only ever makes one or two requests at a time."""
    for attempt in range(retries):
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            time.sleep(delay * (2**attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


def fetch_all_current_players() -> list[str]:
    """Every player on a current NHL roster, for populating a search/select input that can
    only produce valid names (rather than free text a user could mistype)."""
    teams = sorted({t["teamAbbrev"]["default"] for t in _get_json(STANDINGS_URL)["standings"]})

    names = []
    for team in teams:
        roster = _get_json(ROSTER_URL.format(team_abbrev=team))
        for player in roster["forwards"] + roster["defensemen"] + roster["goalies"]:
            names.append(f"{player['firstName']['default']} {player['lastName']['default']}")
        time.sleep(0.3)
    return sorted(set(names))


if __name__ == "__main__":
    for name in ["Connor McDavid", "Auston Matthews", "Cale Makar", "Wayne Gretzky"]:
        try:
            print(name, "->", fetch_current_player_stats(name))
        except PlayerNotFoundError as e:
            print(name, "-> NOT FOUND:", e)
