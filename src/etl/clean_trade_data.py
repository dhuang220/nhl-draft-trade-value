"""Clean the raw NHL trade dataset (data/raw/trades_raw.csv) into a tidy, long-format table
of individual assets, one row per player/pick/cash-or-other item acquired in each trade.

Each trade row in the raw data has two free-text cells - "Team One Acquisitions" and
"Team Two Acquisitions" - where every line is one asset that team received. The text is
scraped from a wiki-style source and is inconsistent: mixed capitalization, typos
("Conditonal", "pcik"), unbalanced parentheses, and assets nested inside a
"future considerations (...)" wrapper that itself needs to be unwrapped and re-classified.

parse_acquisition_line() turns one such line into a structured dict. build_tidy_trades()
explodes every trade's acquisition cells into that long format and adds a trade_id linking
rows back to the trade they came from.
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_TRADES_CSV = RAW_DIR / "trades_raw.csv"
CLEAN_TRADES_CSV = PROCESSED_DIR / "trades_clean.csv"

# --- Regexes -----------------------------------------------------------------------------

# A "future considerations" line's payload lives in a trailing "(...)". Real data sometimes
# drops the closing paren ("future considerations (1988 3rd round pick (#62-Daniel Gauthier)"),
# so the unwrap logic below only strips one trailing ")" rather than requiring balance.
FUTURE_CONSIDERATIONS_RE = re.compile(r"^future\s+considerations?\b\s*(.*)$", re.IGNORECASE)

# Disambiguates two same-named players by birth year, e.g. "Petr (b.1976) Sykora" - the
# marker sits *inside* the name, so it has to be cut out before the name can be matched.
BIRTH_YEAR_RE = re.compile(r"\(\s*b\.\s*(?P<year>\d{4})\s*\)")

PICK_KEYWORD_RE = re.compile(r"\bpicks?\b", re.IGNORECASE)
ROUND_KEYWORD_RE = re.compile(r"\bround\b", re.IGNORECASE)
ROUND_DIGIT_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s+round", re.IGNORECASE)
ROUND_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
ROUND_WORD_RE = re.compile(
    r"\b(" + "|".join(ROUND_WORDS) + r")\s+round", re.IGNORECASE
)
PICK_OVERALL_RE = re.compile(r"#(\d+)\s*-\s*([^)]+)")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Typo-tolerant: catches "Conditional", "Conditionnal", "Conditonal" (real typos in the data).
CONDITIONAL_RE = re.compile(r"\bcondit\w*", re.IGNORECASE)

# Word-boundary is load-bearing here: a naive substring search for "cash" matches inside
# player names like "Jason Bacashihua" ("Ba-CASH-ihua"). \b prevents that false positive.
CASH_WORD_RE = re.compile(r"\bcash\b", re.IGNORECASE)
CASH_AMOUNT_RE = re.compile(r"\$\s*[\d][\d,.\s]*(?:[KM])?")

# Player-name matching. Only the *first* token is required to start uppercase; interior tokens
# may be a known lowercase name-connector (Dutch/French "van", "de", "St." ...) so that names
# like "Trevor van Riemsdyk" and "Calvin de Haan" still match even though a naive
# "every word is capitalized" rule would reject the lowercase connector and misfire.
_INITIALS = r"[A-Z]\.(?:[A-Z]\.)+"  # "J.D.", "T.J."
_WORD_CAP = r"[A-Z][^\W\d_]*(?:['’`-][^\W\d_]+)*\.?"  # "O`Brien", "Fernström", "St."
_NAME_WORD = rf"(?:{_INITIALS}|{_WORD_CAP})"
_CONNECTOR = r"(?:[Vv]an|[Dd]er|[Dd]e|[Ll]a|[Vv]on|[Ll]e|[Dd]i|[Dd]u|dit|[Ss]t\.?)"
_SUFFIX = r"(?:Jr\.?|Sr\.?|II|III|IV)"
NAME_RE = re.compile(
    rf"^{_NAME_WORD}(?:\s+(?:{_CONNECTOR}|{_NAME_WORD}))*(?:\s+{_SUFFIX})?$"
)

_EMPTY_VALUES = {"", "none", "n/a", "-", "nothing"}


def _base_result(asset_type: str, subtype: str) -> dict:
    return {
        "type": asset_type,
        "subtype": subtype,
        "name": None,
        "birth_year": None,
        "pick_year": None,
        "pick_round": None,
        "pick_overall": None,
        "pick_player": None,
        "is_conditional": False,
        "cash_amount": None,
    }


def _parse_core(text: str) -> dict:
    """Classify already-unwrapped text (no more "future considerations (...)" layer) as
    player / pick / other."""
    birth_year = None
    m = BIRTH_YEAR_RE.search(text)
    work = text
    if m:
        birth_year = int(m.group("year"))
        work = re.sub(r"\s+", " ", (text[: m.start()] + " " + text[m.end() :])).strip()

    if work.lower().strip() in _EMPTY_VALUES:
        result = _base_result("other", "none")
        result["birth_year"] = birth_year
        return result

    if PICK_KEYWORD_RE.search(work) or ROUND_KEYWORD_RE.search(work):
        result = _base_result("pick", "pick")
        result["is_conditional"] = bool(CONDITIONAL_RE.search(work))
        year_m = YEAR_RE.search(work)
        result["pick_year"] = int(year_m.group(0)) if year_m else None
        round_m = ROUND_DIGIT_RE.search(work)
        if round_m:
            result["pick_round"] = int(round_m.group(1))
        else:
            round_word_m = ROUND_WORD_RE.search(work)
            if round_word_m:
                result["pick_round"] = ROUND_WORDS[round_word_m.group(1).lower()]
        overall_m = PICK_OVERALL_RE.search(work)
        if overall_m:
            result["pick_overall"] = int(overall_m.group(1))
            result["pick_player"] = overall_m.group(2).strip().rstrip(")").strip()
        result["birth_year"] = birth_year
        return result

    if CASH_WORD_RE.search(work) or ("$" in work and re.search(r"\d", work)):
        result = _base_result("other", "cash")
        amount_m = CASH_AMOUNT_RE.search(work)
        result["cash_amount"] = amount_m.group(0).strip() if amount_m else None
        result["birth_year"] = birth_year
        return result

    name_part = work
    subtype = "trade"
    lowered = work.lower()
    if lowered.startswith("rights to "):
        name_part = work[len("rights to ") :].strip()
        subtype = "rights"
    elif lowered.startswith("loan of "):
        name_part = re.split(r"\bfor\b", work[len("loan of ") :], maxsplit=1)[0].strip()
        subtype = "loan"

    if name_part and NAME_RE.match(name_part):
        result = _base_result("player", subtype)
        result["name"] = name_part
        result["birth_year"] = birth_year
        return result

    result = _base_result("other", "unclassified")
    result["birth_year"] = birth_year
    return result


def parse_acquisition_line(line: str) -> dict:
    """Parse one line of an acquisitions cell into a structured dict.

    Returns a dict with keys: raw, type ("player"/"pick"/"other"), subtype, name,
    birth_year, pick_year, pick_round, pick_overall, pick_player, is_conditional,
    cash_amount, wrapped_in_future_considerations.

    Handles, based on patterns found in data/raw/trades_raw.csv:
      - "future considerations (X)" - unwraps and re-classifies X (recursing one level;
        a bare "future considerations" with no parenthetical is type "other").
      - Word-boundary cash matching, so "Jason Bacashihua" (a player) isn't misread as a
        cash line just because the substring "cash" appears inside "Bacashihua".
      - Lowercase name-connectors ("van", "de", "der", "la", "von", "le", "di", "du", "st")
        so names like "Trevor van Riemsdyk" / "Calvin de Haan" still match as players.
      - The "(b.YYYY)" birth-year disambiguator embedded mid-name, e.g.
        "Petr (b.1976) Sykora", which is stripped out to recover the plain name.
    """
    if line is None:
        raw = ""
    else:
        raw = line.strip()

    if not raw:
        result = _base_result("other", "empty")
        result["raw"] = raw
        result["wrapped_in_future_considerations"] = False
        return result

    fc_m = FUTURE_CONSIDERATIONS_RE.match(raw)
    if fc_m:
        remainder = fc_m.group(1).strip()
        if remainder == "":
            result = _base_result("other", "future_considerations")
        elif remainder.startswith("("):
            inner = remainder[1:]
            if inner.endswith(")"):
                inner = inner[:-1]
            inner = inner.strip()
            result = (
                _parse_core(inner) if inner else _base_result("other", "future_considerations")
            )
        elif re.match(r"^and\s+cash$", remainder, re.IGNORECASE):
            result = _base_result("other", "future_considerations_and_cash")
        else:
            result = _base_result("other", "future_considerations")
        result["raw"] = raw
        result["wrapped_in_future_considerations"] = True
        return result

    result = _parse_core(raw)
    result["raw"] = raw
    result["wrapped_in_future_considerations"] = False
    return result


# --- Tidy dataframe ------------------------------------------------------------------------

_SIDES = [
    ("team_one", "Team One", "Team One Acquisitions", "Team Two"),
    ("team_two", "Team Two", "Team Two Acquisitions", "Team One"),
]


def build_tidy_trades(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Explode the raw trades table (one row per trade, two free-text acquisition cells)
    into a tidy long-format table with one row per acquired asset."""
    records = []
    for trade_id, row in enumerate(raw_df.to_dict(orient="records"), start=1):
        for side, acquiring_col, acq_cell_col, counterparty_col in _SIDES:
            cell = row.get(acq_cell_col)
            if pd.isna(cell):
                continue
            for line in str(cell).split("\n"):
                line = line.strip()
                if not line:
                    continue
                parsed = parse_acquisition_line(line)
                records.append(
                    {
                        "trade_id": trade_id,
                        "season": row.get("Season"),
                        "date": row.get("Date"),
                        "month": row.get("Month"),
                        "side": side,
                        "acquiring_team": row.get(acquiring_col),
                        "counterparty_team": row.get(counterparty_col),
                        **parsed,
                    }
                )

    columns = [
        "trade_id", "season", "date", "month", "side", "acquiring_team", "counterparty_team",
        "type", "subtype", "name", "birth_year", "pick_year", "pick_round", "pick_overall",
        "pick_player", "is_conditional", "cash_amount", "wrapped_in_future_considerations",
        "raw",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"Total parsed assets: {total:,}")
    print()
    print("Parse rate by type:")
    for asset_type, count in df["type"].value_counts().items():
        print(f"  {asset_type:<8} {count:>6,}  ({count / total * 100:5.1f}%)")
    print()

    other_df = df[df["type"] == "other"]
    other_pct = len(other_df) / total * 100
    print(f'"other" share of all assets: {len(other_df):,} / {total:,} ({other_pct:.1f}%)')
    print()
    print('Breakdown within "other" by subtype:')
    for subtype, count in other_df["subtype"].value_counts().items():
        print(f"  {subtype:<28} {count:>6,}  ({count / len(other_df) * 100:5.1f}% of other)")


def main() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW_TRADES_CSV)
    tidy_df = build_tidy_trades(raw_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(CLEAN_TRADES_CSV, index=False)
    print(f"Wrote {len(tidy_df):,} rows to {CLEAN_TRADES_CSV}")
    print()
    print_summary(tidy_df)
    return tidy_df


if __name__ == "__main__":
    main()
