"""Match a player name from the draft history dataset against the live NHL roster feed,
even when the two sources spell the same person's name differently - seen with accented
letters ("Lafreniere" vs "Lafrenière") and nicknames ("Mitchell" vs "Mitch" Marner).

A same-surname, prefix-related-first-name match is treated as the same player. This can
theoretically produce a false positive (a real "Cal Smith" wrongly matched to an unrelated
"Callahan Smith"), but that's a rare, accepted tradeoff for closing a much more common gap -
the same kind of graceful-approximation call made throughout this project's name handling.
"""

import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

# Common English nicknames that don't share a prefix with their formal name (a
# prefix check alone misses "Charlie"/"Charles" - they diverge after "charl").
# Not exhaustive - the fuzzy-similarity fallback below catches most anything
# else, at the cost of occasionally needing a close-enough spelling.
_NICKNAME_GROUPS = [
    {"charles", "charlie", "chuck", "chick"},
    {"william", "will", "bill", "billy", "willy"},
    {"robert", "rob", "bob", "bobby", "robbie"},
    {"richard", "rick", "dick", "ricky"},
    {"james", "jim", "jimmy", "jamie"},
    {"john", "jack", "johnny", "jon", "jonathan"},
    {"michael", "mike", "mikey", "mick"},
    {"matthew", "matt", "matty"},
    {"nicholas", "nicolas", "nick", "nico", "nicky"},
    {"vincent", "vince", "vinnie", "vinny", "vin"},
    {"alexander", "alex", "sasha"},
    {"anthony", "tony"},
    {"christopher", "chris"},
    {"zachary", "zach", "zack", "zac"},
    {"benjamin", "ben", "benny"},
    {"samuel", "sam", "sammy"},
    {"jacob", "jake"},
    {"joseph", "joe", "joey"},
    {"timothy", "tim", "timmy"},
    {"gregory", "greg"},
    {"edward", "ed", "eddie", "ted", "teddy"},
    {"thomas", "tom", "tommy"},
    {"daniel", "dan", "danny"},
    {"david", "dave", "davey"},
    {"steven", "stephen", "steve"},
    {"andrew", "andy", "drew"},
    {"patrick", "pat", "paddy"},
    {"kenneth", "ken", "kenny"},
    {"donald", "don", "donnie"},
    {"douglas", "doug"},
    {"gerald", "gerry", "jerry"},
    {"harold", "harry"},
    {"lawrence", "larry"},
    {"raymond", "ray"},
    {"ronald", "ron", "ronnie"},
    {"russell", "russ"},
    {"walter", "walt"},
]
_NICKNAME_LOOKUP = {name: group for group in _NICKNAME_GROUPS for name in group}

# How close two first names sharing a last name must be to count as the same
# spelling by coincidence (a real accent/typo variant) rather than two
# different people who happen to share a common surname.
_FUZZY_THRESHOLD = 0.75


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.strip().lower()


def _split_name(name: str) -> tuple[str, str]:
    parts = _normalize(name).split()
    return parts[0], parts[-1]  # first, last - middle names/suffixes are ignored


def _first_names_match(a: str, b: str) -> bool:
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    if _NICKNAME_LOOKUP.get(a) == _NICKNAME_LOOKUP.get(b) and a in _NICKNAME_LOOKUP:
        return True
    return SequenceMatcher(None, a, b).ratio() >= _FUZZY_THRESHOLD


def build_last_name_index(names: list[str]) -> dict[str, list[str]]:
    index = defaultdict(list)
    for name in names:
        first, last = _split_name(name)
        index[last].append(first)
    return index


def is_same_player(name: str, last_name_index: dict[str, list[str]]) -> bool:
    first, last = _split_name(name)
    candidates = last_name_index.get(last, [])
    return any(_first_names_match(first, c) for c in candidates)
