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


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.strip().lower()


def _split_name(name: str) -> tuple[str, str]:
    parts = _normalize(name).split()
    return parts[0], parts[-1]  # first, last - middle names/suffixes are ignored


def build_last_name_index(names: list[str]) -> dict[str, list[str]]:
    index = defaultdict(list)
    for name in names:
        first, last = _split_name(name)
        index[last].append(first)
    return index


def is_same_player(name: str, last_name_index: dict[str, list[str]]) -> bool:
    first, last = _split_name(name)
    candidates = last_name_index.get(last, [])
    return any(first.startswith(c) or c.startswith(first) for c in candidates)
