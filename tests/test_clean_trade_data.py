"""Tests for src/etl/clean_trade_data.py.

Covers the real line formats and edge cases found in data/raw/trades_raw.csv:
plain player names, "rights to"/"loan of" players, draft picks (digit and word-spelled
rounds, conditional/typo variants, overall-pick + drafted-player extraction), cash amounts,
the nested "future considerations (...)" wrapper (including unbalanced parens), the
"(b.YYYY)" birth-year name disambiguator, lowercase name-connectors, and the
"Jason Bacashihua" cash false-positive.
"""

import pandas as pd
import pytest

from src.etl.clean_trade_data import build_tidy_trades, parse_acquisition_line


# --- Player lines ----------------------------------------------------------------------

def test_plain_player_name():
    result = parse_acquisition_line("Corey Perry")
    assert result["type"] == "player"
    assert result["subtype"] == "trade"
    assert result["name"] == "Corey Perry"


def test_rights_to_player():
    result = parse_acquisition_line("rights to Alyn McCauley")
    assert result["type"] == "player"
    assert result["subtype"] == "rights"
    assert result["name"] == "Alyn McCauley"


def test_loan_of_player_strips_trailing_duration():
    result = parse_acquisition_line("loan of Albert Leduc for remainder of 1933-34 season")
    assert result["type"] == "player"
    assert result["subtype"] == "loan"
    assert result["name"] == "Albert Leduc"


def test_player_name_with_initials():
    result = parse_acquisition_line("J.D. Dudek")
    assert result["type"] == "player"
    assert result["name"] == "J.D. Dudek"


def test_player_name_with_suffix():
    result = parse_acquisition_line("Ab DeMarco Jr.")
    assert result["type"] == "player"
    assert result["name"] == "Ab DeMarco Jr."


def test_player_name_with_apostrophe():
    result = parse_acquisition_line("Kevin O'Sullivan")
    assert result["type"] == "player"
    assert result["name"] == "Kevin O'Sullivan"


def test_player_name_with_backtick_apostrophe():
    result = parse_acquisition_line("Willie O`Ree")
    assert result["type"] == "player"
    assert result["name"] == "Willie O`Ree"


def test_player_name_with_unicode_letters():
    result = parse_acquisition_line("Melvin Fernström")
    assert result["type"] == "player"
    assert result["name"] == "Melvin Fernström"


# --- Lowercase name-connector handling --------------------------------------------------

def test_lowercase_van_connector():
    result = parse_acquisition_line("Trevor van Riemsdyk")
    assert result["type"] == "player"
    assert result["name"] == "Trevor van Riemsdyk"


def test_capitalized_van_connector_still_matches():
    result = parse_acquisition_line("James Van Riemsdyk")
    assert result["type"] == "player"
    assert result["name"] == "James Van Riemsdyk"


def test_lowercase_de_connector():
    result = parse_acquisition_line("Calvin de Haan")
    assert result["type"] == "player"
    assert result["name"] == "Calvin de Haan"


# --- Birth-year disambiguator ------------------------------------------------------------

def test_birth_year_disambiguator_1976():
    result = parse_acquisition_line("Petr (b.1976) Sykora")
    assert result["type"] == "player"
    assert result["name"] == "Petr Sykora"
    assert result["birth_year"] == 1976


def test_birth_year_disambiguator_distinguishes_two_players():
    a = parse_acquisition_line("Petr (b.1976) Sykora")
    b = parse_acquisition_line("Petr (b.1978) Sykora")
    assert a["name"] == b["name"] == "Petr Sykora"
    assert a["birth_year"] != b["birth_year"]
    assert {a["birth_year"], b["birth_year"]} == {1976, 1978}


# --- Cash lines, including the word-boundary false positive -----------------------------

def test_jason_bacashihua_is_not_cash():
    """'Bacashihua' contains the substring 'cash' but is a player, not a cash asset -
    a naive substring search for 'cash' misfires here."""
    result = parse_acquisition_line("Jason Bacashihua")
    assert result["type"] == "player"
    assert result["name"] == "Jason Bacashihua"
    assert result["subtype"] != "cash"


def test_bare_cash():
    result = parse_acquisition_line("cash")
    assert result["type"] == "other"
    assert result["subtype"] == "cash"


def test_cash_amount_extracted():
    result = parse_acquisition_line("$10K cash")
    assert result["type"] == "other"
    assert result["subtype"] == "cash"
    assert result["cash_amount"] == "$10K"


def test_cash_amount_with_space_thousands_separator():
    result = parse_acquisition_line("$2 500 cash")
    assert result["type"] == "other"
    assert result["subtype"] == "cash"
    assert result["cash_amount"] == "$2 500"


# --- Draft picks -----------------------------------------------------------------------

def test_pick_with_digit_round_and_overall_and_drafted_player():
    result = parse_acquisition_line("1977 3rd round pick (#43-Alain Cote)")
    assert result["type"] == "pick"
    assert result["pick_year"] == 1977
    assert result["pick_round"] == 3
    assert result["pick_overall"] == 43
    assert result["pick_player"] == "Alain Cote"
    assert result["is_conditional"] is False


def test_pick_with_word_spelled_round():
    result = parse_acquisition_line("1992 eleventh round pick (#259-Wade Salzman)")
    assert result["type"] == "pick"
    assert result["pick_round"] == 11
    assert result["pick_year"] == 1992


def test_conditional_pick_typo_conditonal():
    """'Conditonal' (missing an 'i') is a real typo in the source data."""
    result = parse_acquisition_line("Conditonal 2012 or 2013 2nd round pick")
    assert result["type"] == "pick"
    assert result["is_conditional"] is True
    assert result["pick_round"] == 2


def test_conditional_pick_typo_conditionnal():
    result = parse_acquisition_line("Conditionnal 2013 5th round pick")
    assert result["type"] == "pick"
    assert result["is_conditional"] is True


def test_pick_typo_pcik_still_detected_via_round():
    result = parse_acquisition_line("2013 4th round pcik")
    assert result["type"] == "pick"
    assert result["pick_round"] == 4


def test_generic_conditional_pick_no_round():
    result = parse_acquisition_line("Conditional pick")
    assert result["type"] == "pick"
    assert result["is_conditional"] is True
    assert result["pick_round"] is None


# --- Nested "future considerations (...)" unwrapping -------------------------------------

def test_future_considerations_bare_is_other():
    result = parse_acquisition_line("future considerations")
    assert result["type"] == "other"
    assert result["subtype"] == "future_considerations"
    assert result["wrapped_in_future_considerations"] is True


def test_future_considerations_case_insensitive_singular():
    result = parse_acquisition_line("Future consideration")
    assert result["type"] == "other"
    assert result["subtype"] == "future_considerations"


def test_future_considerations_unwraps_nested_pick():
    result = parse_acquisition_line(
        "future considerations (1977 3rd round pick (#43-Alain Cote))"
    )
    assert result["type"] == "pick"
    assert result["pick_year"] == 1977
    assert result["pick_round"] == 3
    assert result["pick_overall"] == 43
    assert result["pick_player"] == "Alain Cote"
    assert result["wrapped_in_future_considerations"] is True
    assert result["raw"] == "future considerations (1977 3rd round pick (#43-Alain Cote))"


def test_future_considerations_unwraps_nested_player():
    result = parse_acquisition_line("future considerations (rights to Theoren Fleury)")
    assert result["type"] == "player"
    assert result["subtype"] == "rights"
    assert result["name"] == "Theoren Fleury"
    assert result["wrapped_in_future_considerations"] is True


def test_future_considerations_unwraps_nested_cash():
    result = parse_acquisition_line("future considerations (cash)")
    assert result["type"] == "other"
    assert result["subtype"] == "cash"
    assert result["wrapped_in_future_considerations"] is True


def test_future_considerations_handles_missing_closing_paren():
    """Real row: the source data drops the outer closing paren here."""
    result = parse_acquisition_line(
        "future considerations (1988 3rd round pick (#62-Daniel Gauthier)"
    )
    assert result["type"] == "pick"
    assert result["pick_year"] == 1988
    assert result["pick_round"] == 3
    assert result["pick_overall"] == 62
    assert result["pick_player"] == "Daniel Gauthier"


def test_future_considerations_and_cash():
    result = parse_acquisition_line("future considerations and cash")
    assert result["type"] == "other"
    assert result["subtype"] == "future_considerations_and_cash"
    assert result["wrapped_in_future_considerations"] is True


def test_non_future_considerations_line_not_wrapped():
    result = parse_acquisition_line("Corey Perry")
    assert result["wrapped_in_future_considerations"] is False


# --- Other/unclassified catch-all --------------------------------------------------------

def test_none_placeholder_is_other():
    result = parse_acquisition_line("none")
    assert result["type"] == "other"
    assert result["subtype"] == "none"


def test_descriptive_sentence_is_unclassified_other():
    result = parse_acquisition_line(
        "Islanders promised to not draft certain players in 1972 expansion draft"
    )
    assert result["type"] == "other"
    assert result["subtype"] == "unclassified"


def test_empty_line_is_other():
    result = parse_acquisition_line("")
    assert result["type"] == "other"
    assert result["subtype"] == "empty"


def test_whitespace_only_line_is_other():
    result = parse_acquisition_line("   \n  ")
    assert result["type"] == "other"
    assert result["subtype"] == "empty"


# --- raw field always preserved -----------------------------------------------------------

def test_raw_field_preserves_original_text_with_surrounding_whitespace_stripped():
    result = parse_acquisition_line("  Corey Perry  \n")
    assert result["raw"] == "Corey Perry"


# --- build_tidy_trades ----------------------------------------------------------------

def test_build_tidy_trades_explodes_multiline_cells_and_assigns_trade_id():
    raw_df = pd.DataFrame(
        [
            {
                "Season": "2024-25",
                "Date": "March 7, 2025",
                "Month": "March",
                "Team One": "Anaheim Ducks",
                "Team One Acquisitions": "Oliver Kylington \n",
                "Team Two": "New York Islanders",
                "Team Two Acquisitions": "Future considerations \n",
            },
            {
                "Season": "1999-00",
                "Date": "January 1, 2000",
                "Month": "January",
                "Team One": "Team A",
                "Team One Acquisitions": "Player One\nPlayer Two",
                "Team Two": "Team B",
                "Team Two Acquisitions": "1999 3rd round pick (#80-Some Guy)",
            },
        ]
    )

    tidy = build_tidy_trades(raw_df)

    # 1 (Kylington) + 1 (future considerations) + 2 (Player One/Two) + 1 (pick) = 5 rows.
    assert len(tidy) == 5

    # Each trade's rows share one trade_id, and trade_id increments per source row.
    assert set(tidy[tidy["trade_id"] == 1]["side"]) == {"team_one", "team_two"}
    trade_2 = tidy[tidy["trade_id"] == 2]
    assert len(trade_2) == 3

    kylington = tidy[tidy["raw"] == "Oliver Kylington"].iloc[0]
    assert kylington["type"] == "player"
    assert kylington["acquiring_team"] == "Anaheim Ducks"
    assert kylington["counterparty_team"] == "New York Islanders"
    assert kylington["side"] == "team_one"

    names = set(trade_2[trade_2["type"] == "player"]["name"])
    assert names == {"Player One", "Player Two"}

    pick_row = trade_2[trade_2["type"] == "pick"].iloc[0]
    assert pick_row["pick_year"] == 1999
    assert pick_row["pick_round"] == 3
    assert pick_row["pick_overall"] == 80
    assert pick_row["pick_player"] == "Some Guy"
    assert pick_row["acquiring_team"] == "Team B"
    assert pick_row["counterparty_team"] == "Team A"


def test_build_tidy_trades_skips_missing_acquisition_cells():
    raw_df = pd.DataFrame(
        [
            {
                "Season": "1990-91",
                "Date": "January 1, 1991",
                "Month": "January",
                "Team One": "Team A",
                "Team One Acquisitions": "Player One",
                "Team Two": "Team B",
                "Team Two Acquisitions": float("nan"),
            },
        ]
    )
    tidy = build_tidy_trades(raw_df)
    assert len(tidy) == 1
    assert tidy.iloc[0]["side"] == "team_one"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
