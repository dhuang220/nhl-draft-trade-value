from src.etl.fetch_current_stats import _disambiguate_duplicate_names


def _player(first: str, last: str, position: str) -> dict:
    return {"firstName": {"default": first}, "lastName": {"default": last}, "positionCode": position}


def test_unique_names_are_left_untouched():
    players = [_player("Connor", "McDavid", "C"), _player("Cale", "Makar", "D")]
    assert _disambiguate_duplicate_names(players) == ["Connor McDavid", "Cale Makar"]


def test_duplicate_name_gets_position_suffix():
    # Vancouver currently rosters two active "Elias Pettersson"s - a defenseman and a center.
    players = [_player("Elias", "Pettersson", "C"), _player("Elias", "Pettersson", "D")]
    assert _disambiguate_duplicate_names(players) == ["Elias Pettersson (C)", "Elias Pettersson (D)"]


def test_only_the_colliding_names_are_suffixed():
    players = [_player("Elias", "Pettersson", "C"), _player("Elias", "Pettersson", "D"), _player("Cale", "Makar", "D")]
    assert _disambiguate_duplicate_names(players) == [
        "Elias Pettersson (C)", "Elias Pettersson (D)", "Cale Makar",
    ]
