from src.features.name_matching import build_last_name_index, is_same_player


def test_exact_match():
    index = build_last_name_index(["Connor McDavid"])
    assert is_same_player("Connor McDavid", index)


def test_accent_mismatch_still_matches():
    index = build_last_name_index(["Alexis Lafrenière"])
    assert is_same_player("Alexis Lafreniere", index)


def test_nickname_prefix_matches_both_directions():
    index = build_last_name_index(["Mitch Marner"])
    assert is_same_player("Mitchell Marner", index)

    index = build_last_name_index(["Mitchell Marner"])
    assert is_same_player("Mitch Marner", index)


def test_different_last_name_does_not_match():
    index = build_last_name_index(["Connor McDavid"])
    assert not is_same_player("Connor Bedard", index)


def test_same_last_name_unrelated_first_name_does_not_match():
    index = build_last_name_index(["Steve Smith"])
    assert not is_same_player("John Smith", index)
