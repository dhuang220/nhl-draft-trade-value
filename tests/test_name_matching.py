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


def test_non_prefix_nickname_matches_via_dictionary():
    # "Charlie" isn't a prefix of "Charles" (they diverge after "Charl") - this
    # needs the nickname-equivalence table, not the prefix check.
    index = build_last_name_index(["Charles McAvoy"])
    assert is_same_player("Charlie McAvoy", index)

    index = build_last_name_index(["Vincent Hinostroza"])
    assert is_same_player("Vinnie Hinostroza", index)


def test_transliteration_variant_matches_via_fuzzy_fallback():
    index = build_last_name_index(["Sergey Bobrovsky"])
    assert is_same_player("Sergei Bobrovsky", index)


def test_unrelated_short_nickname_like_first_name_does_not_match():
    # A same-surname collision between two different real people should not be
    # rescued by the fuzzy fallback just because the names are short.
    index = build_last_name_index(["Jack Smith"])
    assert not is_same_player("Josh Smith", index)
