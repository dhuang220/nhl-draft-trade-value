import pandas as pd

from src.features.value_features import FEATURE_COLUMNS, RATE_STATS, build_feature_matrix, clean_value_training_data


def _row(**overrides):
    row = {
        "Born": "90-06-15",
        "DftYr": 2008,
        "DftRd": 1,
        "Ovrl": 5,
        "Position": "C",
        "GP": 20,
        "G": 4,
        "A": 6,
        "PIM": 10,
        "iHF": 8,
        "iGVA": 2,
        "iTKA": 3,
        "iBLK": 5,
        "CF": 100,
        "CA": 90,
        "xGF": 10.0,
        "xGA": 8.0,
    }
    row.update(overrides)
    return row


def test_birth_year_two_digit_century_disambiguation():
    df = pd.DataFrame([_row(Born="97-01-30"), _row(Born="02-01-15")])
    out = clean_value_training_data(df, season_end_year=2017)
    assert out.loc[0, "age"] == 2017 - 1997
    assert out.loc[1, "age"] == 2017 - 2002


def test_undrafted_player_encoded_with_pinned_sentinel_values():
    df = pd.DataFrame(
        [
            _row(DftYr=2008, DftRd=1, Ovrl=5),
            _row(DftYr=float("nan"), DftRd=float("nan"), Ovrl=float("nan")),
        ]
    )
    out = clean_value_training_data(df)

    assert out.loc[0, "was_drafted"] == 1
    assert out.loc[0, "draft_round"] == 1
    assert out.loc[0, "draft_overall"] == 5

    assert out.loc[1, "was_drafted"] == 0
    assert out.loc[1, "draft_round"] == 0
    assert out.loc[1, "draft_overall"] == 300


def test_per_gp_rate_stat_divides_by_games_played():
    df = pd.DataFrame([_row(GP=2, G=4)])
    out = clean_value_training_data(df)
    assert out.loc[0, "G_per_gp"] == 2.0


def test_per_gp_rate_stat_clips_zero_gp_to_one_instead_of_dividing_by_zero():
    df = pd.DataFrame([_row(GP=0, G=3)])
    out = clean_value_training_data(df)
    # GP is clipped to a floor of 1, so a 0-GP player's rate is just the raw count,
    # not inf/NaN from a literal divide-by-zero.
    assert out.loc[0, "G_per_gp"] == 3.0


def test_build_feature_matrix_returns_exact_columns_in_order_with_no_nans():
    df = pd.DataFrame([_row(), _row(Born="85-11-02", DftYr=float("nan"), DftRd=float("nan"), Ovrl=float("nan"))])
    clean = clean_value_training_data(df)
    matrix = build_feature_matrix(clean)

    assert list(matrix.columns) == FEATURE_COLUMNS
    assert not matrix.isna().any().any()
    # sanity: every RATE_STATS entry produced its per_gp column in the matrix
    for stat in RATE_STATS:
        assert f"{stat}_per_gp" in matrix.columns
