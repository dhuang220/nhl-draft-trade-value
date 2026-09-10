import pandas as pd

from src.features.draft_features import (
    MATURE_CUTOFF_YEAR,
    build_feature_matrix,
    clean_draft_data,
    merge_draft_sources,
)


def _row(**overrides):
    row = {
        "year": 2005,
        "overall_pick": 10,
        "age": 18,
        "position": "C",
        "games_played": 100,
        "point_shares": 5.0,
    }
    row.update(overrides)
    return row


def test_point_shares_nan_for_never_played_becomes_zero_not_dropped():
    df = pd.DataFrame(
        [
            _row(games_played=float("nan"), point_shares=float("nan")),
            _row(games_played=200, point_shares=12.0),
        ]
    )
    out = clean_draft_data(df)

    assert len(out) == 2
    assert out.loc[0, "point_shares"] == 0.0
    assert out.loc[0, "games_played"] == 0.0
    assert out.loc[1, "point_shares"] == 12.0


def test_is_mature_flag_exact_cutoff_boundary():
    df = pd.DataFrame(
        [
            _row(year=MATURE_CUTOFF_YEAR - 1),
            _row(year=MATURE_CUTOFF_YEAR),
            _row(year=MATURE_CUTOFF_YEAR + 1),
        ]
    )
    out = clean_draft_data(df)
    assert out.loc[0, "is_mature"] == True
    assert out.loc[1, "is_mature"] == True
    assert out.loc[2, "is_mature"] == False


def test_position_group_is_goalie_only_for_g_others_are_skater():
    df = pd.DataFrame(
        [
            _row(position="C"),
            _row(position="D"),
            _row(position="G"),
        ]
    )
    out = clean_draft_data(df)
    assert out.loc[0, "position_group"] == "Skater"
    assert out.loc[1, "position_group"] == "Skater"
    assert out.loc[2, "position_group"] == "G"


def test_merge_draft_sources_keeps_2000_to_2020_kaggle_rows_and_all_recent_rows():
    kaggle = pd.DataFrame([_row(year=y) for y in [1999, 2000, 2020, 2021, 2022]])
    recent = pd.DataFrame([_row(year=y) for y in [2023, 2024]])

    out = merge_draft_sources(kaggle, recent)

    assert sorted(out["year"]) == [2000, 2020, 2023, 2024]


def test_merge_draft_sources_excludes_2021_and_2022_from_both_sides():
    # 2021-2022 stay out of the dashboard's dataset either way - the Kaggle file has them,
    # but a caller could in principle also pass recent rows tagged with those years, and
    # they should still be dropped rather than silently included via the Kaggle branch.
    kaggle = pd.DataFrame([_row(year=y) for y in [2021, 2022]])
    recent = pd.DataFrame([_row(year=y) for y in [2023]])

    out = merge_draft_sources(kaggle, recent)

    assert 2021 not in out["year"].values
    assert 2022 not in out["year"].values


def test_build_feature_matrix_one_hot_encodes_position_group_with_drop_first():
    df = pd.DataFrame([_row(position="C"), _row(position="G")])
    clean = clean_draft_data(df)
    matrix = build_feature_matrix(clean)

    assert "position_group" not in matrix.columns
    assert "overall_pick" in matrix.columns
    assert "age" in matrix.columns
    # drop_first=True over {"G", "Skater"} keeps only one dummy column
    dummy_cols = [c for c in matrix.columns if c.startswith("position_group_")]
    assert len(dummy_cols) == 1
    assert dummy_cols[0] == "position_group_Skater"
    assert list(matrix[dummy_cols[0]]) == [True, False]
