import pytest

from src.trade.trade_grader import TradeGrader


@pytest.fixture(scope="module")
def grader():
    return TradeGrader()


def test_future_pick_is_discounted_relative_to_immediate_pick(grader):
    immediate = grader.value_pick(pick_year=2020, pick_round=1, pick_overall=15, is_conditional=False, trade_year=2020)
    future = grader.value_pick(pick_year=2023, pick_round=1, pick_overall=15, is_conditional=False, trade_year=2020)
    assert future.value < immediate.value


def test_conditional_pick_is_discounted_relative_to_unconditional(grader):
    unconditional = grader.value_pick(pick_year=2020, pick_round=2, pick_overall=None, is_conditional=False, trade_year=2020)
    conditional = grader.value_pick(pick_year=2020, pick_round=2, pick_overall=None, is_conditional=True, trade_year=2020)
    assert conditional.value < unconditional.value


def test_unmatched_player_reports_none_not_zero(grader):
    asset = grader.value_player_career_fallback("Definitely Not A Real Player Xyzzy")
    assert asset.value is None
    assert asset.confidence == "none"


def test_2016_17_player_lookup_uses_high_confidence_season_model(grader):
    # Auston Matthews' rookie season is in the training data with a well-known real outcome.
    asset = grader.value_player_2016_17("Auston Matthews")
    assert asset.confidence == "high"
    assert asset.value > 0
