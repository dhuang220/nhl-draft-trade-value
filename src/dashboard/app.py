"""Streamlit dashboard: draft-value exploration and trade grading.

Two tabs. Draft Explorer plots the fitted pick-value curve against a chosen
draft class and surfaces the negative modeling result from train_draft_model.py
(pick number alone beat the richer XGBoost model on held-out draft classes).
Trade Grader wraps TradeGrader to grade a historical trade from trades_clean.csv
or a hypothetical trade built from live player names.
"""

import sys
from datetime import date
from pathlib import Path

# `streamlit run src/dashboard/app.py` puts this file's own directory on
# sys.path, not the repo root - so `import src....` fails unless the root is
# added explicitly first (bites locally too, not just on Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.etl.fetch_current_stats import fetch_all_current_players, fetch_team_logos
from src.features.draft_features import clean_draft_data, merge_draft_sources
from src.features.name_matching import build_last_name_index, is_same_player
from src.trade.trade_grader import HypotheticalPick, TradeGrader, TradeSideGrade

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

st.set_page_config(page_title="NHL Draft Value & Trade Grader", layout="wide")


@st.cache_data
def load_draft_data() -> pd.DataFrame:
    kaggle = pd.read_csv(RAW_DIR / "draft_history_raw.csv")
    # 2023+ classes have no Kaggle equivalent yet - recent_draft_data.csv is fetched live from
    # the NHL API instead (see fetch_recent_draft_data.py); its point_shares is our value
    # model's predicted career value, not the real stat, since these players are too recent
    # for one to exist.
    recent = pd.read_csv(PROCESSED_DIR / "recent_draft_data.csv")
    return clean_draft_data(merge_draft_sources(kaggle, recent))


@st.cache_resource
def load_pick_curve():
    return joblib.load(PROCESSED_DIR / "pick_value_curve.joblib")


@st.cache_resource
def load_games_played_curve():
    return joblib.load(PROCESSED_DIR / "games_played_curve.joblib")


TRADE_GRADER_YEAR_RANGE = (2000, 2026)


@st.cache_data
def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(PROCESSED_DIR / "trades_clean.csv")
    season_start_year = trades["season"].str.slice(0, 4).astype(int)
    lo, hi = TRADE_GRADER_YEAR_RANGE
    return trades[season_start_year.between(lo, hi)]


@st.cache_resource
def load_trade_grader() -> TradeGrader:
    return TradeGrader()


@st.cache_data(ttl=3600)
def load_current_player_names() -> list[str]:
    return fetch_all_current_players()


@st.cache_data(ttl=3600)
def load_team_logos() -> dict[str, str]:
    return fetch_team_logos()


def draft_explorer_tab():
    draft = load_draft_data()
    curve = load_pick_curve()
    games_curve = load_games_played_curve()
    # A plain exact-match against current roster names misses real active players -
    # the draft dataset spells names differently than the NHL's live API for accented
    # letters ("Lafreniere" vs "Lafrenière") and nicknames ("Mitchell" vs "Mitch"
    # Marner) - see name_matching.py.
    current_player_index = build_last_name_index(load_current_player_names())

    year = st.selectbox("Draft year", sorted(draft["year"].unique(), reverse=True))
    class_df = draft[draft["year"] == year].sort_values("overall_pick").copy()

    is_mature = bool(class_df["is_mature"].iloc[0]) if not class_df.empty else False
    if is_mature:
        st.caption(f"{year} class is **mature** (drafted 2012 or earlier) - used to train the pick-value curve.")
    else:
        st.caption(
            f"{year} class is **excluded from model training** (drafted after 2012, so most careers are "
            "still in progress). Below, players are split by whether they're on an NHL roster right now, "
            "not by class year - see the note further down."
        )

    # Per-PLAYER, not per-class: a "mature" class can still have an active veteran (a
    # long career didn't end just because the class turned 10 years old), and a
    # "non-mature" class already has plenty of players who are done (a bust who never
    # stuck, or someone who retired early) whose totals are already final. Checked
    # against today's actual rosters rather than the class-level year cutoff used for
    # training-label safety (is_mature above, which stays a class-level concept).
    class_df["is_active_now"] = class_df["player"].apply(lambda p: is_same_player(p, current_player_index))

    # A rolling mean over THIS class's RETIRED/inactive outcomes only - mixing in
    # still-active players' understated totals would drag the trend down artificially.
    # Not another isotonic fit: a single class has exactly one player per pick number,
    # so there's nothing to average across at a given pick, and forcing monotonicity
    # here would hide the real (and often noisy) shape of one specific class. Computed
    # over the full class before pagination so a page boundary doesn't distort the
    # smoothing at its edges.
    class_df["actual_trend"] = (
        class_df["point_shares"]
        .where(~class_df["is_active_now"])
        .rolling(window=7, center=True, min_periods=1)
        .mean()
    )

    # Pace, not just the raw total: a non-mature player's career is still in progress,
    # so their cumulative Point Shares understates them relative to the curve (which is
    # calibrated on ~15-20 season careers). Point Shares per 82 games played puts a
    # 7-season career on the same footing as a full one for comparison purposes.
    class_df["pace"] = class_df["point_shares"] / class_df["games_played"].clip(lower=1) * 82

    # Projected career value: pace held constant over a REALISTIC assumed career
    # length for this pick slot (from games_played_curve, fit on mature careers at
    # the same slot) rather than the player's own incomplete games total. This is
    # what actually puts a still-active player on the same footing as the curve -
    # comparing his raw (unfinished) total to a full-career curve never can.
    benchmark_games = games_curve.predict(class_df["overall_pick"])
    class_df["projected_value"] = class_df["pace"] * benchmark_games / 82

    max_pick = int(class_df["overall_pick"].max())
    chunk_size = 20
    chunk_bounds = [(start, min(start + chunk_size - 1, max_pick)) for start in range(1, max_pick + 1, chunk_size)]
    chunk_labels = {bounds: f"Picks {bounds[0]}-{bounds[1]}" for bounds in chunk_bounds}
    chosen_bounds = st.selectbox("Pick range", chunk_bounds, format_func=lambda b: chunk_labels[b])
    lo, hi = chosen_bounds

    page_df = class_df[class_df["overall_pick"].between(lo, hi)]

    fig = go.Figure()
    pick_range = list(range(lo, hi + 1))
    curve_values = curve.predict(pick_range)
    fig.add_trace(go.Scatter(
        x=pick_range, y=curve_values, mode="lines", name="Expected value (pick curve)",
        line=dict(color="#e15759", width=3),
    ))
    retired_df = page_df[~page_df["is_active_now"]]
    active_df = page_df[page_df["is_active_now"]]

    fig.add_trace(go.Scatter(
        x=page_df["overall_pick"], y=page_df["actual_trend"], mode="lines",
        name="Retired/inactive players' actual trend",
        line=dict(color="#59a14f", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=retired_df["overall_pick"], y=retired_df["point_shares"], mode="markers", name="Actual outcome (final)",
        text=retired_df["player"],
        hovertemplate="%{text}<br>Pick %{x}<br>Career Point Shares: %{y:.1f}<extra></extra>",
        marker=dict(color="#4e79a7", size=9, line=dict(width=1, color="rgba(0,0,0,0.3)")),
    ))
    # Only the pace-projected points for still-active players - their raw "so far"
    # total is the understated number we're specifically telling the user not to
    # trust here, so plotting it too would just be clutter alongside the fair
    # comparison. The raw number is still one hover away rather than gone entirely.
    fig.add_trace(go.Scatter(
        x=active_df["overall_pick"], y=active_df["projected_value"], mode="markers",
        name="Still active - projected at current pace",
        text=active_df["player"], customdata=np.stack([active_df["point_shares"], active_df["pace"]], axis=-1),
        hovertemplate=(
            "%{text}<br>Pick %{x}<br>Projected career Point Shares: %{y:.1f}"
            "<br>Actual so far: %{customdata[0]:.1f} (pace: %{customdata[1]:.1f} PS/82GP)<extra></extra>"
        ),
        marker=dict(color="#b07aa1", size=11, symbol="diamond", line=dict(width=1, color="rgba(0,0,0,0.3)")),
    ))
    fig.update_layout(
        title=f"{year} draft class: actual outcome vs. expected pick value ({chunk_labels[chosen_bounds]})",
        xaxis_title="Overall pick", yaxis_title="Career Point Shares",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig, width='stretch')

    team_logos = load_team_logos()
    page_df = page_df.copy()
    page_df["team_logo"] = page_df["team"].map(team_logos)

    any_active_on_page = bool(page_df["is_active_now"].any())

    table_cols = ["team_logo", "overall_pick", "player", "position", "point_shares", "pace"]
    rename_map = {
        "team_logo": "Team", "overall_pick": "Pick", "player": "Player", "position": "Pos",
        "point_shares": "Career PS so far", "pace": "Pace (PS/82GP)",
    }
    if any_active_on_page:
        table_cols.append("projected_value")
        rename_map["projected_value"] = "Projected career PS"
    table_cols.append("is_active_now")
    rename_map["is_active_now"] = "Active now"

    table = page_df[table_cols].rename(columns=rename_map)
    round_cols = {"Pace (PS/82GP)": 1, **({"Projected career PS": 1} if any_active_on_page else {})}
    st.dataframe(
        table.round(round_cols), width='stretch', hide_index=True,
        column_config={"Team": st.column_config.ImageColumn("Team", width="small")},
    )
    if any_active_on_page:
        st.caption(
            "The purple diamonds (and 'Projected career PS' column) show what each still-active "
            "player's career total would be if their current pace continued for a realistic full "
            "career at their draft slot - the fair comparison against the red curve, regardless of "
            "which year they were drafted. See the expander below for why."
        )

    with st.expander("Why can a legendary player look like he's \"underperforming\" the pick curve?"):
        st.markdown(
            "The pick curve is trained on **mature** classes (2000-2012) - players with 15-20+ "
            "complete seasons, so it represents *full-career* totals. A player who's still active "
            "today has only played part of a career so far, so his raw total understates him against "
            "that curve - which is exactly why still-active players are shown as a projected pace "
            "point (purple diamond) here, not their raw total.\n\n"
            "**Connor McDavid** (2015, pick #1) is the clearest example: his 82.4 career Point Shares "
            "cover only 487 games (about 6 82-game seasons) - if plotted directly, that would land "
            "well below the curve's ~103 full-career expectation for pick #1.\n\n"
            "His **pace** is 82.4 / 487 games x 82 = **13.9 Point Shares per 82 games**. Pick #1 "
            "picks who complete a full career play about 887 games on average - so at his current "
            "pace over a realistic career length, McDavid projects to 13.9 x 887 / 82 = **~150 "
            "career Point Shares**, well *above* the curve's 103 expectation. That's the purple "
            "diamond: the same player, same pace, just given a fair career length to work with "
            "instead of his still-in-progress raw total."
        )

    with st.expander("Why did the fancier model lose to a straight-line baseline?"):
        st.markdown(
            "`train_draft_model.py` trained an XGBoost model (pick number + age + position) on "
            "2000-2009 draft classes and tested it on unseen 2010-2012 classes, against a simple "
            "log-linear baseline using pick number alone:\n\n"
            "| Model | MAE (career PS) | R² |\n"
            "|---|---|---|\n"
            "| Baseline (pick number only) | 9.68 | **0.209** |\n"
            "| XGBoost (pick + age + position) | **9.05** | 0.096 |\n\n"
            "The XGBoost model had a slightly lower average error, but explained *less* of the "
            "variance out-of-sample - it overfit patterns in the training draft classes that didn't "
            "generalize. SHAP confirms why: mean |SHAP value| for `overall_pick` (7.66) dwarfs "
            "`position_group` (1.23) and `age` (0.59). Pick number is doing almost all the work; "
            "the extra features mostly added noise on new draft classes."
        )


def _render_side(grade: TradeSideGrade, team_logos: dict[str, str]):
    with st.container(border=True):
        header_col, metric_col = st.columns([1, 3])
        logo_url = team_logos.get(grade.team)
        if logo_url:
            header_col.image(logo_url, width=56)
        metric_col.metric(grade.team, f"{grade.total_value:+.2f}")

        rows = [
            {"Photo": a.image_url, "Asset": a.label, "Value": a.value, "Confidence": a.confidence, "Source": a.source}
            for a in grade.assets
        ]
        df = pd.DataFrame(rows)
        st.dataframe(
            df, width='stretch', hide_index=True,
            column_config={"Photo": st.column_config.ImageColumn("Photo", width="small")},
        )
        if grade.unvalued_assets:
            st.warning(
                f"{len(grade.unvalued_assets)} asset(s) came back unvalued (confidence \"none\"): "
                + ", ".join(a.label for a in grade.unvalued_assets)
            )


def _render_grade_comparison(grades: dict[str, TradeSideGrade]):
    team_logos = load_team_logos()
    cols = st.columns(len(grades))
    for col, (_, grade) in zip(cols, grades.items()):
        with col:
            _render_side(grade, team_logos)

    fig = go.Figure(go.Bar(
        x=[g.team for g in grades.values()],
        y=[g.total_value for g in grades.values()],
        marker_color=["#4e79a7", "#f28e2b"][: len(grades)],
    ))
    fig.update_layout(
        title="Total value by side", yaxis_title="Total graded value", template="plotly_white", margin=dict(t=60)
    )
    st.plotly_chart(fig, width='stretch')


def _trade_label(trade_id: int, rows: pd.DataFrame) -> str:
    date = rows["date"].iloc[0]
    sides = rows.groupby("side")["acquiring_team"].first()
    teams = " <-> ".join(sides.sort_index().tolist())
    return f"{date}: {teams}"


def historical_trade_mode():
    trades = load_trades()
    grader = load_trade_grader()

    season = st.selectbox("Season", sorted(trades["season"].unique(), reverse=True))
    season_trades = trades[trades["season"] == season]

    trade_ids = sorted(season_trades["trade_id"].unique())
    labels = {tid: _trade_label(tid, season_trades[season_trades["trade_id"] == tid]) for tid in trade_ids}
    chosen_id = st.selectbox("Trade", trade_ids, format_func=lambda tid: labels[tid])

    trade_rows = season_trades[season_trades["trade_id"] == chosen_id]
    with st.expander("Raw trade rows"):
        st.dataframe(trade_rows[["side", "acquiring_team", "type", "subtype", "raw"]], width='stretch', hide_index=True)

    grades = grader.grade_historical_trade(trade_rows)
    _render_grade_comparison(grades)


def _pick_builder(side_key: str, trade_year: int) -> list[HypotheticalPick]:
    picks_key = f"{side_key}_picks"
    st.session_state.setdefault(picks_key, [])

    with st.expander("Add a draft pick"):
        year_col, round_col, cond_col = st.columns(3)
        year = year_col.number_input(
            "Year", min_value=trade_year, max_value=trade_year + 7,
            value=trade_year + 1, key=f"{side_key}_pick_year",
        )
        pick_round = round_col.selectbox("Round", list(range(1, 8)), key=f"{side_key}_pick_round")
        conditional = cond_col.checkbox("Conditional", key=f"{side_key}_pick_conditional")
        if st.button("Add pick", key=f"{side_key}_add_pick"):
            st.session_state[picks_key].append(HypotheticalPick(year=int(year), round=pick_round, conditional=conditional))
            st.rerun()

    for i, pick in enumerate(st.session_state[picks_key]):
        label = f"{pick.year} round {pick.round} pick" + (" (conditional)" if pick.conditional else "")
        text_col, remove_col = st.columns([4, 1])
        text_col.write(f"- {label}")
        if remove_col.button("Remove", key=f"{side_key}_remove_pick_{i}"):
            st.session_state[picks_key].pop(i)
            st.rerun()

    return st.session_state[picks_key]


def hypothetical_trade_mode():
    grader = load_trade_grader()

    with st.spinner("Loading current rosters..."):
        player_names = load_current_player_names()

    trade_year = st.number_input(
        "Trade year", min_value=date.today().year, max_value=date.today().year + 3,
        value=date.today().year, key="hypothetical_trade_year",
        help="Only affects pick discounting (further-out picks are worth less). Player values always "
        "use each player's current real stats - there's no way to fetch a player's stats for a season "
        "that hasn't happened yet.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Side A**")
        side_a_players = st.multiselect("Players", player_names, key="side_a_players")
        side_a_picks = _pick_builder("side_a", trade_year)
    with col_b:
        st.markdown("**Side B**")
        side_b_players = st.multiselect("Players", player_names, key="side_b_players")
        side_b_picks = _pick_builder("side_b", trade_year)

    side_a = [*side_a_players, *side_a_picks]
    side_b = [*side_b_players, *side_b_picks]

    if st.button("Grade trade", disabled=not (side_a and side_b)):
        with st.spinner("Fetching live stats and grading..."):
            grades = grader.grade_hypothetical_trade(side_a, side_b, trade_year=trade_year)
        _render_grade_comparison(grades)
        st.caption(
            "Every player name above is a real current roster player, but the live-stats lookup uses "
            "a separate NHL API that occasionally formats a name differently (suffixes, accents) - on "
            "the rare mismatch, that player falls back to career draft Point Shares instead of live "
            "stats, and shows up as unvalued only if that also fails. Picks are discounted for how "
            "far out they are from the trade year above, and for being conditional."
        )


def trade_grader_tab():
    mode = st.radio("Mode", ["Browse a historical trade", "Build a hypothetical trade"], horizontal=True)
    st.divider()
    if mode == "Browse a historical trade":
        historical_trade_mode()
    else:
        hypothetical_trade_mode()


def main():
    st.title("NHL Draft Value & Trade Grading")
    tab1, tab2 = st.tabs(["Draft Explorer", "Trade Grader"])
    with tab1:
        draft_explorer_tab()
    with tab2:
        trade_grader_tab()


if __name__ == "__main__":
    main()
