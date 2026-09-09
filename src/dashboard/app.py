"""Streamlit dashboard: draft-value exploration and trade grading.

Two tabs. Draft Explorer plots the fitted pick-value curve against a chosen
draft class and surfaces the negative modeling result from train_draft_model.py
(pick number alone beat the richer XGBoost model on held-out draft classes).
Trade Grader wraps TradeGrader to grade a historical trade from trades_clean.csv
or a hypothetical trade built from live player names.
"""

import sys
from pathlib import Path

# `streamlit run src/dashboard/app.py` puts this file's own directory on
# sys.path, not the repo root - so `import src....` fails unless the root is
# added explicitly first (bites locally too, not just on Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.draft_features import clean_draft_data
from src.trade.trade_grader import TradeGrader, TradeSideGrade

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

st.set_page_config(page_title="NHL Draft Value & Trade Grader", layout="wide")


@st.cache_data
def load_draft_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / "draft_history_raw.csv")
    raw = raw[(raw["year"] >= 2000) & (raw["year"] <= 2020)]
    return clean_draft_data(raw)


@st.cache_resource
def load_pick_curve():
    return joblib.load(PROCESSED_DIR / "pick_value_curve.joblib")


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


def draft_explorer_tab():
    draft = load_draft_data()
    curve = load_pick_curve()

    year = st.selectbox("Draft year", sorted(draft["year"].unique(), reverse=True))
    class_df = draft[draft["year"] == year].sort_values("overall_pick")

    is_mature = bool(class_df["is_mature"].iloc[0]) if not class_df.empty else False
    if is_mature:
        st.caption(f"{year} class is **mature** (drafted 2012 or earlier) - graded on full career Point Shares.")
    else:
        st.caption(
            f"{year} class is **too early to grade** (drafted after 2012) - careers are still in progress, "
            "so low Point Shares here may just mean \"too soon to tell,\" not \"bad pick.\" Excluded from model training."
        )

    fig = go.Figure()
    pick_range = list(range(1, int(class_df["overall_pick"].max()) + 1))
    curve_values = curve.predict(pick_range)
    fig.add_trace(go.Scatter(
        x=pick_range, y=curve_values, mode="lines", name="Expected value (pick curve)",
        line=dict(color="#e15759", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=class_df["overall_pick"], y=class_df["point_shares"], mode="markers", name="Actual outcome",
        text=class_df["player"], hovertemplate="%{text}<br>Pick %{x}<br>Point Shares %{y:.1f}<extra></extra>",
        marker=dict(
            color="#4e79a7" if is_mature else "#bab0ac",
            size=9,
            line=dict(width=1, color="rgba(0,0,0,0.3)"),
        ),
    ))
    fig.update_layout(
        title=f"{year} draft class: actual outcome vs. expected pick value",
        xaxis_title="Overall pick", yaxis_title="Career Point Shares",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    table = class_df[["overall_pick", "player", "position", "point_shares", "is_mature"]].rename(
        columns={"overall_pick": "Pick", "player": "Player", "position": "Pos", "point_shares": "Career PS", "is_mature": "Mature"}
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

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


def _render_side(grade: TradeSideGrade):
    st.markdown(f"**{grade.team}** - total value: `{grade.total_value:+.2f}`")
    rows = [
        {"Asset": a.label, "Value": a.value, "Confidence": a.confidence, "Source": a.source}
        for a in grade.assets
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if grade.unvalued_assets:
        st.warning(
            f"{len(grade.unvalued_assets)} asset(s) came back unvalued (confidence \"none\"): "
            + ", ".join(a.label for a in grade.unvalued_assets)
        )


def _render_grade_comparison(grades: dict[str, TradeSideGrade]):
    cols = st.columns(len(grades))
    for col, (_, grade) in zip(cols, grades.items()):
        with col:
            _render_side(grade)

    fig = go.Figure(go.Bar(
        x=[g.team for g in grades.values()],
        y=[g.total_value for g in grades.values()],
        marker_color=["#4e79a7", "#f28e2b"][: len(grades)],
    ))
    fig.update_layout(
        title="Total value by side", yaxis_title="Total graded value", template="plotly_white", margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True)


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
        st.dataframe(trade_rows[["side", "acquiring_team", "type", "subtype", "raw"]], use_container_width=True, hide_index=True)

    grades = grader.grade_historical_trade(trade_rows)
    _render_grade_comparison(grades)


def hypothetical_trade_mode():
    grader = load_trade_grader()

    col_a, col_b = st.columns(2)
    with col_a:
        side_a_text = st.text_area("Side A players (one per line)", placeholder="Connor McDavid\nEvan Bouchard")
    with col_b:
        side_b_text = st.text_area("Side B players (one per line)", placeholder="Auston Matthews")

    side_a = [n.strip() for n in side_a_text.splitlines() if n.strip()]
    side_b = [n.strip() for n in side_b_text.splitlines() if n.strip()]

    if st.button("Grade trade", disabled=not (side_a and side_b)):
        with st.spinner("Fetching live stats and grading..."):
            grades = grader.grade_hypothetical_trade(side_a, side_b)
        _render_grade_comparison(grades)
        st.caption(
            "Names that don't resolve to an exact live NHL player fall back to career draft Point "
            "Shares, and if that also fails they show up as unvalued rather than silently dropped."
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
