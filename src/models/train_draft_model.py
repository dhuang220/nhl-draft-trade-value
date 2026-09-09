"""Train the draft-prospect value model.

Time-based split, not random: in real use we always predict an upcoming draft
class using only past classes. A random split would let the model train on
2011 picks and get evaluated on 2005 picks - it would look good but wouldn't
measure what we actually care about (can this model generalize to a draft
class it has never seen), and could leak era-specific effects (e.g. rule
changes, expansion) across the split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.features.draft_features import FEATURE_COLUMNS, build_feature_matrix, clean_draft_data

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "draft_history_raw.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "draft_model.joblib"

TRAIN_YEARS = (2000, 2009)
TEST_YEARS = (2010, 2012)


def train_and_evaluate():
    raw = pd.read_csv(RAW_PATH)
    raw = raw[(raw["year"] >= 2000) & (raw["year"] <= 2020)]
    clean = clean_draft_data(raw)
    mature = clean[clean["is_mature"]]

    train = mature[mature["year"].between(*TRAIN_YEARS)]
    test = mature[mature["year"].between(*TEST_YEARS)]

    X_train, y_train = build_feature_matrix(train), train["point_shares"]
    X_test, y_test = build_feature_matrix(test), test["point_shares"]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # in case a position is absent in test years

    baseline = LinearRegression()
    baseline.fit(np.log1p(train[["overall_pick"]]), y_train)
    baseline_pred = baseline.predict(np.log1p(test[["overall_pick"]]))

    gbm = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)

    print(f"Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} drafts ({len(train)} picks)")
    print(f"Test:  {TEST_YEARS[0]}-{TEST_YEARS[1]} drafts ({len(test)} picks, unseen draft classes)\n")
    for name, pred in [("Baseline (pick number only)", baseline_pred), ("XGBoost (pick + age + position)", gbm_pred)]:
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"{name:35s}  MAE={mae:.2f} career Point Shares   R^2={r2:.3f}")

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
    print("\n=== Mean |SHAP value| (average impact on predicted career value) ===")
    print(mean_abs_shap.to_string())

    # Refit on ALL mature data (not just the training window) for the deployed model.
    X_all, y_all = build_feature_matrix(mature), mature["point_shares"]
    final_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    final_model.fit(X_all, y_all)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": list(X_all.columns)}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return final_model, mature


if __name__ == "__main__":
    train_and_evaluate()
