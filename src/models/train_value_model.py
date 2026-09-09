"""Train the composite on-ice player value model.

Target: Point Shares (PS), an existing performance-value stat, for the 2016-17
season. Salary is deliberately NOT a feature here - it's compared against the
model's output afterward (see src/models/surplus_value.py) rather than blended
into what counts as "valuable," so the model measures performance, not cost.
"""

from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.value_features import FEATURE_COLUMNS, build_feature_matrix, clean_value_training_data

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "value_model.joblib"
PLAYER_VALUES_PATH = PROCESSED_DIR / "player_values_2016_17.csv"


def load_training_data() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download("camnugent/predict-nhl-player-salaries"))
    return pd.read_csv(dataset_path / "train.csv", encoding="latin1")


def train_and_evaluate():
    raw = load_training_data()
    clean = clean_value_training_data(raw)
    X = build_feature_matrix(clean)
    # Predict the RATE (Point Shares per game played), not the season total.
    # Our features are also rates (per-GP), and linear models can only combine
    # features additively - they can't represent total = rate * GP on their own.
    # Matching target and feature scale sidesteps that; we multiply back by GP
    # afterward to evaluate (and later apply the model) in total-value terms.
    y_rate = clean["PS"] / clean["GP"].clip(lower=1)
    gp = clean["GP"]

    X_train, X_test, y_train, y_test, gp_train, gp_test = train_test_split(
        X, y_rate, gp, test_size=0.2, random_state=42
    )

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    # RidgeCV standardizes implicitly via the pipeline below (Ridge's penalty
    # treats every coefficient the same, so features must be on the same scale
    # first - otherwise a feature like CF (0-2300) would be penalized far more
    # harshly than age (19-45) just because of its units, not its importance).
    alphas = np.logspace(-2, 3, 50)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5))
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    print("=== Holdout performance, in total-Point-Shares terms (rate * GP) ===")
    y_test_total = y_test * gp_test
    for name, pred in [("Plain linear regression", baseline_pred), ("Ridge (regularized)", ridge_pred)]:
        pred_total = pred * gp_test
        mae = mean_absolute_error(y_test_total, pred_total)
        r2 = r2_score(y_test_total, pred_total)
        print(f"{name:28s}  MAE={mae:.2f} Point Shares   R^2={r2:.3f}")

    chosen_alpha = ridge.named_steps["ridgecv"].alpha_
    print(f"\nRidgeCV picked alpha={chosen_alpha:.3g} via 5-fold cross-validation")

    coefs = pd.Series(ridge.named_steps["ridgecv"].coef_, index=FEATURE_COLUMNS).sort_values()
    print("\n=== Standardized coefficients (comparable across features) ===")
    print(coefs.to_string())

    # Holdout evaluation is done - refit on every row for the deployed model.
    # There's no more need to keep data back once we're not measuring
    # generalization anymore, and more training data only helps the fit.
    final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5))
    final_model.fit(X, y_rate)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    predicted_rate = final_model.predict(X)
    values = clean[["First Name", "Last Name", "Position", "GP", "Salary", "PS"]].copy()
    values["position_group"] = clean["position_group"]
    values["predicted_value_total"] = predicted_rate * gp
    values.to_csv(PLAYER_VALUES_PATH, index=False)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved per-player predicted values to {PLAYER_VALUES_PATH}")

    return final_model, values


if __name__ == "__main__":
    train_and_evaluate()
