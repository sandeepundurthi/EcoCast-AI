import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib


# ----------------------------
# Load Data
# ----------------------------
def load_features(path: str):
    return pd.read_csv(path)


# ----------------------------
# Split X and y
# ----------------------------
def split_xy(df, target_col="pm2.5"):
    X = df.drop(columns=[target_col, "datetime"])
    y = df[target_col]
    return X, y


# ----------------------------
# Baseline Model
# ----------------------------
def baseline_model(df):
    # naive: predict last value (lag_1)
    y_true = df["pm2.5"]
    y_pred = df["pm2.5_lag_1"]

    return y_true, y_pred


# ----------------------------
# Train XGBoost
# ----------------------------
def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


# ----------------------------
# Evaluate Model
# ----------------------------
def evaluate(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {name} Performance =====")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ----------------------------
# Save Model
# ----------------------------
def save_model(model, path="models/xgboost_model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


# ----------------------------
# Main Training Pipeline
# ----------------------------
def main():
    train_path = "data/features/train_features.csv"
    test_path = "data/features/test_features.csv"

    print("Loading feature datasets...")
    train_df = load_features(train_path)
    test_df = load_features(test_path)

    # --------------------
    # Baseline
    # --------------------
    print("\nRunning baseline model...")
    y_true_base, y_pred_base = baseline_model(test_df)
    evaluate(y_true_base, y_pred_base, name="Baseline (Lag-1)")

    # --------------------
    # ML Model
    # --------------------
    print("\nTraining XGBoost model...")

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    model = train_xgboost(X_train, y_train)

    print("\nEvaluating XGBoost model...")
    y_pred = model.predict(X_test)

    evaluate(y_test, y_pred, name="XGBoost")

    # --------------------
    # Save Model
    # --------------------
    save_model(model)


if __name__ == "__main__":
    main()
