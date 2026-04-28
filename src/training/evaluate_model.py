import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "models/xgboost_model.pkl"
TEST_PATH = "data/features/test_features.csv"
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(path: str):
    return joblib.load(path)


def split_xy(df: pd.DataFrame, target_col: str = "pm2.5"):
    X = df.drop(columns=[target_col, "datetime"])
    y = df[target_col]
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ----------------------------
# Plot 1: Actual vs Predicted
# ----------------------------
def plot_actual_vs_predicted(y_true, y_pred, output_path: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values[:300], label="Actual")
    plt.plot(y_pred[:300], label="Predicted")
    plt.title("Actual vs Predicted PM2.5 (First 300 Test Points)")
    plt.xlabel("Time Index")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


# ----------------------------
# Plot 2: Residual Histogram
# ----------------------------
def plot_residuals(y_true, y_pred, output_path: Path):
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


# ----------------------------
# Plot 3: Feature Importance
# ----------------------------
def plot_feature_importance(model, feature_names, output_path: Path, top_n: int = 15):
    importances = model.feature_importances_

    feat_imp = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    return feat_imp


# ----------------------------
# Worst Predictions Table
# ----------------------------
def save_worst_predictions(df: pd.DataFrame, y_true, y_pred, output_path: Path, top_n: int = 25):
    result_df = df.copy()
    result_df["actual_pm25"] = y_true.values
    result_df["predicted_pm25"] = y_pred
    result_df["absolute_error"] = np.abs(result_df["actual_pm25"] - result_df["predicted_pm25"])

    worst_df = result_df.sort_values("absolute_error", ascending=False).head(top_n)
    worst_df.to_csv(output_path, index=False)
    print(f"Saved worst predictions: {output_path}")

    return worst_df


# ----------------------------
# Save text report
# ----------------------------
def save_evaluation_summary(metrics: dict, feat_imp: pd.DataFrame, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== MODEL EVALUATION SUMMARY =====\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE:  {metrics['mae']:.4f}\n")
        f.write(f"R2:   {metrics['r2']:.4f}\n\n")

        f.write("===== TOP FEATURES =====\n")
        for _, row in feat_imp.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")

    print(f"Saved evaluation summary: {output_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()

    print("Loading test dataset...")
    test_df = load_test_data(TEST_PATH)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Preparing features...")
    X_test, y_test = split_xy(test_df)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    print("Computing evaluation metrics...")
    metrics = compute_metrics(y_test, y_pred)
    print("\n===== XGBoost Evaluation =====")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"R2:   {metrics['r2']:.4f}")

    print("\nCreating evaluation plots...")
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        FIGURES_DIR / "actual_vs_predicted.png"
    )

    plot_residuals(
        y_test,
        y_pred,
        FIGURES_DIR / "residual_distribution.png"
    )

    feat_imp = plot_feature_importance(
        model,
        X_test.columns,
        FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    save_worst_predictions(
        test_df,
        y_test,
        y_pred,
        REPORTS_DIR / "worst_predictions.csv",
        top_n=25
    )

    save_evaluation_summary(
        metrics,
        feat_imp,
        REPORTS_DIR / "evaluation_summary.txt"
    )

    print("\nStep 5 completed successfully.")


if __name__ == "__main__":
    main()
