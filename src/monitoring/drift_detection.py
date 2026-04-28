import pandas as pd
import numpy as np
from pathlib import Path


TRAIN_PATH = "data/features/train_features.csv"
TEST_PATH = "data/features/test_features.csv"
REPORT_PATH = "reports/drift_report.txt"


# ----------------------------
# Load Data
# ----------------------------
def load_data(path):
    return pd.read_csv(path)


# ----------------------------
# Drift Metric (Mean + Std)
# ----------------------------
def calculate_drift(train_col, test_col):
    train_mean = train_col.mean()
    test_mean = test_col.mean()

    train_std = train_col.std()
    test_std = test_col.std()

    mean_shift = abs(train_mean - test_mean)
    std_shift = abs(train_std - test_std)

    return mean_shift, std_shift


# ----------------------------
# Detect Drift
# ----------------------------
def detect_drift(train_df, test_df, threshold=0.2):
    drift_results = []

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == "pm2.5":
            continue

        mean_shift, std_shift = calculate_drift(
            train_df[col],
            test_df[col]
        )

        drift_flag = (mean_shift > threshold) or (std_shift > threshold)

        drift_results.append({
            "feature": col,
            "mean_shift": round(mean_shift, 4),
            "std_shift": round(std_shift, 4),
            "drift_detected": drift_flag
        })

    return pd.DataFrame(drift_results)


# ----------------------------
# Save Report
# ----------------------------
def save_report(df):
    Path("reports").mkdir(exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write("===== DATA DRIFT REPORT =====\n\n")

        drifted = df[df["drift_detected"] == True]

        f.write(f"Total features checked: {len(df)}\n")
        f.write(f"Features with drift: {len(drifted)}\n\n")

        f.write("=== DRIFTED FEATURES ===\n")
        for _, row in drifted.iterrows():
            f.write(
                f"{row['feature']} | mean_shift={row['mean_shift']} | std_shift={row['std_shift']}\n"
            )

    print(f"Drift report saved to: {REPORT_PATH}")


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading datasets...")
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    print("Detecting drift...")
    drift_df = detect_drift(train_df, test_df)

    print("\nSample drift results:")
    print(drift_df.head())

    save_report(drift_df)


if __name__ == "__main__":
    main()
