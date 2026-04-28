import pandas as pd
import numpy as np
from pathlib import Path


# ----------------------------
# 1. Time Features
# ----------------------------
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# ----------------------------
# 2. Lag Features
# ----------------------------
def create_lag_features(df: pd.DataFrame, target_col="pm2.5") -> pd.DataFrame:
    df = df.copy()

    lags = [1, 3, 6, 12, 24]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


# ----------------------------
# 3. Rolling Features
# ----------------------------
def create_rolling_features(df: pd.DataFrame, target_col="pm2.5") -> pd.DataFrame:
    df = df.copy()

    windows = [3, 6, 12, 24]

    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(window=w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(window=w).std()

    return df


# ----------------------------
# 4. Weather Interaction Features
# ----------------------------
def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_dewp_diff"] = df["temp"] - df["dewp"]
    df["wind_strength"] = df["iws"]
    df["pressure_temp"] = df["pres"] * df["temp"]

    return df


# ----------------------------
# 5. Encode Categorical
# ----------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "cbwd" in df.columns:
        df = pd.get_dummies(df, columns=["cbwd"], drop_first=True)

    return df


# ----------------------------
# 6. Handle Outliers (Clipping)
# ----------------------------
def clip_target(df: pd.DataFrame, target_col="pm2.5") -> pd.DataFrame:
    df = df.copy()

    upper = df[target_col].quantile(0.99)
    df[target_col] = np.clip(df[target_col], 0, upper)

    return df


# ----------------------------
# 7. Drop NaNs from Lagging
# ----------------------------
def drop_na_after_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)


# ----------------------------
# 8. Full Feature Pipeline
# ----------------------------
def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = clip_target(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_weather_features(df)
    df = encode_categorical(df)

    df = drop_na_after_features(df)

    return df


# ----------------------------
# 9. Save Features
# ----------------------------
def save_features(df: pd.DataFrame, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved features to: {output}")


# ----------------------------
# 10. Train/Test Feature Split
# ----------------------------
def split_features(df: pd.DataFrame, test_size=0.2):
    df = df.sort_values("datetime").reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test


if __name__ == "__main__":
    input_path = "data/processed/air_quality_clean.csv"
    df = pd.read_csv(input_path)

    features_df = build_feature_pipeline(df)

    train_df, test_df = split_features(features_df)

    save_features(train_df, "data/features/train_features.csv")
    save_features(test_df, "data/features/test_features.csv")

    print("Feature engineering complete.")
    print(f"Final shape: {features_df.shape}")
