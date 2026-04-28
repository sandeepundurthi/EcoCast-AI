import pandas as pd
import numpy as np
from pathlib import Path


def summarize_dataset(df: pd.DataFrame) -> dict:
    summary = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary


def target_summary(df: pd.DataFrame, target_col: str = "pm2.5") -> dict:
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe")

    target = df[target_col]
    return {
        "count": int(target.count()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "q1": float(target.quantile(0.25)),
        "q3": float(target.quantile(0.75)),
    }


def check_datetime_continuity(df: pd.DataFrame, datetime_col: str = "datetime") -> dict:
    if datetime_col not in df.columns:
        raise ValueError(f"{datetime_col} not found in dataframe")

    dt = pd.to_datetime(df[datetime_col])
    diffs = dt.sort_values().diff().dropna()

    hourly_gaps = (diffs != pd.Timedelta(hours=1)).sum()

    return {
        "start": str(dt.min()),
        "end": str(dt.max()),
        "non_hourly_gaps": int(hourly_gaps),
    }


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ValueError(f"{column} not found in dataframe")

    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((series < lower) | (series > upper)).sum()

    return {
        "column": column,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "num_outliers": int(outliers),
    }


def fill_feature_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"pm2.5", "year", "month", "day", "hour", "no"}
    fill_cols = [col for col in numeric_cols if col not in excluded]

    for col in fill_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    if "cbwd" in df.columns and df["cbwd"].isnull().sum() > 0:
        df["cbwd"] = df["cbwd"].fillna(df["cbwd"].mode()[0])

    return df


def time_based_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("datetime").reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def save_text_report(report: dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for section, content in report.items():
            f.write(f"===== {section.upper()} =====\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{content}\n")
            f.write("\n")

    print(f"Validation report saved to: {output_file}")
