import pandas as pd
from pathlib import Path


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine year/month/day/hour into a single datetime column if present.
    """
    required_cols = ["year", "month", "day", "hour"]

    if all(col in df.columns for col in required_cols):
        df["datetime"] = pd.to_datetime(df[required_cols])
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all column names to lowercase.
    """
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()


def sort_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort data by datetime if present.
    """
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


def handle_missing_target(df: pd.DataFrame, target_col: str = "pm2.5") -> pd.DataFrame:
    """
    Remove rows where the target is missing.
    """
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    return df


def basic_clean_pipeline(df: pd.DataFrame, target_col: str = "pm2.5") -> pd.DataFrame:
    """
    Run the basic cleaning steps.
    """
    df = standardize_column_names(df)
    df = create_datetime_column(df)
    df = remove_duplicates(df)
    df = sort_by_datetime(df)
    df = handle_missing_target(df, target_col=target_col)
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataframe to CSV.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved processed data to: {output_file}")


if __name__ == "__main__":
    input_path = "data/raw/air_quality.csv"
    output_path = "data/processed/air_quality_clean.csv"

    df = pd.read_csv(input_path)
    clean_df = basic_clean_pipeline(df, target_col="pm2.5")
    save_processed_data(clean_df, output_path)

    print("\n===== CLEANED DATA SHAPE =====")
    print(clean_df.shape)

    print("\n===== CLEANED DATA HEAD =====")
    print(clean_df.head())
