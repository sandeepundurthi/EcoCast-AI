import pandas as pd
from pathlib import Path


def load_raw_air_quality_data(file_path: str) -> pd.DataFrame:
    """
    Load raw air quality dataset from CSV.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(path)
    return df


def inspect_dataframe(df: pd.DataFrame) -> None:
    """
    Print a quick inspection summary of the dataset.
    """
    print("\n===== DATAFRAME SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== HEAD =====")
    print(df.head())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)


if __name__ == "__main__":
    file_path = "data/raw/air_quality.csv"
    df = load_raw_air_quality_data(file_path)
    inspect_dataframe(df)
