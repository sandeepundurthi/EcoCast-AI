from src.ingestion.load_data import load_raw_air_quality_data, inspect_dataframe
from src.preprocessing.clean_data import basic_clean_pipeline, save_processed_data
from src.preprocessing.validate_data import (
    summarize_dataset,
    target_summary,
    check_datetime_continuity,
    detect_outliers_iqr,
    fill_feature_missing_values,
    time_based_train_test_split,
    save_text_report,
)

from src.features.build_features import (
    build_feature_pipeline,
    split_features,
    save_features,
)


def main():
    raw_path = "data/raw/air_quality.csv"
    processed_path = "data/processed/air_quality_clean.csv"
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    report_path = "reports/data_validation_report.txt"

    print("Loading raw dataset...")
    df = load_raw_air_quality_data(raw_path)

    print("Inspecting raw dataset...")
    inspect_dataframe(df)

    print("\nRunning cleaning pipeline...")
    clean_df = basic_clean_pipeline(df, target_col="pm2.5")

    print("\nFilling missing feature values if any...")
    clean_df = fill_feature_missing_values(clean_df)

    print("\nSaving cleaned dataset...")
    save_processed_data(clean_df, processed_path)

    print("\nGenerating validation summaries...")
    dataset_info = summarize_dataset(clean_df)
    target_info = target_summary(clean_df, target_col="pm2.5")
    datetime_info = check_datetime_continuity(clean_df, datetime_col="datetime")
    outlier_info = detect_outliers_iqr(clean_df, column="pm2.5")

    report = {
        "dataset_summary": dataset_info,
        "target_summary": target_info,
        "datetime_continuity": datetime_info,
        "pm25_outlier_check": outlier_info,
    }

    save_text_report(report, report_path)

    print("\nCreating time-based train/test split...")
    train_df, test_df = time_based_train_test_split(clean_df, test_size=0.2)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nBuilding feature dataset...")
    features_df = build_feature_pipeline(clean_df)

    train_feat, test_feat = split_features(features_df)

    save_features(train_feat, "data/features/train_features.csv")
    save_features(test_feat, "data/features/test_features.csv")

    print("\nPipeline completed successfully.")
    print(f"Final cleaned shape: {clean_df.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Feature dataset shape: {features_df.shape}")


if __name__ == "__main__":
    main()
