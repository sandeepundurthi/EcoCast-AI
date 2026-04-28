import pandas as pd


MODEL_FEATURES = [
    "no",
    "year",
    "month",
    "day",
    "hour",
    "dewp",
    "temp",
    "pres",
    "iws",
    "is",
    "ir",
    "day_of_week",
    "is_weekend",
    "pm2.5_lag_1",
    "pm2.5_lag_3",
    "pm2.5_lag_6",
    "pm2.5_lag_12",
    "pm2.5_lag_24",
    "pm2.5_roll_mean_3",
    "pm2.5_roll_std_3",
    "pm2.5_roll_mean_6",
    "pm2.5_roll_std_6",
    "pm2.5_roll_mean_12",
    "pm2.5_roll_std_12",
    "pm2.5_roll_mean_24",
    "pm2.5_roll_std_24",
    "temp_dewp_diff",
    "wind_strength",
    "pressure_temp",
    "cbwd_NW",
    "cbwd_SE",
    "cbwd_cv",
]


def normalize_input_dict(data: dict) -> dict:
    """
    Convert API-safe field names into model feature names.
    """
    renamed = data.copy()

    if "is_" in renamed:
        renamed["is"] = renamed.pop("is_")

    dot_name_map = {
        "pm2_5_lag_1": "pm2.5_lag_1",
        "pm2_5_lag_3": "pm2.5_lag_3",
        "pm2_5_lag_6": "pm2.5_lag_6",
        "pm2_5_lag_12": "pm2.5_lag_12",
        "pm2_5_lag_24": "pm2.5_lag_24",
        "pm2_5_roll_mean_3": "pm2.5_roll_mean_3",
        "pm2_5_roll_std_3": "pm2.5_roll_std_3",
        "pm2_5_roll_mean_6": "pm2.5_roll_mean_6",
        "pm2_5_roll_std_6": "pm2.5_roll_std_6",
        "pm2_5_roll_mean_12": "pm2.5_roll_mean_12",
        "pm2_5_roll_std_12": "pm2.5_roll_std_12",
        "pm2_5_roll_mean_24": "pm2.5_roll_mean_24",
        "pm2_5_roll_std_24": "pm2.5_roll_std_24",
    }

    for api_name, model_name in dot_name_map.items():
        if api_name in renamed:
            renamed[model_name] = renamed.pop(api_name)

    return renamed


def prepare_features(data: dict) -> pd.DataFrame:
    """
    Convert incoming request JSON into a DataFrame with exact model column order.
    """
    normalized = normalize_input_dict(data)

    row = {}
    for col in MODEL_FEATURES:
        row[col] = normalized.get(col, 0)

    return pd.DataFrame([row])


def get_health_risk(pm25: float) -> tuple[str, str]:
    """
    Simple PM2.5-based health risk mapping.
    """
    if pm25 <= 12:
        return "Good", "Air quality is satisfactory with little or no health risk."
    if pm25 <= 35.4:
        return "Moderate", "Air quality is acceptable, but sensitive individuals should take care."
    if pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "Sensitive groups may experience health effects."
    if pm25 <= 150.4:
        return "Unhealthy", "Everyone may begin to experience health effects."
    if pm25 <= 250.4:
        return "Very Unhealthy", "Health alert: the risk of health effects increases for everyone."
    return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."
