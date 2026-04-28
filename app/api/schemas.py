from pydantic import BaseModel


class PredictionRequest(BaseModel):
    no: int
    year: int
    month: int
    day: int
    hour: int
    dewp: float
    temp: float
    pres: float
    iws: float
    is_: int
    ir: int
    day_of_week: int
    is_weekend: int
    pm2_5_lag_1: float
    pm2_5_lag_3: float
    pm2_5_lag_6: float
    pm2_5_lag_12: float
    pm2_5_lag_24: float
    pm2_5_roll_mean_3: float
    pm2_5_roll_std_3: float
    pm2_5_roll_mean_6: float
    pm2_5_roll_std_6: float
    pm2_5_roll_mean_12: float
    pm2_5_roll_std_12: float
    pm2_5_roll_mean_24: float
    pm2_5_roll_std_24: float
    temp_dewp_diff: float
    wind_strength: float
    pressure_temp: float
    cbwd_NW: int = 0
    cbwd_SE: int = 0
    cbwd_cv: int = 0


class PredictionResponse(BaseModel):
    predicted_pm25: float


class HealthRiskResponse(BaseModel):
    predicted_pm25: float
    risk_category: str
    health_message: str
