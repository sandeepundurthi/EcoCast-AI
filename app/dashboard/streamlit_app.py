import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def build_payload(
    no,
    year,
    month,
    day,
    hour,
    dewp,
    temp,
    pres,
    iws,
    is_,
    ir,
    cbwd,
    pm2_5_lag_1,
    pm2_5_lag_3,
    pm2_5_lag_6,
    pm2_5_lag_12,
    pm2_5_lag_24,
    pm2_5_roll_mean_3,
    pm2_5_roll_std_3,
    pm2_5_roll_mean_6,
    pm2_5_roll_std_6,
    pm2_5_roll_mean_12,
    pm2_5_roll_std_12,
    pm2_5_roll_mean_24,
    pm2_5_roll_std_24,
):
    day_of_week_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    weekday_name = st.session_state.get("weekday_name", "Monday")
    day_of_week = day_of_week_map[weekday_name]
    is_weekend = 1 if day_of_week in [5, 6] else 0

    cbwd_nw = 1 if cbwd == "NW" else 0
    cbwd_se = 1 if cbwd == "SE" else 0
    cbwd_cv = 1 if cbwd == "cv" else 0

    payload = {
        "no": no,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "dewp": dewp,
        "temp": temp,
        "pres": pres,
        "iws": iws,
        "is_": is_,
        "ir": ir,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "pm2_5_lag_1": pm2_5_lag_1,
        "pm2_5_lag_3": pm2_5_lag_3,
        "pm2_5_lag_6": pm2_5_lag_6,
        "pm2_5_lag_12": pm2_5_lag_12,
        "pm2_5_lag_24": pm2_5_lag_24,
        "pm2_5_roll_mean_3": pm2_5_roll_mean_3,
        "pm2_5_roll_std_3": pm2_5_roll_std_3,
        "pm2_5_roll_mean_6": pm2_5_roll_mean_6,
        "pm2_5_roll_std_6": pm2_5_roll_std_6,
        "pm2_5_roll_mean_12": pm2_5_roll_mean_12,
        "pm2_5_roll_std_12": pm2_5_roll_std_12,
        "pm2_5_roll_mean_24": pm2_5_roll_mean_24,
        "pm2_5_roll_std_24": pm2_5_roll_std_24,
        "temp_dewp_diff": temp - dewp,
        "wind_strength": iws,
        "pressure_temp": pres * temp,
        "cbwd_NW": cbwd_nw,
        "cbwd_SE": cbwd_se,
        "cbwd_cv": cbwd_cv,
    }

    return payload


def main():
    st.set_page_config(
        page_title="EcoCast AI Dashboard",
        page_icon="🌍",
        layout="wide"
    )

    st.title("🌍 EcoCast AI")
    st.subheader("Air Quality Forecasting and Health Risk Dashboard")

    st.markdown(
        """
        This dashboard sends environmental feature inputs to the FastAPI backend
        and returns:
        - predicted PM2.5
        - health risk category
        - health advisory message
        """
    )

    with st.sidebar:
        st.header("System")
        st.write(f"API Base URL: `{API_BASE_URL}`")

        if st.button("Check API Health"):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=10)
                if response.status_code == 200:
                    st.success("API is healthy")
                    st.json(response.json())
                else:
                    st.error("API health check failed")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")

    st.markdown("---")
    st.header("Input Environmental Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        no = st.number_input("Record ID (no)", min_value=1, value=40000)
        year = st.number_input("Year", min_value=2010, max_value=2035, value=2014)
        month = st.number_input("Month", min_value=1, max_value=12, value=12)
        day = st.number_input("Day", min_value=1, max_value=31, value=20)
        hour = st.number_input("Hour", min_value=0, max_value=23, value=14)
        weekday_name = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=5
        )
        st.session_state["weekday_name"] = weekday_name

    with col2:
        dewp = st.number_input("Dew Point (dewp)", value=-5.0)
        temp = st.number_input("Temperature (temp)", value=3.0)
        pres = st.number_input("Pressure (pres)", value=1020.0)
        iws = st.number_input("Wind Speed (iws)", value=15.0)
        is_ = st.number_input("Snow Hours (is_)", min_value=0, value=0)
        ir = st.number_input("Rain Hours (ir)", min_value=0, value=0)
        cbwd = st.selectbox("Wind Direction (cbwd)", ["NW", "SE", "cv", "NE"])

    with col3:
        pm2_5_lag_1 = st.number_input("PM2.5 Lag 1", value=82.0)
        pm2_5_lag_3 = st.number_input("PM2.5 Lag 3", value=76.0)
        pm2_5_lag_6 = st.number_input("PM2.5 Lag 6", value=65.0)
        pm2_5_lag_12 = st.number_input("PM2.5 Lag 12", value=58.0)
        pm2_5_lag_24 = st.number_input("PM2.5 Lag 24", value=61.0)

    st.markdown("### Rolling Statistics")
    col4, col5 = st.columns(2)

    with col4:
        pm2_5_roll_mean_3 = st.number_input("PM2.5 Rolling Mean 3", value=79.0)
        pm2_5_roll_std_3 = st.number_input("PM2.5 Rolling Std 3", value=6.0)
        pm2_5_roll_mean_6 = st.number_input("PM2.5 Rolling Mean 6", value=74.0)
        pm2_5_roll_std_6 = st.number_input("PM2.5 Rolling Std 6", value=10.0)

    with col5:
        pm2_5_roll_mean_12 = st.number_input("PM2.5 Rolling Mean 12", value=68.0)
        pm2_5_roll_std_12 = st.number_input("PM2.5 Rolling Std 12", value=14.0)
        pm2_5_roll_mean_24 = st.number_input("PM2.5 Rolling Mean 24", value=64.0)
        pm2_5_roll_std_24 = st.number_input("PM2.5 Rolling Std 24", value=16.0)

    payload = build_payload(
        no,
        year,
        month,
        day,
        hour,
        dewp,
        temp,
        pres,
        iws,
        is_,
        ir,
        cbwd,
        pm2_5_lag_1,
        pm2_5_lag_3,
        pm2_5_lag_6,
        pm2_5_lag_12,
        pm2_5_lag_24,
        pm2_5_roll_mean_3,
        pm2_5_roll_std_3,
        pm2_5_roll_mean_6,
        pm2_5_roll_std_6,
        pm2_5_roll_mean_12,
        pm2_5_roll_std_12,
        pm2_5_roll_mean_24,
        pm2_5_roll_std_24,
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Predict PM2.5", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=20)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction successful")
                    st.metric("Predicted PM2.5", result["predicted_pm25"])
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"API request error: {e}")

    with c2:
        if st.button("Get Health Risk", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/health-risk", json=payload, timeout=20)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Health risk prediction successful")
                    st.metric("Predicted PM2.5", result["predicted_pm25"])
                    risk = result["risk_category"]

                    if risk == "Good":
                      st.success(risk)
                    elif risk == "Moderate":
                      st.info(risk)
                    elif risk == "Unhealthy for Sensitive Groups":
                      st.warning(risk)
                    else:
                      st.error(risk)
                    st.info(result["health_message"])
                else:
                    st.error(f"Health risk request failed: {response.text}")
            except Exception as e:
                st.error(f"API request error: {e}")

    with st.expander("Show API Payload"):
        st.json(payload)


if __name__ == "__main__":
    main()
