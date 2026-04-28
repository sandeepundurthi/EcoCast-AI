# EcoCast AI 🌍

### End-to-End Air Quality Forecasting & Health Risk Alert System

EcoCast AI is a **production-style machine learning project** that predicts **PM2.5 air pollution levels** using time-series forecasting and environmental signals. The system includes a complete ML lifecycle: data pipeline, feature engineering, model training, API deployment, interactive dashboard, and drift monitoring.

Built to simulate a **real AI Engineer / ML Engineer production system**.

---

# 🚀 Key Features

✅ Forecast hourly **PM2.5 pollution levels**
✅ Generate **health-risk alerts** based on predicted air quality
✅ Automated data cleaning + validation pipeline
✅ Advanced time-series feature engineering
✅ XGBoost model with strong predictive performance
✅ FastAPI backend for real-time inference
✅ Streamlit dashboard for interactive predictions
✅ Feature importance + residual diagnostics
✅ Data drift detection for monitoring

---

# 📊 Model Performance

| Metric | Baseline (Lag-1) | XGBoost    |
| ------ | ---------------- | ---------- |
| RMSE   | 20.78            | **10.00**  |
| MAE    | 11.50            | **5.93**   |
| R²     | 0.9480           | **0.9880** |

### Improvement Achieved

* **52% RMSE reduction** over baseline
* **48% MAE reduction** over baseline

---

# 🧠 Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Deployment

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Monitoring

* Custom Drift Detection

### Visualization

* Matplotlib
* Plotly

---

# 🏗️ System Architecture

```text
Raw Data
   ↓
Cleaning + Validation
   ↓
Feature Engineering
   ↓
XGBoost Model
   ↓
FastAPI API
   ↓
Streamlit Dashboard
   ↓
Drift Monitoring
```

---

# 📁 Project Structure

```text
ecocast-ai/
│
├── app/
│   ├── api/
│   └── dashboard/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── training/
│   └── monitoring/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ecocast-ai.git
cd ecocast-ai
```

---

## 2️⃣ Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset

Uses the **Beijing PM2.5 Air Quality Dataset** containing:

* PM2.5 levels
* Temperature
* Pressure
* Dew point
* Wind direction
* Wind speed
* Rain / Snow

Place dataset here:

```text
data/raw/air_quality.csv
```

---

# 🚀 How to Run

---

## Step 1 — Run Data Pipeline

```bash
python run_pipeline.py
```

Creates:

```text
data/processed/
data/features/
reports/
```

---

## Step 2 — Train Model

```bash
python src/training/train_model.py
```

Creates:

```text
models/xgboost_model.pkl
```

---

## Step 3 — Evaluate Model

```bash
python src/training/evaluate_model.py
```

Creates:

```text
reports/figures/
reports/evaluation_summary.txt
```

---

## Step 4 — Start FastAPI Backend

```bash
uvicorn app.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Step 5 — Start Streamlit Dashboard

```bash
streamlit run app/dashboard/streamlit_app.py
```

---

# 🖥️ Dashboard Features

* Enter environmental conditions
* Predict PM2.5 instantly
* Get health-risk classification
* View live backend API connection

---

# 📈 Explainability

Generated artifacts:

* Feature Importance Plot
* Actual vs Predicted Plot
* Residual Distribution
* Worst Prediction Cases CSV

---

# 🛡️ Drift Monitoring

Run:

```bash
python src/monitoring/drift_detection.py
```

Detects feature distribution changes between training and incoming data.

Useful for:

* model degradation alerts
* retraining triggers
* production monitoring

---

# 💼 Resume Impact

Built a production-ready end-to-end AI system for air-quality forecasting using XGBoost, FastAPI, and Streamlit, achieving **52% RMSE reduction** over baseline with explainability and drift monitoring.

---

# 🔮 Future Improvements

* Docker deployment
* AWS / GCP hosting
* Real-time OpenAQ API integration
* SHAP explainability dashboard
* Automated retraining pipeline
* CI/CD with GitHub Actions

---

# 👨‍💻 Author

**Sandeep Undurthi**

* Data Scientist / AI Engineer
* Python | ML | FastAPI | Streamlit | XGBoost

---

# ⭐ If you found this useful

Star the repository ⭐ and connect with me on LinkedIn.
