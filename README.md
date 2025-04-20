# 🌍 PolluCast

This project implements an **end-to-end MLOps pipeline** to collect, track, analyze, and predict environmental pollution levels using real-time data. Built as a university course project, it combines data engineering, machine learning, model deployment, and monitoring into a cohesive system.

---

## 📌 Features

- 🔄 **Real-time data fetching** from OpenWeatherMap and AirVisual APIs.
- 🧊 **Data versioning** using [DVC](https://dvc.org/).
- 📊 **Time-series forecasting** with LSTM to predict pollution trends (e.g., AQI).
- ⚙️ **Model tracking** with MLflow (metrics like RMSE, MAE).
- 🚀 **Model deployment** using FastAPI.
- 📈 **Live monitoring** with Prometheus + Grafana.
- 🔐 .env for API key protection and environment configuration.

---

## 🛠 Tech Stack

| Component       | Tech Used                    |
|----------------|------------------------------|
| Language        | Python 3                     |
| Data Collection | REST APIs (OpenWeatherMap) |
| Version Control | Git + DVC                    |
| Model Type      | LSTM (Keras/TensorFlow)      |
| Experiment Tracking | MLflow                  |
| Deployment      | FastAPI                      |
| Monitoring      | Prometheus + Grafana         |

---

## 📂 Project Structure

PolluCast/ ├── .dvc/, .dvcignore # DVC config and tracking ├── .env, .gitignore # Environment & git exclusions ├── csv_data/, data/ # Raw and processed data ├── fetch_data.py # API data fetching script ├── preprocessing.py # Data cleaning & prep ├── json-to-csv.py # Format conversion utility ├── data.dvc # DVC tracked data file ├── mlops_project_description.pdf # Project guidelines ├── integration-documentation.docx # Setup documentation ├── requirements.txt # All dependencies ├── development/ # Model training & MLflow code ├── prometheus_grafana/ # Monitoring setup └── README.md # You are here 📖

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/talhasiddik/PolluCast
```
### 2.  Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Set Up Environment
Create a .env file:

OPENWEATHER_API_KEY=your_api_key_here
AIRVISUAL_API_KEY=your_api_key_here

### 4. Fetch & Version Data
```bash
python fetch_data.py
dvc add data/
dvc push
```

### 5. Train Model (with MLflow Logging)
```bash
cd development
mlflow run model.py
```
### 6. Deploy API
```bash
python development/app.py
```
### 7. Start Monitoring
Run Prometheus & Grafana containers as per prometheus_grafana/ setup instructions.

📈 Sample Output
📉 LSTM predictions of AQI trends.

🧠 Tracked experiments (loss curves, metrics) in MLflow UI.

📊 Real-time performance dashboards in Grafana.

### 📚 Documentation
 integration-documentation.docx – setup and DVC integration/full project breakdown.


### 🙋‍♂️ Author
Talha Siddik

### ⚠️ Disclaimer
This project is for educational purposes only.

