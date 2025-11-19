Breach Predictor — AI-Powered Data Breach Risk Prediction System  

An AI-powered system that predicts data breach risk for companies using machine learning, real-world breach datasets, synthetic company profiles, and dark web intelligence.  
This project includes a full ML pipeline, data lake, prediction API, landing page, dashboard, and monitoring tools.

---

  Overview  

Breach Predictor analyzes company attributes (industry, size, exposure) and generates:  
✔ Risk score (%)  
✔ Prediction classification  
✔ Forecasted breach trends  
✔ Alerts for high-risk companies  
✔ Dashboards for interactive analysis  

Built as a **SaaS-ready prototype** with Streamlit + Flask + Python ML.

---

Features  

   Machine Learning  
- Trained on **920 real breaches** (HIBP)  
- **500 synthetic companies**  
- Random Forest model → **71% accuracy**  
- Predictive forecasting with confidence intervals  
- Model monitoring + drift detection  

  Data Pipeline  
- Collects breach data from Have I Been Pwned (API)  
- Generates synthetic company profiles  
- Processes & validates using schemas  
- Organizes a full **data lake** (raw → processed → enriched → analytics)

 Web Interfaces  
- **Streamlit Dashboard**  
- **Flask API** (`flask_app.py`)  
- **Landing Page (HTML/CSS/JS)`**  
- **Cyber Calculator Web App**  
- Additional interactive tools  

 Alerts & Monitoring  
- Auto-alerts for high-risk predictions  
- Model drift monitoring  
- Dark web threat scanning  

 Testing  
- Forecasting tests  
- Batch prediction tests  
- Dashboard tests
  
 Cloud Storage: Why Model & Data Are Stored in S3

This project uses **AWS S3 for storing large files** instead of keeping them inside the GitHub repository.

 Why?
Some project assets are large and should not be stored in Git directly, including:
- `breach_predictor_model.pkl` (trained ML model)
- `breach_data.csv` (920 breach records)
- `company_profiles.csv` (500 synthetic profiles)

GitHub has a 100 MB limit and storing large datasets in the repository causes performance issues, so these files are stored in **Amazon S3**.

How It Works
When the project runs, it automatically downloads the required files from S3:

```python
from download_s3_files import download_files_from_s3
download_files_from_s3()


---

Project Structure  
breach_predictor/
│
├── dashboard.py                 # Streamlit dashboard  
├── flask_app.py                 # Flask prediction API  
├── train.py                     # Train ML models  
├── ml_model.py                  # ML logic (training + prediction)
├── predictive_forecasting.py    # Forecasting model  
├── alert_system.py              # High-risk alerts  
├── model_monitor.py             # Model performance tracking  
├── data_collector.py            # HIBP + synthetic data collection  
├── data_processor.py            # Data cleaning + transformation  
├── data_validator.py            # Schema-based validation  
├── data_catalog.py              # SQLite metadata store  
│
├── data_lake/                   # Data lake layers  
│   ├── raw/  
│   ├── processed/  
│   ├── enriched/  
│   └── analytics/  
│
├── landing_page/                # Marketing landing page (HTML/CSS/JS)  
├── web_app/                     # General web app interface  
├── cyber_calc/                  # Cyber risk calculator tool  
│
├── models/                      # Saved ML models (.pkl)  
├── metrics/                     # Model performance metrics  
├── logs/                        # Log files  
│
├── breach_data.csv              # 920 breach records  
├── company_profiles.csv         # 500 synthetic profiles  
│
├── requirements.txt             # Python dependencies  
├── .gitignore                   # Ignore venv + junk  
└── README.md                    # Documentation  


