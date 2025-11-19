# test_load.py
from download_s3_files import download_files_from_s3
import pickle
import pandas as pd

print("Downloading files from S3...")
download_files_from_s3()

print("Loading model...")
with open("breach_predictor_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded:", type(model))

print("Reading CSVs...")
df_breach = pd.read_csv("breach_data.csv")
df_comp = pd.read_csv("company_profiles.csv")
print("breach_data rows:", len(df_breach))
print("company_profiles rows:", len(df_comp))
print(df_breach.head())
print(df_comp.head())
