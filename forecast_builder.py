#!/usr/bin/env python3
"""
Real-Time Breach Attack Forecasting Builder
==========================================

This script builds and runs real attack forecasting models using your company data.
It trains machine learning models on historical breach patterns and generates
predictions for future attack likelihoods.

Usage:
    python forecast_builder.py

Requirements:
    - company_profiles.csv: Company data with columns:
      company_name, industry, company_size, data_sensitivity, security_budget, employee_count

    - breach_data.csv: Historical breach data with columns:
      company, breach_date, records_affected, risk_score

Output:
    - Trained forecasting model saved to models/predictive_forecaster.pkl
    - Forecast report saved to forecast_report.json
    - Console output with detailed predictions
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from predictive_forecasting import PredictiveAttackForecaster

def load_data_files():
    """Load company and breach data files"""
    print("📂 Loading data files...")

    try:
        company_df = pd.read_csv('company_profiles.csv')
        breach_df = pd.read_csv('breach_data.csv')
        print(f"✅ Loaded {len(company_df)} companies and {len(breach_df)} breach records")
        return company_df, breach_df
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please ensure these files exist in the current directory:")
        print("   - company_profiles.csv")
        print("   - breach_data.csv")
        return None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def validate_data(company_df, breach_df):
    """Basic data validation"""
    print("🔍 Validating data...")

    errors = []

    # Check company data
    required_company_cols = ['company_name', 'industry', 'company_size', 'data_sensitivity', 'security_budget', 'employee_count']
    missing_cols = [col for col in required_company_cols if col not in company_df.columns]
    if missing_cols:
        errors.append(f"Company data missing columns: {missing_cols}")

    # Check breach data
    required_breach_cols = ['company', 'breach_date', 'records_affected', 'risk_score']
    missing_cols = [col for col in required_breach_cols if col not in breach_df.columns]
    if missing_cols:
        errors.append(f"Breach data missing columns: {missing_cols}")

    # Check data types
    if 'breach_date' in breach_df.columns:
        try:
            pd.to_datetime(breach_df['breach_date'])
        except:
            errors.append("breach_date column must be in YYYY-MM-DD format")

    if errors:
        print("❌ Data validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("✅ Data validation passed")
    return True

def build_forecast_model(company_df, breach_df):
    """Build and train the forecasting model"""
    print("\n🤖 Building forecasting model...")

    # Initialize forecaster
    forecaster = PredictiveAttackForecaster()

    # Prepare data
    forecast_data = forecaster.prepare_forecasting_data(company_df, breach_df)

    # Train model
    print("🎯 Training model on your data...")
    metrics = forecaster.train_forecasting_model(forecast_data)

    # Save model
    forecaster.save_model()

    return forecaster, metrics

def generate_forecast_report(forecaster, company_df, breach_df):
    """Generate comprehensive forecast report"""
    print("\n🔮 Generating forecast report...")

    # Generate report
    report = forecaster.generate_attack_forecast_report(company_df, breach_df)

    # Save to file
    with open('forecast_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report

def display_results(report):
    """Display forecast results in a readable format"""
    print("\n" + "="*60)
    print("🎯 FORECAST RESULTS")
    print("="*60)

    summary = report['summary']
    predictions = report['predictions']

    print("\n📊 SUMMARY STATISTICS:")
    print(f"   Total Companies Analyzed: {summary['total_companies']}")
    print(f"   Critical Risk Companies: {summary['critical_risk']}")
    print(f"   High Risk Companies: {summary['high_risk']}")
    print(f"   Average Risk Score: {summary.get('average_risk_score', 0):.2f}")
    print(f"   Total Records Affected: {summary.get('total_records_affected', 0):.0f}")
    print(f"   Companies at Risk (90 days): {summary['companies_at_risk_90_days']}")

    print("\n🏢 INDIVIDUAL COMPANY FORECASTS:")
    print(f"{'Company':<12} {'Risk Level':<12} {'Risk Score':<12} {'90-Day Risk':<12}")
    print("-" * 70)

    for _, pred in predictions.iterrows():
        risk_icon = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(pred['risk_level'], '⚪')

        print(f"{pred['company_name']:<12} {pred['risk_level']:<12} {pred['risk_score']:<12.2f} {pred['risk_probability_90_days']:<12.2f}")

    print(f"\n💾 Report saved to: forecast_report.json")
    print(f"📅 Generated at: {report['generated_at']}")

def main():
    """Main forecasting pipeline"""
    print("🚀 REAL-TIME BREACH ATTACK FORECASTING BUILDER")
    print("=" * 55)

    # Load data
    company_df, breach_df = load_data_files()
    if company_df is None or breach_df is None:
        return

    # Validate data
    if not validate_data(company_df, breach_df):
        return

    # Build model
    forecaster, metrics = build_forecast_model(company_df, breach_df)

    # Generate report
    report = generate_forecast_report(forecaster, company_df, breach_df)

    # Display results
    display_results(report)

    print("\n✅ Forecasting complete!")
    print("💡 Use the Streamlit dashboard to visualize results interactively")

if __name__ == "__main__":
    main()
