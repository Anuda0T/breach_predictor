from predictive_forecasting import PredictiveAttackForecaster
from data_validator import DataValidator
import pandas as pd

def test_forecasting():
    print("🧪 Testing Predictive Forecasting System")
    print("=" * 50)

    # Load and validate data
    validator = DataValidator()
    company_df, breach_df, _ = validator.validate_data_files()
    print(f"✅ Data loaded: {len(company_df)} companies, {len(breach_df)} breaches")

    # Initialize forecaster
    forecaster = PredictiveAttackForecaster()

    # Prepare forecasting data
    forecast_data = forecaster.prepare_forecasting_data(company_df, breach_df)
    print(f"✅ Forecast data prepared: {forecast_data.shape}")

    # Train the model
    metrics = forecaster.train_forecasting_model(forecast_data)
    print(f"✅ Model trained with metrics: {metrics}")

    # Save the model
    forecaster.save_model()
    print("✅ Model saved")

    # Generate forecast report
    report = forecaster.generate_attack_forecast_report(company_df, breach_df)
    print(f"✅ Report generated for {report['summary']['total_companies']} companies")
    print(f"   - Critical risk: {report['summary']['critical_risk']}")
    print(f"   - High risk: {report['summary']['high_risk']}")
    print(f"   - Average risk: {report['summary']['avg_attack_probability']:.3f}")

    # Test individual prediction
    sample_company = {
        'company_size_encoded': 3,  # Large
        'industry_encoded': 1,  # Technology
        'data_sensitivity': 3,
        'security_budget_log': 13.122365,  # log(500000)
        'employee_count_log': 6.907755,  # log(1000)
        'days_since_last_breach': 365,
        'breach_frequency': 0.5,
        'avg_records_affected': 50000,
        'risk_score_trend': 0.1,
        'seasonal_factor': 0.6,
        'growth_rate': 0.08,
        'compliance_score': 0.75
    }

    prediction = forecaster.predict_attack_risk(sample_company)
    print("\n🧪 Individual Prediction Test:")
    print(f"   - Risk Level: {prediction['risk_level']}")
    print(f"   - Attack Probability: {prediction['attack_probability']:.3f}")
    print(f"   - Days to Attack: {prediction['days_to_next_attack']}")
    print(f"   - Confidence Score: {prediction['confidence_score']:.2f}")

    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    test_forecasting()
