import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class PredictiveAttackForecaster:
    """
    Advanced predictive attack forecasting system that analyzes breach patterns
    and predicts future attack likelihoods based on temporal trends and company characteristics.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'company_size_encoded', 'industry_encoded', 'data_sensitivity',
            'security_budget_log', 'employee_count_log', 'days_since_last_breach',
            'breach_frequency', 'avg_records_affected', 'risk_score_trend',
            'seasonal_factor', 'growth_rate', 'compliance_score'
        ]

    def prepare_forecasting_data(self, company_df, breach_df):
        """
        Prepare comprehensive dataset for attack forecasting
        """
        print("🔮 Preparing forecasting data...")

        # Convert breach dates
        breach_df['breach_date'] = pd.to_datetime(breach_df['breach_date'])

        # Create time series features for each company
        forecast_data = []

        for _, company in company_df.iterrows():
            company_name = company.get('company_name', company.get('company', f"Company_{company.name}"))
            company_breaches = breach_df[breach_df['company'] == company_name].copy()

            if len(company_breaches) == 0:
                # No breach history - create baseline prediction
                baseline_features = self._create_baseline_features(company)
                forecast_data.append(baseline_features)
            else:
                # Create time series features
                company_features = self._create_time_series_features(company, company_breaches)
                forecast_data.extend(company_features)

        return pd.DataFrame(forecast_data)

    def _create_baseline_features(self, company):
        """Create baseline features for companies with no breach history"""
        company_name = company.get('company_name', company.get('company', f"Company_{company.name}"))
        return {
            'company_name': company_name,
            'company_size_encoded': self._encode_company_size(company['company_size']),
            'industry_encoded': self._encode_industry(company['industry']),
            'data_sensitivity': company['data_sensitivity'],
            'security_budget_log': np.log1p(company['security_budget']),
            'employee_count_log': np.log1p(company['employee_count']),
            'days_since_last_breach': 365 * 2,  # Assume 2 years since last breach
            'breach_frequency': 0.1,  # Very low frequency
            'avg_records_affected': 10000,  # Average breach size
            'risk_score_trend': 0,  # No trend
            'seasonal_factor': 0.5,  # Neutral seasonal factor
            'growth_rate': 0.05,  # Moderate growth
            'compliance_score': self._calculate_compliance_score(company),
            'target_attack_probability': 0.1,  # Low baseline probability
            'target_days_to_next_attack': 730  # 2 years
        }

    def _create_time_series_features(self, company, company_breaches):
        """Create time series features for companies with breach history"""
        features_list = []
        current_date = datetime.now()
        company_name = company.get('company_name', company.get('company', f"Company_{company.name}"))
        # Sort breaches by date
        company_breaches = company_breaches.sort_values('breach_date')

        # Calculate time-based features
        days_since_last_breach = (current_date - company_breaches['breach_date'].max()).days
        breach_frequency = len(company_breaches) / max(days_since_last_breach / 365, 1)
        avg_records_affected = company_breaches['records_affected'].mean()

        # Calculate risk score trend
        if len(company_breaches) > 1:
            risk_scores = company_breaches['risk_score'].values
            risk_score_trend = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
        else:
            risk_score_trend = 0

        # Seasonal factor based on breach dates
        seasonal_factor = self._calculate_seasonal_factor(company_breaches)

        # Growth rate (simplified)
        growth_rate = 0.05  # Could be calculated from company data

        # Create feature set
        features = {
            'company_name': company_name,
            'company_size_encoded': self._encode_company_size(company['company_size']),
            'industry_encoded': self._encode_industry(company['industry']),
            'data_sensitivity': company['data_sensitivity'],
            'security_budget_log': np.log1p(company['security_budget']),
            'employee_count_log': np.log1p(company['employee_count']),
            'days_since_last_breach': days_since_last_breach,
            'breach_frequency': breach_frequency,
            'avg_records_affected': avg_records_affected,
            'risk_score_trend': risk_score_trend,
            'seasonal_factor': seasonal_factor,
            'growth_rate': growth_rate,
            'compliance_score': self._calculate_compliance_score(company),
        }
        # Calculate target variables
        features['target_attack_probability'] = self._calculate_attack_probability(features)
        features['target_days_to_next_attack'] = self._predict_days_to_next_attack(features)

        features_list.append(features)
        return features_list

    def _encode_company_size(self, size):
        """Encode company size to numeric"""
        size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
        return size_map.get(size, 2)

    def _encode_industry(self, industry):
        """Encode industry to numeric"""
        industry_map = {
            'Technology': 1, 'Healthcare': 2, 'Finance': 3, 'Retail': 4,
            'Education': 5, 'Government': 6, 'Manufacturing': 7, 'Energy': 8
        }
        return industry_map.get(industry, 0)

    def _calculate_compliance_score(self, company):
        """Calculate compliance score based on company characteristics"""
        score = 0

        # Security budget factor
        if company['security_budget'] > 500000:
            score += 0.3
        elif company['security_budget'] > 100000:
            score += 0.2
        else:
            score += 0.1

        # Company size factor
        if company['company_size'] == 'Large':
            score += 0.2
        elif company['company_size'] == 'Medium':
            score += 0.15

        # Data sensitivity factor
        score += (company['data_sensitivity'] - 1) * 0.1

        return min(score, 1.0)

    def _calculate_seasonal_factor(self, breaches):
        """Calculate seasonal factor based on breach timing"""
        if len(breaches) < 2:
            return 0.5

        # Extract months of breaches
        breach_months = breaches['breach_date'].dt.month

        # Calculate concentration in certain months
        month_counts = breach_months.value_counts()
        max_month_count = month_counts.max()
        seasonal_factor = max_month_count / len(breaches)

        return seasonal_factor

    def _calculate_attack_probability(self, features):
        """Calculate attack probability based on features"""
        base_prob = 0.1

        # Risk factors
        if features['data_sensitivity'] >= 3:
            base_prob += 0.2
        if features['days_since_last_breach'] < 365:
            base_prob += 0.15
        if features['breach_frequency'] > 0.5:
            base_prob += 0.1
        if features['compliance_score'] < 0.3:
            base_prob += 0.1

        # Protective factors
        if features['security_budget_log'] > 12:  # log(500k) ≈ 13.1
            base_prob -= 0.1
        if features['compliance_score'] > 0.7:
            base_prob -= 0.05

        return np.clip(base_prob, 0.01, 0.95)

    def _predict_days_to_next_attack(self, features):
        """Predict days to next attack"""
        base_days = 730  # 2 years

        # Adjust based on risk factors
        if features['breach_frequency'] > 1:
            base_days *= 0.5
        elif features['breach_frequency'] > 0.5:
            base_days *= 0.7

        if features['days_since_last_breach'] < 180:
            base_days *= 0.8

        if features['compliance_score'] < 0.3:
            base_days *= 0.9

        return max(int(base_days), 30)  # Minimum 30 days

    def train_forecasting_model(self, forecast_data):
        """
        Train the predictive forecasting model
        """
        print("🎯 Training predictive attack forecasting model...")

        # Prepare features and targets
        X = forecast_data[self.feature_columns]
        y_prob = forecast_data['target_attack_probability']
        y_days = forecast_data['target_days_to_next_attack']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train probability prediction model
        self.probability_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.probability_model.fit(X_scaled, y_prob)

        # Train days prediction model
        self.days_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.days_model.fit(X_scaled, y_days)

        # Evaluate models
        tscv = TimeSeriesSplit(n_splits=3)
        prob_scores = []
        days_scores = []

        for train_idx, test_idx in tscv.split(X_scaled):
            # Probability model
            self.probability_model.fit(X_scaled[train_idx], y_prob.iloc[train_idx])
            prob_pred = self.probability_model.predict(X_scaled[test_idx])
            prob_scores.append(mean_absolute_error(y_prob.iloc[test_idx], prob_pred))

            # Days model
            self.days_model.fit(X_scaled[train_idx], y_days.iloc[train_idx])
            days_pred = self.days_model.predict(X_scaled[test_idx])
            days_scores.append(mean_absolute_error(y_days.iloc[test_idx], days_pred))

        print(f"Probability MAE: {np.mean(prob_scores):.2f}")
        print(f"Days MAE: {np.mean(days_scores):.0f}")
        return {
            'probability_mae': np.mean(prob_scores),
            'days_mae': np.mean(days_scores)
        }

    def predict_attack_risk(self, features):
        """
        Predict attack risk for a company using pre-computed features
        """
        # Prepare features DataFrame
        features_df = pd.DataFrame([features])

        # Scale features
        features_scaled = self.scaler.transform(features_df[self.feature_columns])

        # Make predictions
        attack_probability = self.probability_model.predict(features_scaled)[0]
        days_to_attack = self.days_model.predict(features_scaled)[0]

        # Calculate risk level
        risk_level = self._calculate_risk_level(attack_probability, days_to_attack)

        return {
            'attack_probability': attack_probability,
            'days_to_next_attack': int(days_to_attack),
            'risk_level': risk_level,
            'confidence_score': self._calculate_confidence(attack_probability, days_to_attack)
        }

    def _calculate_risk_level(self, probability, days):
        """Calculate risk level based on probability and time to attack"""
        if probability > 0.7 or days < 90:
            return 'Critical'
        elif probability > 0.5 or days < 180:
            return 'High'
        elif probability > 0.3 or days < 365:
            return 'Medium'
        else:
            return 'Low'

    def _calculate_confidence(self, probability, days):
        """Calculate prediction confidence"""
        # Simplified confidence calculation
        base_confidence = 0.7

        # Adjust based on prediction extremes
        if probability > 0.8 or probability < 0.1:
            base_confidence += 0.1
        if days > 1000 or days < 60:
            base_confidence += 0.1

        return min(base_confidence, 0.95)

    def save_model(self, filename="predictive_forecaster.pkl"):
        """Save the forecasting model"""
        model_data = {
            'probability_model': self.probability_model,
            'days_model': self.days_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }

        filepath = self.model_dir / filename
        joblib.dump(model_data, filepath)
        print(f"💾 Forecasting model saved to {filepath}")

    def load_model(self, filename="predictive_forecaster.pkl"):
        """Load the forecasting model"""
        filepath = self.model_dir / filename
        if filepath.exists():
            try:
                model_data = joblib.load(filepath)
                self.probability_model = model_data['probability_model']
                self.days_model = model_data['days_model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                print(f"📂 Forecasting model loaded from {filepath}")
                return True
            except Exception as e:
                print(f"⚠️ Error loading model: {str(e)}")
                print("🔄 Model will be retrained...")
                return False
        return False

    def generate_attack_forecast_report(self, company_df, breach_df, days_ahead=365):
        """
        Generate comprehensive attack forecast report
        """
        print(f"🔮 Generating {days_ahead}-day attack forecast report...")

        forecast_data = self.prepare_forecasting_data(company_df, breach_df)

        if self.probability_model is None:
            raise ValueError("Model not trained or loaded. Please train or load the forecasting model first.")

        # Generate predictions for all companies
        predictions = []
        for _, company in forecast_data.iterrows():
            company_data = dict(company)
            prediction = self.predict_attack_risk(company_data)
            prediction['company_name'] = company['company_name']
            predictions.append(prediction)

        predictions_df = pd.DataFrame(predictions)

        # Generate summary statistics
        summary = {
            'total_companies': len(predictions_df),
            'critical_risk': len(predictions_df[predictions_df['risk_level'] == 'Critical']),
            'high_risk': len(predictions_df[predictions_df['risk_level'] == 'High']),
            'avg_attack_probability': predictions_df['attack_probability'].mean(),
            'avg_days_to_attack': predictions_df['days_to_next_attack'].mean(),
            'companies_at_risk_90_days': len(predictions_df[predictions_df['days_to_next_attack'] <= 90])
        }

        return {
            'predictions': predictions_df,
            'summary': summary,
            'forecast_period_days': days_ahead,
            'generated_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    from data_validator import DataValidator

    # Initialize forecaster
    forecaster = PredictiveAttackForecaster()

    # Load and validate data
    validator = DataValidator()
    company_df, breach_df, _ = validator.validate_data_files()

    # Prepare forecasting data
    forecast_data = forecaster.prepare_forecasting_data(company_df, breach_df)

    # Train the model
    metrics = forecaster.train_forecasting_model(forecast_data)
    print("Training completed with metrics:", metrics)

    # Save the model
    forecaster.save_model()

    # Generate forecast report
    report = forecaster.generate_attack_forecast_report(company_df, breach_df)
    print("Forecast report generated for", report['summary']['total_companies'], "companies")
