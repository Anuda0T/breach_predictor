import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import shap
from model_monitor import ModelMonitor

class BreachPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_importance = {}
        self.explainer = None
        self.feature_names = None
        self.monitor = ModelMonitor()
        
    def prepare_data(self):
        """Prepare and enhance data for ML training"""
        print("📊 Preparing data for machine learning...")
        
        try:
            # Validate data first
            from data_validator import DataValidator
            validator = DataValidator()
            company_df, breach_df, validation_stats = validator.validate_data_files()
            
            # Log validation results
            print("\n=== Data Validation Results ===")
            print(f"Companies: {validation_stats['companies']['valid']} valid, "
                  f"{validation_stats['companies']['invalid']} invalid")
            print(f"Breaches: {validation_stats['breaches']['valid']} valid, "
                  f"{validation_stats['breaches']['invalid']} invalid")
            
            if validation_stats['companies']['invalid'] > 0 or validation_stats['breaches']['invalid'] > 0:
                validator.save_validation_report()
                print("⚠️ Some data failed validation. Check data_validation_report.json for details.")
            
            # Continue with valid data only
            company_df = self._engineer_features(company_df)
            breach_df = self._enhance_breach_data(breach_df)
            
            training_data = self._create_training_dataset(company_df, breach_df)
            
            print(f"\n✅ Data prepared: {len(training_data)} samples")
            return training_data
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            print("Using synthetic data for demonstration...")
            return self._create_synthetic_data()
    
    def _engineer_features(self, company_df):
        """Create advanced features for ML"""
       
        size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        industry_mapping = {
            'Technology': 0, 'Finance': 1, 'Healthcare': 2, 
            'Education': 3, 'Retail': 4, 'Government': 5
        }
        
        company_df['company_size_encoded'] = company_df['company_size'].map(size_mapping)
        company_df['industry_encoded'] = company_df['industry'].map(industry_mapping)
        
        
        company_df['employee_to_budget_ratio'] = company_df['employee_count'] / company_df['security_budget']
        company_df['high_sensitivity_risk'] = (company_df['data_sensitivity'] >= 2).astype(int)
        
        return company_df
    
    def _enhance_breach_data(self, breach_df):
        """Enhance breach data with additional features"""
        
        breach_df['breach_severity'] = breach_df['records_affected'].apply(
            lambda x: 0 if x < 100000 else (1 if x < 1000000 else 2)
        )
        
        
        industry_risk = {
            'Healthcare': 3, 'Finance': 3, 'Government': 2, 
            'Education': 2, 'Retail': 1, 'Technology': 1
        }
        
        breach_df['industry_risk_score'] = breach_df['industry'].map(industry_risk)
        
        return breach_df
    
    def _create_training_dataset(self, company_df, breach_df):
        """Create training dataset with labels"""
        training_samples = []

        # Enhanced breach pattern analysis
        industry_breach_patterns = breach_df.groupby('industry').agg({
            'risk_score': 'mean',
            'records_affected': ['sum', 'mean', 'count']
        }).reset_index()

        # Flatten column names
        industry_breach_patterns.columns = [
            'industry', 'industry_breach_risk', 'breach_volume_total',
            'breach_volume_avg', 'breach_count'
        ]

        # Add critical breach ratio separately (using risk_score as proxy for severity)
        critical_breaches = breach_df.groupby('industry').agg({
            'risk_score': lambda x: (x >= 4).sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        critical_breaches = critical_breaches.rename(columns={'risk_score': 'critical_breach_ratio'})

        # Merge critical breach ratio
        industry_breach_patterns = industry_breach_patterns.merge(
            critical_breaches[['industry', 'critical_breach_ratio']],
            on='industry',
            how='left'
        )

        # Merge with company data
        company_enhanced = company_df.merge(
            industry_breach_patterns,
            on='industry',
            how='left'
        )

        # Fill missing values with industry averages or defaults
        company_enhanced['industry_breach_risk'] = company_enhanced['industry_breach_risk'].fillna(0)
        company_enhanced['breach_volume_total'] = company_enhanced['breach_volume_total'].fillna(0)
        company_enhanced['breach_volume_avg'] = company_enhanced['breach_volume_avg'].fillna(0)
        company_enhanced['breach_count'] = company_enhanced['breach_count'].fillna(0)
        company_enhanced['critical_breach_ratio'] = company_enhanced['critical_breach_ratio'].fillna(0)

        # Also fill any remaining NaN values in numeric columns
        numeric_columns = company_enhanced.select_dtypes(include=[np.number]).columns
        company_enhanced[numeric_columns] = company_enhanced[numeric_columns].fillna(0)

        # Create features for each company
        for _, company in company_enhanced.iterrows():
            # Base features
            features = [
                company['company_size_encoded'],
                company['industry_encoded'],
                company['data_sensitivity'],
                company['employee_to_budget_ratio'],
                company['high_sensitivity_risk'],
            ]

            # Breach history features
            features.extend([
                company['industry_breach_risk'],
                np.log1p(company['breach_volume_total']),  # log(1+x) to handle zeros
                np.log1p(company['breach_volume_avg']),
                company['breach_count'],
                company['critical_breach_ratio'],
            ])

            # Financial features
            features.extend([
                company['security_budget'] / 10000,  # Normalized budget
                company['security_budget'] / company['employee_count'],  # Budget per employee
            ])

            # Risk indicators
            high_risk_industry = company['industry'] in ['Healthcare', 'Finance']
            features.append(int(high_risk_industry))

            # Label based on breach history
            label = 1 if company['has_breach_history'] else 0

            training_samples.append({
                'features': features,
                'label': label,
                'company_id': company['company_id']
            })

        return training_samples
    
    def _create_synthetic_data(self):
        """Create synthetic training data if real data is unavailable"""
        print("🔄 Creating synthetic training data...")
        
        training_samples = []
        np.random.seed(42)
        
        for i in range(200):  
            company_size = np.random.choice([0, 1, 2])  
            industry = np.random.choice([0, 1, 2, 3, 4, 5]) 
            data_sensitivity = np.random.randint(1, 4)
            
            
            is_high_risk_industry = industry in [1, 2]  
            has_high_sensitivity = data_sensitivity >= 2
            
            
            base_risk = 0.3 if is_high_risk_industry else 0.1
            sensitivity_boost = 0.4 if has_high_sensitivity else 0.0
            size_risk = company_size * 0.1 
            
            breach_probability = base_risk + sensitivity_boost + size_risk
            
            
            label = 1 if np.random.random() < breach_probability else 0
            
            features = [
                company_size,
                industry,
                data_sensitivity,
                np.random.uniform(0.1, 2.0),
                int(has_high_sensitivity),
                np.random.uniform(0.5, 3.0),
                np.random.uniform(10, 15),    
                np.random.uniform(1, 100)    
            ]
            
            training_samples.append({
                'features': features,
                'label': label,
                'company_id': i + 1
            })
        
        print(f" Created {len(training_samples)} synthetic training samples")
        return training_samples
    
    def train(self):
        """Train the machine learning model"""
        print(" Training Machine Learning Model...")
        
        
        training_data = self.prepare_data()
        
        
        X = [sample['features'] for sample in training_data]
        y = [sample['label'] for sample in training_data]
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        

        self.model.fit(X_train, y_train)
        
       
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with {accuracy:.2%} accuracy")
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))
        
        
        feature_names = [
            'Company Size', 'Industry', 'Data Sensitivity', 'Budget Ratio',
            'High Sensitivity', 'Industry Risk', 'Breach Volume Total', 'Breach Volume Avg',
            'Breach Count', 'Critical Breach Ratio', 'Security Budget', 'Budget Per Employee', 'High Risk Industry'
        ]
        
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        print("\n Feature Importance:")
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.3f}")
        
        # Initialize SHAP explainer
        self.initialize_shap_explainer(X_train)
        
        # Save the model and feature names
        model_data = {
            'model': self.model,
            'feature_names': feature_names,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, 'breach_predictor_model.pkl')
        print(" Model saved as 'breach_predictor_model.pkl'")
        
        return accuracy
    
    def predict_risk(self, company_features):
        """Predict breach risk for a new company"""
        # Check for data drift
        if self.feature_names:
            features_df = pd.DataFrame([company_features], columns=self.feature_names)
            drift_report = self.monitor.check_data_drift(features_df)
            if drift_report.get('drift_detected', False):
                self.monitor.logger.warning("Data drift detected in prediction request")

        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            risk_probability = self.model.predict_proba([company_features])[0][1]
        else:
            risk_probability = float(self.model.predict([company_features])[0])

        # Log prediction
        if self.feature_names:
            features_dict = dict(zip(self.feature_names, company_features))
            self.monitor.log_prediction(features_dict, risk_probability)

        return risk_probability

    def batch_predict_risk(self, companies_data):
        """Predict breach risk for multiple companies"""
        if not isinstance(companies_data, list):
            companies_data = [companies_data]

        predictions = []
        for company_features in companies_data:
            risk_probability = self.predict_risk(company_features)
            predictions.append({
                'risk_probability': risk_probability,
                'risk_level': self._get_risk_level(risk_probability),
                'features': company_features
            })

        return predictions

    def _get_risk_level(self, risk_probability):
        """Convert risk probability to risk level"""
        risk_percentage = risk_probability * 100
        if risk_percentage < 20:
            return "LOW"
        elif risk_percentage < 50:
            return "MEDIUM"
        elif risk_percentage < 75:
            return "HIGH"
        else:
            return "CRITICAL"
            
    def load_model(self, model_path='breach_predictor_model.pkl'):
        """Load a trained model and its metadata"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
    
    def initialize_shap_explainer(self, X_train):
        """Initialize SHAP explainer with training data"""
        self.feature_names = X_train.columns.tolist()
        # Using TreeExplainer since we're using RandomForestClassifier
        self.explainer = shap.TreeExplainer(self.model)
        
    def get_shap_values(self, features):
        """Calculate SHAP values for given features"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap_explainer first.")

        # Convert features to DataFrame if it's not already
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features], columns=self.feature_names)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        # For binary classification, shap_values is a list with two elements
        # We want the second element (probability of positive class)
        if isinstance(shap_values, list):
            shap_values_array = shap_values[1]
            base_value = self.explainer.expected_value[1]
        else:
            shap_values_array = shap_values
            base_value = self.explainer.expected_value

        # Create a simple object with base_values and values attributes
        class SHAPResult:
            def __init__(self, base_values, values):
                self.base_values = base_values
                self.values = values

        return SHAPResult(base_value, shap_values_array[0]), self.feature_names


if __name__ == "__main__":
    print(" STARTING DAY 2: MACHINE LEARNING TRAINING")
    print("=" * 50)
    
    predictor = BreachPredictor()
    accuracy = predictor.train()
    
    print(f"\n DAY 2 COMPLETED!")
    print(f" Model Accuracy: {accuracy:.2%}")
    print(f" Ready to predict breach risks!")
    print(f" Model saved for future use")
    
    
    print(f"\n DEMO PREDICTION:")
    sample_company = [1, 2, 3, 0.5, 1, 2.5, 12.0, 50.0]  
    risk = predictor.predict_risk(sample_company)
    print(f"Sample company breach risk: {risk:.2%}")