import os
import json
import time
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import joblib

from ml_model import BreachPredictor
from data_validator import DataValidator

class ModelTrainer:
    def __init__(self, 
                 model_dir: str = "models",
                 data_dir: str = "data",
                 random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
    def train(self):
        """Run the complete training pipeline"""
        print("🚀 Starting model training pipeline...")
        
        # Validate and load data
        print("\n1️⃣ Validating input data...")
        validator = DataValidator()
        company_df, breach_df, validation_stats = validator.validate_data_files()
        
        # Initialize model
        print("\n2️⃣ Initializing model...")
        predictor = BreachPredictor()
        
        # Prepare features
        print("\n3️⃣ Preparing features...")
        training_data = predictor._create_training_dataset(
            predictor._engineer_features(company_df),
            predictor._enhance_breach_data(breach_df)
        )

        # Convert list of dicts to DataFrame
        training_df = pd.DataFrame(training_data)

        # Split data
        print("\n4️⃣ Splitting train/test data...")
        X = pd.DataFrame(training_df['features'].tolist())
        y = training_df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        print("\n5️⃣ Training model...")
        predictor.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        print("\n6️⃣ Initializing SHAP explainer...")
        predictor.initialize_shap_explainer(X_train)
        
        # Evaluate model
        print("\n7️⃣ Evaluating model performance...")
        metrics = self._evaluate_model(predictor.model, X_test, y_test)
        
        # Save model and metadata
        print("\n8️⃣ Saving model artifacts...")
        model_path = self._save_model_artifacts(predictor, metrics, validation_stats)
        
        print(f"\n✅ Training pipeline completed! Model saved to: {model_path}")
        print(f"   Test set accuracy: {metrics['accuracy']:.2%}")
        return metrics
    
    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        classification_metrics = classification_report(
            y_test, y_pred, output_dict=True
        )
        
        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        metrics = {
            'accuracy': classification_metrics['accuracy'],
            'precision': classification_metrics['1']['precision'],
            'recall': classification_metrics['1']['recall'],
            'f1': classification_metrics['1']['f1-score'],
            'roc_auc': auc(fpr, tpr),
            'pr_auc': auc(recall, precision),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_size': len(y_test)
        }
        
        return metrics
    
    def _save_model_artifacts(self, predictor, metrics, validation_stats):
        """Save model and metadata with atomic write"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_name = f"breach_predictor_model_{timestamp}.pkl"
        metrics_name = f"model_metrics_{timestamp}.json"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            
            # Save model with metadata
            model_data = {
                'model': predictor.model,
                'feature_names': predictor.feature_names,
                'feature_importance': predictor.feature_importance
            }
            tmp_model_path = tmp_dir / model_name
            joblib.dump(model_data, tmp_model_path)
            
            # Save metrics
            metrics_data = {
                'metrics': metrics,
                'validation_stats': validation_stats,
                'model_info': {
                    'type': predictor.model.__class__.__name__,
                    'params': predictor.model.get_params()
                }
            }
            tmp_metrics_path = tmp_dir / metrics_name
            with open(tmp_metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Atomically move files to final location
            final_model_path = self.model_dir / model_name
            final_metrics_path = self.model_dir / metrics_name
            
            shutil.move(str(tmp_model_path), str(final_model_path))
            shutil.move(str(tmp_metrics_path), str(final_metrics_path))
            
            # Update symlink to latest model (Windows compatibility)
            latest_symlink = self.model_dir / 'breach_predictor_model.pkl'
            if latest_symlink.exists():
                latest_symlink.unlink()
            try:
                os.symlink(str(final_model_path), str(latest_symlink))
            except OSError:
                # On Windows, copy the file instead of symlinking
                shutil.copy2(str(final_model_path), str(latest_symlink))
            
            return final_model_path

if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        metrics = trainer.train()
        print("\n📊 Final Model Metrics:")
        print(f"Accuracy:   {metrics['accuracy']:.2%}")
        print(f"Precision:  {metrics['precision']:.2%}")
        print(f"Recall:     {metrics['recall']:.2%}")
        print(f"F1 Score:   {metrics['f1']:.2%}")
        print(f"ROC AUC:    {metrics['roc_auc']:.2%}")
        print(f"PR AUC:     {metrics['pr_auc']:.2%}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise