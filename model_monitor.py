import logging
import time
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import ks_2samp

class ModelMonitor:
    def __init__(self, 
                 log_dir: str = "logs",
                 metrics_dir: str = "metrics",
                 alert_threshold: float = 0.1):
        # Setup directories
        self.log_dir = Path(log_dir)
        self.metrics_dir = Path(metrics_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.setup_logging()
        
        # Initialize metrics storage
        self.baseline_stats = {}
        self.alert_threshold = alert_threshold
        self.load_baseline_stats()
        
    def setup_logging(self):
        """Configure logging with both file and console handlers"""
        log_file = self.log_dir / f"model_monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('ModelMonitor')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_baseline_stats(self):
        """Load baseline statistics for drift detection"""
        baseline_file = self.metrics_dir / "baseline_stats.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                self.baseline_stats = json.load(f)
                self.logger.info("Loaded baseline statistics")
    
    def save_baseline_stats(self, training_data: pd.DataFrame):
        """Save baseline statistics from training data"""
        stats = {
            'feature_stats': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for column in training_data.columns:
            col_data = training_data[column]
            stats['feature_stats'][column] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75))
            }
        
        baseline_file = self.metrics_dir / "baseline_stats.json"
        with open(baseline_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.baseline_stats = stats
        self.logger.info("Saved new baseline statistics")
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift using KS test"""
        if not self.baseline_stats:
            self.logger.warning("No baseline stats available for drift detection")
            return {}
        
        drift_detected = False
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {}
        }
        
        for column in current_data.columns:
            if column not in self.baseline_stats['feature_stats']:
                continue
                
            # Perform Kolmogorov-Smirnov test
            baseline_stats = self.baseline_stats['feature_stats'][column]
            ks_statistic, p_value = ks_2samp(
                current_data[column],
                np.random.normal(
                    baseline_stats['mean'],
                    baseline_stats['std'],
                    len(current_data)
                )
            )
            
            is_drifting = p_value < self.alert_threshold
            if is_drifting:
                drift_detected = True
                
            drift_report['features'][column] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'is_drifting': is_drifting
            }
        
        drift_report['drift_detected'] = drift_detected
        
        # Log drift detection results
        if drift_detected:
            self.logger.warning("Data drift detected!")
            drifting_features = [
                f for f, stats in drift_report['features'].items()
                if stats['is_drifting']
            ]
            self.logger.warning(f"Drifting features: {', '.join(drifting_features)}")
        
        return drift_report
    
    def log_prediction(self, 
                      features: Dict[str, Any],
                      prediction: float,
                      actual: Optional[float] = None):
        """Log a single prediction with optional actual value"""
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': float(prediction)
        }
        
        if actual is not None:
            prediction_log['actual'] = float(actual)
        
        log_file = self.metrics_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(prediction_log) + '\n')
    
    def check_model_health(self) -> Dict[str, Any]:
        """Check model performance metrics"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        try:
            # Load today's predictions
            log_file = self.metrics_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            if not log_file.exists():
                return health_report
            
            predictions = []
            actuals = []
            
            with open(log_file) as f:
                for line in f:
                    record = json.loads(line)
                    if 'actual' in record:
                        predictions.append(record['prediction'])
                        actuals.append(record['actual'])
            
            if predictions and actuals:
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                predictions_binary = [1 if p >= 0.5 else 0 for p in predictions]
                actuals = [1 if a >= 0.5 else 0 for a in actuals]
                
                health_report['metrics'] = {
                    'accuracy': float(accuracy_score(actuals, predictions_binary)),
                    'precision': float(precision_score(actuals, predictions_binary)),
                    'recall': float(recall_score(actuals, predictions_binary))
                }
                
                # Check for performance degradation
                if health_report['metrics']['accuracy'] < 0.7:  # threshold
                    self.logger.warning("Model performance below threshold!")
                    
        except Exception as e:
            self.logger.error(f"Error checking model health: {e}")
        
        return health_report

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status including model health and drift"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'warnings': []
        }
        
        # Check model health
        health_report = self.check_model_health()
        if health_report.get('metrics', {}).get('accuracy', 1.0) < 0.7:
            status['status'] = 'degraded'
            status['warnings'].append('Model performance below threshold')
        
        # Check recent drift reports
        drift_file = self.metrics_dir / f"drift_{datetime.now().strftime('%Y%m%d')}.json"
        if drift_file.exists():
            with open(drift_file) as f:
                drift_report = json.load(f)
                if drift_report.get('drift_detected', False):
                    status['status'] = 'warning'
                    status['warnings'].append('Data drift detected')
        
        return status

if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()
    
    # Check system status
    status = monitor.get_system_status()
    print("\n=== System Status ===")
    print(f"Status: {status['status']}")
    if status['warnings']:
        print("Warnings:")
        for warning in status['warnings']:
            print(f"- {warning}")
    
    # Example prediction logging
    monitor.log_prediction(
        features={'feature1': 1.0, 'feature2': 2.0},
        prediction=0.8,
        actual=1.0
    )