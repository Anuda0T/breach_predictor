import pandas as pd
import sys
sys.path.append('.')
from dashboard import BreachPredictionDashboard

def test_dashboard_functionality():
    # Test data validation
    dashboard = BreachPredictionDashboard()
    test_df = pd.read_csv('test_companies.csv')

    print('Testing data validation...')
    validation = dashboard._validate_uploaded_data(test_df)
    print(f'Validation result: {validation}')

    # Test batch analysis
    print('\nTesting batch analysis...')
    results = dashboard._analyze_uploaded_companies(test_df)
    print(f'Analysis completed for {len(results)} companies')
    for result in results[:2]:  # Show first 2 results
        print(f'{result["company_name"]}: {result["risk_level"]} ({result["risk_probability"]*100:.1f}%)')

    print('\nTesting completed successfully!')

if __name__ == "__main__":
    test_dashboard_functionality()
