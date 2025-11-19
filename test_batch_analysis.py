import pandas as pd
import sys
sys.path.append('.')
from dashboard import BreachPredictionDashboard

def test_batch_analysis():
    # Test batch analysis with valid data
    dashboard = BreachPredictionDashboard()
    valid_df = pd.DataFrame({
        'company_name': ['TestCorp', 'AnotherCorp'],
        'industry': ['Technology', 'Finance'],
        'company_size': ['Large', 'Medium'],
        'data_sensitivity': [2, 3],
        'security_budget': [100000, 200000],
        'employee_count': [500, 800]
    })

    print('Testing batch analysis...')
    results = dashboard._analyze_uploaded_companies(valid_df)
    print(f'Analysis completed for {len(results)} companies')
    for result in results:
        print(f'{result["company_name"]}: {result["risk_level"]} ({result["risk_probability"]*100:.1f}%)')

    print('\nTesting risk level conversion...')
    test_probs = [0.1, 0.3, 0.6, 0.8]
    for prob in test_probs:
        level = dashboard._get_risk_level(prob)
        print(f'{prob*100:.1f}% -> {level}')

if __name__ == "__main__":
    test_batch_analysis()
