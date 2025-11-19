import requests
import pandas as pd

class AdvancedDataCollector:
    def __init__(self):
        self.breach_data = []
        self.company_profiles = []
    
    def collect_breach_data(self):
        
        print(" Collecting advanced breach data...")
        
        try:
            
            response = requests.get(
                "https://haveibeenpwned.com/api/v3/breaches",
                headers={"User-Agent": "Advanced-Research-Project"}
            )
            
            if response.status_code == 200:
                breaches = response.json()
                print(f" Collected {len(breaches)} breaches from HIBP")
                
                
                for breach in breaches:  # Collect all available breaches
                    enhanced_breach = {
                        'company': breach['Name'],
                        'domain': breach.get('Domain', ''),
                        'breach_date': breach.get('BreachDate', ''),
                        'records_affected': breach.get('PwnCount', 0),
                        'data_classes': breach.get('DataClasses', []),
                        'industry': self._classify_industry(breach.get('Domain', '')),
                        'risk_score': self._calculate_initial_risk(breach)
                    }
                    self.breach_data.append(enhanced_breach)
                
                return True
                
        except Exception as e:
            print(f" Error collecting data: {e}")
            return False
    
    def _classify_industry(self, domain):
        """Classify company by industry based on domain"""
        domain = domain.lower()
        if 'health' in domain or 'medical' in domain or 'hospital' in domain:
            return 'Healthcare'
        elif 'bank' in domain or 'finance' in domain or 'credit' in domain:
            return 'Finance'
        elif 'edu' in domain or 'university' in domain:
            return 'Education'
        elif 'gov' in domain:
            return 'Government'
        elif 'store' in domain or 'shop' in domain or 'market' in domain:
            return 'Retail'
        else:
            return 'Technology'
    
    def _calculate_initial_risk(self, breach):
        """Calculate initial risk score based on breach details"""
        risk_score = 0
        
       
        records = breach.get('PwnCount', 0)
        if records > 10000000:
            risk_score += 3
        elif records > 1000000:
            risk_score += 2
        elif records > 100000:
            risk_score += 1
        
        
        data_classes = breach.get('DataClasses', [])
        sensitive_data = ['Passwords', 'Email addresses', 'Credit cards', 'Social security numbers']
        
        for data_type in data_classes:
            if any(sensitive in data_type for sensitive in sensitive_data):
                risk_score += 2
        
        return min(risk_score, 5)  
    
    def generate_company_profiles(self):
        """Generate synthetic company profiles for ML training"""
        print("🏢 Generating company profiles...")

        industries = ['Healthcare', 'Finance', 'Education', 'Retail', 'Technology', 'Government']
        company_sizes = ['Small', 'Medium', 'Large']

        # First, try to load extended company data for more realism
        try:
            extended_df = pd.read_csv('test_companies_extended.csv')
            print(f"📊 Loading {len(extended_df)} extended company profiles for enhanced realism")

            for _, row in extended_df.iterrows():
                profile = {
                    'company_id': len(self.company_profiles) + 1,
                    'company_name': row['company_name'],
                    'company_size': row['company_size'],
                    'industry': row['industry'],
                    'employee_count': row['employee_count'],
                    'data_sensitivity': row['data_sensitivity'],
                    'security_budget': row['security_budget'],
                    'annual_revenue': row.get('annual_revenue', 0),
                    'headquarters_location': row.get('headquarters_location', ''),
                    'compliance_certifications': row.get('compliance_certifications', ''),
                    'remote_workers_percentage': row.get('remote_workers_percentage', 0),
                    'last_security_audit': row.get('last_security_audit', ''),
                    'has_breach_history': False  # Will be set based on breach data
                }
                self.company_profiles.append(profile)
        except FileNotFoundError:
            print("⚠️ Extended company data not found, using synthetic generation only")
        except Exception as e:
            print(f"⚠️ Error loading extended data: {e}, using synthetic generation only")

        # Generate additional synthetic profiles to reach target
        target_profiles = 500
        current_count = len(self.company_profiles)

        for i in range(target_profiles - current_count):
            profile = {
                'company_id': current_count + i + 1,
                'company_size': company_sizes[i % 3],
                'industry': industries[i % len(industries)],
                'employee_count': [50, 500, 5000][i % 3],
                'data_sensitivity': (i % 3) + 1,
                'security_budget': [10000, 100000, 1000000][i % 3],
                'has_breach_history': i % 4 == 0
            }
            self.company_profiles.append(profile)

        print(f"Generated {len(self.company_profiles)} company profiles")
        return self.company_profiles
    
    def save_data(self):
       
        breach_df = pd.DataFrame(self.breach_data)
        company_df = pd.DataFrame(self.company_profiles)
        
        breach_df.to_csv('breach_data.csv', index=False)
        company_df.to_csv('company_profiles.csv', index=False)
        
        print("💾 Data saved successfully!")
        return breach_df, company_df


if __name__ == "__main__":
    collector = AdvancedDataCollector()
    
    if collector.collect_breach_data():
        collector.generate_company_profiles()
        breach_df, company_df = collector.save_data()
        
        print(f"\n DAY 1 COMPLETED!")
        print(f" Breach Records: {len(breach_df)}")
        print(f" Company Profiles: {len(company_df)}")
        print(f" Sample Risk Scores calculated")
        print(f"Data saved for ML training")
    else:
        print(" Data collection failed - using synthetic data only")
        collector.generate_company_profiles()
        collector.save_data()