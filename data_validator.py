import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from data_schemas import CompanyProfile, BreachData

class DataValidator:
    def __init__(self):
        self.validation_stats = {
            'companies': {'valid': 0, 'invalid': 0, 'errors': []},
            'breaches': {'valid': 0, 'invalid': 0, 'errors': []}
        }
    
    def validate_company_profiles(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate company profiles data against schema"""
        valid_rows = []

        for idx, row in df.iterrows():
            try:
                # Convert row to dict and adapt to schema
                company_data = row.to_dict()
                company_data['company_id'] = str(company_data['company_id'])  # Convert to string
                company_data['company_name'] = f"Company_{company_data['company_id']}"
                company_data['data_sensitivity_level'] = company_data['data_sensitivity']
                del company_data['data_sensitivity']
                del company_data['has_breach_history']  # Remove field not in schema

                CompanyProfile(**company_data)
                valid_rows.append(idx)
                self.validation_stats['companies']['valid'] += 1
            except Exception as e:
                self.validation_stats['companies']['invalid'] += 1
                self.validation_stats['companies']['errors'].append({
                    'row': idx,
                    'data': row.to_dict(),
                    'error': str(e)
                })

        return df.loc[valid_rows], self.validation_stats['companies']
    
    def _map_data_types(self, data_classes_str: str) -> list[str]:
        """Map real breach data types to schema-compliant types"""
        # Parse the data classes string
        try:
            data_classes = json.loads(data_classes_str.replace("'", '"'))
        except:
            # Fallback for malformed strings
            data_classes = [item.strip().strip("'\"") for item in data_classes_str.strip('[]').split(',')]

        # Mapping from real breach data types to schema types
        type_mapping = {
            # Email related
            'Email addresses': 'Email',
            'Email messages': 'Email',

            # Personal Information
            'Names': 'PII',
            'Dates of birth': 'PII',
            'Phone numbers': 'PII',
            'Physical addresses': 'PII',
            'Geographic locations': 'PII',
            'IP addresses': 'PII',
            'Genders': 'PII',
            'Marital statuses': 'PII',
            'Nationalities': 'PII',
            'Spoken languages': 'PII',
            'Relationship statuses': 'PII',
            'Races': 'PII',
            'Salutations': 'PII',
            'Profile photos': 'PII',

            # Financial
            'Purchases': 'Financial',
            'Income levels': 'Financial',
            'Credit card numbers': 'Credit Card',
            'Bank account numbers': 'Financial',
            'Payment information': 'Financial',

            # Health
            'Health records': 'Health',
            'Medical data': 'Health',

            # Credentials
            'Passwords': 'Credentials',
            'Usernames': 'Credentials',
            'Password hints': 'Credentials',

            # Corporate/Internal
            'Job titles': 'Corporate',
            'Employers': 'Corporate',
            'Company data': 'Corporate',
            'Internal documents': 'Internal',
            'Private messages': 'Internal',
            'Website activity': 'Internal',

            # Device/Network
            'Device information': 'Internal',
            'Cellular network names': 'Internal',
            'IMEI numbers': 'Internal',
            'IMSI numbers': 'Internal',
            'Apps installed on devices': 'Internal',

            # Other
            'Cryptocurrency wallet addresses': 'Financial',
            'Social media profiles': 'PII',
            'Homepage URLs': 'PII',
            'Instant messenger identities': 'PII',
            'Partial dates of birth': 'PII',
            'Address book contacts': 'PII',
            'Education levels': 'PII',
            'Job applications': 'Corporate',
            'Religions': 'PII',
            'Sexual orientations': 'PII'
        }

        # Map each data class to schema type
        mapped_types = set()
        for data_class in data_classes:
            data_class = data_class.strip()
            if data_class in type_mapping:
                mapped_types.add(type_mapping[data_class])
            else:
                # Default to PII for unknown types
                mapped_types.add('PII')

        return list(mapped_types)

    def validate_breach_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate breach data against schema"""
        valid_rows = []

        for idx, row in df.iterrows():
            try:
                # Convert row to dict and adapt to schema
                breach_data = row.to_dict()

                # Map CSV columns to schema fields
                breach_data['breach_id'] = f"breach_{idx}"
                breach_data['company_id'] = str(idx % 100 + 1)  # Map to company IDs 1-100
                breach_data['severity'] = 'Critical' if breach_data['risk_score'] >= 5 else 'High' if breach_data['risk_score'] >= 4 else 'Medium' if breach_data['risk_score'] >= 3 else 'Low'
                breach_data['records_exposed'] = breach_data['records_affected']
                breach_data['breach_type'] = 'Data Breach'
                breach_data['data_types_affected'] = self._map_data_types(breach_data['data_classes'])

                # Convert string date to datetime
                if isinstance(breach_data['breach_date'], str):
                    breach_data['breach_date'] = datetime.strptime(
                        breach_data['breach_date'], '%Y-%m-%d'
                    )

                BreachData(**breach_data)
                valid_rows.append(idx)
                self.validation_stats['breaches']['valid'] += 1
            except Exception as e:
                self.validation_stats['breaches']['invalid'] += 1
                self.validation_stats['breaches']['errors'].append({
                    'row': idx,
                    'data': row.to_dict(),
                    'error': str(e)
                })

        return df.loc[valid_rows], self.validation_stats['breaches']
    
    def validate_data_files(
        self, 
        company_file: str = 'company_profiles.csv',
        breach_file: str = 'breach_data.csv'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Validate both data files and return clean dataframes"""
        
        # Validate company profiles
        company_df = pd.read_csv(company_file)
        valid_companies, company_stats = self.validate_company_profiles(company_df)
        
        # Validate breach data
        breach_df = pd.read_csv(breach_file)
        valid_breaches, breach_stats = self.validate_breach_data(breach_df)
        
        # Additional cross-validation - add company_id column to breaches
        valid_company_ids = set(valid_companies['company_id'])
        valid_breaches['company_id'] = valid_breaches.index.map(lambda x: str(x % 100 + 1))
        valid_breaches = valid_breaches[
            valid_breaches['company_id'].isin(valid_company_ids)
        ]
        
        return valid_companies, valid_breaches, self.validation_stats
    
    def save_validation_report(self, output_file: str = 'data_validation_report.json'):
        """Save validation statistics and errors to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.validation_stats, f, indent=2)
        print(f"Validation report saved to {output_file}")
        
if __name__ == "__main__":
    validator = DataValidator()
    
    try:
        companies, breaches, stats = validator.validate_data_files()
        print("\n=== Validation Results ===")
        print(f"Companies: {stats['companies']['valid']} valid, {stats['companies']['invalid']} invalid")
        print(f"Breaches: {stats['breaches']['valid']} valid, {stats['breaches']['invalid']} invalid")
        
        if stats['companies']['invalid'] > 0 or stats['breaches']['invalid'] > 0:
            print("\nSaving detailed validation report...")
            validator.save_validation_report()
        
    except Exception as e:
        print(f"Error during validation: {e}")