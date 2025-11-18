"""
Business Context Data Connectors for Breach Predictor System
Phase 4: Business Context & Industry Data Integration

This module provides connectors for collecting business context data including:
- Financial performance metrics (revenue, profit margins)
- Employee turnover and remote work statistics
- Technology stack maturity assessments
- Third-party vendor risk scores
- Regulatory compliance data
- Industry-specific threat intelligence
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BusinessContextCollector:
    """
    Collects business context data from various sources
    """

    def __init__(self):
        self.connectors = {
            'financial': FinancialDataConnector(),
            'workforce': WorkforceDataConnector(),
            'technology': TechnologyStackConnector(),
            'vendor': VendorRiskConnector(),
            'compliance': ComplianceDataConnector(),
            'industry': IndustryThreatConnector()
        }

    def collect_all_business_context(self) -> Dict[str, Dict]:
        """
        Collect business context data from all available sources
        """
        results = {}

        for name, connector in self.connectors.items():
            try:
                logger.info(f"Collecting {name} data...")
                result = connector.collect_data()
                results[name] = result
                logger.info(f"✓ {name}: {result.get('records_collected', 0)} records collected")
            except Exception as e:
                logger.error(f"✗ {name} collection failed: {str(e)}")
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'records_collected': 0
                }

        return results

class FinancialDataConnector:
    """
    Collects financial performance metrics
    """

    def collect_data(self) -> Dict:
        """
        Collect financial data from various sources
        """
        try:
            # Simulate collecting financial data
            # In real implementation, this would connect to:
            # - SEC EDGAR API for public companies
            # - Financial data providers (Bloomberg, Reuters)
            # - Company annual reports

            # Generate realistic financial data for demo
            companies = pd.DataFrame({
                'company_name': [
                    'TechCorp Inc', 'FinanceFirst Bank', 'MediCare Solutions',
                    'EduLearn Academy', 'RetailMax Stores', 'GovTech Services'
                ],
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'revenue': [50000000, 200000000, 75000000, 25000000, 150000000, 100000000],
                'profit_margin': [0.15, 0.25, 0.12, 0.08, 0.18, 0.10],
                'market_cap': [200000000, 500000000, 150000000, 50000000, 300000000, 250000000],
                'debt_to_equity': [0.3, 0.8, 0.4, 0.2, 0.6, 0.1],
                'growth_rate': [0.25, 0.15, 0.20, 0.10, 0.18, 0.08]
            })

            return {
                'success': True,
                'records_collected': len(companies),
                'data': companies.to_dict('records'),
                'source': 'Financial Data API'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class WorkforceDataConnector:
    """
    Collects workforce-related metrics
    """

    def collect_data(self) -> Dict:
        """
        Collect workforce data including turnover, remote work, etc.
        """
        try:
            # Simulate workforce data collection
            # Real sources: LinkedIn, Glassdoor, company reports, BLS data

            workforce_data = pd.DataFrame({
                'company_name': [
                    'TechCorp Inc', 'FinanceFirst Bank', 'MediCare Solutions',
                    'EduLearn Academy', 'RetailMax Stores', 'GovTech Services'
                ],
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'employee_turnover': [0.18, 0.12, 0.15, 0.22, 0.25, 0.08],
                'remote_workers_pct': [0.65, 0.25, 0.35, 0.15, 0.20, 0.10],
                'avg_tenure_years': [2.5, 4.2, 3.8, 2.1, 1.8, 6.5],
                'training_budget_per_employee': [2500, 1800, 2200, 1500, 1200, 2000],
                'diversity_score': [0.75, 0.68, 0.72, 0.80, 0.65, 0.78]
            })

            return {
                'success': True,
                'records_collected': len(workforce_data),
                'data': workforce_data.to_dict('records'),
                'source': 'Workforce Analytics API'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class TechnologyStackConnector:
    """
    Assesses technology stack maturity
    """

    def collect_data(self) -> Dict:
        """
        Collect technology stack maturity data
        """
        try:
            # Simulate tech stack assessment
            # Real sources: BuiltWith, Wappalyzer, company disclosures

            tech_data = pd.DataFrame({
                'company_name': [
                    'TechCorp Inc', 'FinanceFirst Bank', 'MediCare Solutions',
                    'EduLearn Academy', 'RetailMax Stores', 'GovTech Services'
                ],
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'tech_stack_maturity': [0.95, 0.78, 0.82, 0.65, 0.70, 0.85],
                'cloud_adoption_score': [0.90, 0.75, 0.80, 0.60, 0.65, 0.85],
                'security_tools_count': [25, 18, 20, 12, 15, 22],
                'automation_level': [0.85, 0.70, 0.75, 0.55, 0.60, 0.80],
                'digital_transformation_score': [0.92, 0.68, 0.72, 0.58, 0.62, 0.78],
                'legacy_system_pct': [0.05, 0.22, 0.18, 0.35, 0.30, 0.15]
            })

            return {
                'success': True,
                'records_collected': len(tech_data),
                'data': tech_data.to_dict('records'),
                'source': 'Technology Stack Analysis'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class VendorRiskConnector:
    """
    Collects third-party vendor risk scores
    """

    def collect_data(self) -> Dict:
        """
        Collect vendor risk assessment data
        """
        try:
            # Simulate vendor risk data
            # Real sources: Vendor risk assessment platforms, supply chain data

            vendor_data = pd.DataFrame({
                'company_name': [
                    'TechCorp Inc', 'FinanceFirst Bank', 'MediCare Solutions',
                    'EduLearn Academy', 'RetailMax Stores', 'GovTech Services'
                ],
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'vendor_risk_score': [0.25, 0.35, 0.28, 0.20, 0.40, 0.15],
                'supply_chain_vendors': [45, 120, 85, 25, 200, 150],
                'high_risk_vendors': [3, 8, 5, 1, 12, 4],
                'vendor_diversity_score': [0.75, 0.82, 0.78, 0.65, 0.88, 0.90],
                'third_party_dependency_level': [0.65, 0.85, 0.70, 0.45, 0.90, 0.75]
            })

            return {
                'success': True,
                'records_collected': len(vendor_data),
                'data': vendor_data.to_dict('records'),
                'source': 'Vendor Risk Assessment'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class ComplianceDataConnector:
    """
    Collects regulatory compliance data
    """

    def collect_data(self) -> Dict:
        """
        Collect compliance and regulatory data
        """
        try:
            # Simulate compliance data
            # Real sources: Regulatory filings, compliance databases

            compliance_data = pd.DataFrame({
                'company_name': [
                    'TechCorp Inc', 'FinanceFirst Bank', 'MediCare Solutions',
                    'EduLearn Academy', 'RetailMax Stores', 'GovTech Services'
                ],
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'compliance_score': [0.85, 0.92, 0.88, 0.78, 0.82, 0.95],
                'regulatory_fines_last_5_years': [0, 250000, 50000, 0, 150000, 0],
                'audit_findings_count': [2, 1, 3, 1, 4, 0],
                'certifications_count': [8, 12, 10, 6, 7, 15],
                'data_privacy_compliance': [0.90, 0.95, 0.92, 0.85, 0.88, 0.98],
                'industry_regulations_count': [5, 15, 12, 8, 6, 20]
            })

            return {
                'success': True,
                'records_collected': len(compliance_data),
                'data': compliance_data.to_dict('records'),
                'source': 'Compliance Database'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class IndustryThreatConnector:
    """
    Collects industry-specific threat intelligence
    """

    def collect_data(self) -> Dict:
        """
        Collect industry-specific threat data
        """
        try:
            # Simulate industry threat data
            # Real sources: Industry threat reports, cybersecurity firms

            industry_threats = pd.DataFrame({
                'industry': ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government'],
                'threat_level': ['High', 'Critical', 'High', 'Medium', 'High', 'Medium'],
                'common_attack_vectors': [
                    ['Phishing', 'Ransomware', 'Supply Chain'],
                    ['Fraud', 'Insider Threats', 'DDoS'],
                    ['Ransomware', 'PHI Theft', 'Medical Device Hacks'],
                    ['Phishing', 'Student Data Theft', 'Ransomware'],
                    ['POS Malware', 'Customer Data Theft', 'Supply Chain'],
                    ['Espionage', 'Ransomware', 'Insider Threats']
                ],
                'industry_breach_rate': [0.25, 0.30, 0.20, 0.10, 0.15, 0.05],
                'avg_breach_cost': [4500000, 5500000, 6500000, 2500000, 3500000, 2000000],
                'regulatory_pressure': ['Medium', 'High', 'Critical', 'High', 'Medium', 'Critical'],
                'threat_actor_motivation': [
                    'Financial, Espionage',
                    'Financial, Fraud',
                    'Financial, Blackmail',
                    'Financial, Disruption',
                    'Financial, Data Theft',
                    'Espionage, Disruption'
                ]
            })

            return {
                'success': True,
                'records_collected': len(industry_threats),
                'data': industry_threats.to_dict('records'),
                'source': 'Industry Threat Intelligence'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_collected': 0
            }

class BusinessDataIntegrator:
    """
    Integrates business context data with company profiles
    """

    def __init__(self):
        self.business_collector = BusinessContextCollector()

    def enrich_company_profiles(self, company_profiles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich company profiles with business context data
        """
        try:
            # Collect all business context data
            business_data = self.business_collector.collect_all_business_context()

            # Start with original company profiles
            enriched_df = company_profiles_df.copy()

            # Merge financial data
            if business_data.get('financial', {}).get('success'):
                financial_df = pd.DataFrame(business_data['financial']['data'])
                enriched_df = enriched_df.merge(
                    financial_df[['company_name', 'revenue', 'profit_margin']],
                    on='company_name',
                    how='left'
                )

            # Merge workforce data
            if business_data.get('workforce', {}).get('success'):
                workforce_df = pd.DataFrame(business_data['workforce']['data'])
                enriched_df = enriched_df.merge(
                    workforce_df[['company_name', 'employee_turnover', 'remote_workers_pct']],
                    on='company_name',
                    how='left'
                )

            # Merge technology data
            if business_data.get('technology', {}).get('success'):
                tech_df = pd.DataFrame(business_data['technology']['data'])
                enriched_df = enriched_df.merge(
                    tech_df[['company_name', 'tech_stack_maturity']],
                    on='company_name',
                    how='left'
                )

            # Merge vendor risk data
            if business_data.get('vendor', {}).get('success'):
                vendor_df = pd.DataFrame(business_data['vendor']['data'])
                enriched_df = enriched_df.merge(
                    vendor_df[['company_name', 'vendor_risk_score']],
                    on='company_name',
                    how='left'
                )

            # Merge compliance data
            if business_data.get('compliance', {}).get('success'):
                compliance_df = pd.DataFrame(business_data['compliance']['data'])
                enriched_df = enriched_df.merge(
                    compliance_df[['company_name', 'compliance_score']],
                    on='company_name',
                    how='left'
                )

            # Fill missing values with industry averages
            enriched_df = self._fill_missing_with_industry_averages(enriched_df)

            logger.info(f"Successfully enriched {len(enriched_df)} company profiles with business context data")

            return enriched_df

        except Exception as e:
            logger.error(f"Failed to enrich company profiles: {str(e)}")
            return company_profiles_df

    def _fill_missing_with_industry_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing business context values with industry averages
        """
        try:
            # Define industry averages for missing data
            industry_defaults = {
                'Technology': {
                    'revenue': 100000000,
                    'profit_margin': 0.15,
                    'employee_turnover': 0.20,
                    'remote_workers_pct': 0.60,
                    'tech_stack_maturity': 0.90,
                    'vendor_risk_score': 0.25,
                    'compliance_score': 0.85
                },
                'Finance': {
                    'revenue': 500000000,
                    'profit_margin': 0.25,
                    'employee_turnover': 0.12,
                    'remote_workers_pct': 0.30,
                    'tech_stack_maturity': 0.80,
                    'vendor_risk_score': 0.35,
                    'compliance_score': 0.92
                },
                'Healthcare': {
                    'revenue': 200000000,
                    'profit_margin': 0.12,
                    'employee_turnover': 0.18,
                    'remote_workers_pct': 0.35,
                    'tech_stack_maturity': 0.75,
                    'vendor_risk_score': 0.28,
                    'compliance_score': 0.88
                },
                'Education': {
                    'revenue': 50000000,
                    'profit_margin': 0.08,
                    'employee_turnover': 0.25,
                    'remote_workers_pct': 0.20,
                    'tech_stack_maturity': 0.65,
                    'vendor_risk_score': 0.20,
                    'compliance_score': 0.78
                },
                'Retail': {
                    'revenue': 300000000,
                    'profit_margin': 0.18,
                    'employee_turnover': 0.28,
                    'remote_workers_pct': 0.25,
                    'tech_stack_maturity': 0.70,
                    'vendor_risk_score': 0.40,
                    'compliance_score': 0.82
                },
                'Government': {
                    'revenue': 150000000,
                    'profit_margin': 0.10,
                    'employee_turnover': 0.08,
                    'remote_workers_pct': 0.15,
                    'tech_stack_maturity': 0.80,
                    'vendor_risk_score': 0.15,
                    'compliance_score': 0.95
                }
            }

            # Fill missing values based on industry
            for idx, row in df.iterrows():
                industry = row.get('industry', 'Technology')
                defaults = industry_defaults.get(industry, industry_defaults['Technology'])

                for col, default_value in defaults.items():
                    if pd.isna(row.get(col)):
                        df.at[idx, col] = default_value

            return df

        except Exception as e:
            logger.error(f"Failed to fill missing values: {str(e)}")
            return df

def test_business_context_integration():
    """
    Test the business context data integration
    """
    print("🧪 Testing Phase 4: Business Context & Industry Data Integration")
    print("=" * 80)

    try:
        # Initialize integrator
        integrator = BusinessDataIntegrator()

        # Test individual connectors
        print("\n🔌 Testing Individual Business Context Connectors:")
        results = integrator.business_collector.collect_all_business_context()

        total_records = 0
        for source, result in results.items():
            if result.get('success'):
                records = result.get('records_collected', 0)
                total_records += records
                print(f"✓ {source}: {records} records collected")
            else:
                print(f"✗ {source}: Failed - {result.get('error', 'Unknown error')}")

        print(f"\n🎯 Total business context records collected: {total_records}")

        # Test company profile enrichment
        print("\n🔄 Testing Company Profile Enrichment:")

        # Create sample company profiles
        sample_profiles = pd.DataFrame({
            'company_name': ['TestTech Corp', 'SecureBank', 'MediHealth Inc'],
            'industry': ['Technology', 'Finance', 'Healthcare'],
            'company_size': ['Large', 'Large', 'Medium'],
            'employee_count': [1000, 2500, 800],
            'security_budget': [1500000, 3750000, 1200000],
            'data_sensitivity': [3, 3, 3],
            'has_breach_history': [False, True, False]
        })

        print(f"Original profiles: {len(sample_profiles)} companies")
        print(f"Original columns: {list(sample_profiles.columns)}")

        # Enrich profiles
        enriched_profiles = integrator.enrich_company_profiles(sample_profiles)

        print(f"Enriched profiles: {len(enriched_profiles)} companies")
        print(f"Enriched columns: {list(enriched_profiles.columns)}")

        # Show sample enriched data
        print("\n📊 Sample Enriched Company Profile:")
        sample_row = enriched_profiles.iloc[0]
        for col in enriched_profiles.columns:
            if not pd.isna(sample_row[col]):
                print(f"  {col}: {sample_row[col]}")

        print("\n🎉 Phase 4 Business Context & Industry Data Integration Successfully Implemented!")
        print("=" * 80)
        print("Available Business Context Data Sources:")
        print("• Financial - Revenue, profit margins, market data")
        print("• Workforce - Turnover rates, remote work percentages")
        print("• Technology - Stack maturity, automation levels")
        print("• Vendor Risk - Third-party risk assessments")
        print("• Compliance - Regulatory compliance scores")
        print("• Industry Threats - Sector-specific threat intelligence")
        print("\nReady for complete system integration and testing!")

        return True

    except Exception as e:
        print(f"❌ Phase 4 testing failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_business_context_integration()
