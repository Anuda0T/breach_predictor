import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
import requests
from datetime import datetime
import logging

class DataProcessor:
    """Advanced data processor for company breach risk analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Column name mappings for flexible validation
        self.column_mappings = {
            'company_name': ['company_name', 'company name', 'company', 'organization', 'org', 'name'],
            'industry': ['industry', 'sector', 'industry_sector', 'business_sector'],
            'company_size': ['company_size', 'size', 'company size', 'org_size', 'organization_size'],
            'data_sensitivity': ['data_sensitivity', 'sensitivity', 'data sensitivity', 'data_sens', 'sens_level'],
            'security_budget': ['security_budget', 'budget', 'security budget', 'sec_budget', 'cyber_budget'],
            'employee_count': ['employee_count', 'employees', 'employee count', 'emp_count', 'staff_count']
        }

        # Industry normalization mappings
        self.industry_normalization = {
            'tech': 'Technology', 'it': 'Technology', 'software': 'Technology',
            'fin': 'Finance', 'banking': 'Finance', 'financial': 'Finance',
            'health': 'Healthcare', 'medical': 'Healthcare', 'hospital': 'Healthcare',
            'edu': 'Education', 'school': 'Education', 'university': 'Education',
            'gov': 'Government', 'government': 'Government', 'public': 'Government',
            'retail': 'Retail', 'shopping': 'Retail', 'commerce': 'Retail'
        }

        # Size normalization mappings
        self.size_normalization = {
            'small': 'Small', 's': 'Small', '1-100': 'Small', 'small business': 'Small',
            'medium': 'Medium', 'm': 'Medium', '101-1000': 'Medium', 'mid-size': 'Medium',
            'large': 'Large', 'l': 'Large', '1000+': 'Large', 'enterprise': 'Large', 'big': 'Large'
        }

    def validate_and_map_columns(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, str], List[str]]:
        """Flexible column validation and mapping - now processes ALL columns"""
        warnings = []
        column_mapping = {}

        # Find matches for required columns
        for required_col, possible_names in self.column_mappings.items():
            found = False
            for col in df.columns:
                col_lower = col.lower().strip()
                if any(possible_name.lower() in col_lower or col_lower in possible_name.lower()
                      for possible_name in possible_names):
                    column_mapping[required_col] = col
                    found = True
                    break
            if not found:
                available_cols = list(df.columns)
                warnings.append(f"Could not find column for '{required_col}'. Expected one of: {possible_names}. Available columns in file: {available_cols}. Will analyze all available columns.")

        # Always return True to allow processing of all columns
        return True, column_mapping, warnings

    def preprocess_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive data preprocessing pipeline"""
        processed_df = df.copy()
        preprocessing_report = {
            'original_rows': len(df),
            'missing_values_handled': {},
            'outliers_detected': {},
            'data_quality_score': 0.0,
            'warnings': [],
            'transformations': []
        }

        # Rename columns to standard names
        processed_df = processed_df.rename(columns=column_mapping)

        # Handle missing values
        processed_df, missing_report = self._handle_missing_values(processed_df)
        preprocessing_report['missing_values_handled'] = missing_report

        # Clean and normalize text fields
        processed_df, text_report = self._clean_text_fields(processed_df)
        preprocessing_report['transformations'].extend(text_report)

        # Normalize categorical values
        processed_df, cat_report = self._normalize_categorical_values(processed_df)
        preprocessing_report['transformations'].extend(cat_report)

        # Handle outliers
        processed_df, outlier_report = self._handle_outliers(processed_df)
        preprocessing_report['outliers_detected'] = outlier_report

        # Validate data ranges and types
        validation_report = self._validate_data_ranges(processed_df)
        preprocessing_report['warnings'].extend(validation_report['warnings'])

        # Calculate data quality score
        preprocessing_report['data_quality_score'] = self._calculate_data_quality_score(processed_df)

        preprocessing_report['final_rows'] = len(processed_df)

        return processed_df, preprocessing_report

    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values with appropriate imputation strategies"""
        df_clean = df.copy()
        missing_report = {}

        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                if col in ['company_name']:
                    # Drop rows with missing company names
                    df_clean = df_clean.dropna(subset=[col])
                    missing_report[col] = f"Dropped {missing_count} rows with missing values"
                elif col in ['data_sensitivity', 'security_budget', 'employee_count']:
                    # Numeric imputation
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        missing_report[col] = f"Imputed {missing_count} missing values with median: {median_val}"
                    else:
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col] = df_clean[col].fillna(mode_val[0])
                            missing_report[col] = f"Imputed {missing_count} missing values with mode: {mode_val[0]}"
                elif col in ['industry', 'company_size']:
                    # Categorical imputation
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                        missing_report[col] = f"Imputed {missing_count} missing values with mode: {mode_val[0]}"
                else:
                    # For other columns, keep as is but log
                    missing_report[col] = f"Left {missing_count} missing values unchanged"

        return df_clean, missing_report

    def _clean_text_fields(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and standardize text fields"""
        df_clean = df.copy()
        transformations = []

        # Clean company names
        if 'company_name' in df_clean.columns:
            original_count = len(df_clean)
            df_clean['company_name'] = df_clean['company_name'].astype(str).str.strip()
            df_clean['company_name'] = df_clean['company_name'].str.title()
            transformations.append("Cleaned and title-cased company names")

        # Clean industry names
        if 'industry' in df_clean.columns:
            df_clean['industry'] = df_clean['industry'].astype(str).str.strip()
            df_clean['industry'] = df_clean['industry'].str.lower()
            transformations.append("Normalized industry names to lowercase")

        return df_clean, transformations

    def _normalize_categorical_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Normalize categorical values to standard formats"""
        df_norm = df.copy()
        transformations = []

        # Normalize industry
        if 'industry' in df_norm.columns:
            df_norm['industry'] = df_norm['industry'].map(self.industry_normalization).fillna(df_norm['industry'])
            transformations.append("Normalized industry categories")

        # Normalize company size
        if 'company_size' in df_norm.columns:
            df_norm['company_size'] = df_norm['company_size'].map(self.size_normalization).fillna(df_norm['company_size'])
            transformations.append("Normalized company size categories")

        return df_norm, transformations

    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers in numeric columns"""
        df_clean = df.copy()
        outlier_report = {}

        numeric_cols = ['security_budget', 'employee_count', 'data_sensitivity']

        for col in numeric_cols:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                # Use IQR method for outlier detection
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    outlier_report[col] = {
                        'count': int(outlier_count),
                        'percentage': round(outlier_count / len(df_clean) * 100, 2),
                        'bounds': [lower_bound, upper_bound]
                    }

                    # Cap outliers at bounds
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

        return df_clean, outlier_report

    def _validate_data_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data ranges and business rules"""
        warnings = []

        # Validate data sensitivity
        if 'data_sensitivity' in df.columns:
            invalid_sens = ~df['data_sensitivity'].between(1, 3)
            if invalid_sens.any():
                warnings.append(f"Found {invalid_sens.sum()} companies with invalid data sensitivity (should be 1-3)")

        # Validate security budget
        if 'security_budget' in df.columns:
            negative_budget = df['security_budget'] < 0
            if negative_budget.any():
                warnings.append(f"Found {negative_budget.sum()} companies with negative security budget")

        # Validate employee count
        if 'employee_count' in df.columns:
            invalid_emp = df['employee_count'] <= 0
            if invalid_emp.any():
                warnings.append(f"Found {invalid_emp.sum()} companies with invalid employee count (<=0)")

        return {'warnings': warnings}

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        score = 100.0

        # Completeness score (100 - percentage of missing values)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100
        score = min(score, completeness)

        # Validity score based on business rules
        validity_score = 100.0

        if 'data_sensitivity' in df.columns:
            valid_sens = df['data_sensitivity'].between(1, 3).sum()
            validity_score *= (valid_sens / len(df))

        if 'security_budget' in df.columns:
            valid_budget = (df['security_budget'] >= 0).sum()
            validity_score *= (valid_budget / len(df))

        if 'employee_count' in df.columns:
            valid_emp = (df['employee_count'] > 0).sum()
            validity_score *= (valid_emp / len(df))

        score = min(score, validity_score)

        return round(score, 2)

    def enhance_with_external_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhance data with external API calls (placeholder for real implementation)"""
        enhanced_df = df.copy()
        enhancement_report = {
            'api_calls_made': 0,
            'data_enhanced': 0,
            'errors': []
        }

        # Placeholder for external data enhancement
        # In a real implementation, this would call APIs like:
        # - Company size verification
        # - Industry classification
        # - Financial data
        # - Recent breach history

        try:
            # Example: Add current timestamp for "last_updated"
            enhanced_df['data_last_updated'] = datetime.now().strftime('%Y-%m-%d')
            enhancement_report['data_enhanced'] = len(enhanced_df)

        except Exception as e:
            enhancement_report['errors'].append(f"External data enhancement failed: {str(e)}")

        return enhanced_df, enhancement_report

    def generate_quality_dashboard_data(self, preprocessing_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for quality dashboard visualization"""
        return {
            'data_quality_score': preprocessing_report.get('data_quality_score', 0),
            'rows_processed': preprocessing_report.get('final_rows', 0),
            'missing_values_handled': len(preprocessing_report.get('missing_values_handled', {})),
            'outliers_detected': sum(len(v) if isinstance(v, dict) else 0 for v in preprocessing_report.get('outliers_detected', {}).values()),
            'warnings_count': len(preprocessing_report.get('warnings', [])),
            'transformations_count': len(preprocessing_report.get('transformations', []))
        }
