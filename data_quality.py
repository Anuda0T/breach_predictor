"""
Data Quality Validation System
Phase 1: Data Architecture Foundation

Provides comprehensive data quality validation with configurable rules,
anomaly detection, and real-time quality monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from pathlib import Path
import re
from collections import defaultdict
import statistics

class QualityRuleType(Enum):
    """Types of quality validation rules"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    CUSTOM = "custom"

class QualitySeverity(Enum):
    """Quality issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityRule:
    """Data quality validation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: QualityRuleType
    severity: QualitySeverity
    column: Optional[str] = None  # None for table-level rules
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    validation_function: Optional[Callable] = None

@dataclass
class QualityIssue:
    """Data quality issue"""
    rule_id: str
    column: Optional[str]
    issue_type: str
    severity: QualitySeverity
    description: str
    affected_records: int
    percentage_affected: float
    sample_values: List[Any] = field(default_factory=list)
    suggested_fix: str = ""

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    dataset_id: str
    total_records: int
    quality_score: float
    issues: List[QualityIssue] = field(default_factory=list)
    rule_results: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0

class DataQualityValidator:
    """
    Comprehensive data quality validation system
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, QualityRule] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default quality validation rules"""

        # Completeness rules
        self.add_rule(QualityRule(
            rule_id="null_check",
            name="Null Value Check",
            description="Check for null/missing values in columns",
            rule_type=QualityRuleType.COMPLETENESS,
            severity=QualitySeverity.MEDIUM,
            validation_function=self._check_null_values
        ))

        # Accuracy rules
        self.add_rule(QualityRule(
            rule_id="data_type_check",
            name="Data Type Validation",
            description="Validate data types match expected schema",
            rule_type=QualityRuleType.ACCURACY,
            severity=QualitySeverity.HIGH,
            validation_function=self._check_data_types
        ))

        # Validity rules
        self.add_rule(QualityRule(
            rule_id="range_check",
            name="Value Range Validation",
            description="Check if numeric values are within expected ranges",
            rule_type=QualityRuleType.VALIDITY,
            severity=QualitySeverity.MEDIUM,
            parameters={"min_value": None, "max_value": None},
            validation_function=self._check_value_ranges
        ))

        self.add_rule(QualityRule(
            rule_id="pattern_check",
            name="Pattern Validation",
            description="Validate string values match expected patterns",
            rule_type=QualityRuleType.VALIDITY,
            severity=QualitySeverity.MEDIUM,
            parameters={"pattern": None},
            validation_function=self._check_patterns
        ))

        # Uniqueness rules
        self.add_rule(QualityRule(
            rule_id="duplicate_check",
            name="Duplicate Record Check",
            description="Check for duplicate records",
            rule_type=QualityRuleType.UNIQUENESS,
            severity=QualitySeverity.HIGH,
            validation_function=self._check_duplicates
        ))

        # Consistency rules
        self.add_rule(QualityRule(
            rule_id="format_consistency",
            name="Format Consistency",
            description="Check for consistent formatting across records",
            rule_type=QualityRuleType.CONSISTENCY,
            severity=QualitySeverity.LOW,
            validation_function=self._check_format_consistency
        ))

        # Timeliness rules
        self.add_rule(QualityRule(
            rule_id="date_validity",
            name="Date Validity",
            description="Check if dates are valid and not in future",
            rule_type=QualityRuleType.TIMELINESS,
            severity=QualitySeverity.MEDIUM,
            validation_function=self._check_date_validity
        ))

    def add_rule(self, rule: QualityRule):
        """Add a custom quality rule"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added quality rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """Remove a quality rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed quality rule: {rule_id}")

    def validate_dataset(self,
                        df: pd.DataFrame,
                        dataset_id: str,
                        custom_rules: List[QualityRule] = None) -> QualityReport:
        """
        Validate dataset quality using all enabled rules

        Args:
            df: DataFrame to validate
            dataset_id: Dataset identifier
            custom_rules: Additional custom rules to apply

        Returns:
            Comprehensive quality report
        """
        start_time = datetime.now()
        issues = []
        rule_results = {}

        # Combine default and custom rules
        all_rules = list(self.rules.values())
        if custom_rules:
            all_rules.extend(custom_rules)

        # Filter enabled rules
        enabled_rules = [rule for rule in all_rules if rule.enabled]

        # Execute each rule
        for rule in enabled_rules:
            try:
                rule_issues = self._execute_rule(rule, df)
                issues.extend(rule_issues)
                rule_results[rule.rule_id] = {
                    'success': True,
                    'issues_found': len(rule_issues),
                    'execution_time': 0.0  # Could be measured
                }
            except Exception as e:
                self.logger.error(f"Error executing rule {rule.rule_id}: {str(e)}")
                rule_results[rule.rule_id] = {
                    'success': False,
                    'error': str(e),
                    'issues_found': 0
                }

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(df, issues)

        execution_time = (datetime.now() - start_time).total_seconds()

        return QualityReport(
            dataset_id=dataset_id,
            total_records=len(df),
            quality_score=quality_score,
            issues=issues,
            rule_results=rule_results,
            execution_time=execution_time
        )

    def _execute_rule(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Execute a single quality rule"""
        if rule.validation_function:
            return rule.validation_function(rule, df)
        else:
            self.logger.warning(f"No validation function for rule: {rule.rule_id}")
            return []

    def _check_null_values(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for null/missing values"""
        issues = []

        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                percentage = (null_count / len(df)) * 100
                severity = QualitySeverity.HIGH if percentage > 20 else QualitySeverity.MEDIUM

                # Get sample of null values context
                sample_values = []
                if null_count > 0:
                    non_null_sample = df[column].dropna().head(3).tolist()
                    sample_values = non_null_sample

                issues.append(QualityIssue(
                    rule_id=rule.rule_id,
                    column=column,
                    issue_type="null_values",
                    severity=severity,
                    description=f"Found {null_count} null values ({percentage:.1f}%)",
                    affected_records=null_count,
                    percentage_affected=percentage,
                    sample_values=sample_values,
                    suggested_fix="Consider imputation or removal of null values"
                ))

        return issues

    def _check_data_types(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check data type consistency"""
        issues = []

        for column in df.columns:
            # Try to infer expected type from data
            sample = df[column].dropna().head(100)

            if len(sample) == 0:
                continue

            # Check for mixed types in object columns
            if df[column].dtype == 'object':
                type_counts = sample.apply(type).value_counts()
                if len(type_counts) > 1:
                    most_common_type = type_counts.index[0]
                    inconsistent_count = len(sample) - type_counts[most_common_type]
                    percentage = (inconsistent_count / len(sample)) * 100

                    if percentage > 10:  # More than 10% inconsistent
                        issues.append(QualityIssue(
                            rule_id=rule.rule_id,
                            column=column,
                            issue_type="mixed_data_types",
                            severity=QualitySeverity.MEDIUM,
                            description=f"Mixed data types detected ({percentage:.1f}% inconsistent)",
                            affected_records=inconsistent_count,
                            percentage_affected=percentage,
                            sample_values=[str(t) for t in type_counts.index[:3]],
                            suggested_fix="Standardize data types or split into separate columns"
                        ))

        return issues

    def _check_value_ranges(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check if numeric values are within expected ranges"""
        issues = []

        min_val = rule.parameters.get('min_value')
        max_val = rule.parameters.get('max_value')

        if min_val is None and max_val is None:
            return issues  # No range specified

        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            values = df[column].dropna()

            if len(values) == 0:
                continue

            out_of_range = pd.Series(dtype=bool)

            if min_val is not None:
                out_of_range = out_of_range | (values < min_val)

            if max_val is not None:
                out_of_range = out_of_range | (values > max_val)

            if not out_of_range.empty:
                out_of_range_count = out_of_range.sum()
                if out_of_range_count > 0:
                    percentage = (out_of_range_count / len(values)) * 100

                    # Get sample out-of-range values
                    sample_values = values[out_of_range].head(5).tolist()

                    issues.append(QualityIssue(
                        rule_id=rule.rule_id,
                        column=column,
                        issue_type="out_of_range",
                        severity=QualitySeverity.MEDIUM,
                        description=f"Found {out_of_range_count} values outside range [{min_val}, {max_val}] ({percentage:.1f}%)",
                        affected_records=out_of_range_count,
                        percentage_affected=percentage,
                        sample_values=sample_values,
                        suggested_fix=f"Clamp values to range [{min_val}, {max_val}] or investigate data source"
                    ))

        return issues

    def _check_patterns(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check if string values match expected patterns"""
        issues = []

        pattern = rule.parameters.get('pattern')
        if not pattern:
            return issues

        try:
            compiled_pattern = re.compile(pattern)
        except re.error:
            self.logger.error(f"Invalid regex pattern: {pattern}")
            return issues

        for column in df.columns:
            if df[column].dtype != 'object':
                continue

            values = df[column].dropna().astype(str)

            if len(values) == 0:
                continue

            matches = values.str.match(compiled_pattern)
            non_matching = ~matches
            non_matching_count = non_matching.sum()

            if non_matching_count > 0:
                percentage = (non_matching_count / len(values)) * 100

                # Get sample non-matching values
                sample_values = values[non_matching].head(5).tolist()

                issues.append(QualityIssue(
                    rule_id=rule.rule_id,
                    column=column,
                    issue_type="pattern_mismatch",
                    severity=QualitySeverity.MEDIUM,
                    description=f"Found {non_matching_count} values not matching pattern ({percentage:.1f}%)",
                    affected_records=non_matching_count,
                    percentage_affected=percentage,
                    sample_values=sample_values,
                    suggested_fix="Clean or standardize string formats"
                ))

        return issues

    def _check_duplicates(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for duplicate records"""
        issues = []

        # Check for complete row duplicates
        duplicate_rows = df.duplicated()
        duplicate_count = duplicate_rows.sum()

        if duplicate_count > 0:
            percentage = (duplicate_count / len(df)) * 100

            severity = QualitySeverity.CRITICAL if percentage > 5 else QualitySeverity.HIGH

            issues.append(QualityIssue(
                rule_id=rule.rule_id,
                column=None,  # Table-level issue
                issue_type="duplicate_rows",
                severity=severity,
                description=f"Found {duplicate_count} duplicate rows ({percentage:.1f}%)",
                affected_records=duplicate_count,
                percentage_affected=percentage,
                sample_values=[],  # Could add sample duplicate rows
                suggested_fix="Remove duplicate records or investigate data source"
            ))

        # Check for duplicate values in key columns (heuristic)
        for column in df.columns:
            if df[column].dtype == 'object' or column.lower() in ['id', 'key', 'name']:
                value_counts = df[column].value_counts()
                duplicates = value_counts[value_counts > 1]
                if len(duplicates) > 0:
                    total_duplicates = duplicates.sum() - len(duplicates)  # Subtract unique values
                    percentage = (total_duplicates / len(df)) * 100

                    if percentage > 10:  # More than 10% duplicates
                        issues.append(QualityIssue(
                            rule_id=rule.rule_id,
                            column=column,
                            issue_type="duplicate_values",
                            severity=QualitySeverity.LOW,
                            description=f"High duplication in {column}: {len(duplicates)} values repeated",
                            affected_records=total_duplicates,
                            percentage_affected=percentage,
                            sample_values=duplicates.head(3).index.tolist(),
                            suggested_fix="Review data collection process"
                        ))

        return issues

    def _check_format_consistency(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for format consistency in string columns"""
        issues = []

        for column in df.columns:
            if df[column].dtype != 'object':
                continue

            values = df[column].dropna().astype(str)

            if len(values) == 0:
                continue

            # Check for inconsistent case patterns
            case_patterns = values.str.match(r'^[A-Z]')  # Starts with capital
            title_case_count = case_patterns.sum()
            total_count = len(values)

            if total_count > 10:  # Only check for larger datasets
                consistency_ratio = title_case_count / total_count

                if consistency_ratio < 0.8 and consistency_ratio > 0.2:  # Mixed case
                    issues.append(QualityIssue(
                        rule_id=rule.rule_id,
                        column=column,
                        issue_type="inconsistent_case",
                        severity=QualitySeverity.LOW,
                        description=f"Inconsistent capitalization ({consistency_ratio:.1f} ratio)",
                        affected_records=int(total_count * (1 - consistency_ratio)),
                        percentage_affected=(1 - consistency_ratio) * 100,
                        sample_values=values.head(3).tolist(),
                        suggested_fix="Standardize text case formatting"
                    ))

        return issues

    def _check_date_validity(self, rule: QualityRule, df: pd.DataFrame) -> List[QualityIssue]:
        """Check date validity and timeliness"""
        issues = []

        for column in df.columns:
            # Try to identify date columns
            if df[column].dtype == 'object':
                sample_values = df[column].dropna().head(10).astype(str)

                # Check if values look like dates
                date_like = 0
                for val in sample_values:
                    try:
                        pd.to_datetime(val)
                        date_like += 1
                    except (ValueError, TypeError):
                        pass

                if date_like >= 3:  # At least 3 date-like values
                    # Try to convert column to datetime
                    try:
                        date_series = pd.to_datetime(df[column], errors='coerce')
                        valid_dates = date_series.dropna()

                        if len(valid_dates) < len(df[column]):
                            invalid_count = len(df[column]) - len(valid_dates)
                            percentage = (invalid_count / len(df[column])) * 100

                            issues.append(QualityIssue(
                                rule_id=rule.rule_id,
                                column=column,
                                issue_type="invalid_dates",
                                severity=QualitySeverity.MEDIUM,
                                description=f"Found {invalid_count} invalid date values ({percentage:.1f}%)",
                                affected_records=invalid_count,
                                percentage_affected=percentage,
                                sample_values=df[column][date_series.isnull()].head(3).tolist(),
                                suggested_fix="Fix date formats or remove invalid dates"
                            ))

                        # Check for future dates
                        future_dates = valid_dates[valid_dates > datetime.now()]
                        if len(future_dates) > 0:
                            percentage = (len(future_dates) / len(valid_dates)) * 100

                            issues.append(QualityIssue(
                                rule_id=rule.rule_id,
                                column=column,
                                issue_type="future_dates",
                                severity=QualitySeverity.MEDIUM,
                                description=f"Found {len(future_dates)} future dates ({percentage:.1f}%)",
                                affected_records=len(future_dates),
                                percentage_affected=percentage,
                                sample_values=future_dates.head(3).dt.strftime('%Y-%m-%d').tolist(),
                                suggested_fix="Verify date data source or remove future dates"
                            ))

                    except Exception as e:
                        self.logger.debug(f"Could not process dates in column {column}: {str(e)}")

        return issues

    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score"""
        base_score = 100.0

        # Completeness penalty
        null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        base_score -= null_percentage * 30  # Max 30 points for completeness

        # Issue-based penalties
        severity_weights = {
            QualitySeverity.LOW: 1,
            QualitySeverity.MEDIUM: 3,
            QualitySeverity.HIGH: 5,
            QualitySeverity.CRITICAL: 10
        }

        total_penalty = 0
        for issue in issues:
            weight = severity_weights[issue.severity]
            penalty = min(issue.percentage_affected * weight / 10, 10)  # Cap per issue
            total_penalty += penalty

        base_score -= min(total_penalty, 40)  # Max 40 points for issues

        return max(0.0, round(base_score, 2))

    def detect_anomalies(self, df: pd.DataFrame, historical_data: List[pd.DataFrame] = None) -> List[QualityIssue]:
        """
        Detect statistical anomalies in data compared to historical patterns

        Args:
            df: Current dataset
            historical_data: List of historical datasets for comparison

        Returns:
            List of anomaly issues
        """
        issues = []

        if not historical_data:
            return issues

        # Combine historical data
        try:
            historical_combined = pd.concat(historical_data, ignore_index=True)

            for column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    continue

                current_values = df[column].dropna()
                historical_values = historical_combined[column].dropna()

                if len(current_values) < 10 or len(historical_values) < 10:
                    continue

                # Calculate statistics
                current_mean = current_values.mean()
                historical_mean = historical_values.mean()
                historical_std = historical_values.std()

                if historical_std == 0:
                    continue

                # Check for significant deviation (more than 3 standard deviations)
                deviation = abs(current_mean - historical_mean) / historical_std

                if deviation > 3:
                    issues.append(QualityIssue(
                        rule_id="statistical_anomaly",
                        column=column,
                        issue_type="statistical_anomaly",
                        severity=QualitySeverity.HIGH,
                        description=f"Statistical anomaly detected: {deviation:.1f} std deviations from historical mean",
                        affected_records=len(current_values),
                        percentage_affected=100.0,
                        sample_values=[current_mean, historical_mean, historical_std],
                        suggested_fix="Investigate data source for anomalies"
                    ))

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")

        return issues

    def generate_quality_dashboard_data(self, report: QualityReport) -> Dict[str, Any]:
        """Generate data for quality monitoring dashboard"""
        return {
            'quality_score': report.quality_score,
            'total_records': report.total_records,
            'total_issues': len(report.issues),
            'issues_by_severity': {
                severity.value: len([i for i in report.issues if i.severity == severity])
                for severity in QualitySeverity
            },
            'issues_by_type': {
                issue_type: len([i for i in report.issues if i.issue_type == issue_type])
                for issue_type in set(i.issue_type for i in report.issues)
            },
            'execution_time': report.execution_time,
            'generated_at': report.generated_at.isoformat()
        }

# Convenience functions
def create_quality_validator() -> DataQualityValidator:
    """Create a data quality validator with default rules"""
    return DataQualityValidator()

def validate_data_quality(df: pd.DataFrame, dataset_id: str) -> QualityReport:
    """Convenience function to validate data quality"""
    validator = DataQualityValidator()
    return validator.validate_dataset(df, dataset_id)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data with quality issues
    data = {
        'name': ['John', 'Jane', None, 'Bob', 'Alice'],
        'age': [25, 30, 35, -5, 40],  # Invalid negative age
        'email': ['john@example.com', 'jane@', 'bob@example.com', 'alice@example.com', 'charlie@invalid'],
        'score': [85, 92, 78, 95, 200],  # Score > 100
        'date': ['2023-01-01', '2023-02-01', 'invalid', '2023-04-01', '2025-01-01']  # Future date
    }

    df = pd.DataFrame(data)

    # Validate quality
    validator = DataQualityValidator()
    report = validator.validate_dataset(df, "sample_dataset")

    print(f"Quality Score: {report.quality_score}%")
    print(f"Total Issues: {len(report.issues)}")

    for issue in report.issues:
        print(f"- {issue.severity.value.upper()}: {issue.description} ({issue.column})")
