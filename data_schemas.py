from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime

class CompanyProfile(BaseModel):
    company_id: str
    company_name: str
    industry: str
    company_size: str
    data_sensitivity_level: int
    security_budget: float
    employee_count: int

class BreachData(BaseModel):
    breach_id: str
    company_id: str
    breach_date: datetime
    severity: str
    records_exposed: int
    breach_type: str
    data_types_affected: List[str]

    @validator('data_types_affected')
    def validate_data_types(cls, v):
        allowed_types = {'Financial', 'PII', 'Credentials', 'Credit Card', 'Internal', 'Health', 'Corporate', 'Email', 'SSN'}
        if not isinstance(v, list):
            raise ValueError('data_types_affected must be a list')
        for item in v:
            if item not in allowed_types:
                raise ValueError(f'Invalid data type: {item}. Must be one of {allowed_types}')
        return v
