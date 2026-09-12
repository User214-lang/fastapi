from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "monthly_fee": 29.99,
                    "usage_hours": 120.5,
                    "support_requests": 2,
                    "account_age_months": 15,
                    "failed_payments": 0,
                    "region": "europe",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1
                }
            ]
        }
    }

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    churn_prediction: int
    probability: List[float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn_prediction": 0,
                    "probability": [0.9980580756384752, 0.0019419243615249005]
                }
            ]
        }
    }

class TrainingConfigChurn(BaseModel):
    model_type: str
    hyperparameters: dict

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class ModelMetrics(BaseModel):
    accuracy: float
    f1: float
    roc_auc: Optional[float] = None

class TrainingRecord(BaseModel):
    trained_at: datetime
    model_type: str
    hyperparameters: dict
    metrics: ModelMetrics
