DATASET_PATH = "data/churn_dataset.csv"
CHURN_MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/churn_model_meta.json"
HISTORY_PATH = "models/training_history.json"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

ALL_FEATURES_ORDER = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]

TARGET_COLUMN = "churn"
