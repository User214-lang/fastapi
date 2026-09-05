import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from schemas import DatasetRowChurn

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled"
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method"
]

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def parse_dataset(df: pd.DataFrame) -> List[DatasetRowChurn]:
    rows = []
    for _, row in df.iterrows():
        obj = DatasetRowChurn(**row.to_dict())
        rows.append(obj)
    return rows


def load_validated_dataset(path: str) -> pd.DataFrame:
    """
    Загрузка CSV, датасет не пуст,
    валидация через pydantic, возвращает DataFrame.
    """

    df = load_dataset(path)
    if df.empty:
        raise ValueError("Dataset is empty (0 rows)")

    parse_dataset(df)
    return df

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Отделяет целевую переменную, обрабатывает пропуски"""

    if "churn" not in df.columns:
        raise ValueError("Колонка 'churn' не найдена в датасете")

    y = df["churn"]
    X = df.drop(columns=["churn"])

    numeric_features = NUMERIC_FEATURES.copy()
    categorical_features = CATEGORICAL_FEATURES.copy()

    for col in numeric_features:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_features:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, y, numeric_features, categorical_features

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_class_distribution(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Распределение классов churn в train/test выборках"""
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    train_dist = {int(k): int(v) for k, v in train_counts.items()}
    test_dist = {int(k): int(v) for k, v in test_counts.items()}

    return {
        "train": train_dist,
        "test": test_dist
    }
