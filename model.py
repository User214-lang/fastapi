import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from typing import Tuple
from dataset import load_validated_dataset, prepare_data, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
#import joblib

def build_model() -> Pipeline:
    """Создание пайплайна"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]
    )
    return pipeline


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    return pipeline

def train_model(data_path: str = "data/churn_dataset.csv") -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_validated_dataset(data_path)
    X, y, _, _ = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model = train_churn_model(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
