import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple
from dataset import load_validated_dataset, prepare_data, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
import joblib
import json, os
from datetime import datetime

CHURN_MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/churn_model_meta.json"

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

def train_model(data_path: str = "data/churn_dataset.csv"):
    df = load_validated_dataset(data_path)
    X, y, _, _ = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model = train_churn_model(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {"accuracy": accuracy, "f1": f1}
    training_time = datetime.now().isoformat()

    save_churn_model(model, metrics=metrics, training_time=training_time)

    return model, X_train, X_test, y_train, y_test

def save_churn_model(model, path: str = CHURN_MODEL_PATH, metrics: dict = None, training_time: str = None):
    joblib.dump(model, path)
    if metrics is not None or training_time is not None:
        meta = {}
        if metrics is not None:
            meta['metrics'] = metrics
        if training_time is not None:
            meta['training_time'] = training_time

        meta_path = path.replace('.joblib', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

def load_churn_model(path: str = CHURN_MODEL_PATH):
    if not os.path.exists(path):
        return None, None, None

    model = joblib.load(path)

    metrics = None
    training_time = None
    meta_path = path.replace('.joblib', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        metrics = meta.get('metrics')
        training_time = meta.get('training_time')

    return model, metrics, training_time




