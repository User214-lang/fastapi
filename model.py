import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Tuple
from dataset import load_validated_dataset, prepare_data, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
import joblib
import json, os
from datetime import datetime

CHURN_MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/churn_model_meta.json"
HISTORY_PATH = "models/training_history.json"

def build_model(model_type: str = "logreg", hyperparameters: dict = None) -> Pipeline:
    """Создание пайплайна"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = create_model(model_type, hyperparameters)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )
    return pipeline


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "logreg", hyperparameters: dict = None) -> Pipeline:
    pipeline = build_model(model_type, hyperparameters)
    pipeline.fit(X_train, y_train)
    return pipeline

def train_model(data_path: str = "data/churn_dataset.csv", model_type: str = "logreg", hyperparameters: dict = None):
    df = load_validated_dataset(data_path)
    X, y, _, _ = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model = train_churn_model(X_train, y_train, model_type, hyperparameters)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    training_time = datetime.now().isoformat()

    save_churn_model(model, metrics=metrics, training_time=training_time, 
                     model_type=model_type, hyperparameters=hyperparameters) #!!!

    return model, X_train, X_test, y_train, y_test

def save_churn_model(model, path: str = CHURN_MODEL_PATH, metrics: dict = None,
                     training_time: str = None, model_type: str = None,
                     hyperparameters: dict = None):
    joblib.dump(model, path)
    if metrics is not None or training_time is not None or model_type is not None or hyperparameters is not None:
        meta = {}
        if metrics is not None:
            meta['metrics'] = metrics
        if training_time is not None:
            meta['training_time'] = training_time
        if model_type is not None:
            meta['model_type'] = model_type
        if hyperparameters is not None:
            meta['hyperparameters'] = hyperparameters

        meta_path = path.replace('.joblib', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

def load_churn_model(path: str = CHURN_MODEL_PATH):
    if not os.path.exists(path):
        return None, None, None, None, None

    model = joblib.load(path)

    metrics = None
    training_time = None
    model_type = None
    hyperparameters = None

    meta_path = path.replace('.joblib', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        metrics = meta.get('metrics')
        training_time = meta.get('training_time')
        model_type = meta.get('model_type')
        hyperparameters = meta.get('hyperparameters')

    return model, metrics, training_time, model_type, hyperparameters

def create_model(model_type: str = "logreg", hyperparameters: dict = None):
    """
    Создаёт модель по заданному типу и гиперпараметрам"""
    if hyperparameters is None:
        hyperparameters = {}

    if model_type == "logreg":
        default_params = {"max_iter": 1000, "random_state": 42}
        default_params.update(hyperparameters)
        return LogisticRegression(**default_params)

    elif model_type == "random_forest":
        default_params = {"n_estimators": 100, "random_state": 42}
        default_params.update(hyperparameters)
        return RandomForestClassifier(**default_params)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def append_training_record(record: dict, path: str = HISTORY_PATH) -> None:
    """Добавляет запись в историю обучений модели(json)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            history = json.load(f)
    history.append(record)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def load_training_history(path: str = HISTORY_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)
