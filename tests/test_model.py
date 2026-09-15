import json
import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from dataset import prepare_data
from model import (
    create_model,
    build_model,
    train_churn_model,
    save_churn_model,
    load_churn_model,
    append_training_record,
    load_training_history,
    CHURN_MODEL_PATH,
)

def test_create_model_logreg():
    model = create_model("logreg", {})
    assert isinstance(model, LogisticRegression)


def test_create_model_random_forest():
    model = create_model("random_forest", {})
    assert isinstance(model, RandomForestClassifier)


def test_create_model_applies_hyperparameters():
    model = create_model("random_forest", {"n_estimators": 50, "max_depth": 4})
    assert model.n_estimators == 50
    assert model.max_depth == 4


def test_create_model_unknown_type_raises():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        create_model("unknown_model", {})

def test_build_model_returns_pipeline():
    pipeline = build_model("logreg", {})
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

def test_build_model_classifier_type_matches():
    pipeline = build_model("random_forest", {})
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)

    pipeline = build_model("logreg", {})
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)

def test_train_churn_model(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    pipeline = train_churn_model(X, y, "logreg", {})
    assert isinstance(pipeline, Pipeline)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)

def test_train_churn_model_predict_proba_shape(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    pipeline = train_churn_model(X, y, "logreg", {})
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)

def test_save_and_load_model_roundtrip(sample_df, tmp_path):
    X, y, _, _ = prepare_data(sample_df)
    pipeline = train_churn_model(X, y, "logreg", {})

    path = str(tmp_path / "model.joblib")
    metrics = {"accuracy": 0.9, "f1": 0.5, "roc_auc": 0.8}
    save_churn_model(pipeline, path=path, metrics=metrics,
                     training_time="2026-09-15T12:00:00",
                     model_type="logreg", hyperparameters={"C": 1.0})

    model, m, t, mt, hp = load_churn_model(path)
    assert model is not None
    assert m == metrics
    assert t == "2026-09-15T12:00:00"
    assert mt == "logreg"
    assert hp == {"C": 1.0}


def test_load_churn_model_missing_returns_nones(tmp_path):
    missing = str(tmp_path / "no_model.joblib")
    result = load_churn_model(missing)
    assert result == (None, None, None, None, None)


def test_save_churn_model_creates_meta_json(sample_df, tmp_path):
    X, y, _, _ = prepare_data(sample_df)
    pipeline = train_churn_model(X, y, "logreg", {})

    path = str(tmp_path / "model.joblib")
    save_churn_model(pipeline, path=path, metrics={"accuracy": 1.0},
                     model_type="logreg")

    meta_path = path.replace(".joblib", "_meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["metrics"] == {"accuracy": 1.0}
    assert meta["model_type"] == "logreg"

def test_append_and_load_history(tmp_path):
    path = str(tmp_path / "history.json")
    record = {
        "trained_at": "2026-09-15T12:00:00",
        "model_type": "logreg",
        "hyperparameters": {"C": 1.0},
        "metrics": {"accuracy": 0.9, "f1": 0.5, "roc_auc": 0.8},
    }
    append_training_record(record, path=path)

    history = load_training_history(path)
    assert len(history) == 1
    assert history[0] == record


def test_append_multiple_records(tmp_path):
    path = str(tmp_path / "history.json")
    for i in range(3):
        append_training_record(
            {"trained_at": f"2026-09-15T12:0{i}:00", "model_type": "logreg",
             "hyperparameters": {}, "metrics": {"accuracy": 0.9}},
            path=path,
        )
    history = load_training_history(path)
    assert len(history) == 3


def test_load_history_missing_file_returns_empty(tmp_path):
    missing = str(tmp_path / "no_history.json")
    assert load_training_history(missing) == []
