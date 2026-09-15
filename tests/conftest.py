import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as main_module
import model as model_module
from main import app

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "monthly_fee":        [20.0, 50.0, 80.0, 30.0, 90.0, 40.0, 25.0, 70.0, 15.0, 60.0,
                               55.0, 35.0, 85.0, 45.0, 75.0, 65.0, 10.0, 95.0, 22.0, 48.0],
        "usage_hours":        [100.0, 40.0, 10.0, 80.0, 5.0, 60.0, 95.0, 20.0, 120.0, 30.0,
                               70.0, 45.0, 8.0, 90.0, 15.0, 50.0, 130.0, 3.0, 110.0, 55.0],
        "support_requests":   [1, 3, 7, 2, 8, 4, 1, 6, 0, 5,
                               2, 3, 7, 1, 8, 4, 0, 9, 1, 3],
        "account_age_months": [30, 12, 3, 40, 2, 20, 35, 5, 50, 15,
                               25, 18, 4, 45, 6, 22, 60, 1, 38, 10],
        "failed_payments":    [0, 1, 3, 0, 4, 1, 0, 2, 0, 1,
                               1, 0, 3, 0, 4, 1, 0, 5, 0, 2],
        "region":             ["europe", "asia", "africa", "america", "africa", "asia", "europe",
                               "america", "europe", "asia",
                               "africa", "europe", "america", "asia", "africa", "europe",
                               "america", "africa", "europe", "asia"],
        "device_type":        ["mobile", "desktop", "tablet"] * 6 + ["mobile", "desktop"],
        "payment_method":     ["card", "paypal", "crypto"] * 6 + ["card", "paypal"],
        "autopay_enabled":    [1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                               0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        "churn":              [0, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                               0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """sample_df, сохранённый в CSV внутри tmp_path."""
    path = tmp_path / "churn.csv"
    sample_df.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "CHURN_MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(model_module, "HISTORY_PATH",      str(tmp_path / "history.json"))
    with TestClient(app) as client:
        yield client

@pytest.fixture
def valid_customer():
    """Корректный JSON для одного клиента"""
    return {
        "monthly_fee": 29.99,
        "usage_hours": 120.5,
        "support_requests": 2,
        "account_age_months": 15,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    }