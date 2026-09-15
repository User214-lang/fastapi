import pytest
from fastapi.testclient import TestClient
import model as model_module
from main import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "CHURN_MODEL_PATH",
                        str(tmp_path / "model.joblib"))
    monkeypatch.setattr(model_module, "HISTORY_PATH",
                        str(tmp_path / "history.json"))

    with TestClient(app) as client:
        yield client

def test_full_pipeline_train_status_predict(client):
    # При старте модель на обучена
    status = client.get("/model/status").json()
    assert status["is_trained"] is False

    # Обучение
    train_response = client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {"C": 0.5}},
    )
    assert train_response.status_code == 200
    metrics = train_response.json()
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Проверяем что модель обучена
    status = client.get("/model/status").json()
    assert status["is_trained"] is True
    assert status["model_type"] == "logreg"
    assert status["hyperparameters"] == {"C": 0.5}
    assert status["trained_at"] is not None
    assert status["metrics"] == metrics

    # Предсказание
    customer = {
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
    response = client.post("/predict", json=customer)
    assert response.status_code == 200

    prediction = response.json()
    assert prediction["churn_prediction"] in (0, 1)
    assert isinstance(prediction["probability"], list)
    assert len(prediction["probability"]) == 2
    assert abs(sum(prediction["probability"]) - 1.0) < 1e-6


def test_full_pipeline_list_of_customers(client):
    """Предсказание для конкретного списка клиентов"""

    client.post("/model/train",
                json={"model_type": "logreg", "hyperparameters": {}})

    customers = [
        {
            "monthly_fee": 29.99, "usage_hours": 120.5,
            "support_requests": 2, "account_age_months": 15,
            "failed_payments": 0, "region": "europe",
            "device_type": "mobile", "payment_method": "card",
            "autopay_enabled": 1,
        },
        {
            "monthly_fee": 89.99, "usage_hours": 12.5,
            "support_requests": 7, "account_age_months": 3,
            "failed_payments": 3, "region": "africa",
            "device_type": "desktop", "payment_method": "paypal",
            "autopay_enabled": 0,
        },
    ]
    response = client.post("/predict", json=customers)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert item["churn_prediction"] in (0, 1)
        assert len(item["probability"]) == 2


def test_predict_without_training_returns_503(client):
    """Без обучения /predict должен вернуть 503"""
    customer = {
        "monthly_fee": 29.99, "usage_hours": 120.5,
        "support_requests": 2, "account_age_months": 15,
        "failed_payments": 0, "region": "europe",
        "device_type": "mobile", "payment_method": "card",
        "autopay_enabled": 1,
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "SERVICE_UNAVAILABLE"


def test_train_then_retrain_updates_status(client):
    """Повторное обучение обновляет метаданные"""
    client.post("/model/train",
                json={"model_type": "logreg", "hyperparameters": {}})
    status1 = client.get("/model/status").json()

    client.post("/model/train",
                json={"model_type": "random_forest",
                      "hyperparameters": {"n_estimators": 50, "max_depth": 4}})
    status2 = client.get("/model/status").json()

    assert status2["model_type"] == "random_forest"
    assert status2["hyperparameters"] == {"n_estimators": 50, "max_depth": 4}
    assert status2["trained_at"] != status1["trained_at"]
