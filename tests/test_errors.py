import main as main_module

def test_503(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "SERVICE_UNAVAILABLE"
    assert "message" in body
    assert "details" in body

def test_predict_invalid_type_422(client, valid_customer):
    bad = valid_customer.copy()
    bad["monthly_fee"] = "not_a_number"
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert isinstance(body["details"], list)
    assert len(body["details"]) > 0


def test_predict_missing_field_422(client, valid_customer):
    bad = valid_customer.copy()
    del bad["monthly_fee"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"


def test_predict_extra_field_422(client, valid_customer):
    """Проверка на extra='forbid"""
    bad = valid_customer.copy()
    bad["extra_field"] = "boom"
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"


def test_train_empty_body_422(client):
    response = client.post("/model/train", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"

def test_missing_dataset_404(client, monkeypatch):
    monkeypatch.setattr(main_module, "DATASET_PATH",
                        "/nonexistent/path/churn.csv")
    response = client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "NOT_FOUND"

def test_train_with_empty_dataset_400(client, monkeypatch, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "monthly_fee,usage_hours,support_requests,account_age_months,"
        "failed_payments,region,device_type,payment_method,autopay_enabled,churn\n"
    )
    monkeypatch.setattr(main_module, "DATASET_PATH", str(empty))

    response = client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "DATA_ERROR"
    assert "empty" in body["message"].lower()

def test_200(client, monkeypatch, sample_csv,
                                            valid_customer):
    monkeypatch.setattr(main_module, "DATASET_PATH", sample_csv)
    client.post("/model/train",
                json={"model_type": "logreg", "hyperparameters": {}})

    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    body = response.json()
    assert "churn_prediction" in body
    assert "probability" in body
