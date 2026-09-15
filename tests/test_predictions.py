import random
from fastapi.testclient import TestClient
from main import app
import pytest


client = TestClient(app)

REGIONS = ["europe", "asia", "america", "africa"]
DEVICES = ["mobile", "desktop", "tablet"]
PAYMENT_METHODS = ["card", "paypal", "crypto"]

def generate_random_customer():
    return {
        "monthly_fee": round(random.uniform(10, 100), 2),
        "usage_hours": round(random.uniform(0, 200), 1),
        "support_requests": random.randint(0, 10),
        "account_age_months": random.randint(1, 60),
        "failed_payments": random.randint(0, 5),
        "region": random.choice(REGIONS),
        "device_type": random.choice(DEVICES),
        "payment_method": random.choice(PAYMENT_METHODS),
        "autopay_enabled": random.randint(0, 1)
    }

@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    status = client.get("/model/status").json()
    if not status.get("is_trained"):
        client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {}})

def test_random_customers():
    n=10
    print(f"Генерация {n} случайных клиентов...\n")
    for i in range(n):
        customer = generate_random_customer()
        response = client.post("/predict", json=customer)
        assert response.status_code == 200, response.text
        result = response.json()
        prob_churn = result["probability"][1]
        assert "churn_prediction" in result
        assert "probability" in result
        assert result["churn_prediction"] in (0, 1)
        assert 0.0 <= prob_churn <= 1.0
