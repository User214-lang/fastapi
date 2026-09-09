import random
import requests

URL = "http://127.0.0.1:8000/predict"

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

def test_random_customers(n=10):
    print(f"Генерация {n} случайных клиентов...\n")
    for i in range(n):
        customer = generate_random_customer()
        response = requests.post(URL, json=customer)
        if response.status_code == 200:
            result = response.json()
            prob_churn = result["probability"][1]
            print(f"Клиент {i+1}: {customer}")
            print(f"  -> Прогноз: {result['churn_prediction']}, вероятность ухода: {prob_churn:.3f}")
            print("---")
        else:
            print(f"Ошибка для клиента {i+1}: {response.status_code} {response.text}\n")

if __name__ == "__main__":
    test_random_customers(20)
