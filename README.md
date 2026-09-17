# Churn Prediction Service

ML-сервис на FastAPI для предсказания оттока клиентов (churn).

---

## 1. Цель сервиса

Сервис предсказывает вероятность того, что клиент уйдёт из компании в следующем месяце.

| Понятие | Значение |
|---------|----------|
| Целевая переменная | `churn` |
| `churn = 0` | Клиент останется |
| `churn = 1` | Клиент уйдёт |
| Тип задачи | Бинарная классификация |
| Модели | LogisticRegression, RandomForest, CatBoost |
| Выход | Класс (`0`/`1`) и вероятности классов |

Сервис позволяет:

- обучить модель на подготовленном датасете;
- получить предсказание для одного клиента или группы клиентов;
- посмотреть метрики качества модели (`accuracy`, `f1`, `roc_auc`);
- вести историю всех обучений с параметрами и метриками.

---

## 2. Формат датасета `churn_dataset.csv`

Датасет находится в папке `data/`.

| Признак | Тип | Описание | Пример |
|---------|-----|----------|--------|
| `monthly_fee` | float | Ежемесячная стоимость тарифа | `29.99` |
| `usage_hours` | float | Часов использования за месяц | `120.5` |
| `support_requests` | int | Обращений в техподдержку | `2` |
| `account_age_months` | int | Возраст аккаунта в месяцах | `15` |
| `failed_payments` | int | Неудачных платежей | `0` |
| `region` | string | Регион: `europe`, `asia`, `america`, `africa` | `europe` |
| `device_type` | string | Устройство: `mobile`, `desktop`, `tablet` | `mobile` |
| `payment_method` | string | Способ оплаты: `card`, `paypal`, `crypto` | `card` |
| `autopay_enabled` | int | Автоплатёж: `0` — выключен, `1` — включён | `1` |
| `churn` | int | Целевая переменная: `0` или `1` | `0` |

**Пример:**

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
29.99,120.5,2,15,0,europe,mobile,card,1,0
89.99,12.5,7,3,3,africa,desktop,paypal,0,1
```

**Разделение признаков:**

| Тип | Признаки |
|-----|----------|
| Числовые | `monthly_fee`, `usage_hours`, `support_requests`, `account_age_months`, `failed_payments`, `autopay_enabled` |
| Категориальные | `region`, `device_type`, `payment_method` |
| Целевая | `churn` |

Предобработка:

- числовые признаки масштабируются через `StandardScaler`;
- категориальные кодируются через `OneHotEncoder`;
- пропуски в числовых заполняются медианой;
- пропуски в категориальных заполняются модой.

---

## 3. Запуск локально

### Шаг 1. Клонировать репозиторий

```bash
git clone git@github.com:User214-lang/fastapi.git
cd fastapi
```

### Шаг 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Шаг 3. Запустить сервер

```bash
uvicorn main:app --reload
```

Ожидаемый вывод:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Шаг 4. Открыть документацию

| Адрес | Назначение |
|-------|------------|
| `http://127.0.0.1:8000/docs` | Swagger UI, интерактивная документация |
| `http://127.0.0.1:8000/health` | Проверка состояния сервиса |
| `http://127.0.0.1:8000/` | Корневой эндпоинт |

### Шаг 5. Обучить модель

Первый запуск — модель не обучена. Нужно вызвать `/model/train` (см. [раздел 9](#9-примеры-запросов)).

---

## 5. Запуск в Docker

### Шаг 1. Собрать образ

```bash
docker build -t churn-service:latest .
```

### Шаг 2. Запустить контейнер

```bash
docker run -d \
    --name churn \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    churn-service:latest
```

### Шаг 3. Проверить работу

```bash
docker ps
curl http://localhost:8000/health
```

Ожидается:

```
CONTAINER ID   IMAGE                  STATUS
xxxxx          churn-service:latest   Up X seconds (healthy)
```

```json
{"status":"ok","model_loaded":true,"dataset_loaded":true}
```

### Шаг 4. Просмотр логов

```bash
docker logs churn
docker logs churn --tail 20
docker logs -f churn
```

---

## 6. Работа с сервисом

Типовой сценарий работы:

| Шаг | Действие | Эндпоинт |
|-----|----------|----------|
| 1 | Проверить, что сервис работает | `GET /health` |
| 2 | Проверить, обучена ли модель | `GET /model/status` |
| 3 | Обучить модель (если не обучена) | `POST /model/train` |
| 4 | Получить предсказание | `POST /predict` |
| 5 | Посмотреть метрики | `GET /model/metrics` |
| 6 | Посмотреть историю обучений | `GET /model/metrics` |

### Процесс обучения модели

1. Отправляется запрос `POST /model/train` с телом, где указан `model_type` и `hyperparameters`.
2. Сервер загружает датасет `data/churn_dataset.csv`.
3. Данные валидируются и разделяются на обучающую и тестовую выборки (80/20) со стратификацией по `churn`.
4. Создаётся пайплайн: `ColumnTransformer` (StandardScaler + OneHotEncoder) плюс классификатор.
5. Модель обучается на тренировочной части.
6. Вычисляются метрики на тестовой части: `accuracy`, `f1`, `roc_auc`.
7. Модель сохраняется в `models/churn_model.joblib`.
8. Метаданные (метрики, время, тип модели, гиперпараметры) сохраняются в `models/churn_model_meta.json`.
9. Запись об обучении добавляется в `models/training_history.json`.
10. В ответ возвращаются метрики.

### Процесс предсказания

1. Отправляется запрос `POST /predict` с признаками клиента (или списком клиентов).
2. Сервер проверяет, что модель обучена.
3. Данные преобразуются в таблицу и подаются в модель.
4. Модель возвращает класс (0/1) и вероятности `[P(класс 0), P(класс 1)]`.
5. Ответ возвращается клиенту.

---

## 7. Описание эндпоинтов

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/` | Проверка, что сервис запущен |
| GET | `/health` | Состояние: модель и датасет |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Предсказание для одного клиента или списка |
| POST | `/model/train` | Обучение модели |
| GET | `/model/status` | Статус текущей модели |
| GET | `/model/schema` | Список признаков и их типов |
| GET | `/model/metrics` | Метрики последнего обучения и история |
| GET | `/dataset/info` | Сводка о датасете |
| GET | `/dataset/preview` | Первые строки датасета |
| GET | `/dataset/split-info` | Размеры train/test и распределение классов |

---

## 8. Формат ответов и ошибок

### Успешный ответ `/predict`

```json
{
  "churn_prediction": 0,
  "probability": [0.9980580756384752, 0.0019419243615249005]
}
```

| Поле | Значение |
|------|----------|
| `churn_prediction` | Предсказанный класс: `0` или `1` |
| `probability[0]` | Вероятность того, что клиент останется |
| `probability[1]` | Вероятность того, что клиент уйдёт |

### Формат ошибок

Все ошибки возвращаются в json:

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Ошибка валидации данных",
  "details": null
}
```

| Поле | Значение |
|------|----------|
| `code` | Машинный код ошибки |
| `message` | Описание |
| `details` | Дополнительная информация (может быть `null`) |

### Возможные коды ошибок

| HTTP | `code` | Когда возникает |
|------|--------|-----------------|
| 400 | `DATA_ERROR` | Пустой датасет, отсутствует колонка `churn` |
| 404 | `NOT_FOUND` | Файл датасета не найден |
| 422 | `VALIDATION_ERROR` | Неверный тип поля, пропущено поле, лишнее поле |
| 500 | `INTERNAL_ERROR` | Непредвиденная ошибка |
| 503 | `SERVICE_UNAVAILABLE` | Модель ещё не обучена |

---

## 9. Примеры запросов

### 9.1 Обучение модели

**Запрос к model/train:**

**Поддерживаемые `model_type`:**

| Значение | Модель |
|----------|--------|
| `logreg` | LogisticRegression |
| `random_forest` | RandomForestClassifier |
| `catboost` | CatBoostClassifier |

**logreg**
```bash
curl -X POST http://127.0.0.1:8000/model/train \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "logreg",
       "hyperparameters": {"C": 0.5}
     }'
```

**random forest**

```bash
curl -X POST http://127.0.0.1:8000/model/train \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "random_forest",
       "hyperparameters": {"n_estimators": 100, "max_depth": 6}
     }'
```

**Ожидаемый ответ(logreg):**

```json
{
  "accuracy": 0.7875,
  "f1": 0.0449438202247191,
  "roc_auc": 0.6091179999225975
}
```

| Метрика | Что показывает |
|---------|---------------|
| `accuracy` | Доля правильных ответов |
| `f1` | Баланс точности и полноты по классу `churn = 1` |
| `roc_auc` | Качество ранжирования по вероятности |

### 9.2 Предсказание для одного клиента

**Запрос:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "monthly_fee": 29.99,
       "usage_hours": 120.5,
       "support_requests": 2,
       "account_age_months": 15,
       "failed_payments": 0,
       "region": "europe",
       "device_type": "mobile",
       "payment_method": "card",
       "autopay_enabled": 1
     }'
```

**Ожидаемый ответ:**

```json
{
  "churn_prediction": 0,
  "probability": [0.9980580756384752, 0.0019419243615249005]
}
```

### 9.3 Предсказание для списка клиентов

**Запрос:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '[
       {
         "monthly_fee": 29.99, "usage_hours": 120.5,
         "support_requests": 2, "account_age_months": 15,
         "failed_payments": 0, "region": "europe",
         "device_type": "mobile", "payment_method": "card",
         "autopay_enabled": 1
       },
       {
         "monthly_fee": 89.99, "usage_hours": 12.5,
         "support_requests": 7, "account_age_months": 3,
         "failed_payments": 3, "region": "africa",
         "device_type": "desktop", "payment_method": "paypal",
         "autopay_enabled": 0
       }
     ]'
```

**Ожидаемый ответ:**

```json
[
  {"churn_prediction": 0, "probability": [0.998, 0.002]},
  {"churn_prediction": 1, "probability": [0.120, 0.880]}
]
```

### 9.4 Проверка статуса модели

```bash
curl http://127.0.0.1:8000/model/status
```

**Ожидаемый ответ:**

```json
{
  "is_trained": true,
  "trained_at": "2026-09-15T15:52:48.782179",
  "metrics": {"accuracy": 0.8, "f1": 0.024, "roc_auc": 0.622},
  "model_type": "logreg",
  "hyperparameters": {"C": 0.5}
}
```

### 9.5 Просмотр истории обучений

```bash
curl "http://127.0.0.1:8000/model/metrics?limit=5"
```

Фильтр по типу модели:

```bash
curl "http://127.0.0.1:8000/model/metrics?model_type=random_forest"
```

### 9.6 Проверка состояния сервиса

```bash
curl http://127.0.0.1:8000/health
```

**Ожидаемый ответ:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "dataset_loaded": true
}
```

Если модель не обучена — `model_loaded: false`, `status: "not ready"`.

### 9.7 Список признаков и типов

```bash
curl http://127.0.0.1:8000/model/schema
```

**Ожидаемый ответ:**

```json
{
  "monthly_fee": "float",
  "usage_hours": "float",
  "support_requests": "int",
  "account_age_months": "int",
  "failed_payments": "int",
  "region": "str",
  "device_type": "str",
  "payment_method": "str",
  "autopay_enabled": "int"
}
```

---

## 10. Логирование

Сервис пишет логи в двух форматах:

| Куда | Как посмотреть |
|------|----------------|
| stdout | `docker logs churn` или `uvicorn` |
| Файл | `logs/churn_service.log` |

Формат — JSON:

```json
{
  "timestamp": "2026-09-15T17:46:48",
  "level": "INFO",
  "logger": "churn_service",
  "message": "Модель загружена: тип=random_forest, обучена=2026-09-15T15:52:48.782179"
}
```

Логируются события:

| Событие | Уровень |
|---------|---------|
| Загрузка модели при старте | INFO |
| Модель не найдена | WARNING |
| Запуск и завершение обучения | INFO |
| Запрос `/predict` | INFO |
| Успешное предсказание | INFO |
| Ошибки данных | WARNING |
| Ошибки HTTP | WARNING |
| Непредвиденные ошибки | ERROR |

---

## Быстрая проверка всех эндпоинтов

```bash
for url in \
    "http://127.0.0.1:8000/" \
    "http://127.0.0.1:8000/health" \
    "http://127.0.0.1:8000/model/status" \
    "http://127.0.0.1:8000/model/schema" \
    "http://127.0.0.1:8000/model/metrics" \
    "http://127.0.0.1:8000/dataset/info" \
    "http://127.0.0.1:8000/dataset/preview" \
    "http://127.0.0.1:8000/dataset/split-info" \
    "http://127.0.0.1:8000/docs"; do
  printf "%-45s -> " "$url"
  curl -s -o /dev/null -w "%{http_code}\n" "$url"
done
```

Все строки должны вернуть `200`.

---
