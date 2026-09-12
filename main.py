from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from schemas import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponse
from dataset import load_dataset, load_validated_dataset, prepare_data, split_data, get_class_distribution
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import train_model, save_churn_model, load_churn_model, append_training_record, load_training_history
from contextlib import asynccontextmanager
from typing import Union, List, Any
from dataset import ALL_FEATURES_ORDER
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import json, os
from datetime import datetime

MODEL = None
MODEL_METRICS = None
TRAINED_TIME = None
MODEL_TYPE = None
MODEL_HYPERPARAMS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS
    MODEL, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS = load_churn_model()
    if MODEL is not None:
        print("Модель успешно загружена из файла")
    else:
        print("Модель не найдена. Обучите модель через POST /model/train")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post("/predict", 
          response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]],
          responses = {
        503: {
            "model": ErrorResponse,
            "description": "Модель не обучена",
            "content": {
                "application/json": {
                    "example": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Модель ещё не обучена",
                        "details": None
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Ошибка валидации входных данных",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Ошибка валидации данных",
                        "details": [
                            {
                                "loc": ["body", "monthly_fee"],
                                "msg": "value is not a valid float",
                                "type": "type_error.float"
                            }
                        ]
                    }
                }
            }
        }
    }
)
def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]]):
    global MODEL

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель еще не обучена")

    if isinstance(features, FeatureVectorChurn):
        features_list = [features]
        single_input = True
    else:
        features_list = features
        single_input = False

    data = [f.model_dump() for f in features_list]
    df = pd.DataFrame(data)
    missing_cols = set(ALL_FEATURES_ORDER) - set(df.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    df = df[ALL_FEATURES_ORDER]

    predictions = MODEL.predict(df)
    probabilities = MODEL.predict_proba(df)

    results = []
    for i, pred in enumerate(predictions):
        prob = probabilities[i].tolist()
        results.append({
            "churn_prediction": int(pred),
            "probability": prob
        })

    if single_input:
        return results[0]
    else:
        return results

@app.get("/dataset/preview")

def dataset_preview():
        try:
            df = load_dataset("data/churn_dataset.csv")
            preview = df.head(10).to_dict(orient="records")
            return {"rows": preview}

        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Dataset file not found")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/info")
def dataset_info():

    try:

        df = load_dataset("data/churn_dataset.csv")
        rows = len(df)
        cols = len(df.columns)
        features = df.columns.tolist()

        if "churn" not in df.columns:
            raise HTTPException(status_code=422, detail="Column 'churn' not found in dataset")
        churn_counts = df["churn"].value_counts().to_dict()

        return {
            "rows": rows,
            "columns": cols,
            "feature_names": features,
            "churn_distribution": churn_counts
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/split-info")
def dataset_split_info():
    try:
        df = load_validated_dataset("data/churn_dataset.csv")
        X, y, _, _ = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        train_shape = list(X_train.shape)
        test_shape = list(X_test.shape)
        dist = get_class_distribution(y_train, y_test)

        return {
            "train_shape": train_shape,
            "test_shape": test_shape,
            "churn_distribution": {
                "train": dist["train"],
                "test": dist["test"]
            }
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train",
responses={
        404: {
            "model": ErrorResponse,
            "description": "Датасет не найден",
            "content": {
                "application/json": {
                    "example": {
                        "code": "NOT_FOUND",
                        "message": "Dataset file not found",
                        "details": None
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Ошибка валидации входных данных",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Ошибка валидации данных",
                        "details": [
                            {
                                "loc": ["body", "model_type"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Внутренняя ошибка сервера",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": None
                    }
                }
            }
        }
    }
)
def train_model_endpoint(config: TrainingConfigChurn):
    global MODEL, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS
    try:
        model, X_train, X_test, y_train, y_test = train_model(
            model_type=config.model_type,
            hyperparameters=config.hyperparameters
        )

        MODEL = model
        _, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS = load_churn_model()

        record = {
            "trained_at": TRAINED_TIME,
            "model_type": MODEL_TYPE,
            "hyperparameters": MODEL_HYPERPARAMS,
            "metrics": MODEL_METRICS,
        }
        append_training_record(record)

        
        return MODEL_METRICS
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
def model_status():
    return {
        "is_trained": MODEL is not None,
        "trained_at": TRAINED_TIME,
        "metrics": MODEL_METRICS,
        "model_type": MODEL_TYPE,
        "hyperparameters": MODEL_HYPERPARAMS
    }

@app.get("/model/schema")
def model_schema():
    schema = {}
    for name, field in FeatureVectorChurn.model_fields.items():
        schema[name] = field.annotation.__name__
    return schema

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        error = exc.detail
    else:
        code_map = {
            404: "NOT_FOUND",
            422: "VALIDATION_ERROR",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE"
        }
        error = ErrorResponse(
            code=code_map.get(exc.status_code, "HTTP_ERROR"),
            message=str(exc.detail),
            details=None
        ).model_dump()

    return JSONResponse(status_code=exc.status_code, content=error)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error = ErrorResponse(
        code = "VALIDATION_ERROR", 
        message = "Ошибка валидации данных", 
        details = exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content=error.model_dump()
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    error = ErrorResponse(
        code = "DATA_ERROR", 
        message = str(exc),
        details = None
    )

@app.get("/model/metrics")
def model_metrics(limit: int = 10, model_type: str = None):
    history = load_training_history()

    if model_type:
        history = [r for r in history if r.get("model_type") == model_type]

    recent = history[-limit:] if limit > 0 else []

    return {
        "last_metrics": MODEL_METRICS,
        "recent_records": recent,
    }
