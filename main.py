from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from schemas import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn, TrainingConfigChurn
from dataset import load_dataset, load_validated_dataset, prepare_data, split_data, get_class_distribution
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from model import train_model, save_churn_model, load_churn_model
from contextlib import asynccontextmanager
from typing import Union, List, Any
from dataset import ALL_FEATURES_ORDER

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

@app.post("/predict", response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]])
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

@app.post("/model/train")
def train_model_endpoint(config: TrainingConfigChurn):
    global MODEL, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS
    try:
        model, X_train, X_test, y_train, y_test = train_model(
            model_type=config.model_type,
            hyperparameters=config.hyperparameters
        )

        MODEL = model
        _, MODEL_METRICS, TRAINED_TIME, MODEL_TYPE, MODEL_HYPERPARAMS = load_churn_model()
        
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
