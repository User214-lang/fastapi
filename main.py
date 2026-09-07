from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import load_dataset, load_validated_dataset, prepare_data, split_data, get_class_distribution
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from model import train_model, save_churn_model, load_churn_model
from contextlib import asynccontextmanager

MODEL = None
MODEL_METRICS = None
TRAINED_TIME = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_METRICS, TRAINED_TIME
    MODEL, MODEL_METRICS, TRAINED_TIME = load_churn_model()
    if MODEL is not None:
        print("Модель успешно загружена из файла")
    else:
        print("Модель не найдена. Обучите модель через POST /model/train")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features.model_dump()

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
def train_model_endpoint():
    global MODEL, MODEL_METRICS, TRAINED_TIME
    try:
        model, X_train, X_test, y_train, y_test = train_model()
        MODEL = model
        _, MODEL_METRICS, TRAINED_TIME = load_churn_model()
        
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
        "metrics": MODEL_METRICS
    }


