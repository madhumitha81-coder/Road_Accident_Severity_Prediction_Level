import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
SERVING_DIR = PROJECT_ROOT / "artifacts" / "serving"
PIPELINE_PATH = SERVING_DIR / "training_pipeline.pkl"
METADATA_PATH = SERVING_DIR / "metadata.json"


app = FastAPI(
    title="Accident Severity Intelligence",
    version="1.0.0",
    description="FastAPI service for road traffic accident severity prediction.",
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_ASSET_CACHE: dict[str, Any] = {
    "version": None,
    "pipeline": None,
    "metadata": None,
}


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


def load_assets():
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Serving bundle not found. Run `conda run -n accident python trained_model.py` to train and save the latest model."
        )

    version = (
        PIPELINE_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
    )

    if _ASSET_CACHE["version"] != version:
        with open(PIPELINE_PATH, "rb") as pipeline_file:
            _ASSET_CACHE["pipeline"] = pickle.load(pipeline_file)
        with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
            _ASSET_CACHE["metadata"] = json.load(metadata_file)
        _ASSET_CACHE["version"] = version

    return _ASSET_CACHE["pipeline"], _ASSET_CACHE["metadata"]


def normalize_features(raw_features: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    feature_columns = metadata["feature_columns"]
    numeric_columns = set(metadata["numeric_columns"])
    allowed_columns = set(feature_columns)

    unexpected_fields = sorted(
        key for key, value in raw_features.items() if key not in allowed_columns and value not in ("", None)
    )
    if unexpected_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected fields received: {', '.join(unexpected_fields)}.",
        )

    normalized_row: dict[str, Any] = {}
    for column in feature_columns:
        value = raw_features.get(column)

        if isinstance(value, str):
            value = value.strip()

        if value in ("", None):
            normalized_row[column] = None
            continue

        if column in numeric_columns:
            try:
                normalized_row[column] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid numeric value for '{column}'.") from exc
        else:
            normalized_row[column] = str(value)

    return pd.DataFrame([normalized_row], columns=feature_columns)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        _, metadata = load_assets()
        serving_ready = True
        error_message = None
    except FileNotFoundError as exc:
        metadata = None
        serving_ready = False
        error_message = str(exc)

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "serving_ready": serving_ready,
            "metadata": metadata,
            "error_message": error_message,
        },
    )


@app.get("/api/health")
async def health():
    try:
        _, metadata = load_assets()
        return {
            "status": "ok",
            "serving_ready": True,
            "model_name": metadata["model_name"],
            "updated_at": metadata["created_at"],
        }
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "missing_model",
                "serving_ready": False,
                "message": "Run `conda run -n accident python trained_model.py` first.",
            },
        )


@app.get("/api/metadata")
async def metadata():
    try:
        _, model_metadata = load_assets()
        return {
            "serving_ready": True,
            "metadata": model_metadata,
        }
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "serving_ready": False,
                "message": str(exc),
            },
        )


@app.post("/api/predict")
async def predict(payload: PredictionRequest):
    try:
        pipeline, metadata = load_assets()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features_df = normalize_features(payload.features, metadata)
    prediction = pipeline.predict(features_df)[0]

    probabilities: list[dict[str, Any]] = []
    if hasattr(pipeline, "predict_proba"):
        predicted_probabilities = pipeline.predict_proba(features_df)[0]
        for label, probability in zip(metadata["class_labels"], predicted_probabilities):
            probabilities.append(
                {
                    "label": label,
                    "probability": float(probability),
                }
            )
        probabilities.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "prediction": str(prediction),
        "probabilities": probabilities,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }
