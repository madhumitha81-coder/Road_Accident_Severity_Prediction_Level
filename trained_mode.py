import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from source.feature_config import (
    REQUESTED_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    get_training_drop_columns,
    humanize_column_name,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILES = [
    PROJECT_ROOT / "Datasets" / "raw" / "RTA Dataset.csv",
    PROJECT_ROOT / "Datasets" / "raw" / "RTADatasetE1.csv",
]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACT_DIR / "runs"
SERVING_DIR = ARTIFACT_DIR / "serving"
PROCESSED_DATA_PATH = ARTIFACT_DIR / "processed_data_from_raw.csv"
MODEL_READY_DATA_PATH = ARTIFACT_DIR / "data.csv"
TRAIN_DATA_PATH = ARTIFACT_DIR / "train.csv"
VALIDATION_DATA_PATH = ARTIFACT_DIR / "test.csv"
ROOT_MODEL_PATH = ARTIFACT_DIR / "model.pkl"
ROOT_PREPROCESSOR_PATH = ARTIFACT_DIR / "preprecessor.pkl"
ROOT_PIPELINE_PATH = ARTIFACT_DIR / "training_pipeline.pkl"
ROOT_METRICS_PATH = ARTIFACT_DIR / "metrics.json"

MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": 1,
}


def normalize_text(value):
    if pd.isna(value):
        return np.nan

    cleaned = " ".join(str(value).strip().split())
    if cleaned.lower() in {"", "nan", "null"}:
        return np.nan

    return cleaned


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def save_dataframe(path, dataframe):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def load_raw_data():
    frames = []

    for raw_file in RAW_FILES:
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing raw file: {raw_file}")

        frame = pd.read_csv(raw_file)
        frame.columns = frame.columns.str.strip().str.lower()

        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = frame[column].map(normalize_text)

        for column in ["number_of_vehicles_involved", "number_of_casualties"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return data


def build_artifact_paths():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    SERVING_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "run_model": run_dir / "model.pkl",
        "run_preprocessor": run_dir / "preprecessor.pkl",
        "run_pipeline": run_dir / "training_pipeline.pkl",
        "run_metrics": run_dir / "metrics.json",
        "run_metadata": run_dir / "metadata.json",
        "root_model": ROOT_MODEL_PATH,
        "root_preprocessor": ROOT_PREPROCESSOR_PATH,
        "root_pipeline": ROOT_PIPELINE_PATH,
        "root_metrics": ROOT_METRICS_PATH,
        "serving_model": SERVING_DIR / "model.pkl",
        "serving_preprocessor": SERVING_DIR / "preprecessor.pkl",
        "serving_pipeline": SERVING_DIR / "training_pipeline.pkl",
        "serving_metadata": SERVING_DIR / "metadata.json",
    }


def main():
    print("[trained_mode.py] Loading raw CSV files", flush=True)
    data = load_raw_data()
    save_dataframe(PROCESSED_DATA_PATH, data)
    print(f"[trained_mode.py] Combined data shape: {data.shape}", flush=True)

    features = data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    training_drop_columns = get_training_drop_columns()
    dropped_columns = [column for column in training_drop_columns if column in features.columns]
    requested_excluded_columns = [
        column for column in REQUESTED_EXCLUDED_COLUMNS if column in features.columns
    ]
    features = features.drop(columns=dropped_columns, errors="ignore")
    target = data[TARGET_COLUMN].copy()

    model_ready_data = features.copy()
    model_ready_data[TARGET_COLUMN] = target.values
    save_dataframe(MODEL_READY_DATA_PATH, model_ready_data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    print(f"[trained_mode.py] Train shape: {X_train.shape}", flush=True)
    print(f"[trained_mode.py] Validation shape: {X_valid.shape}", flush=True)

    train_frame = X_train.copy()
    train_frame[TARGET_COLUMN] = y_train.values
    validation_frame = X_valid.copy()
    validation_frame[TARGET_COLUMN] = y_valid.values
    save_dataframe(TRAIN_DATA_PATH, train_frame)
    save_dataframe(VALIDATION_DATA_PATH, validation_frame)

    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_options = {
        column: sorted(value for value in features[column].dropna().unique().tolist())
        for column in categorical_columns
    }
    field_labels = {
        column: humanize_column_name(column)
        for column in [*features.columns.tolist(), *requested_excluded_columns]
    }

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehotencoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = RandomForestClassifier(**MODEL_PARAMS)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("[trained_mode.py] Training model from scratch", flush=True)
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    valid_predictions = pipeline.predict(X_valid)

    train_accuracy = float(accuracy_score(y_train, train_predictions))
    valid_accuracy = float(accuracy_score(y_valid, valid_predictions))
    train_f1 = float(f1_score(y_train, train_predictions, average="weighted", zero_division=0))
    valid_f1 = float(f1_score(y_valid, valid_predictions, average="weighted", zero_division=0))

    artifact_paths = build_artifact_paths()

    metrics = {
        "train_accuracy": train_accuracy,
        "validation_accuracy": valid_accuracy,
        "train_f1_weighted": train_f1,
        "validation_f1_weighted": valid_f1,
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_files": [str(path) for path in RAW_FILES],
        "raw_data_paths": [str(path) for path in RAW_FILES],
        "generated_processed_path": str(PROCESSED_DATA_PATH),
        "run_directory": str(artifact_paths["run_dir"]),
        "target_column": TARGET_COLUMN,
        "requested_excluded_columns": requested_excluded_columns,
        "dropped_columns": dropped_columns,
        "feature_columns": list(features.columns),
        "field_labels": field_labels,
        "feature_summary": {
            "numerical_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_options": categorical_options,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "model_name": "RandomForestClassifier",
        "model_params": MODEL_PARAMS,
        "metrics": metrics,
        "train_metrics": {
            "accuracy": train_accuracy,
            "f1_weighted": train_f1,
        },
        "validation_metrics": {
            "accuracy": valid_accuracy,
            "f1_weighted": valid_f1,
        },
        "class_labels": [str(label) for label in pipeline.classes_],
        "artifacts": {
            "processed_data": str(PROCESSED_DATA_PATH),
            "model_ready_data": str(MODEL_READY_DATA_PATH),
            "train_data": str(TRAIN_DATA_PATH),
            "validation_data": str(VALIDATION_DATA_PATH),
            "root_model": str(artifact_paths["root_model"]),
            "root_preprocessor": str(artifact_paths["root_preprocessor"]),
            "root_pipeline": str(artifact_paths["root_pipeline"]),
            "root_metrics": str(artifact_paths["root_metrics"]),
            "serving_directory": str(SERVING_DIR),
        },
    }

    save_pickle(artifact_paths["run_model"], pipeline.named_steps["model"])
    save_pickle(artifact_paths["run_preprocessor"], pipeline.named_steps["preprocessor"])
    save_pickle(artifact_paths["run_pipeline"], pipeline)
    save_json(artifact_paths["run_metrics"], metadata)
    save_json(artifact_paths["run_metadata"], metadata)

    save_pickle(artifact_paths["root_model"], pipeline.named_steps["model"])
    save_pickle(artifact_paths["root_preprocessor"], pipeline.named_steps["preprocessor"])
    save_pickle(artifact_paths["root_pipeline"], pipeline)
    save_json(artifact_paths["root_metrics"], metadata)

    save_pickle(artifact_paths["serving_model"], pipeline.named_steps["model"])
    save_pickle(artifact_paths["serving_preprocessor"], pipeline.named_steps["preprocessor"])
    save_pickle(artifact_paths["serving_pipeline"], pipeline)
    save_json(artifact_paths["serving_metadata"], metadata)

    print(f"[trained_mode.py] Root model saved to: {artifact_paths['root_model']}", flush=True)
    print(f"[trained_mode.py] Root preprocessor saved to: {artifact_paths['root_preprocessor']}", flush=True)
    print(f"[trained_mode.py] Root pipeline saved to: {artifact_paths['root_pipeline']}", flush=True)
    print(f"[trained_mode.py] Root metrics saved to: {artifact_paths['root_metrics']}", flush=True)
    print(f"[trained_mode.py] Run artifacts saved to: {artifact_paths['run_dir']}", flush=True)
    print(f"[trained_mode.py] Serving bundle updated in: {SERVING_DIR}", flush=True)
    print(f"[trained_mode.py] Requested exclusions: {requested_excluded_columns}", flush=True)
    print(f"[trained_mode.py] Active training features: {list(features.columns)}", flush=True)
    print(f"[trained_mode.py] Train accuracy: {train_accuracy:.6f}", flush=True)
    print(f"[trained_mode.py] Validation accuracy: {valid_accuracy:.6f}", flush=True)
    print(f"[trained_mode.py] Train F1: {train_f1:.6f}", flush=True)
    print(f"[trained_mode.py] Validation F1: {valid_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
