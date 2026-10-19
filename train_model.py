"""Train and save PhishGuard's URL-only Linear SVM model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from phishguard_pipeline import (
    LABEL_NAMES,
    LEXICAL_FEATURE_COLUMNS,
    fit_feature_pipeline,
    normalize_url,
    transform_urls,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "phiusiil.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhishGuard from the existing local PhiUSIIL CSV."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30_000,
        help="Stratified row count for training and evaluation; use 0 for all rows.",
    )
    return parser.parse_args()


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Keep the existing PhiUSIIL CSV in data/raw."
        )

    data = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    if data[["URL", "label"]].isna().any().any():
        raise ValueError("The URL or label column contains missing values.")

    data["label"] = pd.to_numeric(data["label"], errors="raise").astype("int8")
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError(
            "Expected the already-converted labels 0=legitimate and 1=phishing."
        )
    data["URL"] = data["URL"].map(normalize_url)
    return data


def train_and_save(sample_size: int = 30_000) -> dict[str, object]:
    """Fit the training-only pipeline and write model plus test-split metadata."""
    if sample_size < 0:
        raise ValueError("sample_size must be 0 (all data) or a positive integer.")
    data = load_training_data()
    dataset_rows = len(data)
    if sample_size and sample_size < len(data):
        if sample_size < 10:
            raise ValueError("sample_size must be at least 10 for stratified sampling.")
        data, _ = train_test_split(
            data,
            train_size=sample_size,
            stratify=data["label"],
            random_state=RANDOM_STATE,
        )

    urls = data["URL"].reset_index(drop=True)
    labels = data["label"].reset_index(drop=True)
    train_urls, test_urls, y_train, y_test = train_test_split(
        urls,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    vectorizer, scaler, train_matrix = fit_feature_pipeline(train_urls)
    test_matrix = transform_urls(test_urls, vectorizer, scaler)
    estimator = LinearSVC(C=1.0, max_iter=2_000, random_state=RANDOM_STATE)
    estimator.fit(train_matrix, y_train)

    predictions = estimator.predict(test_matrix)
    decision_scores = estimator.decision_function(test_matrix)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "confusion_matrix_labels": ["Legitimate (0)", "Potential Phishing (1)"],
        "confusion_matrix": matrix.tolist(),
    }

    bundle = {
        "model_name": "LinearSVC",
        "estimator": estimator,
        "vectorizer": vectorizer,
        "lexical_scaler": scaler,
        "lexical_feature_columns": LEXICAL_FEATURE_COLUMNS,
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "uses_probability": False,
    }
    metadata: dict[str, object] = {
        "model_name": "LinearSVC (character TF-IDF + lexical features)",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI PhiUSIIL Phishing URL (Website), ID 967, local CSV",
        "dataset_rows": int(dataset_rows),
        "experiment_rows": int(len(data)),
        "train_rows": int(len(train_urls)),
        "test_rows": int(len(test_urls)),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "features": {
            "input_column": "URL",
            "target_column": "label",
            "target_used_without_remapping": True,
            "character_tfidf": {
                "analyzer": "char",
                "ngram_range": [3, 5],
                "min_df": 2,
                "max_features": 100_000,
                "dtype": "float32",
                "sublinear_tf": True,
            },
            "lexical_feature_columns": LEXICAL_FEATURE_COLUMNS,
            "lexical_scaling": "MaxAbsScaler fitted on training rows",
            "combined_matrix": "Sparse character TF-IDF plus scaled lexical features",
        },
        "estimator": {
            "class": "sklearn.svm.LinearSVC",
            "parameters": estimator.get_params(),
            "supports_probability": False,
        },
        "evaluation": {
            "description": "Metrics from the held-out stratified test split; not guaranteed real-world performance.",
            **metrics,
        },
        "scikit_learn_version": sklearn.__version__,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved evaluation metadata to {METADATA_PATH}")
    print(
        "Held-out metrics: "
        + ", ".join(
            f"{metric}={metrics[metric]:.4f}"
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        )
    )
    return metadata


if __name__ == "__main__":
    arguments = parse_args()
    try:
        train_and_save(arguments.sample_size)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Training failed: {error}") from error
