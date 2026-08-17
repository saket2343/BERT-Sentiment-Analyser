"""
Model 1: TF-IDF + Logistic Regression baseline.

Run standalone with:
    python -m src.baseline
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PATH, TFIDF_MODEL_DIR, SEED
from src.data import load_and_prepare
from src.seed import set_seed


def build_pipeline(
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True,
    C=1.0,
) -> Pipeline:
    """Build the configurable TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=C,
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, texts, labels) -> dict:
    preds = pipeline.predict(texts)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, output_dict=True)
    return {"metrics": metrics, "confusion_matrix": cm, "report": report, "preds": preds}


def train_and_evaluate(data_path: str = DATA_PATH, output_dir: str = TFIDF_MODEL_DIR):
    set_seed(SEED)

    _, _, splits = load_and_prepare(data_path)

    print("\nTraining TF-IDF + Logistic Regression baseline...")
    start = time.time()
    pipeline = build_pipeline()
    pipeline.fit(splits.train_text, splits.train_labels)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")

    print("\nEvaluating on the held-out test set...")
    result = evaluate_pipeline(pipeline, splits.test_text, splits.test_labels)

    print("\nTF-IDF + Logistic Regression -- Test Set Results")
    for k, v in result["metrics"].items():
        print(f"    {k:10s}: {v:.4f}")
    print(f"    confusion_matrix: {result['confusion_matrix']}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "tfidf_logreg_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved pipeline to {model_path}")

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "training_time_seconds": train_time,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")

    return pipeline, result


def load_pipeline(model_dir: str = TFIDF_MODEL_DIR) -> Pipeline:
    model_path = os.path.join(model_dir, "tfidf_logreg_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved TF-IDF pipeline found at {model_path}. "
            f"Run `python -m src.baseline` first."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    train_and_evaluate()
