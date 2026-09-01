"""
Evaluates the calibrated model on the held-out test split, overall and
per-language, reporting accuracy/precision/recall/F1/ROC-AUC/confusion
matrix plus calibration/abstention metrics (coverage, abstention rate,
selective risk, accuracy-on-accepted).

Writes training/data/evaluation_report.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_DIR = Path(__file__).parent / "data"
BACKEND_MODELS_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"


def evaluate_subset(y_true, proba_ai, threshold):
    confidence = np.maximum(proba_ai, 1 - proba_ai)
    predicted = (proba_ai >= 0.5).astype(int)
    accepted_mask = confidence >= threshold

    metrics = {"n": int(len(y_true))}
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba_ai))
    metrics["accuracy"] = float(accuracy_score(y_true, predicted))
    metrics["precision"] = float(precision_score(y_true, predicted, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, predicted, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, predicted, zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(y_true, predicted).tolist()

    metrics["coverage"] = float(np.mean(accepted_mask))
    metrics["abstention_rate"] = float(1 - metrics["coverage"])
    if accepted_mask.sum() > 0:
        metrics["accuracy_on_accepted"] = float(
            accuracy_score(y_true[accepted_mask], predicted[accepted_mask])
        )
        metrics["selective_risk"] = float(1 - metrics["accuracy_on_accepted"])
    else:
        metrics["accuracy_on_accepted"] = None
        metrics["selective_risk"] = None

    return metrics


def main() -> None:
    model_path = BACKEND_MODELS_DIR / "calibrated_model.joblib"
    metadata_path = BACKEND_MODELS_DIR / "model_metadata.json"
    features_path = DATA_DIR / "features.csv"

    if not model_path.exists() or not features_path.exists():
        raise SystemExit("Run extract_features.py, train.py, and calibrate.py first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_names"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = metadata.get("threshold", 0.65)

    df = pd.read_csv(features_path)
    test_df = df[df["split"] == "test"]

    report = {"threshold": threshold, "overall": None, "by_language": {}}

    if len(test_df) > 0:
        X_test = test_df[feature_cols].values
        y_test = (test_df["label"] == "ai").astype(int).values
        proba_ai = model.predict_proba(X_test)[:, 1]
        report["overall"] = evaluate_subset(y_test, proba_ai, threshold)

        for language, group in test_df.groupby("language"):
            X_lang = group[feature_cols].values
            y_lang = (group["label"] == "ai").astype(int).values
            proba_lang = model.predict_proba(X_lang)[:, 1]
            report["by_language"][language] = evaluate_subset(y_lang, proba_lang, threshold)
    else:
        print("WARNING: test split is empty -- add more data per class/language.")

    out_path = DATA_DIR / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
