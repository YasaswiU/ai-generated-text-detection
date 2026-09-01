"""
Calibrates the raw model saved by train.py using CalibratedClassifierCV, and
selects an abstention threshold on the validation split by evaluating
selective risk at a range of thresholds.

Outputs:
  backend/models/calibrated_model.joblib  (bundle: {model, feature_names})
  backend/models/model_metadata.json      (version, threshold, method, etc.)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

DATA_DIR = Path(__file__).parent / "data"
BACKEND_MODELS_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"
FEATURE_COLUMNS_EXCLUDE = {"language", "label", "split"}

DATASET_VERSION = "v1"
FEATURE_VERSION = "1.0.0"


def load_split(df: pd.DataFrame, split: str, feature_cols):
    subset = df[df["split"] == split]
    X = subset[feature_cols].values
    y = (subset["label"] == "ai").astype(int).values
    return X, y


def select_threshold(y_true: np.ndarray, proba_ai: np.ndarray):
    """
    Sweeps candidate thresholds and reports coverage / selective risk for
    each, choosing the smallest threshold that keeps selective risk <= 0.10
    (i.e. <=10% error on the predictions the model is willing to make),
    falling back to the threshold with the lowest selective risk if none
    meet that bar (e.g. on a tiny demo dataset).
    """
    confidence = np.maximum(proba_ai, 1 - proba_ai)
    predicted_label = (proba_ai >= 0.5).astype(int)
    correct = (predicted_label == y_true).astype(int)

    results = []
    for threshold in np.arange(0.5, 0.96, 0.05):
        accepted = confidence >= threshold
        coverage = float(np.mean(accepted))
        if accepted.sum() == 0:
            selective_risk = 1.0
        else:
            selective_risk = 1.0 - float(np.mean(correct[accepted]))
        results.append(
            {"threshold": round(float(threshold), 2), "coverage": coverage, "selective_risk": selective_risk}
        )

    # Prefer the lowest threshold achieving selective_risk <= 0.10; else the
    # threshold with the globally lowest selective risk.
    acceptable = [r for r in results if r["selective_risk"] <= 0.10]
    if acceptable:
        chosen = min(acceptable, key=lambda r: r["threshold"])
    else:
        chosen = min(results, key=lambda r: r["selective_risk"])

    return chosen["threshold"], results


def main() -> None:
    raw_path = DATA_DIR / "best_model_raw.joblib"
    features_path = DATA_DIR / "features.csv"
    if not raw_path.exists() or not features_path.exists():
        raise SystemExit("Run extract_features.py and train.py first.")

    bundle = joblib.load(raw_path)
    raw_model = bundle["model"]
    feature_cols = bundle["feature_names"]

    df = pd.read_csv(features_path)
    X_train, y_train = load_split(df, "train", feature_cols)
    X_val, y_val = load_split(df, "val", feature_cols)

    n_val = len(y_val)
    # sigmoid (Platt scaling) suits small/medium datasets; isotonic needs
    # more data or it overfits. We pick based on validation set size.
    method = "sigmoid" if n_val < 1000 else "isotonic"
    cv_folds = min(3, max(2, len(set(y_train))))

    print(f"Calibrating with method='{method}' ...")
    try:
        calibrated = CalibratedClassifierCV(raw_model, method=method, cv="prefit")
        calibrated.fit(X_val if n_val > 0 else X_train, y_val if n_val > 0 else y_train)
    except Exception as exc:  # noqa: BLE001
        print(f"Calibration with held-out val set failed ({exc}); falling back to cv=2 on train.")
        calibrated = CalibratedClassifierCV(raw_model, method=method, cv=2)
        calibrated.fit(X_train, y_train)

    if n_val > 0 and len(set(y_val)) > 1:
        proba_ai = calibrated.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, proba_ai)
        threshold, sweep = select_threshold(y_val, proba_ai)
    else:
        print("WARNING: validation split too small/uniform for a meaningful "
              "threshold sweep (expected with --demo data). Using default threshold=0.65.")
        proba_ai = calibrated.predict_proba(X_train)[:, 1]
        brier = brier_score_loss(y_train, proba_ai) if len(set(y_train)) > 1 else float("nan")
        threshold, sweep = 0.65, []

    print(f"Selected abstention threshold: {threshold}")
    print(f"Brier score on validation: {brier:.4f}" if brier == brier else "Brier score: n/a")

    BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_out_path = BACKEND_MODELS_DIR / "calibrated_model.joblib"
    joblib.dump({"model": calibrated, "feature_names": feature_cols}, model_out_path)

    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S"),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "supported_languages": ["en", "te", "hi"],
        "feature_version": FEATURE_VERSION,
        "threshold": threshold,
        "calibration_method": method,
        "dataset_version": DATASET_VERSION,
        "base_model": bundle.get("model_name", "unknown"),
        "threshold_sweep": sweep,
    }
    metadata_out_path = BACKEND_MODELS_DIR / "model_metadata.json"
    with open(metadata_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved calibrated model to {model_out_path}")
    print(f"Saved metadata to {metadata_out_path}")


if __name__ == "__main__":
    main()
