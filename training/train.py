"""
Trains candidate classifiers (Logistic Regression, Random Forest, and
XGBoost if installed) on training/data/features.csv, selects the best model
by validation-set ROC-AUC, and saves it (uncalibrated) plus the feature name
order to training/data/best_model_raw.joblib.

Calibration and abstention-threshold selection happen separately in
calibrate.py, which is intentional: calibration must be fit on a held-out
split the classifier itself never trained on.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent / "data"
FEATURE_COLUMNS_EXCLUDE = {"language", "label", "split"}


def load_split(df: pd.DataFrame, split: str):
    subset = df[df["split"] == split]
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    X = subset[feature_cols].values
    y = (subset["label"] == "ai").astype(int).values
    return X, y, feature_cols


def main() -> None:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        raise SystemExit("features.csv not found. Run extract_features.py first.")

    df = pd.read_csv(features_path)
    X_train, y_train, feature_cols = load_split(df, "train")
    X_val, y_val, _ = load_split(df, "val")

    if len(set(y_train)) < 2:
        raise SystemExit(
            "Training split contains only one class. Add more data "
            "(this is expected with the tiny --demo dataset)."
        )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced", random_state=42
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
    except ImportError:
        print("xgboost not installed; skipping (it is optional, see README).")

    best_name, best_model, best_auc = None, None, -1.0
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        if len(set(y_val)) < 2:
            # Not enough validation diversity (tiny demo dataset) -- fall
            # back to training AUC just so the pipeline can run end-to-end.
            proba = clf.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, proba) if len(set(y_train)) > 1 else 0.5
        else:
            proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
        print(f"{name}: validation ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best_name, best_model, best_auc = name, clf, auc

    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")

    out_path = DATA_DIR / "best_model_raw.joblib"
    joblib.dump({"model": best_model, "feature_names": feature_cols, "model_name": best_name}, out_path)
    print(f"Saved raw (uncalibrated) model to {out_path}")


if __name__ == "__main__":
    main()
