"""
Ablation experiments comparing feature groups, as required for the project
report:

  Experiment 1: Stylometric features only
  Experiment 2: Pseudo-perplexity only
  Experiment 3: Stylometry + pseudo-perplexity
  Experiment 4: Stylometry + pseudo-perplexity + curvature
  Experiment 5: Calibrated final model + abstention

Writes training/data/model_comparison.csv and a bar-chart PNG comparing
validation ROC-AUC across experiments.
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
BACKEND_MODELS_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"

STYLOMETRY_COLS_HINT_EXCLUDE = {"language", "label", "split", "pp_mean", "pp_median", "pp_std"}
CURVATURE_COLS = [
    "mean_first_difference",
    "std_first_difference",
    "mean_abs_first_difference",
    "mean_second_difference",
    "std_second_difference",
    "mean_abs_second_difference",
    "max_absolute_curvature",
    "curvature_sign_changes",
]
PERPLEXITY_COLS = ["pp_mean", "pp_median", "pp_std"]


def _fit_eval(X_train, y_train, X_val, y_val):
    if len(set(y_train)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    if len(set(y_val)) < 2:
        proba = clf.predict_proba(X_train)[:, 1]
        return roc_auc_score(y_train, proba)
    proba = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def main() -> None:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        raise SystemExit("Run extract_features.py first.")

    df = pd.read_csv(features_path)
    all_feature_cols = [c for c in df.columns if c not in {"language", "label", "split"}]
    stylometry_cols = [c for c in all_feature_cols if c not in CURVATURE_COLS + PERPLEXITY_COLS]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"] if (df["split"] == "val").any() else train_df
    y_train = (train_df["label"] == "ai").astype(int).values
    y_val = (val_df["label"] == "ai").astype(int).values

    experiments = {
        "1_stylometry_only": stylometry_cols,
        "2_pseudo_perplexity_only": PERPLEXITY_COLS,
        "3_stylometry_plus_pseudo_perplexity": stylometry_cols + PERPLEXITY_COLS,
        "4_stylometry_plus_pp_plus_curvature": stylometry_cols + PERPLEXITY_COLS + CURVATURE_COLS,
    }

    rows = []
    for name, cols in experiments.items():
        auc = _fit_eval(train_df[cols].values, y_train, val_df[cols].values, y_val)
        rows.append({"experiment": name, "n_features": len(cols), "validation_roc_auc": auc})
        print(f"{name}: {len(cols)} features, validation ROC-AUC = {auc:.4f}" if auc == auc else f"{name}: n/a")

    # Experiment 5: reuse the already-trained, calibrated final model if present.
    calibrated_path = BACKEND_MODELS_DIR / "calibrated_model.joblib"
    if calibrated_path.exists():
        bundle = joblib.load(calibrated_path)
        model = bundle["model"]
        feature_cols = bundle["feature_names"]
        if len(set(y_val)) > 1:
            proba = model.predict_proba(val_df[feature_cols].values)[:, 1]
            auc = roc_auc_score(y_val, proba)
        else:
            auc = float("nan")
        rows.append(
            {"experiment": "5_calibrated_final_model_with_abstention", "n_features": len(feature_cols), "validation_roc_auc": auc}
        )
    else:
        print("calibrated_model.joblib not found; skipping experiment 5 (run calibrate.py first).")

    comparison_df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "model_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison table to {out_csv}")

    valid = comparison_df.dropna(subset=["validation_roc_auc"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(valid["experiment"], valid["validation_roc_auc"], color="#3B4C6B")
        ax.set_xlabel("Validation ROC-AUC")
        ax.set_xlim(0, 1)
        ax.set_title("Feature Ablation Comparison")
        fig.tight_layout()
        out_png = DATA_DIR / "model_comparison.png"
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
