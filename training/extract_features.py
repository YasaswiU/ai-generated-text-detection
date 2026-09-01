"""
Reads training/data/manifest.csv and extracts the same feature vector used
at inference time (backend/app/features/feature_extractor.py), saving the
result to training/data/features.csv.

We deliberately import the backend's feature-extraction code directly
(rather than re-implementing it here) so training and inference can never
drift out of sync.
"""
import sys
from pathlib import Path

import pandas as pd

# Make the backend package importable from this sibling directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.features.feature_extractor import (  # noqa: E402
    build_feature_vector,
    extract_all_features,
    get_feature_names,
)
from app.ml.model_loader import load_perplexity_model  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("manifest.csv not found. Run prepare_dataset.py first.")

    manifest = pd.read_csv(manifest_path)
    settings = get_settings()

    print(f"Loading pseudo-perplexity model '{settings.perplexity_model_name}'...")
    tokenizer, model = load_perplexity_model()
    if tokenizer is None:
        print(
            "WARNING: perplexity model failed to load (no network access?). "
            "Continuing with stylometry-only features -- pp_* and curvature "
            "columns will be zero. This is fine for a local pipeline smoke "
            "test but not for a real model."
        )

    feature_rows = []
    stylometric_keys = None

    for i, row in manifest.iterrows():
        stylo, pp_summary, curvature, _segments = extract_all_features(
            row["text"], tokenizer=tokenizer, model=model, device=settings.inference_device
        )
        if stylometric_keys is None:
            stylometric_keys = list(stylo.keys())

        vector = build_feature_vector(stylo, pp_summary, curvature)
        feature_rows.append(vector)

        if (i + 1) % 25 == 0 or (i + 1) == len(manifest):
            print(f"  extracted features for {i + 1}/{len(manifest)} documents")

    feature_names = get_feature_names(stylometric_keys)
    features_df = pd.DataFrame(feature_rows, columns=feature_names)
    features_df["language"] = manifest["language"].values
    features_df["label"] = manifest["label"].values
    features_df["split"] = manifest["split"].values

    out_path = DATA_DIR / "features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Wrote {len(features_df)} feature rows ({len(feature_names)} features) to {out_path}")


if __name__ == "__main__":
    main()
