"""
Top-level prediction orchestration used by app/api/routes.py.

Ties together: feature extraction -> classifier -> calibration/abstention ->
human-readable explanation, and assembles the final AnalyzeResponse payload.
"""
import logging
from typing import Any, Dict

import numpy as np

from app.core.config import get_settings
from app.features.feature_extractor import build_feature_vector, extract_all_features
from app.ml.calibration import decide_with_abstention
from app.ml.model_loader import load_classifier, load_metadata, load_perplexity_model
from app.preprocessing.language_detector import detect_language
from app.preprocessing.text_cleaner import split_sentences

logger = logging.getLogger(__name__)


def _build_explanation(label: str, abstained: bool) -> str:
    if abstained:
        return (
            "The available evidence was insufficient for a reliable "
            "classification, so the system abstained from making a "
            "definitive prediction."
        )
    if label == "AI-GENERATED":
        return (
            "The model identified patterns in sentence structure, word "
            "predictability, and stylistic consistency that are commonly "
            "associated with AI-generated text. This prediction is "
            "probabilistic and should not be considered definitive proof "
            "of authorship."
        )
    return (
        "The model identified patterns in sentence structure, word choice, "
        "and predictability that are more commonly associated with "
        "human-written text. This prediction is probabilistic and should "
        "not be considered definitive proof of authorship."
    )


def run_analysis(text: str, requested_language: str) -> Dict[str, Any]:
    settings = get_settings()
    metadata = load_metadata()

    language_code, language_name = detect_language(text, requested_language)

    tokenizer, model = load_perplexity_model()
    stylometric_features, pp_summary, curvature_features, segments = extract_all_features(
        text, tokenizer=tokenizer, model=model, device=settings.inference_device
    )

    classifier_bundle = load_classifier()
    if classifier_bundle is None:
        raise RuntimeError(
            "No trained model is available on this server yet. Run the "
            "training pipeline (see training/train.py) and restart the API."
        )

    classifier = classifier_bundle["model"]
    feature_order = classifier_bundle["feature_names"]

    raw_vector = build_feature_vector(stylometric_features, pp_summary, curvature_features)
    vector_by_name = dict(
        zip(
            list(stylometric_features.keys()) + ["pp_mean", "pp_median", "pp_std"] + [
                "mean_first_difference",
                "std_first_difference",
                "mean_abs_first_difference",
                "mean_second_difference",
                "std_second_difference",
                "mean_abs_second_difference",
                "max_absolute_curvature",
                "curvature_sign_changes",
            ],
            raw_vector,
        )
    )
    # Reorder defensively to match the exact order the model was trained on,
    # in case feature_extractor.py's dict ordering ever drifts.
    ordered_vector = np.array([[vector_by_name[name] for name in feature_order]])

    calibrated_proba_ai = float(classifier.predict_proba(ordered_vector)[0][1])
    threshold = float(metadata.get("threshold", 0.65))

    label, confidence, abstained = decide_with_abstention(calibrated_proba_ai, threshold)
    explanation = _build_explanation(label, abstained)

    sentence_count = max(len(split_sentences(text)), 1)
    word_count = int(stylometric_features.get("word_count", 0))

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "abstained": abstained,
        "language": language_name,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "stylometric_features": stylometric_features,
        "pseudo_perplexity": pp_summary,
        "curvature": curvature_features,
        "segments": segments,
        "explanation": explanation,
        "model_version": metadata.get("model_version", "unknown"),
    }
