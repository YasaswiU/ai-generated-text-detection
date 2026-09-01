"""
Orchestrates the full feature pipeline: stylometry + pseudo-perplexity +
curvature, into a single flat feature vector matching the vector the
training pipeline produces (see training/extract_features.py).

The two most important invariants this module preserves:
  1. Feature ORDER and NAMES must exactly match training/extract_features.py
     (both import get_feature_names() from here for that reason).
  2. This function never raises on ordinary text; text-length validation
     happens earlier in app/api/routes.py.
"""
import statistics
from typing import Dict, List, Tuple

from app.features.curvature import compute_curvature_features
from app.features.perplexity import pseudo_perplexity_for_segments
from app.features.stylometry import extract_stylometric_features
from app.preprocessing.text_cleaner import split_sentences

_NUM_SEGMENTS = 5


def _segment_text(text: str, num_segments: int = _NUM_SEGMENTS) -> List[str]:
    """Splits text into up to `num_segments` roughly equal sentence groups."""
    sentences = split_sentences(text)
    if not sentences:
        return [text]
    if len(sentences) <= num_segments:
        return sentences

    chunk_size = max(1, len(sentences) // num_segments)
    segments = [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]
    return segments[:num_segments] if len(segments) > num_segments else segments


# Ordered list of the flat, numeric feature names fed into the classifier.
# Stylometric feature names are dynamic in principle, but stable in practice
# for a given version of extract_stylometric_features(); FEATURE_VERSION in
# model_metadata.json should be bumped if this list ever changes.
FEATURE_VERSION = "1.0.0"

_PERPLEXITY_FEATURE_NAMES = ["pp_mean", "pp_median", "pp_std"]
_CURVATURE_FEATURE_NAMES = [
    "mean_first_difference",
    "std_first_difference",
    "mean_abs_first_difference",
    "mean_second_difference",
    "std_second_difference",
    "mean_abs_second_difference",
    "max_absolute_curvature",
    "curvature_sign_changes",
]


def get_feature_names(stylometric_keys: List[str]) -> List[str]:
    return list(stylometric_keys) + _PERPLEXITY_FEATURE_NAMES + _CURVATURE_FEATURE_NAMES


def extract_all_features(
    text: str, tokenizer=None, model=None, device: str = "cpu"
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    Returns (stylometric_features, pseudo_perplexity_summary, curvature_features, segments)

    `segments` is a list of {"segment": i, "score": pseudo_perplexity} used
    for the frontend's segment-level predictability chart.
    """
    stylometric_features = extract_stylometric_features(text)

    segment_texts = _segment_text(text)

    if tokenizer is not None and model is not None:
        segment_scores = pseudo_perplexity_for_segments(segment_texts, tokenizer, model, device)
    else:
        segment_scores = []

    if not segment_scores:
        # Degrades gracefully (e.g. during tests) rather than crashing.
        pseudo_perplexity_summary = {"mean": 0.0, "median": 0.0, "std": 0.0}
    else:
        pseudo_perplexity_summary = {
            "mean": float(statistics.mean(segment_scores)),
            "median": float(statistics.median(segment_scores)),
            "std": float(statistics.pstdev(segment_scores)) if len(segment_scores) > 1 else 0.0,
        }

    curvature_features = compute_curvature_features(segment_scores)

    segments = [
        {"segment": i + 1, "score": round(score, 3)} for i, score in enumerate(segment_scores)
    ]

    return stylometric_features, pseudo_perplexity_summary, curvature_features, segments


def build_feature_vector(
    stylometric_features: Dict[str, float],
    pseudo_perplexity_summary: Dict[str, float],
    curvature_features: Dict[str, float],
) -> List[float]:
    """Flattens the three feature groups into the ordered vector the model expects."""
    vector = list(stylometric_features.values())
    vector += [
        pseudo_perplexity_summary["mean"],
        pseudo_perplexity_summary["median"],
        pseudo_perplexity_summary["std"],
    ]
    vector += [curvature_features[name] for name in _CURVATURE_FEATURE_NAMES]
    return vector
