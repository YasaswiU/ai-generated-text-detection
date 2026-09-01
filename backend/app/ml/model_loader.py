"""
Loads and caches heavyweight resources exactly once per worker process:
  1. The trained scikit-learn classifier + calibration wrapper (joblib).
  2. The XLM-RoBERTa tokenizer + model used for pseudo-perplexity.
  3. model_metadata.json (version, threshold, calibration method, etc).

Using functools.lru_cache means the first request in a worker pays the
loading cost; every subsequent request in that worker reuses the cached
objects. This directly satisfies the "do not reload the model on every
request" performance requirement.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# backend/app/ml/model_loader.py -> parents[2] == backend/
# Resolving model_dir relative to the backend package (rather than the
# process's current working directory) means the API behaves the same
# whether it's started as `uvicorn app.main:app` from backend/, imported by
# pytest from the repo root, or run inside the Docker image.
_BACKEND_DIR = Path(__file__).resolve().parents[2]


def _resolve_model_dir() -> Path:
    settings = get_settings()
    configured = Path(settings.model_dir)
    return configured if configured.is_absolute() else _BACKEND_DIR / configured


@lru_cache
def load_metadata() -> Dict[str, Any]:
    settings = get_settings()
    metadata_path = _resolve_model_dir() / "model_metadata.json"
    if not metadata_path.exists():
        logger.warning("model_metadata.json not found at %s; using defaults.", metadata_path)
        return {
            "model_version": "untrained",
            "training_date": None,
            "supported_languages": settings.supported_languages,
            "feature_version": "1.0.0",
            "threshold": 0.65,
            "calibration_method": "none",
            "dataset_version": "none",
        }
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def load_classifier() -> Optional[Any]:
    """
    Returns the calibrated scikit-learn classifier saved by training/train.py
    + training/calibrate.py, or None if no trained model is present yet
    (e.g. a fresh clone of the repo before `python training/train.py` runs).
    """
    settings = get_settings()
    model_path = _resolve_model_dir() / "calibrated_model.joblib"
    if not model_path.exists():
        logger.warning(
            "No trained model found at %s. /api/analyze will return an error "
            "until a model is trained (see training/README or the main README's "
            "'Model Training' section).",
            model_path,
        )
        return None
    return joblib.load(model_path)


@lru_cache
def load_perplexity_model():
    """
    Loads the XLM-RoBERTa tokenizer + masked-language-model used for
    pseudo-perplexity. This download happens once on first use and is then
    cached by both this lru_cache and Hugging Face's local model cache.
    """
    settings = get_settings()
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(settings.perplexity_model_name)
        model = AutoModelForMaskedLM.from_pretrained(settings.perplexity_model_name)
        model.to(settings.inference_device)
        model.eval()
        return tokenizer, model
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to load perplexity model '%s'. Pseudo-perplexity and "
            "curvature features will be unavailable until this is resolved "
            "(check network access / disk space on first startup).",
            settings.perplexity_model_name,
        )
        return None, None
