"""Tests for backend/app/ml/calibration.py and schema validation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.ml.calibration import decide_with_abstention
from app.preprocessing.text_cleaner import validate_length
from app.schemas.prediction import AnalyzeRequest


def test_high_confidence_ai_prediction():
    label, confidence, abstained = decide_with_abstention(0.95, threshold=0.65)
    assert label == "AI-GENERATED"
    assert not abstained
    assert confidence == pytest.approx(0.95)


def test_high_confidence_human_prediction():
    label, confidence, abstained = decide_with_abstention(0.05, threshold=0.65)
    assert label == "HUMAN-WRITTEN"
    assert not abstained
    assert confidence == pytest.approx(0.95)


def test_low_confidence_abstains():
    label, confidence, abstained = decide_with_abstention(0.55, threshold=0.65)
    assert label == "UNCERTAIN"
    assert abstained


def test_validate_length_rejects_empty():
    with pytest.raises(ValueError):
        validate_length("", min_length=50, max_length=1000)


def test_validate_length_rejects_too_short():
    with pytest.raises(ValueError):
        validate_length("short", min_length=50, max_length=1000)


def test_validate_length_rejects_too_long():
    with pytest.raises(ValueError):
        validate_length("a" * 2000, min_length=50, max_length=1000)


def test_validate_length_accepts_valid_text():
    validate_length("a" * 100, min_length=50, max_length=1000)  # should not raise


def test_analyze_request_schema_accepts_auto_language():
    request = AnalyzeRequest(text="Some sample text here.", language="auto")
    assert request.language == "auto"


def test_analyze_request_schema_rejects_unsupported_language():
    with pytest.raises(Exception):
        AnalyzeRequest(text="Some sample text here.", language="fr")
