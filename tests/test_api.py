"""
Integration tests for the FastAPI app (backend/app/main.py) using FastAPI's
TestClient. These exercise /health and /api/analyze, including validation
errors, without needing a real network call to a running server.

NOTE: these tests require a trained model at backend/models/calibrated_model.joblib
(produced by training/train.py + training/calibrate.py). If no model is
present, the analyze test is skipped rather than failing the suite, so a
fresh clone of the repo can still run `pytest` successfully.
"""
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

_MODEL_PRESENT = (BACKEND_DIR / "models" / "calibrated_model.joblib").exists()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "supported_languages" in body


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_analyze_rejects_empty_text():
    response = client.post("/api/analyze", json={"text": "", "language": "auto"})
    assert response.status_code == 422


def test_analyze_rejects_too_short_text():
    response = client.post("/api/analyze", json={"text": "hi", "language": "auto"})
    assert response.status_code == 422


def test_analyze_rejects_malformed_request():
    response = client.post("/api/analyze", json={"language": "auto"})  # missing "text"
    assert response.status_code == 422


def test_analyze_rejects_unsupported_language():
    response = client.post(
        "/api/analyze", json={"text": "a" * 100, "language": "fr"}
    )
    assert response.status_code == 422


@pytest.mark.skipif(not _MODEL_PRESENT, reason="No trained model present; run the training pipeline first.")
def test_analyze_returns_expected_schema():
    text = (
        "The implementation of renewable energy infrastructure represents a "
        "critical component of sustainable development strategies today."
    )
    response = client.post("/api/analyze", json={"text": text, "language": "auto"})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"AI-GENERATED", "HUMAN-WRITTEN", "UNCERTAIN"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert "disclaimer" in body
    assert "pseudo_perplexity" in body
    assert "curvature" in body
