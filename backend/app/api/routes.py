"""
API route definitions.

Privacy note (see README "Privacy" section): request handlers here log only
metadata (timing, language, prediction, model version, word count) -- never
the submitted text itself.
"""
import logging
import time

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.ml.model_loader import load_classifier, load_metadata
from app.ml.predictor import run_analysis
from app.preprocessing.text_cleaner import validate_length
from app.schemas.prediction import AnalyzeRequest, AnalyzeResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(payload: AnalyzeRequest) -> AnalyzeResponse:
    settings = get_settings()
    start = time.perf_counter()

    try:
        validate_length(payload.text, settings.min_text_length, settings.max_text_length)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = run_analysis(payload.text, payload.language)
    except RuntimeError as exc:
        # Model not trained/loaded yet -- a useful, non-crashing error.
        logger.error("Analysis unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001
        # Never leak stack traces / internal details to the client.
        logger.exception("Unexpected error during analysis.")
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the text right now. Please try again.",
        ) from None

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "analyze_request duration_ms=%s language=%s prediction=%s "
        "model_version=%s word_count=%s",
        duration_ms,
        result["language"],
        result["prediction"],
        result["model_version"],
        result["word_count"],
    )

    return AnalyzeResponse(**result)


def get_health_status() -> HealthResponse:
    """Shared by both the top-level GET /health route (see app/main.py)."""
    settings = get_settings()
    classifier = load_classifier()
    metadata = load_metadata()
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None,
        model_version=metadata.get("model_version"),
        supported_languages=settings.supported_languages,
    )
