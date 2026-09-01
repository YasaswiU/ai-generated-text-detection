"""
Pydantic request/response schemas for the /api/analyze endpoint.

Keeping these in one place guarantees the FastAPI response always matches
what the React frontend expects (see frontend/src/services/api.js).
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

LanguageChoice = Literal["auto", "en", "te", "hi"]
PredictionLabel = Literal["AI-GENERATED", "HUMAN-WRITTEN", "UNCERTAIN"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The submission to analyze.")
    language: LanguageChoice = Field(
        default="auto",
        description="Language of the text, or 'auto' to auto-detect.",
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip() if value else value


class PseudoPerplexityFeatures(BaseModel):
    mean: float
    median: float
    std: float


class CurvatureFeatures(BaseModel):
    mean_first_difference: float
    std_first_difference: float
    mean_abs_first_difference: float
    mean_second_difference: float
    std_second_difference: float
    mean_abs_second_difference: float
    max_absolute_curvature: float
    curvature_sign_changes: int


class SegmentScore(BaseModel):
    segment: int
    score: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: PredictionLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    abstained: bool
    language: str
    word_count: int
    sentence_count: int
    stylometric_features: Dict[str, float]
    pseudo_perplexity: PseudoPerplexityFeatures
    curvature: CurvatureFeatures
    segments: List[SegmentScore]
    explanation: str
    model_version: str
    disclaimer: str = (
        "AI-text detection is probabilistic and should not be treated as "
        "definitive proof of authorship."
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    supported_languages: List[str]
