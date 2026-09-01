"""
FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload

In production (Render/Docker) this is started via the Dockerfile's CMD,
listening on 0.0.0.0 and the $PORT environment variable.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import get_health_status, router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.schemas.prediction import HealthResponse

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s | CORS origins: %s",
        settings.app_name,
        settings.app_version,
        settings.cors_origin_list,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multilingual AI-generated text analysis with confidence-aware "
        "predictions. AI-text detection is probabilistic and should not be "
        "treated as definitive proof of authorship."
    ),
    lifespan=lifespan,
)

# CORS: production frontend origin(s) come from the CORS_ORIGINS env var.
# localhost is only included by default for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_health_status()


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
