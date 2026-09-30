"""FastAPI application factory (CT-23).

There is no module-level ``app`` object: run with
``uvicorn ticket_classifier.api.app:create_app --factory``. The promoted model
version is loaded once at startup and never reloaded while the process runs
(LAC-24); if it cannot be loaded the API still starts without a model (LAC-21).
"""

import logging
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request

from ticket_classifier.api.errors import register_error_handlers
from ticket_classifier.models.base import TicketClassifier
from ticket_classifier.models.loader import load_classifier
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import Settings, get_settings
from ticket_classifier.storage.database import connect, init_schema

logger = logging.getLogger(__name__)

API_KEY_MISSING_WARNING = "API key not configured: /feedback will reject all requests."


@dataclass
class AppState:
    """Per-process API state stored in ``app.state.ctx``."""

    settings: Settings
    classifier: TicketClassifier | None
    model_version: str | None


def _load_promoted(settings: Settings) -> tuple[TicketClassifier | None, str | None]:
    """Load the promoted version once; any failure leaves the API without a model."""
    try:
        version = ModelRegistry(settings.artifacts_dir).get_promoted()
        if version is None:
            logger.error("No promoted model version available; serving without a model.")
            return None, None
        classifier = load_classifier(version, settings.artifacts_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure must not stop startup (LAC-21)
        logger.error(
            "Promoted model could not be loaded (%s: %s); serving without a model.",
            type(exc).__name__,
            exc,
        )
        return None, None
    logger.info("Loaded promoted model version %s (%s).", version.version_id, version.kind)
    return classifier, version.version_id


def _init_database(settings: Settings) -> None:
    conn = connect(settings.db_path)
    try:
        init_schema(conn)
    finally:
        conn.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API. ``settings`` defaults to a fresh ``get_settings()``."""
    resolved = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _init_database(resolved)
        classifier, model_version = _load_promoted(resolved)
        if resolved.api_key is None:
            logger.warning(API_KEY_MISSING_WARNING)
        app.state.ctx = AppState(
            settings=resolved, classifier=classifier, model_version=model_version
        )
        yield

    # Routers import this module for their dependencies; import them here to
    # avoid a circular import at module load time.
    from ticket_classifier.api.routes_health import router as health_router
    from ticket_classifier.api.routes_predict import router as predict_router

    app = FastAPI(title="ticket-classifier", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(predict_router)
    return app


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency returning the ``AppState`` set up by the lifespan."""
    state = getattr(request.app.state, "ctx", None)
    if not isinstance(state, AppState):
        raise RuntimeError("Application state is not initialized.")
    return state


def get_connection(request: Request) -> Iterator[sqlite3.Connection]:
    """FastAPI dependency yielding one SQLite connection per request."""
    conn = connect(get_app_state(request).settings.db_path)
    try:
        yield conn
    finally:
        conn.close()
