"""FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="F1 Regulations RAG",
    description="Answer questions from official FIA Formula 1 regulations.",
    version="0.1.0",
)

app.include_router(router)
