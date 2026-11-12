"""Reranker using Qwen3-Reranker-0.6B via sentence-transformers CrossEncoder."""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    _instance: "Reranker | None" = None

    def __new__(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading reranker %s", settings.reranker_model)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            settings.reranker_model,
            trust_remote_code=True,
            max_length=512,
        )
        self._loaded = True
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        self._load()

        pairs = [(query, c["payload"].get("text", "")) for c in candidates]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        return [
            {**c, "rerank_score": float(s)}
            for s, c in scored[:top_k]
        ]
