"""Dense embeddings via a SentenceTransformer model."""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Singleton-style embedding wrapper. Load once, reuse."""

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading embedding model %s", settings.embedding_model)
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        if settings.embedding_max_seq_length:
            self._model.max_seq_length = settings.embedding_max_seq_length
        self._loaded = True
        logger.info("Embedding model loaded")

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        self._load()
        batch_size = batch_size or settings.embedding_batch_size
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
