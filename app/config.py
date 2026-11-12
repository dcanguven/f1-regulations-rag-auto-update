from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # FIA Source
    fia_catalog_url: str = "https://www.fia.com/regulation/category/110"
    fia_base_url: str = "https://www.fia.com"

    # Qdrant — set qdrant_path for local embedded mode (no Docker needed)
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""          # e.g. "data/qdrant"; overrides qdrant_url when set
    qdrant_collection: str = "f1_regulations_minilm"
    qdrant_lock_wait_seconds: float = 30.0

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_device: str = "cpu"  # "cpu" | "mps" | "cuda"
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 256

    # Reranker
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    enable_reranker: bool = False

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    llm_num_ctx: int = 6144
    llm_max_tokens: int = 320
    llm_timeout: float = 240.0
    llm_keep_alive: str = "30m"
    claim_min_support: float = 0.5
    claim_anchored_min_support: float = 0.2

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    exact_context_max_chunks: int = 12
    exact_context_max_tokens: int = 2400
    exact_fetch_limit: int = 1000
    enable_context_expansion: bool = False

    # Chunking
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32

    pipeline_version: str = "3"
    table_chunk_mode: str = "packed"
    table_row_chunk_max_rows: int = 25

    # System
    default_season: int = 2026
    data_dir: str = "data"
    db_path: str = "data/source_registry.db"
    log_level: str = "INFO"
    log_dir: str = "logs"
    update_interval_hours: float = 20.0

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def db_file(self) -> Path:
        return Path(self.db_path)


settings = Settings()
