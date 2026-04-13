from __future__ import annotations

import os


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


class Settings:
    # Embeddings
    embed_provider: str = (os.getenv("PLAG_EMBED_PROVIDER", "hf_api") or "hf_api").strip().lower()
    hf_token: str | None = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN"))
    hf_model: str = (os.getenv("PLAG_HF_API_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") or "").strip()
    hf_batch_size: int = _env_int("PLAG_HF_BATCH_SIZE", 16)

    # Local (sentence-transformers) embeddings
    local_model_name: str = (
        (os.getenv("PLAG_LOCAL_MODEL_NAME") or os.getenv("SENTENCE_TRANSFORMERS_MODEL") or "all-MiniLM-L6-v2").strip()
    )
    local_device: str = (os.getenv("PLAG_LOCAL_DEVICE", "cpu") or "cpu").strip()
    local_batch_size: int = _env_int("PLAG_LOCAL_BATCH_SIZE", 64)

    # Similarity thresholds
    semantic_threshold: float = _env_float("PLAG_SEM_TH", 0.55)
    external_semantic_threshold: float = _env_float("PLAG_EXTERNAL_SEM_TH", 0.55)
    # Minimum similarity required to ACCEPT an external semantic match (prevents unrelated source mapping).
    # This should generally be higher than PLAG_EXTERNAL_SEM_TH.
    external_accept_threshold: float = _env_float("PLAG_EXTERNAL_ACCEPT_TH", 0.55)
    exact_threshold: float = _env_float("PLAG_EXACT_TH", 0.85)

    # External retrieval
    enable_external_checks: bool = _env_bool("PLAG_ENABLE_EXTERNAL_CHECKS", True)
    enable_semantic_scholar_search: bool = _env_bool("PLAG_ENABLE_SEMANTIC_SCHOLAR_SEARCH", True)
    semantic_scholar_api_key: str | None = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip() or None

    enable_openalex: bool = _env_bool("PLAG_ENABLE_OPENALEX", True)
    openalex_api_key: str | None = os.getenv("OPENALEX_API_KEY")
    openalex_mailto: str | None = os.getenv("OPENALEX_MAILTO")

    enable_arxiv: bool = _env_bool("PLAG_ENABLE_ARXIV", False)

    enable_tavily: bool = _env_bool("PLAG_ENABLE_TAVILY", False)
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    tavily_search_depth: str = (os.getenv("TAVILY_SEARCH_DEPTH", "basic") or "basic").strip()

    enable_opencitations: bool = _env_bool("PLAG_ENABLE_OPENCITATIONS", False)
    opencitations_token: str | None = os.getenv("OPENCITATIONS_TOKEN")

    external_max_queries: int = _env_int("PLAG_EXTERNAL_MAX_QUERIES", 3)
    external_per_query_results: int = _env_int("PLAG_EXTERNAL_PER_QUERY_RESULTS", 10)
    external_max_docs: int = _env_int("PLAG_EXTERNAL_MAX_DOCS", 50)

    # Chunking / compute guards
    max_chunks: int = _env_int("PLAG_MAX_CHUNKS", 120)
    embed_max_chunks: int = _env_int("PLAG_EMBED_MAX_CHUNKS", 20)

    # Debug
    debug: bool = _env_bool("PLAG_DEBUG", False)

    # Cache
    cache_dir: str = (os.getenv("PLAG_CACHE_DIR", "") or "").strip() or ".plag_cache"
    external_cache_ttl_seconds: int = _env_int("PLAG_EXTERNAL_CACHE_TTL_SECONDS", 86400)

    # Qdrant (optional semantic index/search)
    use_qdrant: bool = _env_bool("PLAG_USE_QDRANT", True)
    qdrant_url: str | None = (os.getenv("QDRANT_URL") or os.getenv("PLAG_QDRANT_URL"))
    qdrant_host: str | None = (os.getenv("QDRANT_HOST") or os.getenv("PLAG_QDRANT_HOST"))
    qdrant_port: int = _env_int("QDRANT_PORT", _env_int("PLAG_QDRANT_PORT", 6333))
    qdrant_api_key: str | None = (os.getenv("QDRANT_API_KEY") or os.getenv("PLAG_QDRANT_API_KEY"))
    qdrant_collection: str = (os.getenv("PLAG_QDRANT_COLLECTION", "plagiarism_sources") or "plagiarism_sources").strip()
    qdrant_upsert_batch_size: int = _env_int("PLAG_QDRANT_UPSERT_BATCH_SIZE", 64)
    qdrant_search_top_k: int = _env_int("PLAG_QDRANT_TOP_K", 5)
    qdrant_cleanup_after_request: bool = _env_bool("PLAG_QDRANT_CLEANUP", True)
    qdrant_payload_text_max_chars: int = _env_int("PLAG_QDRANT_PAYLOAD_TEXT_MAX_CHARS", 2000)

    # Hybrid reranking (vector score + lexical overlap)
    hybrid_enabled: bool = _env_bool("PLAG_HYBRID_ENABLED", True)
    hybrid_alpha: float = _env_float("PLAG_HYBRID_ALPHA", 0.75)

    # Document-level (title) match
    clean_query_max_chars: int = _env_int("PLAG_CLEAN_QUERY_MAX_CHARS", 150)
    document_title_match_threshold: float = _env_float("PLAG_DOC_TITLE_MATCH_TH", 0.85)
    # When doc-level similarity exceeds this threshold, override scoring as a full-document match.
    # This is intentionally lower than PLAG_DOC_TITLE_MATCH_TH (which is used for strict early-exit).
    document_title_override_threshold: float = _env_float("PLAG_DOC_TITLE_OVERRIDE_TH", 0.50)
    document_title_search_limit: int = _env_int("PLAG_DOC_TITLE_SEARCH_LIMIT", 10)

    # Persistent external paper index in Qdrant
    qdrant_external_enabled: bool = _env_bool("PLAG_QDRANT_EXTERNAL_ENABLED", True)
    qdrant_external_collection: str = (
        os.getenv("PLAG_QDRANT_EXTERNAL_COLLECTION", "external_papers") or "external_papers"
    ).strip()
    qdrant_external_source_filter: str = (os.getenv("PLAG_QDRANT_EXTERNAL_SOURCE_FILTER", "") or "").strip()


settings = Settings()
