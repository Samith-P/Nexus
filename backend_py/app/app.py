from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path

# Add engine path for imports
sys.path.insert(0, str(Path(__file__).parent / "AI‑Powered-Topic-Selection-Engine"))

import engine as engine_mod  # type: ignore
from engine import generate_topics  # type: ignore

app = FastAPI(
    title="Research Topic Selection Engine",
    description="AI-powered topic recommendation using academic APIs + policy alignment",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/topic-recommendation")
def topic_recommendation(payload: dict):
    """Generate recommended research topics based on query and policies."""
    return generate_topics(payload)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "topic-engine"}

@app.get("/", include_in_schema=False)
def root():
    """Root endpoint."""
    return {
        "message": "Research Topic Selection Engine API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "topic_recommendation": "/topic-recommendation"
        }
    }


@app.get("/debug", include_in_schema=False)
def debug_config():
    """Debug endpoint to verify which code/config is running (no secrets)."""

    if os.getenv("ENABLE_DEBUG_ENDPOINT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return {"enabled": False}

    try:
        bad_title = "Technology Innovation in Commerce Department"
        check = bool(engine_mod._looks_like_research_topic(bad_title, "EV adoption challenges"))
    except Exception:
        check = None

    return {
        "engine_file": getattr(engine_mod, "__file__", None),
        "sem_min_hard": os.getenv("SEMANTIC_MIN_HARD", "0.40"),
        "sem_min_rank": os.getenv("SEMANTIC_MIN", "0.40"),
        "policy_hit_min": os.getenv("POLICY_HIT_MIN", "0.25"),
        "topic_shape_allows_bad_title": check,
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
