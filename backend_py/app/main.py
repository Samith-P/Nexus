# app/main.py
# FastAPI entry point for the Literature Review Engine

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.literature import router as literature_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Nexus Journal — Literature Review Engine",
    description=(
        "AI-driven Literature Review Module that analyzes research papers, "
        "extracts insights, identifies research gaps, maps related works, "
        "and generates structured review reports."
    ),
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(literature_router)


@app.get("/")
async def root():
    return {
        "service": "Nexus Journal — Literature Review Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "review_async": "POST /api/literature/review",
            "review_sync": "POST /api/literature/review/sync",
            "status": "GET /api/literature/status/{task_id}",
            "results": "GET /api/literature/results/{task_id}",
            "export": "GET /api/literature/export/{task_id}",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
