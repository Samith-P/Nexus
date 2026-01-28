from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

from dotenv import load_dotenv

from recommender import JournalRecommender
from gemini.gemini_client import format_abstract_with_gemini


# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")


app = FastAPI(
    title="AI-Powered Journal Recommendation API",
    version="1.1.0"
)

# -----------------------------
# Existing Journal Recommender
# -----------------------------

recommender = JournalRecommender()


class RecommendRequest(BaseModel):
    abstract: str = Field(..., min_length=30)
    top_k: int = Field(5, ge=1, le=10)


class RecommendResponse(BaseModel):
    recommendations: list


@app.post("/recommend/journals", response_model=RecommendResponse)
def recommend_journals(req: RecommendRequest):
    """
    Existing endpoint.
    Takes raw abstract and returns recommended journals.
    """
    recs = recommender.recommend(req.abstract, req.top_k)
    return {"recommendations": recs}


# -----------------------------
# Phase 1: Gemini Formatting
# -----------------------------

class GeminiFormatRequest(BaseModel):
    abstract: str = Field(..., min_length=30)


class GeminiFormatResponse(BaseModel):
    primary_research_area: str
    secondary_areas: list[str]
    methods: list[str]
    application_domains: list[str]
    key_concepts: list[str]
    condensed_summary: str


@app.post(
    "/gemini-formatting-first",
    response_model=GeminiFormatResponse
)
def gemini_formatting_first(req: GeminiFormatRequest):
    """
    Phase 1 endpoint.
    Uses Gemini with automatic model fallback to
    convert raw abstract into structured JSON.
    """

    try:
        structured = format_abstract_with_gemini(
            abstract=req.abstract,
            api_key=GEMINI_API_KEY
        )
        return structured

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini formatting failed: {str(e)}"
        )
