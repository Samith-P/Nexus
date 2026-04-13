from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
from typing import List
from dotenv import load_dotenv

from .gemini.gemini_client import format_abstract_with_gemini

try:
    from recommender import JournalRecommender
except:
    from .recommender import JournalRecommender
# Load env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")


# -------- Router --------
router = APIRouter(prefix="/journals", tags=["Journal Recommendation"])

# Initialize once
recommender = JournalRecommender()


# ==============================
# PHASE 1: Gemini Formatting
# ==============================

class GeminiFormatRequest(BaseModel):
    abstract: str = Field(..., min_length=30)


class GeminiFormatResponse(BaseModel):
    primary_research_area: str
    secondary_areas: List[str]
    methods: List[str]
    application_domains: List[str]
    key_concepts: List[str]
    condensed_summary: str


@router.post("/gemini-formatting", response_model=GeminiFormatResponse)
def gemini_formatting(req: GeminiFormatRequest):
    try:
        return format_abstract_with_gemini(
            abstract=req.abstract,
            api_key=GEMINI_API_KEY
        )
    except Exception as e:
        raise HTTPException(500, f"Gemini formatting failed: {str(e)}")


# ==============================
# PHASE 3: Journal Recommendations
# ==============================

class JournalRecommendationItem(BaseModel):
    title: str
    type: str
    sjr: float
    quartile: str
    h_index: int
    citations_per_doc_2y: float
    publisher: str
    open_access: str
    country: str
    semantic_score: float
    final_score: float
    explanation: str


class RecommendJournalsRequest(BaseModel):
    abstract: str = Field(..., min_length=30)
    top_k: int = Field(10, ge=1, le=20)


class RecommendJournalsResponse(BaseModel):
    query_text: str
    semantic_model: str
    journals: List[JournalRecommendationItem]
    metadata: dict


@router.post("/recommend", response_model=RecommendJournalsResponse)
def recommend_journals(req: RecommendJournalsRequest):
    try:
        gemini_output = format_abstract_with_gemini(
            abstract=req.abstract,
            api_key=GEMINI_API_KEY
        )

        recommendations = recommender.recommend(
            gemini_output=gemini_output,
            top_k=req.top_k,
            search_depth=max(50, req.top_k * 3)
        )

        return RecommendJournalsResponse(
            query_text=f"{gemini_output.get('primary_research_area')} - {gemini_output.get('condensed_summary')}",
            semantic_model="all-MiniLM-L6-v2",
            journals=[JournalRecommendationItem(**j) for j in recommendations],
            metadata={
                "total_journals_indexed": 13946,
                "scoring_formula": "0.55*semantic + 0.25*sjr + 0.20*citations",
                "explanation_type": "deterministic"
            }
        )

    except ValueError as e:
        raise HTTPException(400, f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {str(e)}")


# ==============================
# System Info
# ==============================

class SystemInfoResponse(BaseModel):
    version: str
    phases: List[str]
    journal_count: int
    embedding_model: str
    status: str


@router.get("/system-info", response_model=SystemInfoResponse)
def system_info():
    return SystemInfoResponse(
        version="3.0.0",
        phases=["Phase 1", "Phase 3"],
        journal_count=13946,
        embedding_model="all-MiniLM-L6-v2",
        status="active"
    )


# ==============================
# Legacy
# ==============================

class RecommendRequest(BaseModel):
    abstract: str = Field(..., min_length=30)
    top_k: int = Field(5, ge=1, le=10)


class RecommendResponse(BaseModel):
    recommendations: list


@router.post("/legacy", response_model=RecommendResponse, deprecated=True)
def recommend_journals_legacy(req: RecommendRequest):
    try:
        gemini_output = format_abstract_with_gemini(
            abstract=req.abstract,
            api_key=GEMINI_API_KEY
        )

        recommendations = recommender.recommend(
            gemini_output=gemini_output,
            top_k=req.top_k
        )

        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(500, str(e))