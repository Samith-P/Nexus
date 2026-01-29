from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from typing import List

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
    version="3.0.0"
)

# Initialize recommender once at startup
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


@app.post(
    "/gemini-formatting",
    response_model=GeminiFormatResponse
)
def gemini_formatting(req: GeminiFormatRequest):
    """
    Phase 1: Gemini Formatting
    Converts raw abstract into structured JSON using Gemini.
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
    abstract: str = Field(..., min_length=30, description="Research abstract")
    top_k: int = Field(10, ge=1, le=20, description="Number of recommendations (1-20)")


class RecommendJournalsResponse(BaseModel):
    query_text: str
    semantic_model: str
    journals: List[JournalRecommendationItem]
    metadata: dict


@app.post(
    "/recommend/journals",
    response_model=RecommendJournalsResponse,
    summary="Recommend journals based on research abstract"
)
def recommend_journals(req: RecommendJournalsRequest):
    """
    Phase 3: Journal Recommendations
    
    Pipeline:
    1. Abstract → Gemini formatting (Phase 1)
    2. Gemini output → semantic query (Phase 3)
    3. Query → embedding (all-MiniLM-L6-v2)
    4. Embedding → similarity search against journals_master.json
    5. Results → metric-aware scoring
    6. Scored results → deterministic explanations
    
    Returns top-K recommendations with explanations.
    """
    try:
        # Step 1: Format abstract with Gemini
        gemini_output = format_abstract_with_gemini(
            abstract=req.abstract,
            api_key=GEMINI_API_KEY
        )
        
        # Step 2: Generate recommendations using Phase 3 engine
        recommendations = recommender.recommend(
            gemini_output=gemini_output,
            top_k=req.top_k,
            search_depth=max(50, req.top_k * 3)
        )
        
        # Step 3: Build response
        return RecommendJournalsResponse(
            query_text=f"{gemini_output.get('primary_research_area')} - {gemini_output.get('condensed_summary')}",
            semantic_model="all-MiniLM-L6-v2",
            journals=[JournalRecommendationItem(**journal) for journal in recommendations],
            metadata={
                "total_journals_indexed": 13946,
                "scoring_formula": "0.55*semantic + 0.25*sjr + 0.20*citations",
                "explanation_type": "deterministic"
            }
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )


# ==============================
# System Information
# ==============================

class SystemInfoResponse(BaseModel):
    version: str
    phases: List[str]
    journal_count: int
    embedding_model: str
    status: str


@app.get("/system/info", response_model=SystemInfoResponse)
def system_info():
    """Get system information."""
    return SystemInfoResponse(
        version="3.0.0",
        phases=["Phase 1: Gemini Formatting", "Phase 3: Journal Recommendation"],
        journal_count=13946,
        embedding_model="all-MiniLM-L6-v2",
        status="active"
    )


# Legacy endpoints for backward compatibility
class RecommendRequest(BaseModel):
    abstract: str = Field(..., min_length=30)
    top_k: int = Field(5, ge=1, le=10)


class RecommendResponse(BaseModel):
    recommendations: list


@app.post("/recommend/journals-legacy", response_model=RecommendResponse, deprecated=True)
def recommend_journals_legacy(req: RecommendRequest):
    """
    Legacy endpoint (deprecated).
    Use /recommend/journals instead.
    """
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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Legacy Gemini formatting endpoint
class GeminiFormatRequest(BaseModel):
    abstract: str = Field(..., min_length=30)


class GeminiFormatResponse(BaseModel):
    primary_research_area: str
    secondary_areas: List[str]
    methods: List[str]
    application_domains: List[str]
    key_concepts: List[str]
    condensed_summary: str


@app.post(
    "/gemini-formatting-first",
    response_model=GeminiFormatResponse,
    deprecated=True
)
def gemini_formatting_first(req: GeminiFormatRequest):
    """
    Legacy endpoint (deprecated).
    Use /gemini-formatting instead.
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
