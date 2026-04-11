# app/models/schemas.py
# Pydantic response models for the Literature Review Engine API

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class EvidenceSpan(BaseModel):
    """A specific text span from the paper used as evidence."""
    section: str = ""
    text: str = ""
    page: Optional[int] = None


class InsightResult(BaseModel):
    contributions: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)


class RelatedWork(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None
    source: str = ""  # "semantic_scholar" or "crossref"


class PaperAnalysis(BaseModel):
    """Full analysis result for a single paper."""
    title: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    section_summaries: dict[str, str] = Field(default_factory=dict)
    insights: InsightResult = Field(default_factory=InsightResult)
    gaps: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class ComparisonEntry(BaseModel):
    paper_title: str = ""
    methods: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ComparisonMatrix(BaseModel):
    entries: list[ComparisonEntry] = Field(default_factory=list)
    common_methods: list[str] = Field(default_factory=list)
    differing_methods: list[str] = Field(default_factory=list)


class LiteratureReviewResult(BaseModel):
    """Complete literature review output."""
    papers: list[PaperAnalysis] = Field(default_factory=list)
    comparison_matrix: Optional[ComparisonMatrix] = None
    common_themes: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    related_works: list[RelatedWork] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReviewRequest(BaseModel):
    """API request model."""
    query: Optional[str] = None
    output_language: str = "en"
    fetch_related_works: bool = True


class ReviewStatusResponse(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[LiteratureReviewResult] = None
    error: Optional[str] = None
