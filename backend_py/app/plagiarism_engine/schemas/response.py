from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


MatchType = Literal["external", "internal"]


class MatchItem(BaseModel):
    # Keep only fields shown in the required API output.
    text: str
    source: str
    similarity: float
    type: MatchType
    url: Optional[str] = None
    doi: Optional[str] = None


class MissingCitationItem(BaseModel):
    text: str
    suggested_source: str
    url: Optional[str] = None
    doi: Optional[str] = None


class PlagiarismResponse(BaseModel):
    # Required top-level fields (match the sample response shape)
    report_id: str

    originality_score: float = Field(..., ge=0, le=100)
    plagiarism_percentage: float = Field(..., ge=0, le=100)
    exact_match_percentage: int = Field(default=0, ge=0, le=100)
    semantic_match_percentage: float = Field(default=0, ge=0, le=100)
    citation_coverage_percentage: float = Field(default=0, ge=0, le=100)

    matches: list[MatchItem] = Field(default_factory=list)
    missing_citations: list[MissingCitationItem] = Field(default_factory=list)
    debug: Optional[dict[str, Any]] = None

    # Optional error fields (excluded from normal OK responses)
    status: Optional[Literal["ok", "error"]] = None
    errors: Optional[list[str]] = None

    # Allow extra keys in `debug` without breaking output.
    model_config = {"extra": "allow"}
