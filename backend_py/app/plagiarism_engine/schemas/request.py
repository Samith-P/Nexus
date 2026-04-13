from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


CheckType = Literal["exact", "semantic", "full", "citation"]


class PlagiarismRequest(BaseModel):
    # One of these must be provided (file is handled separately for multipart uploads)
    text: Optional[str] = Field(default=None, description="Raw text to check (when not uploading a file)")
    doi: Optional[str] = Field(default=None, description="DOI to fetch and compare")
    paper_id: Optional[str] = Field(default=None, description="Semantic Scholar paperId to fetch and compare")

    user_id: str = Field(..., min_length=1)
    check_type: CheckType = Field(default="full")
    language: Optional[str] = None


class PlagiarismFileForm(BaseModel):
    """Helper for documenting multipart form fields."""

    user_id: str
    check_type: CheckType = "full"
    language: Optional[str] = None
    doi: Optional[str] = None
    paper_id: Optional[str] = None
    text: Optional[str] = None
