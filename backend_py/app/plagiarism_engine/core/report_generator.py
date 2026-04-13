from __future__ import annotations

from ..schemas.response import MatchItem, MissingCitationItem, PlagiarismResponse


def build_report(
    *,
    report_id: str,
    originality_score: float,
    plagiarism_percentage: float,
    exact_match_percentage: int,
    semantic_match_percentage: float,
    citation_coverage_percentage: float,
    matches: list[MatchItem],
    missing_citations: list[MissingCitationItem],
    debug: dict | None = None,
    status: str | None = None,
    errors: list[str] | None = None,
) -> PlagiarismResponse:
    return PlagiarismResponse(
        report_id=report_id,
        status=status,  # type: ignore[arg-type]
        errors=errors,
        originality_score=originality_score,
        plagiarism_percentage=plagiarism_percentage,
        exact_match_percentage=exact_match_percentage,
        semantic_match_percentage=semantic_match_percentage,
        citation_coverage_percentage=citation_coverage_percentage,
        matches=matches,
        missing_citations=missing_citations,
        debug=debug,
    )
