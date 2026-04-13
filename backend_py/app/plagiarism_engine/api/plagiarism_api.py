from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ..core.engine import run_check
from ..schemas.request import CheckType, PlagiarismRequest
from ..schemas.response import PlagiarismResponse


router = APIRouter(prefix="/plagiarism", tags=["plagiarism"])


def _is_meaningful_form_str(value: str | None) -> bool:
    if value is None:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    # Swagger UI often pre-fills optional fields with the literal placeholder "string".
    # Treat such placeholder values as "not provided".
    return cleaned.lower() not in {"string", "null", "none"}


def _normalize_form_str(value: str | None) -> str | None:
    return value.strip() if _is_meaningful_form_str(value) else None


@router.post("/check", response_model=PlagiarismResponse, response_model_exclude_none=True)
async def check_plagiarism(
    user_id: str = Form(...),
    check_type: CheckType = Form("full"),
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    doi: str | None = Form(default=None),
    paper_id: str | None = Form(default=None),
    language: str | None = Form(default=None),
):
    """Run plagiarism/citation check.

    Provide exactly one of: `file`, `text`, `doi`, `paper_id`.
    Supported upload types: `.pdf`, `.docx`, `.txt`.
    """

    text = _normalize_form_str(text)
    doi = _normalize_form_str(doi)
    paper_id = _normalize_form_str(paper_id)
    language = _normalize_form_str(language)

    file_provided = bool(file is not None and (file.filename or "").strip())
    provided = [file_provided, bool(text), bool(doi), bool(paper_id)]
    if sum(1 for x in provided if x) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly ONE of: file, text, doi, paper_id.")

    file_bytes: bytes | None = None
    filename: str | None = None
    if file is not None:
        filename = file.filename
        file_bytes = await file.read()

    req = PlagiarismRequest(
        user_id=user_id,
        check_type=check_type,
        text=text,
        doi=doi,
        paper_id=paper_id,
        language=language,
    )
    return run_check(req=req, file_bytes=file_bytes, filename=filename)

