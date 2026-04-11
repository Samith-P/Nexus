# app/api/literature.py
# Literature Review API endpoints
# Supports multilingual output via output_language parameter.

import os
import uuid
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.pipeline.orchestrator import LiteratureReviewOrchestrator
from app.pipeline.multilingual import SUPPORTED_LANGUAGES
from app.models.schemas import LiteratureReviewResult, ReviewStatusResponse
from app.utils.export import ReportExporter
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/literature", tags=["Literature Review"])

# In-memory task store (for background processing)
_tasks: dict[str, ReviewStatusResponse] = {}

# Lazy-loaded orchestrator (heavy — loads ML models)
_orchestrator: Optional[LiteratureReviewOrchestrator] = None


def _get_orchestrator() -> LiteratureReviewOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Lazy-loading pipeline orchestrator...")
        _orchestrator = LiteratureReviewOrchestrator()
    return _orchestrator


def _run_analysis(
    task_id: str,
    pdf_paths: list[str],
    query: Optional[str],
    fetch_related: bool,
    output_language: str,
):
    """Background task for running the analysis pipeline."""
    try:
        _tasks[task_id].status = "processing"
        _tasks[task_id].progress = "Loading models..."

        orchestrator = _get_orchestrator()
        _tasks[task_id].progress = "Analyzing papers..."

        result = orchestrator.run(
            pdf_paths,
            query=query,
            fetch_related=fetch_related,
            output_language=output_language,
        )

        _tasks[task_id].status = "completed"
        _tasks[task_id].result = result
        _tasks[task_id].progress = "Done"
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        _tasks[task_id].status = "failed"
        _tasks[task_id].error = str(e)
    finally:
        for path in pdf_paths:
            try:
                os.remove(path)
            except Exception:
                pass


@router.get("/languages")
async def list_languages():
    """List supported output languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@router.post("/review", response_model=ReviewStatusResponse)
async def start_review(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    query: Optional[str] = Form(None),
    output_language: str = Form("en"),
    fetch_related_works: bool = Form(True),
):
    """Upload PDFs and start a literature review analysis.

    - Upload 1-5 PDF files.
    - Optionally provide a research query/theme.
    - Set output_language to get results in Indian languages (hi, te, ur, sa, etc.).
    - Returns a task_id to poll for results.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed.")
    if output_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{output_language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )

    pdf_paths = []
    upload_dir = os.path.join(tempfile.gettempdir(), "nexus_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF.")

        temp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{f.filename}")
        with open(temp_path, "wb") as buf:
            content = await f.read()
            buf.write(content)
        pdf_paths.append(temp_path)

    task_id = uuid.uuid4().hex
    _tasks[task_id] = ReviewStatusResponse(
        task_id=task_id,
        status="queued",
        progress="Waiting to start...",
    )

    background_tasks.add_task(
        _run_analysis, task_id, pdf_paths, query, fetch_related_works, output_language
    )

    logger.info(f"Task {task_id} queued for {len(pdf_paths)} paper(s), lang={output_language}.")
    return _tasks[task_id]


@router.get("/status/{task_id}", response_model=ReviewStatusResponse)
async def get_status(task_id: str):
    """Check the status of a literature review task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return _tasks[task_id]


@router.get("/results/{task_id}", response_model=LiteratureReviewResult)
async def get_results(task_id: str):
    """Get the results of a completed literature review task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    task = _tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=202, detail=f"Task is still {task.status}.")
    return task.result


@router.get("/export/{task_id}")
async def export_report(task_id: str):
    """Export the results of a completed review as a PDF report."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    task = _tasks[task_id]
    if task.status != "completed" or task.result is None:
        raise HTTPException(status_code=202, detail="Results not ready yet.")

    try:
        exporter = ReportExporter()
        output_path = os.path.join(tempfile.gettempdir(), f"nexus_report_{task_id}.pdf")
        exporter.export_pdf(task.result, output_path)

        return FileResponse(
            path=output_path,
            filename="literature_review_report.pdf",
            media_type="application/pdf",
        )
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


@router.post("/review/sync", response_model=LiteratureReviewResult)
async def review_sync(
    files: list[UploadFile] = File(...),
    query: Optional[str] = Form(None),
    output_language: str = Form("en"),
    fetch_related_works: bool = Form(True),
):
    """Synchronous review — waits for completion and returns results directly.

    Use for small reviews (1-2 papers). For larger reviews, use /review (async).

    Args:
        files: 1-5 PDF files to analyze.
        query: Optional research query/theme.
        output_language: Output language code (en, hi, te, ur, sa, bn, ta, ml, kn, mr, gu, pa, or).
        fetch_related_works: Whether to search for related papers.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if output_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{output_language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )

    pdf_paths = []
    upload_dir = os.path.join(tempfile.gettempdir(), "nexus_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF.")
        temp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{f.filename}")
        with open(temp_path, "wb") as buf:
            content = await f.read()
            buf.write(content)
        pdf_paths.append(temp_path)

    try:
        orchestrator = _get_orchestrator()
        result = orchestrator.run(
            pdf_paths,
            query=query,
            fetch_related=fetch_related_works,
            output_language=output_language,
        )
        return result
    finally:
        for p in pdf_paths:
            try:
                os.remove(p)
            except Exception:
                pass
