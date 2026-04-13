# app/api/literature.py
# Literature Review API endpoints
# Supports multilingual output via output_language parameter.

import os
import uuid
import tempfile
from copy import deepcopy
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.pipeline.orchestrator import LiteratureReviewOrchestrator
from app.pipeline.multilingual import SUPPORTED_LANGUAGES
from app.pipeline.translator import TranslationOutputLayer
from app.models.schemas import LiteratureReviewResult, ReviewStatusResponse
from app.utils.export import ReportExporter
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/literature", tags=["Literature Review"])

# In-memory task store (for background processing)
_tasks: dict[str, ReviewStatusResponse] = {}
_export_results: dict[str, LiteratureReviewResult] = {}

# Lazy-loaded orchestrator (heavy — loads ML models)
_orchestrator: Optional[LiteratureReviewOrchestrator] = None


def _get_orchestrator() -> LiteratureReviewOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Lazy-loading pipeline orchestrator...")
        _orchestrator = LiteratureReviewOrchestrator()
        logger.debug("Pipeline orchestrator initialized and cached.")
    return _orchestrator


def _prepare_user_result(
    english_result: LiteratureReviewResult,
    output_language: str,
) -> LiteratureReviewResult:
    logger.debug(
        "Preparing user result output_language=%s papers=%s comparison_matrix=%s",
        output_language,
        len(english_result.papers),
        bool(english_result.comparison_matrix),
    )
    if output_language == "en":
        english_result.output_language_requested = "en"
        english_result.output_language_applied = "en"
        english_result.translation_status = "not_requested"
        english_result.translation_message = ""
        logger.debug("English pass-through selected; no translation performed.")
        return english_result

    translated_result = deepcopy(english_result)

    try:
        translator = TranslationOutputLayer(target_lang=output_language)
        logger.debug("TranslationOutputLayer created for target_language=%s", output_language)
        translator.translate_review_result(translated_result)
        translated_result.output_language_requested = output_language
        translated_result.output_language_applied = output_language
        translated_result.translation_status = "translated"
        translated_result.translation_message = (
            f"Output translated to {SUPPORTED_LANGUAGES[output_language]} in "
            f"{translated_result.translation_time_seconds:.2f}s."
        )
        logger.debug(
            "Translation completed for target_language=%s papers=%s",
            output_language,
            len(translated_result.papers),
        )
        return translated_result
    except Exception as exc:
        logger.warning(
            "Output translation to %s failed. Falling back to English: %s",
            output_language,
            exc,
        )
        logger.debug("Translation fallback exception type=%s", type(exc).__name__)
        english_result.output_language_requested = output_language
        english_result.output_language_applied = "en"
        english_result.translation_status = "failed_fallback_to_english"
        english_result.translation_message = (
            f"Translation to {SUPPORTED_LANGUAGES.get(output_language, output_language)} failed. "
            f"Showing English output instead. Details: {exc}"
        )
        return english_result


def _run_analysis(
    task_id: str,
    pdf_paths: list[str],
    query: Optional[str],
    fetch_related: bool,
    output_language: str,
):
    """Background task for running the analysis pipeline."""
    try:
        logger.debug(
            "Task %s starting analysis pdf_count=%s query_present=%s fetch_related=%s output_language=%s",
            task_id,
            len(pdf_paths),
            bool(query),
            fetch_related,
            output_language,
        )
        _tasks[task_id].status = "processing"
        _tasks[task_id].progress = "Loading models..."

        orchestrator = _get_orchestrator()
        _tasks[task_id].progress = "Analyzing papers..."
        logger.debug("Task %s invoking orchestrator.run(output_language=en)", task_id)

        english_result = orchestrator.run(
            pdf_paths,
            query=query,
            fetch_related=fetch_related,
            output_language="en",
        )
        logger.debug(
            "Task %s analysis complete papers=%s processing_time=%.2fs",
            task_id,
            len(english_result.papers),
            english_result.processing_time_seconds,
        )
        _export_results[task_id] = deepcopy(english_result)

        if output_language != "en":
            _tasks[task_id].progress = f"Translating output to {SUPPORTED_LANGUAGES[output_language]}..."
            logger.debug("Task %s starting translation to %s", task_id, output_language)

        result = _prepare_user_result(english_result, output_language)

        _tasks[task_id].status = "completed"
        _tasks[task_id].result = result
        _tasks[task_id].progress = (
            "Done (English fallback)"
            if result.translation_status == "failed_fallback_to_english"
            else "Done"
        )
        logger.info(f"Task {task_id} completed successfully.")
        logger.debug(
            "Task %s completed output_language_requested=%s output_language_applied=%s translation_status=%s",
            task_id,
            result.output_language_requested,
            result.output_language_applied,
            result.translation_status,
        )

    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        _tasks[task_id].status = "failed"
        _tasks[task_id].error = str(e)
        _export_results.pop(task_id, None)
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
    - Set output_language to get translated user-facing output in the supported multilingual set.
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

    logger.debug(
        "Received async review request file_count=%s query_present=%s output_language=%s fetch_related=%s",
        len(files),
        bool(query),
        output_language,
        fetch_related_works,
    )

    pdf_paths = []
    upload_dir = os.path.join(tempfile.gettempdir(), "nexus_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF.")

        temp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{f.filename}")
        logger.debug("Staging upload filename=%s temp_path=%s", f.filename, temp_path)
        with open(temp_path, "wb") as buf:
            content = await f.read()
            logger.debug("Read upload bytes filename=%s bytes=%s", f.filename, len(content))
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
    logger.debug("Async review task queued task_id=%s upload_dir=%s", task_id, upload_dir)
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
    logger.debug("Results requested task_id=%s status=%s", task_id, task.status)
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
        export_result = _export_results.get(task_id) or task.result
        logger.debug(
            "Exporting report task_id=%s output_path=%s requested_language=%s applied_language=%s source=%s",
            task_id,
            output_path,
            getattr(task.result, "output_language_requested", None),
            getattr(task.result, "output_language_applied", None),
            "cached_english" if task_id in _export_results else "task_result",
        )
        exporter.export_pdf(export_result, output_path)

        return FileResponse(
            path=output_path,
            filename="literature_review_report.pdf",
            media_type="application/pdf",
        )
    except Exception as e:
        logger.exception(f"PDF export failed for task_id={task_id}")
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
        output_language: Output language code from the supported multilingual set.
        fetch_related_works: Whether to search for related papers.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if output_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{output_language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )

    logger.debug(
        "Received sync review request file_count=%s query_present=%s output_language=%s fetch_related=%s",
        len(files),
        bool(query),
        output_language,
        fetch_related_works,
    )

    pdf_paths = []
    upload_dir = os.path.join(tempfile.gettempdir(), "nexus_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF.")
        temp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{f.filename}")
        logger.debug("Staging sync upload filename=%s temp_path=%s", f.filename, temp_path)
        with open(temp_path, "wb") as buf:
            content = await f.read()
            logger.debug("Read sync upload bytes filename=%s bytes=%s", f.filename, len(content))
            buf.write(content)
        pdf_paths.append(temp_path)

    try:
        orchestrator = _get_orchestrator()
        logger.debug("Sync review invoking orchestrator.run on %s file(s)", len(pdf_paths))
        english_result = orchestrator.run(
            pdf_paths,
            query=query,
            fetch_related=fetch_related_works,
            output_language="en",
        )
        logger.debug(
            "Sync review analysis complete papers=%s processing_time=%.2fs",
            len(english_result.papers),
            english_result.processing_time_seconds,
        )
        return _prepare_user_result(english_result, output_language)
    finally:
        for p in pdf_paths:
            try:
                os.remove(p)
            except Exception:
                pass


@router.post("/multilingual/{destination_language}", response_model=LiteratureReviewResult)
async def review_multilingual(
    destination_language: str,
    files: list[UploadFile] = File(...),
    query: Optional[str] = Form(None),
    fetch_related_works: bool = Form(True),
):
    """Dedicated multilingual endpoint.

    - For destination_language='en', returns English output without translation.
    - For non-English destination_language, translates the final payload.
    """
    if destination_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{destination_language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )

    return await review_sync(
        files=files,
        query=query,
        output_language=destination_language,
        fetch_related_works=fetch_related_works,
    )
