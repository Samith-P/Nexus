# Literature Review Engine — Integration Guide

> **For:** Backend developers who need to integrate this module into an existing FastAPI backend.
> **Branch:** `feature/literature-review-engine`
> **Last Tested:** 2026-04-11 (All 24 unit tests + 7 integration tests passing)

---

## Table of Contents
- [Quick Overview](#quick-overview)
- [File Map — What to Copy](#file-map--what-to-copy)
- [Step-by-Step Integration](#step-by-step-integration)
- [Dependencies to Install](#dependencies-to-install)
- [Environment Variables](#environment-variables)
- [API Routes Reference](#api-routes-reference)
- [Frontend Integration Guide](#frontend-integration-guide)
- [JSON Response Schemas](#json-response-schemas)
- [How It Works Internally](#how-it-works-internally)
- [Performance & Resource Notes](#performance--resource-notes)
- [Common Issues & Solutions](#common-issues--solutions)

---

## Quick Overview

This module accepts PDF research papers, runs them through a 7-stage AI pipeline (parse → clean → detect sections → chunk → embed → summarize → extract insights/gaps), fetches related works from Semantic Scholar & CrossRef, and returns a structured JSON result + downloadable PDF report.

**It is self-contained.** You copy the files, install dependencies, register one router, and you're done.

---

## File Map — What to Copy

Copy these folders/files into your existing backend project. The module is **fully self-contained** — no circular dependencies with the rest of the app.

```
your_existing_backend/
├── app/
│   ├── api/
│   │   ├── __init__.py          # Add "# API package" if missing
│   │   └── literature.py        # ⭐ MAIN ROUTER FILE (copy this)
│   │
│   ├── models/
│   │   ├── __init__.py          # Add "# Models package" if missing
│   │   └── schemas.py           # ⭐ ALL PYDANTIC MODELS (copy this)
│   │
│   ├── pipeline/                # ⭐ ENTIRE FOLDER (copy all 12 files)
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main pipeline coordinator
│   │   ├── pdf_parser.py        # PDF → text (PyMuPDF)
│   │   ├── text_cleaner.py      # Text normalization
│   │   ├── section_detector.py  # Detect abstract/intro/methods/etc
│   │   ├── chunker.py           # Split text into chunks
│   │   ├── embedder.py          # all-MiniLM-L6-v2 embeddings
│   │   ├── summarizer.py        # distilbart summarization
│   │   ├── insight_extractor.py # FLAN-T5 insight extraction
│   │   ├── gap_detector.py      # FLAN-T5 gap detection
│   │   ├── multilingual.py      # Multilingual support (parked)
│   │   └── translator.py        # Translation output layer (parked)
│   │
│   ├── integrations/            # ⭐ ENTIRE FOLDER (copy all 3 files)
│   │   ├── __init__.py
│   │   ├── semantic_scholar.py  # Semantic Scholar API client
│   │   └── crossref.py          # CrossRef API client
│   │
│   ├── storage/                 # ⭐ ENTIRE FOLDER (copy all 2 files)
│   │   ├── __init__.py
│   │   └── vector_store.py      # FAISS vector store
│   │
│   └── utils/                   # ⭐ COPY THESE 2 FILES
│       ├── export.py            # PDF report generator
│       └── logger.py            # Logging utility
│
├── tests/
│   └── test_pipeline.py         # 24 unit tests (optional but recommended)
│
└── requirements.txt             # Merge with your existing requirements
```

### Files NOT to copy (these are specific to the standalone version):
- `app/main.py` — You have your own FastAPI app entry point
- `app/app.py` — Unused
- `app/agents/` — Unused
- `app/ml/` — Unused
- `stage-1/` through `stage-7/` — Development stages, not needed
- `test_pdf.py` — One-off PDF debug script
- `*.pdf` — Test output files

---

## Step-by-Step Integration

### Step 1: Copy the Files

Copy the folders listed above into your existing project. Maintain the same directory structure relative to your `app/` directory.

### Step 2: Install Dependencies

Add these to your `requirements.txt` (skip any you already have):

```txt
# PDF parsing
PyMuPDF==1.25.5

# NLP / ML (CPU-only)
transformers==4.51.3
torch>=2.0.0
sentence-transformers==4.1.0
nltk==3.9.1

# Vector store
faiss-cpu==1.11.0
numpy>=1.24.0

# API clients (for related works)
httpx==0.28.1

# Report export
fpdf2==2.9.1

# Environment variables
python-dotenv==1.1.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 3: Register the Router

In your **existing** FastAPI `main.py` (or wherever you create the app), add:

```python
# At the top of your main.py
from dotenv import load_dotenv
load_dotenv()  # Load .env for SEMANTIC_SCHOLAR_API_KEY

# Import the literature review router
from app.api.literature import router as literature_router

# Register it with your app
app.include_router(literature_router)
```

**That's it.** The router mounts at `/api/literature/` with 6 endpoints.

### Step 4: Set Environment Variables

Create or update your `.env` file:
```env
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

Get a free key at: https://www.semanticscholar.org/product/api

> **Without the key**, Semantic Scholar will still work but has aggressive rate limits (1 request/second). CrossRef always works without a key.

### Step 5: Verify It Works

Start your server and test:
```bash
# Check the endpoint exists
curl http://127.0.0.1:8000/api/literature/languages

# Should return:
# {"languages": {"en": "English", "hi": "Hindi", ...}}
```

Then test with a real PDF:
```bash
curl -X POST "http://127.0.0.1:8000/api/literature/review/sync" \
  -F "files=@your_paper.pdf" \
  -F "fetch_related_works=false"
```

---

## Dependencies to Install

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `PyMuPDF` | 1.25.5 | ~30MB | PDF text extraction |
| `transformers` | 4.51.3 | ~100MB | HuggingFace model loading |
| `torch` | ≥2.0.0 | ~200MB (CPU) | PyTorch inference |
| `sentence-transformers` | 4.1.0 | ~50MB | Embedding model |
| `nltk` | 3.9.1 | ~5MB | Sentence tokenization |
| `faiss-cpu` | 1.11.0 | ~30MB | Vector similarity search |
| `httpx` | 0.28.1 | ~1MB | HTTP client for APIs |
| `fpdf2` | 2.9.1 | ~2MB | PDF report generation |
| `python-dotenv` | 1.1.0 | ~50KB | .env file loading |

### ML Models (auto-downloaded on first use)
| Model | Download Size | RAM Usage | Purpose |
|-------|--------------|-----------|---------|
| `all-MiniLM-L6-v2` | ~80MB | ~200MB | Text embeddings (384-dim) |
| `sshleifer/distilbart-cnn-12-6` | ~1.2GB | ~1.5GB | Summarization |
| `google/flan-t5-base` | ~990MB | ~1.2GB | Insights + Gaps (shared) |
| **Total** | **~2.3GB** | **~3GB** | |

> ⚠️ **First request will be slow** (~30-60s) as models download. Subsequent requests reuse cached models.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | API key for higher rate limits. Without it, you get throttled to ~1 req/sec |

---

## API Routes Reference

All routes are prefixed with `/api/literature`.

### 1. `POST /api/literature/review` — Async Review (Recommended for Production)

Upload PDFs and start background processing. Returns immediately with a `task_id`.

**Request:** `multipart/form-data`
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `files` | File[] | ✅ | — | 1-5 PDF files |
| `query` | string | ❌ | null | Optional research theme/query |
| `output_language` | string | ❌ | `"en"` | Output language code |
| `fetch_related_works` | bool | ❌ | `true` | Search Semantic Scholar + CrossRef |

**Response:** `200 OK`
```json
{
  "task_id": "48af9b2b045941b08068fb90dd6df113",
  "status": "queued",
  "progress": "Waiting to start...",
  "result": null,
  "error": null
}
```

**Frontend flow:**
1. POST to `/review` → get `task_id`
2. Poll `GET /status/{task_id}` every 5 seconds
3. When `status === "completed"`, display `result` or redirect to export

---

### 2. `POST /api/literature/review/sync` — Synchronous Review

Same as above but **blocks until completion**. Good for quick tests, not recommended for production (can timeout on slow connections).

**Request:** Same as `/review`

**Response:** `200 OK` — Returns the full `LiteratureReviewResult` directly (no task_id).

---

### 3. `GET /api/literature/status/{task_id}` — Check Task Status

**Response:**
```json
{
  "task_id": "48af9b2b045941b08068fb90dd6df113",
  "status": "completed",       // "queued" | "processing" | "completed" | "failed"
  "progress": "Done",          // Human-readable progress text
  "result": { ... },           // Full results when completed (see schema below)
  "error": null                // Error message if failed
}
```

---

### 4. `GET /api/literature/results/{task_id}` — Get Full Results

Returns just the `LiteratureReviewResult` (without task metadata). Returns `404` if task not found, `202` if still processing.

---

### 5. `GET /api/literature/export/{task_id}` — Download PDF Report

Returns a `application/pdf` file. Filename: `literature_review_report.pdf`.

**Frontend usage:**
```javascript
// Trigger download
const response = await fetch(`/api/literature/export/${taskId}`);
const blob = await response.blob();
const url = window.URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'literature_review_report.pdf';
a.click();
```

---

### 6. `GET /api/literature/languages` — Supported Languages

```json
{
  "languages": {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ur": "Urdu",
    "sa": "Sanskrit",
    "bn": "Bengali",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia"
  }
}
```

> ⚠️ **Note:** Only `"en"` is fully functional. Other languages are listed but the translation model needs to be swapped (current IndicBARTSS doesn't translate well). Use `"en"` for now.

---

## Frontend Integration Guide

### Recommended Flow (React/Next.js)

```
┌─────────┐    POST /review     ┌──────────┐
│  Upload  │ ──────────────────→ │ Backend  │
│   Form   │ ←── { task_id }──  │  (FastAPI)│
└────┬─────┘                    └──────────┘
     │                               │
     │ poll every 5s                 │ background processing
     │ GET /status/{id}              │ (90-380 seconds)
     ▼                               │
┌─────────┐                         │
│ Progress │ ←── { status,          │
│  Screen  │      progress } ───────┘
└────┬─────┘
     │ status === "completed"
     ▼
┌─────────┐
│ Results  │ ← Display papers, insights, gaps
│  Page    │
└────┬─────┘
     │ click "Download PDF"
     ▼
┌─────────┐    GET /export/{id}
│ Download │ ──────────────────→ PDF blob
└─────────┘
```

### React Example Code

```jsx
// 1. Upload Component
const uploadPapers = async (files) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  formData.append('fetch_related_works', 'true');

  const res = await fetch('/api/literature/review', {
    method: 'POST',
    body: formData,
  });
  const data = await res.json();
  return data.task_id;  // Store this for polling
};

// 2. Polling Component
const pollStatus = async (taskId, onComplete, onProgress) => {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/literature/status/${taskId}`);
    const data = await res.json();

    onProgress(data.progress, data.status);

    if (data.status === 'completed') {
      clearInterval(interval);
      onComplete(data.result);
    } else if (data.status === 'failed') {
      clearInterval(interval);
      console.error('Review failed:', data.error);
    }
  }, 5000);  // Poll every 5 seconds

  return () => clearInterval(interval);  // Cleanup function
};

// 3. Download PDF
const downloadReport = async (taskId) => {
  const res = await fetch(`/api/literature/export/${taskId}`);
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'literature_review_report.pdf';
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};
```

### Vite Proxy Setup

If your frontend runs on a different port (e.g., Vite on 5173), add to `vite.config.js`:

```javascript
export default defineConfig({
  server: {
    proxy: {
      '/api/literature': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
```

---

## JSON Response Schemas

### LiteratureReviewResult (Main Response)

```typescript
interface LiteratureReviewResult {
  papers: PaperAnalysis[];                  // Array of per-paper results
  comparison_matrix: ComparisonMatrix | null; // Only when ≥2 papers
  common_themes: string[];                  // Shared themes across papers
  research_gaps: string[];                  // Aggregated gaps
  related_works: RelatedWork[];             // From Semantic Scholar + CrossRef
  processing_time_seconds: number;          // e.g., 254.06
  timestamp: string;                        // ISO 8601
}
```

### PaperAnalysis

```typescript
interface PaperAnalysis {
  title: string;
  sections: Record<string, string>;           // Raw section text
  summary: string;                            // AI-generated summary
  section_summaries: Record<string, string>;  // Per-section summaries
  insights: {
    contributions: string[];   // e.g., "BERT introduces bidirectional pre-training"
    methods: string[];         // e.g., "Masked Language Model"
    results: string[];         // e.g., "BERT achieves SOTA on 11 NLP tasks"
  };
  gaps: string[];              // e.g., "Limited to English data"
  evidence_spans: {
    section: string;           // e.g., "abstract"
    text: string;              // Actual text excerpt
    page: number | null;
  }[];
}
```

### ComparisonMatrix (only when ≥2 papers)

```typescript
interface ComparisonMatrix {
  entries: {
    paper_title: string;
    methods: string[];
    results: string[];
    gaps: string[];
  }[];
  common_methods: string[];    // Methods shared across papers
  differing_methods: string[]; // Methods unique to specific papers
}
```

### RelatedWork

```typescript
interface RelatedWork {
  title: string;
  authors: string[];
  year: number | null;
  abstract: string | null;
  citation_count: number | null;
  url: string | null;
  source: "semantic_scholar" | "crossref";
}
```

---

## How It Works Internally

### Request Lifecycle

```
1. User uploads PDF(s) → API receives multipart/form-data
2. Files saved to temp directory
3. Background thread starts (async) or blocks (sync)
4. For each PDF:
   a. pdf_parser.py    → Extracts text from all pages (PyMuPDF)
   b. text_cleaner.py  → Removes URLs, DOIs, headers, fixes hyphens
   c. section_detector.py → Splits into abstract/intro/methods/results/conclusion
   d. chunker.py       → Splits into ~400-word sentence-aware chunks
   e. embedder.py      → Embeds chunks with all-MiniLM-L6-v2
   f. vector_store.py  → Stores embeddings in FAISS index
   g. summarizer.py    → Hierarchical summarization (distilbart)
   h. insight_extractor.py → Extract contributions/methods/results (FLAN-T5)
   i. gap_detector.py  → Detect research gaps (rules + FLAN-T5)
5. If ≥2 papers → Generate comparison matrix
6. If fetch_related_works → Query Semantic Scholar + CrossRef
7. Return structured JSON result
8. On /export → Generate PDF with fpdf2
```

### Model Loading

- Models are **lazy-loaded** on first request (~30-60s)
- After first load, they stay in memory (~3GB RAM)
- FLAN-T5 is **shared** between InsightExtractor and GapDetector (saves ~1.2GB)
- Server restart clears all models from memory

### Task Storage

- Tasks are stored **in-memory** (Python dict)
- Server restart = all tasks lost
- For production, replace `_tasks` dict in `literature.py` with Redis/SQLite

---

## Performance & Resource Notes

### Processing Times (CPU, AMD Ryzen 5 5600H)

| Scenario | Time | Notes |
|----------|------|-------|
| 1 paper (11 pages) | ~85-120s | Attention Is All You Need |
| 1 paper (25 pages) | ~130s | GPT-3 paper |
| 2 papers | ~250s | BERT + Attention |
| 3 papers | ~250-380s | BERT + Attention + GPT-3 |
| + Related works | +5-15s | API calls to S2 + CrossRef |

### RAM Usage

| Component | RAM |
|-----------|-----|
| FastAPI server | ~100MB |
| Embedding model (MiniLM) | ~200MB |
| Summarizer (distilbart) | ~1.5GB |
| FLAN-T5 (shared) | ~1.2GB |
| **Total after first request** | **~3GB** |

> ⚠️ Server needs at least **6GB free RAM** to run comfortably.

---

## Common Issues & Solutions

### 1. "ModuleNotFoundError: No module named 'app.pipeline'"
**Cause:** Import paths assume `app/` is the top-level package.
**Fix:** Make sure your project root has the `app/` directory and you run from the project root, not inside `app/`.

### 2. "Not enough horizontal space to render a single character" (PDF Export)
**Cause:** fpdf2 v2.x cursor issue. Already fixed in our code.
**Fix:** Make sure you're using the latest `export.py` with `new_x="LMARGIN"`.

### 3. First request takes 60+ seconds
**Cause:** ML models being downloaded/loaded for the first time.
**Fix:** This is normal. Models cache to `~/.cache/huggingface/`. Subsequent requests reuse them.

### 4. Semantic Scholar returns empty results
**Cause:** Rate limited (429) without API key.
**Fix:** Set `SEMANTIC_SCHOLAR_API_KEY` in `.env` file.

### 5. "torch" installation fails / is too large
**Fix:** Install CPU-only torch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 6. Server loses tasks after restart
**Cause:** Tasks stored in-memory.
**Fix:** For production, replace `_tasks` dict in `literature.py` with a database.

### 7. CORS errors from frontend
**Fix:** The router doesn't handle CORS — your main app should have CORS middleware:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 8. "python-dotenv" not loading .env
**Fix:** Add `from dotenv import load_dotenv; load_dotenv()` at the very top of your `main.py`, **before** any other imports.

---

## Quick Verification Checklist

After integration, verify these work:

```bash
# 1. Health check — should list the literature endpoints
curl http://127.0.0.1:8000/

# 2. Languages endpoint
curl http://127.0.0.1:8000/api/literature/languages

# 3. Single paper test (sync — will take ~2 min)
curl -X POST "http://127.0.0.1:8000/api/literature/review/sync" \
  -F "files=@any_paper.pdf" \
  -F "fetch_related_works=false"

# 4. Async + status check
curl -X POST "http://127.0.0.1:8000/api/literature/review" \
  -F "files=@any_paper.pdf" \
  -F "fetch_related_works=false"
# → note the task_id

curl http://127.0.0.1:8000/api/literature/status/{task_id}
# → wait for status: "completed"

# 5. PDF export
curl http://127.0.0.1:8000/api/literature/export/{task_id} --output report.pdf
# → open report.pdf, should be a valid PDF

# 6. Unit tests
python -m pytest tests/test_pipeline.py -v
```

---

## File-by-File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `api/literature.py` | 233 | FastAPI router with 6 endpoints |
| `models/schemas.py` | 80 | All Pydantic models (request/response) |
| `pipeline/orchestrator.py` | ~280 | Main pipeline coordinator |
| `pipeline/pdf_parser.py` | ~40 | PyMuPDF PDF extraction |
| `pipeline/text_cleaner.py` | ~70 | Text normalization |
| `pipeline/section_detector.py` | ~180 | Academic section detection |
| `pipeline/chunker.py` | ~55 | Sentence-aware chunking |
| `pipeline/embedder.py` | ~40 | MiniLM embedding |
| `pipeline/summarizer.py` | ~95 | distilbart summarization |
| `pipeline/insight_extractor.py` | ~110 | FLAN-T5 insight extraction |
| `pipeline/gap_detector.py` | ~115 | Hybrid gap detection |
| `pipeline/multilingual.py` | ~200 | Multilingual support (parked) |
| `pipeline/translator.py` | ~100 | Translation layer (parked) |
| `integrations/semantic_scholar.py` | ~85 | S2 API client |
| `integrations/crossref.py` | ~100 | CrossRef API client |
| `storage/vector_store.py` | ~85 | FAISS vector store |
| `utils/export.py` | ~185 | PDF report generator |
| `utils/logger.py` | 23 | Structured logging utility |

---

*Created: 2026-04-11 | For: Nexus Journal Backend Integration*
