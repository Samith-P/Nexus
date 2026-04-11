# Literature Review Engine — Complete Analysis & Implementation Plan

## 1. What The Documentation Requires

From the RTGS Proposal, the **AI-Driven Literature Review Module** should:

| # | Documented Requirement | Description |
|---|------------------------|-------------|
| 1 | PDF Upload | Accept research papers via file upload |
| 2 | Structure Extraction | Parse Abstract, Methodology, Results, Conclusion sections |
| 3 | Key Insights | Extract contributions, methods, results |
| 4 | Summaries | Section-wise summarization |
| 5 | Research Gaps | Identify what the paper doesn't cover |
| 6 | Related Works Mapping | Use CrossRef/Semantic Scholar APIs to find related papers |
| 7 | Multilingual Support | **IndicBERT/IndicBART** for Telugu, Hindi, Urdu, Sanskrit, English |
| 8 | Cross-lingual Search | Vector DB (FAISS/Qdrant) + multilingual embeddings |
| 9 | Translation Output | Return output in user's preferred language |
| 10 | FastAPI Backend | Microservice with proper API endpoints |
| 11 | Report Export | PDF/Word export of analysis |
| 12 | Multi-paper Comparison | Compare 2-3 papers, find common themes, differences |
| 13 | Redis Caching | Cache repeated analyses |
| 14 | Docker Containerization | Deployable container |

---

## 2. What You've Built (Current State ✅)

A **7-stage pipeline** that processes a single PDF:

| Stage | File | What It Does | Status |
|-------|------|-------------|--------|
| 1 | `stage-1/pdf_parser.py` | Extract raw text from PDF via PyMuPDF | ✅ Working |
| 2 | `stage-2/text_cleaner.py` | Clean text (remove emails, page numbers, headers, fix whitespace) | ✅ Working (⚠️ has hardcoded patterns) |
| 3 | `stage-3/section_detector.py` | Detect academic sections by keyword position | ✅ Working (⚠️ fragile approach) |
| 4 | `stage-4/chunker.py` | Split text into 400-word overlapping chunks | ✅ Working (⚠️ word-level, breaks sentences) |
| 4 | `stage-4/embedder.py` | Embed chunks with `all-MiniLM-L6-v2` | ✅ Working (⚠️ output unused) |
| 5 | `stage-5/summarizer.py` | Hierarchical summarization with `distilbart-cnn-12-6` | ✅ Working |
| 6 | `stage-6/insight_extractor.py` | Extract contributions/methods/results with `flan-t5-base` | ✅ Working (⚠️ has BERT-specific boost) |
| 7 | `stage-7/gap_detector.py` | Hybrid gap detection: rules + FLAN-T5 | ✅ Working (⚠️ hardcoded generic gaps) |

**Supporting infrastructure:**
- Test scripts for each stage (`test_stage1.py` → `test_stage7.py`)
- Empty `app/app.py` scaffold
- Empty `backend_node/` scaffold (gitkeep files only)
- Bare Vite+React frontend (just "Hello world")
- 3 sample PDFs (BERT, Attention is All You Need, GPT-3)

---

## 3. Consensus From All 4 Agent Reviews

### 🟢 What All Agents Agree You Did Well
1. Clear 7-stage modular decomposition — easy to debug and extend
2. Hierarchical summarization approach in `summarizer.py`
3. Hybrid gap detection idea (rules + LLM)
4. Incremental building strategy (simple → stable → improved)
5. Practical handling of resource constraints (token limits, chunk limits)

### 🔴 What All Agents Agree Must Be FIXED (Correctness Issues)

| # | Issue | File(s) | All 4 Agents? |
|---|-------|---------|---------------|
| 1 | **Section detector is fragile** — `str.find()` on lowercased text matches random occurrences of "method" in body text | `section_detector.py` | ✅ All 4 |
| 2 | **BERT-specific `boost_methods()`** — hardcoded keywords for one paper, misleads on others | `insight_extractor.py` | ✅ All 4 |
| 3 | **Hardcoded generic gaps** — "Lack of domain-specific adaptation" appears on every paper | `gap_detector.py` | ✅ All 4 |
| 4 | **Conference-specific header removal** — "Association for Computational Linguistics", "Minneapolis" patterns | `text_cleaner.py` | ✅ Agents 1,2,3 |
| 5 | **Embeddings generated but never used** — dead code in pipeline | `embedder.py` | ✅ All 4 |
| 6 | **Word-level chunking breaks sentences** — degrades summarization quality | `chunker.py` | ✅ Agents 1,3 |
| 7 | **No error propagation between stages** — empty sections produce silent junk | across all stages | ✅ Agents 2,3,4 |
| 8 | **`sys.path` hacks in every test** — fragile, won't work in production | `test_stage*.py` | ✅ Agents 2,3 |
| 9 | **Text lowercased in section_detector** — loses original casing for downstream | `section_detector.py` | ✅ Agents 1,2 |

### 🔶 What All Agents Agree Is MISSING (Per Documentation)

| # | Missing Feature | Priority |
|---|----------------|----------|
| 1 | **FastAPI service / API endpoint** — no backend, only scripts | P0 |
| 2 | **Pipeline orchestrator** — no single function to chain all stages | P0 |
| 3 | **Related works mapping** — no CrossRef/Semantic Scholar integration | P1 |
| 4 | **Multi-paper comparison** — can only process single paper | P1 |
| 5 | **Multilingual support (IndicBERT/IndicBART)** — using English-only models | P1 |
| 6 | **Vector store (FAISS)** — embeddings not stored/searchable | P1 |
| 7 | **Report export (PDF/Word)** — no export capability | P2 |
| 8 | **Structured output schema (Pydantic)** — no defined API contract | P1 |
| 9 | **File upload endpoint** — no way to upload PDFs | P0 |
| 10 | **Evidence spans / citations** — no page-level evidence in output | P2 |
| 11 | **Data storage** — nothing saved (summaries, insights, gaps) | P1 |
| 12 | **Confidence scores** — no quality metrics | P2 |

### 🟡 What's EXTRA (Should Remove/Refactor)

| # | Extra Item | Action |
|---|-----------|--------|
| 1 | `boost_methods()` with BERT-specific keywords | **Remove** — replace with dynamic keyword extraction |
| 2 | Hardcoded "always-on" generic gaps | **Remove** — make fully context-aware |
| 3 | Conference-specific header patterns in cleaner | **Generalize** — use broad patterns |
| 4 | `embedder.py` as dead code | **Wire up** — connect to FAISS for retrieval |

---

## 4. Architecture Restructuring

The current `stage-X/` folder structure with `sys.path` hacks must be replaced with a proper Python package:

```
backend_py/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── literature.py          # POST /api/literature/review endpoint
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Chains all stages, returns structured output
│   │   ├── pdf_parser.py          # Stage 1 (from stage-1/)
│   │   ├── text_cleaner.py        # Stage 2 (from stage-2/)
│   │   ├── section_detector.py    # Stage 3 (from stage-3/, REWRITTEN)
│   │   ├── chunker.py             # Stage 4 (from stage-4/, IMPROVED)
│   │   ├── embedder.py            # Stage 4 (from stage-4/, WIRED UP)
│   │   ├── summarizer.py          # Stage 5 (from stage-5/)
│   │   ├── insight_extractor.py   # Stage 6 (from stage-6/, CLEANED)
│   │   └── gap_detector.py        # Stage 7 (from stage-7/, CLEANED)
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── semantic_scholar.py    # Semantic Scholar API client
│   │   └── crossref.py            # CrossRef API client
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic response models
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py        # FAISS vector store
│   └── utils/
│       ├── __init__.py
│       ├── export.py              # PDF/Word report export
│       └── logger.py              # Structured logging
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py           # Integration tests
│   └── test_stages.py             # Unit tests per stage
├── data/
│   └── sample_papers/             # Existing sample PDFs
├── requirements.txt
├── .env
└── Dockerfile
```

> [!IMPORTANT]
> The old `stage-X/` directories will be **preserved** as archives but all active code moves into `app/pipeline/`. This is non-destructive.

---

## 5. Implementation Plan — Phased Approach

### Phase 1: P0 — Fix Correctness & Restructure (Do First)

#### 1.1 Restructure project into proper Python package
- Move all stage code into `app/pipeline/`
- Add `__init__.py` files everywhere
- Remove `sys.path` hacks

#### 1.2 Fix `section_detector.py` — REWRITE
- Use regex for numbered headings (`1. Introduction`, `III. METHODS`, `3.1 Experimental Setup`)
- Preserve original text case (don't lowercase everything)
- Add line-break-aware heading detection  
- Add fallback: if no headings found, treat whole text as "full_text"
- Separate "Discussion" from "Conclusion"

#### 1.3 Fix `text_cleaner.py` — GENERALIZE
- Remove hardcoded "Association for Computational Linguistics", "Minneapolis", "June" patterns
- Replace with generic academic header/footer removal (page numbers, running headers, copyright notices)
- Add URL removal, DOI removal, figure/table caption markers

#### 1.4 Fix `chunker.py` — SENTENCE-AWARE
- Use `nltk.sent_tokenize()` for sentence boundary detection
- Group sentences into chunks that don't break mid-sentence
- Maintain overlap at sentence boundaries

#### 1.5 Fix `insight_extractor.py` — REMOVE BERT-SPECIFIC CODE
- Delete `boost_methods()` entirely
- Replace with dynamic keyword extraction from the actual text
- Improve prompt templates for better generalization

#### 1.6 Fix `gap_detector.py` — MAKE CONTEXT-AWARE
- Remove all hardcoded "always useful generic" gaps
- Make rule-based gaps context-aware (only trigger if evidence in text)
- Improve LLM prompting to generate multiple gaps instead of one

#### 1.7 Add error propagation
- Each stage checks input validity
- Empty/failed stages produce warnings, not silent empty outputs
- Add structured logging with Python `logging` module

---

### Phase 2: P1 — Build Core Missing Features

#### 2.1 Build Pipeline Orchestrator (`orchestrator.py`)
- One function: `run_literature_review(pdf_paths, query=None)`
- Chains all stages in sequence
- Returns structured `LiteratureReviewResult` Pydantic model
- Supports single AND multi-paper analysis

#### 2.2 Define Pydantic Schemas (`schemas.py`)
```python
class PaperAnalysis:
    title: str
    sections: dict[str, str]
    summary: str
    section_summaries: dict[str, str]
    insights: InsightResult  # contributions, methods, results
    gaps: list[str]
    evidence_spans: list[EvidenceSpan]

class LiteratureReviewResult:
    papers: list[PaperAnalysis]
    comparison_matrix: Optional[ComparisonMatrix]
    common_themes: list[str]
    research_gaps: list[str]
    related_works: list[RelatedWork]
```

#### 2.3 Build FastAPI Service (`main.py` + `api/literature.py`)
- `POST /api/literature/review` — Upload PDFs, get full analysis
- `GET /api/literature/status/{task_id}` — Check analysis progress
- `GET /api/literature/results/{task_id}` — Get results
- File upload handling with `python-multipart`
- Background task processing for long analyses

#### 2.4 Wire Up Embeddings → FAISS Vector Store
- Store chunk embeddings in FAISS index
- Enable similarity search across papers
- Use for cross-paper theme detection

#### 2.5 Add Semantic Scholar API Integration
- Given paper title/keywords, fetch related papers
- Return: title, authors, year, abstract, citation count, URL
- Free API, no key needed (rate limited)

#### 2.6 Add CrossRef API Integration
- Extract DOIs from paper references
- Fetch metadata for cited works
- Build citation graph

#### 2.7 Add Multi-Paper Comparison
- Compare 2-3 papers side by side
- Generate comparison matrix (method × paper)
- Find common themes and differences
- Find gaps across the literature (not just single papers)

#### 2.8 Integrate IndicBERT for Multilingual NLU & Embeddings
- Replace `all-MiniLM-L6-v2` with `ai4bharat/indic-bert` (or `ai4bharat/IndicBERTv2-MLM-only`)
- **Explicitly documented** — proposal says IndicBERT for multilingual understanding
- Supports Telugu, Hindi, Urdu, Sanskrit, English embeddings
- Use for FAISS vector store cross-lingual retrieval
- Fallback: `paraphrase-multilingual-MiniLM-L12-v2` if IndicBERT too heavy locally

#### 2.9 Integrate IndicBART for Multilingual Summarization
- Replace `sshleifer/distilbart-cnn-12-6` with `ai4bharat/IndicBART`
- **Explicitly documented** — proposal says IndicBART for summarization
- Enables summaries in Indic languages
- Add language param: `summarize(text, output_language="en")`
- Fallback: keep distilbart as English-only option

#### 2.10 Add Translation Output Layer
- User selects output language (English, Telugu, Hindi, etc.)
- Use IndicBART seq2seq for translation
- Return all outputs (summaries, insights, gaps) in selected language

---

### Phase 3: P2 — Build Frontend & Export

#### 3.1 Build React Frontend
- **Upload Page**: Drag-and-drop PDF upload (1-3 papers)
- **Analysis Dashboard**: Show structured results
  - Paper summaries with section tabs
  - Insights visualization (contributions, methods, results)
  - Research gaps with evidence
  - Related works list with links
  - Comparison matrix table (if multi-paper)
- **Export Button**: Download PDF/Word report

#### 3.2 Add Report Export (`utils/export.py`)
- Generate PDF report using `reportlab` or `fpdf2`
- Include: summaries, insights, gaps, comparison matrix, related works
- Formatted for academic readability

#### 3.3 Build Node.js Backend (if needed)
- OR: Keep the Python FastAPI as the only backend
- Node backend may serve as API gateway / auth layer if needed

---

### Phase 4: P3 — Polish & Creative Upgrades

#### 4.1 Add Confidence Scores
- Score each insight/gap based on model logits or heuristic overlap

#### 4.2 Creative Features
- **Research Gap Map**: Visual matrix of Domain × Method × Dataset gaps
- **Contradiction Lens**: Show where papers disagree
- **Reviewer Mode**: Export one-page literature brief

---

## 6. What We Will Build NOW (Execution Order)

> [!IMPORTANT]  
> We will execute **Phase 1 and Phase 2** completely, then build the **frontend (Phase 3)**, and add **polish (Phase 4)** if time allows.

### Execution Checklist:
1. ☐ Restructure `backend_py/` into proper package under `app/`
2. ☐ Rewrite `section_detector.py` with regex-based heading detection
3. ☐ Generalize `text_cleaner.py` (remove paper-specific patterns)
4. ☐ Upgrade `chunker.py` to sentence-aware chunking
5. ☐ Clean `insight_extractor.py` (remove `boost_methods()`)
6. ☐ Clean `gap_detector.py` (remove hardcoded generic gaps)
7. ☐ Add structured logging and error propagation
8. ☐ Create Pydantic schemas
9. ☐ Build pipeline orchestrator
10. ☐ Build FastAPI service with file upload
11. ☐ Wire up FAISS vector store
12. ☐ Add Semantic Scholar API integration
13. ☐ Add CrossRef API integration
14. ☐ Add multi-paper comparison
15. ☐ Integrate IndicBERT for multilingual embeddings
16. ☐ Integrate IndicBART for multilingual summarization
17. ☐ Add translation output layer
18. ☐ Build report export (PDF)
19. ☐ Build React frontend
20. ☐ Create `requirements.txt`
21. ☐ Write integration tests
22. ☐ End-to-end test with sample papers

---

## 7. Verification Plan

### Automated Tests
- Run pipeline on BERT paper → verify structured output with all fields populated
- Run pipeline on Attention paper → verify NO BERT-specific outputs appear
- Run pipeline on 2 papers → verify comparison matrix generated
- Test section detection on well-structured and poorly-structured papers
- Test API endpoints with curl/httpx

### Manual Verification
- Upload 3 sample papers via frontend → get structured review
- Verify related works from Semantic Scholar appear
- Export report and verify formatting
- Run with non-NLP paper to verify no bias
