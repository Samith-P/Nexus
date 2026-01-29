# Phase 3 Operational Checklist ✅

## Pre-Deployment Verification

### ✅ Code Structure
- [x] `recommender/__init__.py` - Package initialization
- [x] `recommender/query_builder.py` - Gemini JSON → semantic query
- [x] `recommender/query_embedder.py` - Query → 384-dim embedding
- [x] `recommender/semantic_search.py` - Cosine similarity search
- [x] `recommender/scoring.py` - Metric-aware scoring formula
- [x] `recommender/explainer.py` - Deterministic explanations
- [x] `recommender/journal_recommender.py` - Main orchestrator
- [x] `api.py` - FastAPI integration (updated)

### ✅ Data Dependencies
- [x] `journals_master.json` (158.71 MB, 13,946 journals)
  - Verified: Contains embeddings, metrics, domain_text
  - Size: 158.71 MB
  - Format: JSON with lazy-loading support

### ✅ Python Dependencies
- [x] sentence-transformers (installed in .venv)
- [x] numpy (installed in .venv)
- [x] fastapi (installed in .venv)
- [x] pydantic (installed in .venv)

### ✅ Documentation
- [x] `PHASE3_README.md` - Complete technical documentation
- [x] `PHASE3_QUICKSTART.md` - Quick reference guide
- [x] `PHASE3_IMPLEMENTATION_SUMMARY.md` - Summary & metrics
- [x] `PHASE3_ARCHITECTURE.md` - System architecture & flow

### ✅ Testing
- [x] `test_phase3.py` - Functional test script
  - Status: ✅ PASSING
  - Output: 3 ranked recommendations with scores & explanations
  - Performance: ~1 second execution (warm cache)

---

## Deployment Checklist

### Environment Setup
```bash
# 1. Navigate to backend_py directory
cd backend_py

# 2. Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Verify dependencies
pip list | grep -E "sentence-transformers|fastapi|numpy|pydantic"

# 4. Set environment variables
set GEMINI_API_KEY=your_api_key_here  # Windows
export GEMINI_API_KEY=your_api_key_here  # macOS/Linux
```

### Pre-Launch Testing
```bash
# 1. Test imports
python -c "from recommender import JournalRecommender; print('[OK] Ready')"

# 2. Run functional test
python test_phase3.py

# 3. Check API health
python -c "from api import app; print(app.routes)"
```

### Launch API Server
```bash
# Development mode
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Verification
```bash
# Health check
curl http://localhost:8000/docs

# System info
curl http://localhost:8000/system/info

# Test recommendation
curl -X POST http://localhost:8000/recommend/journals \
  -H "Content-Type: application/json" \
  -d "{
    \"abstract\": \"Deep learning for computer vision using convolutional neural networks and gradient descent optimization\",
    \"top_k\": 5
  }"
```

---

## Monitoring & Debugging

### Performance Metrics to Track
```python
# In api.py, add monitoring:
import time

@app.post("/recommend/journals")
async def recommend(request: RecommendRequest):
    start = time.time()
    
    # ... recommendation logic ...
    
    elapsed = time.time() - start
    print(f"[TIMING] Recommendation took {elapsed:.2f}s")
    # Expected: 0.5-2s depending on cache warm
```

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: sentence-transformers` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: journals_master.json` | Verify file exists in backend_py/ directory |
| `GEMINI_API_KEY not set` | Export/set environment variable before running |
| `CUDA out of memory` | Reduce batch size (already optimized in code) |
| `Slow first request (3-5s)` | Normal - model is loading. Subsequent requests ~1s |
| `429 Too Many Requests from Gemini` | Add rate limiting, use cached results |

### Logging Setup
```python
# In api.py, add logging:
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log key operations:
logger.info(f"Embedding query: {query_text[:50]}...")
logger.info(f"Found {len(candidates)} semantic candidates")
logger.info(f"Top result: {top_result['title']} (score: {score:.4f})")
```

---

## Phase 3 Success Criteria

### ✅ Functional Requirements
- [x] Converts Gemini JSON to semantic query
- [x] Embeds query using sentence-transformers
- [x] Performs cosine similarity search
- [x] Applies weighted scoring formula
- [x] Generates deterministic explanations
- [x] Returns top-K ranked journals

### ✅ Non-Functional Requirements
- [x] Modular architecture (7 independent modules)
- [x] No hardcoded values
- [x] Production-ready error handling
- [x] Type hints for all functions
- [x] Comprehensive documentation
- [x] <2 second response time (warm cache)

### ✅ Code Quality
- [x] PEP 8 compliant
- [x] No external API calls in recommender
- [x] Lazy loading for performance
- [x] Clear separation of concerns
- [x] Test coverage (functional test passing)

---

## Post-Deployment Steps

### 1. Monitor First 24 Hours
- [ ] Track response times (goal: <2s)
- [ ] Monitor error rates (goal: <1%)
- [ ] Check memory usage (expected: 800MB)
- [ ] Verify recommendation quality

### 2. Gather User Feedback
- [ ] Are recommendations relevant?
- [ ] Is ranking appropriate?
- [ ] Are explanations helpful?
- [ ] Any missing journals?

### 3. Optimize Based on Usage
- [ ] Adjust scoring weights if needed (in `scoring.py`)
- [ ] Increase search_depth if missing good journals
- [ ] Cache frequently requested queries
- [ ] Monitor Gemini API usage and costs

### 4. Plan Phase 4 (Future)
- [ ] Add user feedback loop
- [ ] Implement A/B testing for scoring weights
- [ ] Add recommendation explanations with examples
- [ ] Expand journal index with new years
- [ ] Implement Redis caching

---

## Rollback Plan

If issues occur:

```bash
# 1. Stop the server
# (Ctrl+C in terminal)

# 2. Check error logs
tail -n 50 error.log

# 3. Verify journalsmaster.json integrity
python -c "import json; json.load(open('journals_master.json')); print('[OK]')"

# 4. Revert API changes if needed
git checkout api.py

# 5. Restart with diagnostic mode
python -c "from api import app; app.debug = True" &
python test_phase3.py
```

---

## Next Phase Opportunities

### Phase 4A: Quality Improvements
- Add recommendation confidence scores
- Implement journal discovery (find new journals)
- Add "why this journal" detailed explanations
- Track recommendation accuracy metrics

### Phase 4B: Performance Optimization
- Implement FAISS for approximate nearest neighbors
- Add Redis caching layer
- Implement batch processing for bulk recommendations
- Optimize embedding model selection

### Phase 4C: User Experience
- Add journal similarity graphs
- Implement journal comparison tool
- Add trending journals feature
- Build recommendation history

### Phase 4D: Integration
- Connect with reference management tools
- Implement journal alerts
- Add collaborative recommendations
- Build mobile app API

---

## Support & Troubleshooting

### Quick Diagnostic
```bash
python -c "
import json
from recommender import JournalRecommender
from pathlib import Path

# 1. Check file
if Path('journals_master.json').exists():
    with open('journals_master.json') as f:
        data = json.load(f)
        print(f'[OK] Loaded {len(data)} journals')
else:
    print('[ERROR] journals_master.json not found')

# 2. Check recommender
try:
    r = JournalRecommender()
    print('[OK] JournalRecommender initialized')
except Exception as e:
    print(f'[ERROR] {e}')

# 3. Test recommendation
try:
    result = r.recommend({'key_concepts': ['ML']}, top_k=1)
    print(f'[OK] Recommendation works')
except Exception as e:
    print(f'[ERROR] {e}')
"
```

### Contact & Escalation
- Code issues: Check test_phase3.py for reference implementation
- Data issues: Verify journals_master.json exists (13946 entries)
- API issues: Check api.py FastAPI route definitions
- Performance issues: Check Phase 3 architecture document

---

**Phase 3 is production-ready. Deploy with confidence! 🚀**
