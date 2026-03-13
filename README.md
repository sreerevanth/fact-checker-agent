# 🔍 Real-Time Fact-Checking & News Verification Agent

A production-ready, RAG-powered misinformation detection system using Claude AI. Monitors live news feeds, cross-references claims against a continuously updated vector knowledge base, and delivers instant credibility assessments with source attribution.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                          │
│              POST /verify  |  POST /verify/batch                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │    FactCheckAgent     │
          │  (Orchestrator)       │
          └──┬──────────┬─────────┘
             │          │
   ┌──────────▼──┐  ┌───▼──────────────┐
   │ FAISSRetriever│  │  Claude (Sonnet) │
   │ (RAG Engine)  │  │  - Decompose     │
   │               │  │  - Classify      │
   │  384-dim      │  │  - Assess        │
   │  embeddings   │  └──────────────────┘
   └──────┬────────┘
          │
   ┌──────▼────────────────────┐
   │   LiveFeedMonitor          │
   │   (Background Task)        │
   │   ┌────────┐ ┌──────────┐  │
   │   │RSS/Atom│ │ NewsAPI  │  │
   │   │ Feeds  │ │ Client   │  │
   │   └────────┘ └──────────┘  │
   └────────────────────────────┘
          │
   ┌──────▼────────┐
   │ Redis Cache   │  ← Results cached by claim hash (SHA-256)
   └───────────────┘
```

### Key Design Decisions

| Concern | Solution |
|---|---|
| **Semantic retrieval** | FAISS IndexFlatIP with normalized embeddings (cosine similarity) |
| **Embedding model** | `all-MiniLM-L6-v2` — local, 384-dim, fast (no API cost) |
| **LLM reasoning** | Claude Sonnet 4 — multi-step: decompose → classify → assess |
| **Live ingestion** | Background asyncio task polling RSS feeds every 5 min |
| **Caching** | Redis (SHA-256 claim key) with 1h TTL; diskcache fallback |
| **Concurrency** | `asyncio.gather` + semaphores for parallel retrieval & classification |
| **Observability** | Prometheus metrics + structlog JSON logging |

---

## Quickstart

### Option A — Local (Recommended for development)

```bash
# 1. Clone and enter directory
cd fact-checker

# 2. Run setup (creates venv, installs deps, downloads embedding model)
make setup

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Configure API keys
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY

# 5. Start the server
make serve
# → API running at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### Option B — Docker (Recommended for production)

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here
export NEWSAPI_KEY=your_key_here  # optional

# Start full stack (API + Redis + Prometheus + Grafana)
make docker-up

# Services:
# API:        http://localhost:8000
# API Docs:   http://localhost:8000/docs
# Grafana:    http://localhost:3000  (admin/admin)
# Prometheus: http://localhost:9091
```

---

## API Usage

### Verify a single claim

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Scientists have proven that coffee prevents all forms of cancer.",
    "context": "Seen shared on social media",
    "fast_mode": false
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "credibility_level": "LIKELY_FALSE",
    "confidence_score": 0.85,
    "summary": "The claim overstates the evidence. Some studies suggest coffee may reduce risk for certain cancers, but no research supports the 'all forms' assertion.",
    "detailed_analysis": "Multiple independent sources including peer-reviewed research...",
    "supporting_evidence": [...],
    "contradicting_evidence": [...],
    "warnings": ["Absolute language ('all forms') is a common misinformation pattern"],
    "sources_checked": 12,
    "latency_ms": 3420.5
  }
}
```

### Batch verification

```bash
curl -X POST http://localhost:8000/verify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      {"claim": "The Great Wall of China is visible from space."},
      {"claim": "Einstein failed math as a child.", "fast_mode": true}
    ]
  }'
```

### CLI usage

```bash
# Verify from command line
python -m src.main check "The moon landing was faked by NASA"

# Force feed ingestion
python -m src.main ingest

# Run demo against live server
make demo
```

---

## Credibility Levels

| Level | Meaning |
|---|---|
| `VERIFIED` | Strong multi-source confirmation |
| `LIKELY_TRUE` | Mostly supported, minor gaps |
| `UNVERIFIED` | Insufficient evidence either way |
| `MISLEADING` | Contains truth but framing is deceptive |
| `LIKELY_FALSE` | Mostly contradicted by evidence |
| `FALSE` | Definitively debunked |
| `SATIRE` | Satirical content misidentified as news |
| `OUTDATED` | Was true, but no longer accurate |

---

## Configuration

All settings in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API key |
| `NEWSAPI_KEY` | Optional | Live news search from newsapi.org |
| `REDIS_URL` | Optional | Redis for caching (falls back to diskcache) |
| `PRIMARY_MODEL` | Optional | Default: `claude-sonnet-4-20250514` |
| `RETRIEVAL_TOP_K` | Optional | Documents retrieved per query (default: 10) |
| `CACHE_TTL_SECONDS` | Optional | Result cache lifetime (default: 3600) |

---

## Project Structure

```
fact-checker/
├── src/
│   ├── agent/
│   │   └── fact_checker.py      # Main orchestrating agent (decompose→retrieve→classify→assess)
│   ├── rag/
│   │   └── retriever.py         # FAISS vector index + semantic search
│   ├── scrapers/
│   │   └── news_scraper.py      # RSS ingester, NewsAPI client, LiveFeedMonitor
│   ├── api/
│   │   └── app.py               # FastAPI app (routes, middleware, lifespan)
│   ├── utils/
│   │   ├── models.py            # Pydantic data models
│   │   ├── settings.py          # Pydantic-settings config
│   │   ├── cache.py             # Redis/diskcache layer
│   │   └── logging.py           # structlog configuration
│   └── main.py                  # CLI entry point
├── tests/
│   └── test_agent.py            # pytest suite with mocks
├── config/
│   └── trusted_sources.yaml     # Credibility scores for 20+ sources
├── scripts/
│   ├── setup.sh                 # Local dev setup
│   └── demo.py                  # Demo script
├── docker/
│   └── prometheus.yml           # Prometheus scrape config
├── Dockerfile
├── docker-compose.yml           # Full stack (API + Redis + Prometheus + Grafana)
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
# Full test suite with coverage
make test

# Fast (no coverage)
make test-fast

# Lint
make lint
```

---

## Extending the Agent

### Add a new trusted source
Edit `config/trusted_sources.yaml` — add an entry with `name`, `url`, `feed`, `credibility` (0–1), and `category`.

### Add a new data source (e.g., social media API)
1. Create a new client class in `src/scrapers/`
2. Call `retriever.add_documents(docs)` with the normalized document format
3. Optionally wire it into `LiveFeedMonitor.start()`

### Swap the embedding model
Change `EMBEDDING_MODEL` in `.env` and update `FAISS_DIMENSION` to match the model's output dimension. Delete the existing FAISS index files so a fresh one is built.

### Swap the LLM
Change `PRIMARY_MODEL` in `.env`. The agent uses the Anthropic SDK — to use OpenAI models instead, replace the `anthropic.AsyncAnthropic` client in `fact_checker.py` with `openai.AsyncOpenAI`.

---

## Performance Notes

- **Cold start**: ~60–90s (model download on first run; cached in Docker layer)
- **Warm verification**: 2–8s (depends on evidence retrieval + LLM calls)
- **Fast mode** (`fast_mode: true`): ~1–3s (skips LLM evidence classification, uses heuristics)
- **Cache hit**: <50ms
- **Embedding throughput**: ~500 docs/sec on CPU (MiniLM-L6-v2)

---

## License

MIT
