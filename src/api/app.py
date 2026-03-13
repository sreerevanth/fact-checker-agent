"""
FastAPI Application
===================
Exposes the fact-checking agent as a REST API with:
  POST /verify          — single claim verification
  POST /verify/batch    — batch verification (up to 10 claims)
  GET  /health          — service health check
  GET  /metrics         — Prometheus metrics (if enabled)
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from src.agent.fact_checker import FactCheckAgent
from src.rag.retriever import FAISSRetriever
from src.scrapers.news_scraper import LiveFeedMonitor, RSSIngester, SourceRegistry
from src.utils.cache import cache_result, get_cached_result
from src.utils.logging import configure_logging, get_logger
from src.utils.models import (
    BatchVerifyRequest,
    BatchVerifyResponse,
    Claim,
    HealthResponse,
    VerifyRequest,
    VerifyResponse,
)
from src.utils.settings import get_settings

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
log = get_logger(__name__)

# Prometheus metrics
_requests_total = Counter("factcheck_requests_total", "Total verification requests", ["status"])
_latency = Histogram("factcheck_latency_seconds", "Verification latency", buckets=[0.5, 1, 2, 5, 10, 30])

# App-level singletons (set during lifespan)
_retriever: FAISSRetriever | None = None
_agent: FactCheckAgent | None = None
_monitor: LiveFeedMonitor | None = None
_monitor_task: asyncio.Task | None = None
_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _retriever, _agent, _monitor, _monitor_task

    log.info("Starting Fact-Checking Agent service")

    # Initialize RAG retriever
    _retriever = FAISSRetriever()
    await _retriever.initialize()

    # Initialize source registry
    registry = SourceRegistry(str(settings.trusted_sources_config))

    # Seed index with an initial RSS fetch
    ingester = RSSIngester(registry)
    try:
        initial_docs = await ingester.fetch_all()
        if initial_docs:
            added = await _retriever.add_documents(initial_docs)
            log.info("Initial index seeded", docs=added)
    except Exception as exc:
        log.warning("Initial seeding failed", error=str(exc))

    # Create agent
    _agent = FactCheckAgent(_retriever, registry)

    # Start background feed monitor
    _monitor = LiveFeedMonitor(_retriever, registry)
    _monitor_task = asyncio.create_task(_monitor.start())

    log.info("Service ready", index_size=_retriever.index_size)
    yield

    # Shutdown
    log.info("Shutting down service")
    if _monitor:
        _monitor.stop()
    if _monitor_task:
        _monitor_task.cancel()
    if _retriever:
        await _retriever.save()


app = FastAPI(
    title="Real-Time Fact-Checking Agent",
    description="RAG-powered misinformation detection using Claude AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_agent() -> FactCheckAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.primary_model,
        index_size=_retriever.index_size if _retriever else 0,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", tags=["system"])
async def metrics() -> PlainTextResponse:
    if not settings.enable_metrics:
        raise HTTPException(status_code=404)
    return PlainTextResponse(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/verify", response_model=VerifyResponse, tags=["fact-check"])
async def verify_claim(body: VerifyRequest, background_tasks: BackgroundTasks) -> VerifyResponse:
    agent = _get_agent()

    # Check cache
    cached = await get_cached_result(body.claim)
    if cached:
        log.info("Cache hit", claim_preview=body.claim[:60])
        from src.utils.models import VerificationResult
        result = VerificationResult.model_validate_json(cached)
        return VerifyResponse(result=result)

    claim = Claim(
        text=body.claim,
        context=body.context,
        source_url=body.source_url,
        source_name=body.source_name,
    )

    t0 = time.perf_counter()
    try:
        result = await agent.verify(claim, fast=body.fast_mode)
        _requests_total.labels(status="success").inc()
        _latency.observe((time.perf_counter() - t0))
    except Exception as exc:
        _requests_total.labels(status="error").inc()
        log.error("Verification failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    # Cache in background
    background_tasks.add_task(cache_result, body.claim, result.model_dump_json())

    return VerifyResponse(result=result)


@app.post("/verify/batch", response_model=BatchVerifyResponse, tags=["fact-check"])
async def verify_batch(body: BatchVerifyRequest) -> BatchVerifyResponse:
    agent = _get_agent()
    t0 = time.perf_counter()

    async def _one(req: VerifyRequest):
        cached = await get_cached_result(req.claim)
        if cached:
            from src.utils.models import VerificationResult
            return VerificationResult.model_validate_json(cached)
        claim = Claim(text=req.claim, context=req.context, source_url=req.source_url)
        return await agent.verify(claim, fast=req.fast_mode)

    try:
        results = await asyncio.gather(*[_one(r) for r in body.claims], return_exceptions=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    valid = []
    for r in results:
        if isinstance(r, Exception):
            log.error("Batch item failed", error=str(r))
        else:
            valid.append(r)

    return BatchVerifyResponse(
        results=valid,
        total_latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
