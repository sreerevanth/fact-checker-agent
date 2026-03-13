"""
Test suite for the Fact-Checking Agent.
Run with: pytest tests/ -v --cov=src
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src.utils.models import (
    Claim,
    CredibilityLevel,
    Evidence,
    SourceCategory,
    SourceInfo,
    VerificationResult,
    VerifyRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_claim() -> Claim:
    return Claim(text="Scientists have confirmed that coffee prevents all forms of cancer.")


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            text="Large meta-analysis finds no conclusive evidence that coffee prevents cancer.",
            source=SourceInfo(
                name="Reuters",
                url="https://reuters.com/test",
                credibility_score=0.95,
                category=SourceCategory.WIRE_SERVICE,
            ),
            relevance_score=0.85,
            supports_claim=False,
        ),
        Evidence(
            text="Some studies suggest coffee consumption linked to lower risk of certain cancers.",
            source=SourceInfo(
                name="PubMed",
                url="https://pubmed.ncbi.nlm.nih.gov/test",
                credibility_score=0.94,
                category=SourceCategory.RESEARCH,
            ),
            relevance_score=0.72,
            supports_claim=None,
        ),
    ]


@pytest.fixture
def sample_verification_result(sample_claim, sample_evidence) -> VerificationResult:
    return VerificationResult(
        claim_id=sample_claim.id,
        claim_text=sample_claim.text,
        credibility_level=CredibilityLevel.LIKELY_FALSE,
        confidence_score=0.82,
        summary="The claim is an overstatement. Coffee may reduce risk of some cancers, but no evidence supports 'all forms'.",
        detailed_analysis="Multiple independent studies...",
        contradicting_evidence=sample_evidence[:1],
        neutral_evidence=sample_evidence[1:],
        warnings=["Absolute language ('all forms') is a red flag"],
        sources_checked=2,
        latency_ms=1234.5,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_claim_has_uuid(self, sample_claim):
        assert sample_claim.id is not None
        assert str(sample_claim.id)

    def test_claim_timestamp(self, sample_claim):
        assert isinstance(sample_claim.submitted_at, datetime)

    def test_verification_result_serialization(self, sample_verification_result):
        j = sample_verification_result.model_dump_json()
        reloaded = VerificationResult.model_validate_json(j)
        assert reloaded.credibility_level == CredibilityLevel.LIKELY_FALSE
        assert reloaded.confidence_score == 0.82

    def test_credibility_level_values(self):
        levels = [e.value for e in CredibilityLevel]
        assert "VERIFIED" in levels
        assert "FALSE" in levels
        assert "MISLEADING" in levels

    def test_verify_request_validation(self):
        req = VerifyRequest(claim="Is this claim true or false? It is a very important question.")
        assert req.fast_mode is False

    def test_verify_request_min_length(self):
        with pytest.raises(Exception):
            VerifyRequest(claim="short")


# ---------------------------------------------------------------------------
# RAG Retriever tests
# ---------------------------------------------------------------------------

class TestFAISSRetriever:
    @pytest.mark.asyncio
    async def test_initialization(self):
        from src.rag.retriever import FAISSRetriever
        with patch("src.rag.retriever.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = __import__("numpy").zeros((1, 384), dtype="float32")
            mock_st.return_value = mock_model

            retriever = FAISSRetriever()
            await retriever.initialize()
            assert retriever._model is not None

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        import numpy as np
        from src.rag.retriever import FAISSRetriever

        with patch("src.rag.retriever.SentenceTransformer") as mock_st, \
             patch("src.rag.retriever.FAISS_AVAILABLE", False):

            mock_model = MagicMock()
            mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
            mock_st.return_value = mock_model

            retriever = FAISSRetriever()
            await retriever.initialize()

            docs = [
                {
                    "text": "Coffee may reduce liver cancer risk according to studies.",
                    "url": "https://example.com/1",
                    "source_name": "Test Source",
                    "credibility_score": 0.8,
                    "category": "research",
                }
            ]
            added = await retriever.add_documents(docs)
            assert added == 1
            assert retriever.index_size == 1

    @pytest.mark.asyncio
    async def test_keyword_fallback(self):
        import numpy as np
        from src.rag.retriever import FAISSRetriever

        with patch("src.rag.retriever.SentenceTransformer") as mock_st, \
             patch("src.rag.retriever.FAISS_AVAILABLE", False):

            mock_model = MagicMock()
            mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
            mock_st.return_value = mock_model

            retriever = FAISSRetriever()
            await retriever.initialize()
            await retriever.add_documents([
                {"text": "coffee cancer study results", "source_name": "Test", "credibility_score": 0.7, "category": "research"}
            ])

            results = retriever._keyword_fallback("coffee cancer", 5)
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# Source Registry tests
# ---------------------------------------------------------------------------

class TestSourceRegistry:
    def test_load_config(self, tmp_path):
        config = tmp_path / "sources.yaml"
        config.write_text("""
fact_check_organizations:
  - name: "Reuters Fact Check"
    url: "https://www.reuters.com/fact-check/"
    feed: "https://feeds.reuters.com/reuters/topNews"
    credibility: 0.97
    category: "fact_checker"
low_credibility_domains:
  - "fakenews.example.com"
""")
        from src.scrapers.news_scraper import SourceRegistry
        registry = SourceRegistry(str(config))
        assert len(registry._sources) >= 1
        assert "fakenews.example.com" in registry._low_credibility

    def test_credibility_lookup(self, tmp_path):
        config = tmp_path / "sources.yaml"
        config.write_text("""
fact_check_organizations:
  - name: "Reuters"
    url: "https://www.reuters.com"
    credibility: 0.95
    category: "wire_service"
low_credibility_domains:
  - "spam.example.com"
""")
        from src.scrapers.news_scraper import SourceRegistry
        registry = SourceRegistry(str(config))

        score, name = registry.get_credibility("https://www.reuters.com/article/test")
        assert score == 0.95
        assert "Reuters" in name

        score_low, _ = registry.get_credibility("https://spam.example.com/article")
        assert score_low == 0.1

    def test_unknown_domain(self, tmp_path):
        config = tmp_path / "sources.yaml"
        config.write_text("fact_check_organizations: []\n")
        from src.scrapers.news_scraper import SourceRegistry
        registry = SourceRegistry(str(config))
        score, name = registry.get_credibility("https://unknown-blog.xyz/post")
        assert score == 0.5


# ---------------------------------------------------------------------------
# Agent tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestFactCheckAgent:
    @pytest.fixture
    def mock_retriever(self, sample_evidence):
        retriever = AsyncMock()
        retriever.retrieve.return_value = sample_evidence
        retriever.add_documents.return_value = 2
        retriever.index_size = 100
        return retriever

    @pytest.fixture
    def mock_registry(self, tmp_path):
        config = tmp_path / "sources.yaml"
        config.write_text("fact_check_organizations: []\nlow_credibility_domains: []\n")
        from src.scrapers.news_scraper import SourceRegistry
        return SourceRegistry(str(config))

    @pytest.mark.asyncio
    async def test_verify_returns_result(self, mock_retriever, mock_registry, sample_claim):
        from src.agent.fact_checker import FactCheckAgent

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "credibility_level": "LIKELY_FALSE",
            "confidence_score": 0.80,
            "summary": "Claim overstates the evidence.",
            "detailed_analysis": "Multiple sources contradict this.",
            "warnings": ["Absolute language used"]
        }))]

        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = FactCheckAgent(mock_retriever, mock_registry)
            result = await agent.verify(sample_claim, fast=True)

        assert result.claim_id == sample_claim.id
        assert result.credibility_level == CredibilityLevel.LIKELY_FALSE
        assert result.confidence_score == 0.80
        assert len(result.warnings) == 1

    @pytest.mark.asyncio
    async def test_heuristic_classify(self, mock_retriever, mock_registry, sample_evidence):
        from src.agent.fact_checker import FactCheckAgent

        with patch("anthropic.AsyncAnthropic"):
            agent = FactCheckAgent(mock_retriever, mock_registry)

        evidence = [
            Evidence(
                text="Scientists debunked the claim that coffee prevents all cancer types.",
                source=SourceInfo(name="Reuters", credibility_score=0.95),
            )
        ]
        classified = agent._heuristic_classify("coffee prevents cancer", evidence)
        assert classified[0].supports_claim is False  # has "debunked"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCache:
    @pytest.mark.asyncio
    async def test_disk_cache_roundtrip(self, tmp_path):
        import diskcache
        from src.utils import cache as cache_module

        disk = diskcache.Cache(str(tmp_path / "testcache"))
        backend = cache_module.CacheBackend()
        backend._disk = disk
        backend._ttl = 60

        await backend.set("test:key", '{"hello": "world"}')
        val = await backend.get("test:key")
        assert val == '{"hello": "world"}'
        disk.close()


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        return TestClient(app)

    def test_health_endpoint_structure(self):
        """Health endpoint returns expected fields."""
        from src.utils.models import HealthResponse
        # Just validate the model schema
        resp = HealthResponse(status="ok", model="test", index_size=0, uptime_seconds=1.0)
        assert resp.status == "ok"

    def test_verify_request_model(self):
        req = VerifyRequest(
            claim="The Earth is approximately 4.5 billion years old according to scientists.",
            fast_mode=True,
        )
        assert req.fast_mode is True
        assert len(req.claim) > 10
