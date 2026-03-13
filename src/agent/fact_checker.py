"""
Fact-Checking Agent
====================
Orchestrates the full RAG → LLM reasoning pipeline:
  1. Decompose claim into search sub-queries
  2. Retrieve evidence from vector index + live web search
  3. Classify evidence (supports / contradicts / neutral)
  4. Generate structured credibility assessment via Claude
  5. Return VerificationResult
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID

import anthropic

from src.rag.retriever import FAISSRetriever
from src.scrapers.news_scraper import NewsAPIClient, SourceRegistry
from src.utils.logging import get_logger
from src.utils.models import (
    Claim,
    CredibilityLevel,
    Evidence,
    VerificationResult,
)
from src.utils.settings import get_settings

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are a fact-checking assistant. Given a claim, generate 3-5 concise search queries
to retrieve evidence that either supports or contradicts it.

Claim: {claim}

Return ONLY a JSON array of strings (the queries), nothing else.
Example: ["query 1", "query 2", "query 3"]
"""

CLASSIFY_EVIDENCE_PROMPT = """\
You are a rigorous fact-checker. Given a claim and a piece of evidence text, classify
whether the evidence SUPPORTS, CONTRADICTS, or is NEUTRAL toward the claim.

Claim: {claim}

Evidence: {evidence}

Reply with exactly one word: SUPPORTS, CONTRADICTS, or NEUTRAL.
"""

ASSESSMENT_PROMPT = """\
You are an expert fact-checker and investigative journalist. Analyze the following claim
using the provided evidence and produce a thorough credibility assessment.

## Claim
{claim}

## Context
{context}

## Supporting Evidence ({n_support} items)
{supporting}

## Contradicting Evidence ({n_contradict} items)
{contradicting}

## Neutral/Background Evidence ({n_neutral} items)
{neutral}

## Source Credibility Notes
{source_notes}

---

Produce a JSON object with EXACTLY these fields:
{{
  "credibility_level": one of [VERIFIED, LIKELY_TRUE, UNVERIFIED, MISLEADING, LIKELY_FALSE, FALSE, SATIRE, OUTDATED],
  "confidence_score": float between 0.0 and 1.0,
  "summary": "2-3 sentence plain-English verdict for a general audience",
  "detailed_analysis": "Thorough analysis (4-8 sentences): what the evidence shows, source quality, any red flags, logical consistency, what remains uncertain",
  "warnings": ["list of specific concerns or caveats, empty list if none"]
}}

Reasoning guidelines:
- Weight high-credibility sources (fact-checkers, wire services) more heavily
- If evidence directly contradicts from multiple independent sources → FALSE or LIKELY_FALSE
- If only one source contradicts → MISLEADING or LIKELY_FALSE
- If evidence is sparse → UNVERIFIED
- Distinguish between outdated facts (OUTDATED) and never-true claims (FALSE)
- Be conservative: do not call something VERIFIED unless evidence is strong and multi-sourced
- Flag logical fallacies, misleading framing, and missing context in warnings

Return ONLY the JSON object, no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FactCheckAgent:
    """Main orchestrating agent."""

    def __init__(
        self,
        retriever: FAISSRetriever,
        registry: SourceRegistry,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever
        self.registry = registry
        self._client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        self._news_api = NewsAPIClient(self.settings.newsapi_key, registry)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def verify(self, claim: Claim, fast: bool = False) -> VerificationResult:
        t0 = time.perf_counter()
        log.info("Starting verification", claim_id=str(claim.id), fast=fast)

        # Step 1: Decompose claim into search queries
        queries = await self._decompose_claim(claim.text)
        log.debug("Generated search queries", queries=queries)

        # Step 2: Retrieve evidence (parallel)
        evidence_lists = await asyncio.gather(
            *[self.retriever.retrieve(q, top_k=5) for q in queries],
            return_exceptions=True,
        )
        raw_evidence: list[Evidence] = []
        for ev_list in evidence_lists:
            if isinstance(ev_list, list):
                raw_evidence.extend(ev_list)

        # Optionally fetch live results from NewsAPI
        if not fast and self.settings.newsapi_key:
            api_docs = await self._live_search(claim.text)
            if api_docs:
                await self.retriever.add_documents(api_docs)
                live_ev = await self.retriever.retrieve(claim.text, top_k=5)
                raw_evidence.extend(live_ev)

        # Deduplicate by text hash
        seen: set[str] = set()
        unique_evidence: list[Evidence] = []
        for ev in raw_evidence:
            h = hash(ev.text[:100])
            if h not in seen:
                seen.add(h)
                unique_evidence.append(ev)

        # Step 3: Classify evidence
        if not fast:
            unique_evidence = await self._classify_evidence(claim.text, unique_evidence)
        else:
            # Fast mode: heuristic classification
            unique_evidence = self._heuristic_classify(claim.text, unique_evidence)

        supporting = [e for e in unique_evidence if e.supports_claim is True]
        contradicting = [e for e in unique_evidence if e.supports_claim is False]
        neutral = [e for e in unique_evidence if e.supports_claim is None]

        # Step 4: LLM assessment
        result = await self._generate_assessment(claim, supporting, contradicting, neutral)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.latency_ms = round(elapsed_ms, 1)
        result.sources_checked = len(unique_evidence)

        log.info(
            "Verification complete",
            claim_id=str(claim.id),
            verdict=result.credibility_level,
            confidence=result.confidence_score,
            latency_ms=result.latency_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Step 1: Decompose
    # ------------------------------------------------------------------

    async def _decompose_claim(self, claim_text: str) -> list[str]:
        try:
            response = await self._client.messages.create(
                model=self.settings.primary_model,
                max_tokens=256,
                temperature=0.2,
                messages=[{
                    "role": "user",
                    "content": DECOMPOSE_PROMPT.format(claim=claim_text),
                }],
            )
            import json
            text = response.content[0].text.strip()
            queries = json.loads(text)
            if isinstance(queries, list):
                return [str(q) for q in queries[:5]]
        except Exception as exc:
            log.warning("Decompose failed, using claim directly", error=str(exc))
        return [claim_text]

    # ------------------------------------------------------------------
    # Step 2: Live search
    # ------------------------------------------------------------------

    async def _live_search(self, query: str) -> list[dict]:
        try:
            return await self._news_api.search(query, page_size=5)
        except Exception as exc:
            log.warning("Live search failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Step 3: Classify
    # ------------------------------------------------------------------

    async def _classify_evidence(
        self, claim_text: str, evidence: list[Evidence]
    ) -> list[Evidence]:
        """Ask the LLM to classify each piece of evidence (batched)."""
        if not evidence:
            return evidence

        # Run classifications concurrently with a semaphore to avoid rate limits
        sem = asyncio.Semaphore(5)

        async def classify_one(ev: Evidence) -> Evidence:
            async with sem:
                try:
                    response = await self._client.messages.create(
                        model=self.settings.primary_model,
                        max_tokens=10,
                        temperature=0.0,
                        messages=[{
                            "role": "user",
                            "content": CLASSIFY_EVIDENCE_PROMPT.format(
                                claim=claim_text,
                                evidence=ev.text[:500],
                            ),
                        }],
                    )
                    label = response.content[0].text.strip().upper()
                    if label == "SUPPORTS":
                        ev.supports_claim = True
                    elif label == "CONTRADICTS":
                        ev.supports_claim = False
                    else:
                        ev.supports_claim = None
                except Exception as exc:
                    log.debug("Evidence classification failed", error=str(exc))
            return ev

        return list(await asyncio.gather(*[classify_one(e) for e in evidence]))

    def _heuristic_classify(
        self, claim_text: str, evidence: list[Evidence]
    ) -> list[Evidence]:
        """Fast keyword-overlap heuristic for fast_mode."""
        claim_words = set(claim_text.lower().split())
        for ev in evidence:
            text_lower = ev.text.lower()
            # Very simple signal: presence of negation words near claim keywords
            has_negation = any(w in text_lower for w in ["not", "false", "wrong", "debunked", "denied", "hoax"])
            overlap = len(claim_words & set(text_lower.split())) / (len(claim_words) + 1)
            if overlap > 0.3 and not has_negation:
                ev.supports_claim = True
            elif has_negation and overlap > 0.2:
                ev.supports_claim = False
        return evidence

    # ------------------------------------------------------------------
    # Step 4: Assessment
    # ------------------------------------------------------------------

    def _format_evidence_block(self, evidence: list[Evidence], limit: int = 5) -> str:
        if not evidence:
            return "(none)"
        lines = []
        for i, ev in enumerate(evidence[:limit], 1):
            src = f"{ev.source.name} [credibility: {ev.source.credibility_score:.2f}]"
            if ev.source.url:
                src += f" — {ev.source.url}"
            lines.append(f"{i}. [{src}]\n   {ev.text[:300]}")
        return "\n\n".join(lines)

    def _source_notes(self, evidence: list[Evidence]) -> str:
        high = [e for e in evidence if e.source.credibility_score >= 0.85]
        low = [e for e in evidence if e.source.credibility_score < 0.5]
        notes = []
        if high:
            names = ", ".join(set(e.source.name for e in high[:5]))
            notes.append(f"High-credibility sources present: {names}")
        if low:
            names = ", ".join(set(e.source.name for e in low[:3]))
            notes.append(f"Low-credibility sources present: {names}")
        if not notes:
            notes.append("Mixed or unknown source credibility")
        return " | ".join(notes)

    async def _generate_assessment(
        self,
        claim: Claim,
        supporting: list[Evidence],
        contradicting: list[Evidence],
        neutral: list[Evidence],
    ) -> VerificationResult:
        import json as _json

        prompt = ASSESSMENT_PROMPT.format(
            claim=claim.text,
            context=claim.context or "No additional context provided.",
            n_support=len(supporting),
            supporting=self._format_evidence_block(supporting),
            n_contradict=len(contradicting),
            contradicting=self._format_evidence_block(contradicting),
            n_neutral=len(neutral),
            neutral=self._format_evidence_block(neutral, limit=3),
            source_notes=self._source_notes(supporting + contradicting + neutral),
        )

        try:
            response = await self._client.messages.create(
                model=self.settings.primary_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = _json.loads(raw)
        except Exception as exc:
            log.error("Assessment LLM call failed", error=str(exc))
            data = {
                "credibility_level": "UNVERIFIED",
                "confidence_score": 0.0,
                "summary": "Assessment failed due to an internal error.",
                "detailed_analysis": str(exc),
                "warnings": ["Internal error during assessment"],
            }

        return VerificationResult(
            claim_id=claim.id,
            claim_text=claim.text,
            credibility_level=CredibilityLevel(data.get("credibility_level", "UNVERIFIED")),
            confidence_score=float(data.get("confidence_score", 0.0)),
            summary=data.get("summary", ""),
            detailed_analysis=data.get("detailed_analysis", ""),
            supporting_evidence=supporting[:5],
            contradicting_evidence=contradicting[:5],
            neutral_evidence=neutral[:3],
            warnings=data.get("warnings", []),
        )
