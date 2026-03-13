"""
Core data models for the fact-checking agent.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CredibilityLevel(str, Enum):
    VERIFIED = "VERIFIED"           # High confidence — true
    LIKELY_TRUE = "LIKELY_TRUE"     # Mostly supported, minor uncertainties
    UNVERIFIED = "UNVERIFIED"       # Insufficient evidence either way
    MISLEADING = "MISLEADING"       # Contains truth but framing is deceptive
    LIKELY_FALSE = "LIKELY_FALSE"   # Mostly contradicted by evidence
    FALSE = "FALSE"                 # Definitively debunked
    SATIRE = "SATIRE"               # Satirical content misidentified as news
    OUTDATED = "OUTDATED"           # Was true but no longer accurate


class SourceCategory(str, Enum):
    FACT_CHECKER = "fact_checker"
    WIRE_SERVICE = "wire_service"
    INTERNATIONAL = "international"
    NATIONAL = "national"
    HEALTH_AUTHORITY = "health_authority"
    RESEARCH = "research"
    GOVERNMENT = "government"
    SOCIAL_MEDIA = "social_media"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Source / Evidence models
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    name: str
    url: Optional[str] = None
    credibility_score: float = Field(0.5, ge=0.0, le=1.0)
    category: SourceCategory = SourceCategory.UNKNOWN
    published_at: Optional[datetime] = None


class Evidence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    source: SourceInfo
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    supports_claim: Optional[bool] = None  # True=supports, False=refutes, None=neutral
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Claim / Verification models
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    context: Optional[str] = None


class VerificationResult(BaseModel):
    claim_id: UUID
    claim_text: str
    credibility_level: CredibilityLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    summary: str
    detailed_analysis: str
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    neutral_evidence: list[Evidence] = Field(default_factory=list)
    sources_checked: int = 0
    warnings: list[str] = Field(default_factory=list)
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=10, max_length=2000, description="The claim to verify")
    context: Optional[str] = Field(None, max_length=500, description="Optional context")
    source_url: Optional[str] = Field(None, description="URL where claim was found")
    source_name: Optional[str] = Field(None, description="Publication/person making the claim")
    fast_mode: bool = Field(False, description="Faster but less thorough verification")


class VerifyResponse(BaseModel):
    success: bool = True
    result: VerificationResult


class BatchVerifyRequest(BaseModel):
    claims: list[VerifyRequest] = Field(..., min_length=1, max_length=10)


class BatchVerifyResponse(BaseModel):
    success: bool = True
    results: list[VerificationResult]
    total_latency_ms: float


class FeedItem(BaseModel):
    title: str
    url: str
    published_at: Optional[datetime]
    source_name: str
    summary: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    index_size: int
    uptime_seconds: float
    version: str = "1.0.0"
